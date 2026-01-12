// flowfieldgenerator.cpp

#include "flowfieldgenerator.h"
#include "parameterparser.h"
#include "fluentflowimporter.h"

#include <H5Cpp.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <vector>
#include <fmt/format.h>
#include <cstdlib>
#include <omp.h>
#include <limits>
#include <filesystem>



void FlowFieldGenerator::GenerateFlow(const ParameterParser& params,
                                     int gx, int gy, double cellsize, int ox, int oy,
                                     int imageWidth, int imageHeight,
                                     const std::string& projectDirectory)
{
    // Initialize members
    this->gx = gx;
    this->gy = gy;
    this->cellsize = cellsize;
    this->ox = ox;
    this->oy = oy;
    this->imageWidth = imageWidth;
    this->imageHeight = imageHeight;
    this->projectDirectory = projectDirectory;

    // Resize member buffers
    vx_frame.resize(gx * gy);
    vy_frame.resize(gx * gy);

    spdlog::info("FlowFieldGenerator::GenerateFlow: FlowType='{}'", params.FlowType);

    if (params.FlowType == "FLUENT-static") {
        GenerateFluentFlow(params);
    }
    else if (params.FlowType == "FLUENT-transient") {
        GenerateFluentTransient(params);
    }
    else if (params.FlowType.empty()) {
        spdlog::info("No primary flow field requested (FlowType is empty)");
    }
    else {
        throw std::runtime_error("Unknown FlowType: " + params.FlowType);
    }

    // Process optional Wind Data
    if (!params.WindData.empty()) {
        spdlog::info("FlowFieldGenerator: WindData provided (assumed ERA5). Skipping static bake-in to grid_flow.h5 - will be handled by runtime interpolator.");
    }

    spdlog::info("FlowFieldGenerator::GenerateFlow completed");
}

void FlowFieldGenerator::AddWindToFlowFieldHDF5(const std::string& windFilePath, const std::string& outputFlowPath, bool compressFlow)
{
    spdlog::info("Adding Wind Data from {} to {}", windFilePath, outputFlowPath);
    
    if (!std::filesystem::exists(windFilePath)) {
        throw std::runtime_error("Input wind file does not exist: " + windFilePath);
    }

    try {
        // Open Input
        H5::H5File inFile(windFilePath, H5F_ACC_RDONLY);
        H5::DataSet ds_in_vx = inFile.openDataSet("wind_current_vx");
        H5::DataSet ds_in_vy = inFile.openDataSet("wind_current_vy");

        // Get Input Metadata
        int num_frames = 0;
        double time_interval = 0.0;
        // Attributes on dataset as per python script
        ds_in_vx.openAttribute("num_frames").read(H5::PredType::NATIVE_INT, &num_frames);
        ds_in_vx.openAttribute("time_interval").read(H5::PredType::NATIVE_DOUBLE, &time_interval);

         // Get Input Dimensions
        H5::DataSpace inSpace = ds_in_vx.getSpace();
        std::vector<hsize_t> inDims(3);
        inSpace.getSimpleExtentDims(inDims.data(), NULL);
        int widthIn = inDims[1];
        int heightIn = inDims[2];

        // Open Output File (Create if not exists, else Append)
        H5::H5File outFile;
        if (std::filesystem::exists(outputFlowPath)) {
            outFile.openFile(outputFlowPath, H5F_ACC_RDWR);
        } else {
            outFile = H5::H5File(outputFlowPath, H5F_ACC_TRUNC);
            // If we created a new file, we should add minimal root attributes to be valid?
            H5::DataSpace att_space(H5S_SCALAR);
            std::string flowType = "WindOnly";
            H5::StrType str_type(H5::PredType::C_S1, flowType.size() + 1);
            outFile.createGroup("/").createAttribute("flow_type", str_type, att_space).write(str_type, flowType.c_str());
        }

        // Define Output Dimensions
        hsize_t outDims[3] = {static_cast<hsize_t>(num_frames), static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};
        H5::DataSpace outSpace(3, outDims);

        H5::DSetCreatPropList props;
        if (compressFlow) {
            hsize_t chunks[3] = {1, std::min<hsize_t>(gx, 64), std::min<hsize_t>(gy, 64)};
            props.setChunk(3, chunks);
            props.setDeflate(6);
        }

        // Create Datasets in Output
        // Note: If they exist, we might want to overwrite? Or error?
        // HDF5 throws if dataset exists.
        // Let's assume we are adding new data.
        H5::DataSet out_vx, out_vy;
        try {
             out_vx = outFile.createDataSet("wind_current_vx", H5::PredType::NATIVE_FLOAT, outSpace, props);
             out_vy = outFile.createDataSet("wind_current_vy", H5::PredType::NATIVE_FLOAT, outSpace, props);
        } catch (...) {
             // If datasets exist (e.g. re-running preparer), try opening them
             out_vx = outFile.openDataSet("wind_current_vx");
             out_vy = outFile.openDataSet("wind_current_vy");
             // Ideally we should check dimensions match, but assuming overwrite behavior for now
        }

        // Copy Attributes
        H5::DataSpace att_space(H5S_SCALAR);
        int loop_mode = 1; 
        try { ds_in_vx.openAttribute("loop_mode").read(H5::PredType::NATIVE_INT, &loop_mode); } catch(...) {}

        // We can't overwrite attributes easily if they exist, so ignore errors
        try { out_vx.createAttribute("time_interval", H5::PredType::NATIVE_DOUBLE, att_space).write(H5::PredType::NATIVE_DOUBLE, &time_interval); } catch(...) {}
        try { out_vx.createAttribute("num_frames", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &num_frames); } catch(...) {}
        try { out_vx.createAttribute("loop_mode", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &loop_mode); } catch(...) {}

         // Process Data 
        std::vector<double> frame_in(widthIn * heightIn); // Re-use for vx and vy to save memory
        std::vector<float> frame_out(gx * gy);

        for (int f = 0; f < num_frames; f++) {
            hsize_t inOffset[3] = {static_cast<hsize_t>(f), 0, 0};
            hsize_t inCount[3] = {1, static_cast<hsize_t>(widthIn), static_cast<hsize_t>(heightIn)};
            
            inSpace.selectHyperslab(H5S_SELECT_SET, inCount, inOffset);
            H5::DataSpace memSpaceIn(3, inCount);
            
            // Output selection
            hsize_t outOffset[3] = {static_cast<hsize_t>(f), 0, 0};
            hsize_t outCount[3] = {1, static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};
            outSpace.selectHyperslab(H5S_SELECT_SET, outCount, outOffset);
            H5::DataSpace memSpaceOut(3, outCount);

            // Read/Write VX
            ds_in_vx.read(frame_in.data(), H5::PredType::NATIVE_DOUBLE, memSpaceIn, inSpace);
            
            // Extract
             for(int i = 0; i < gx; ++i) {
                for(int j = 0; j < gy; ++j) {
                    int src_i = i + ox;
                    int src_j = j + oy;
                    size_t dstIdx = j + (size_t)i * gy;
                    
                    if(src_i >= 0 && src_i < widthIn && src_j >= 0 && src_j < heightIn) {
                         // Column-major in file input [Width, Height] -> x * strideY + y
                         // My python script writes [Time, Width, Height]
                         // x is dim 1, y is dim 2. 
                         // Flattened index: x * Height + y.
                         size_t srcIdx = src_i * heightIn + src_j;
                         frame_out[dstIdx] = frame_in[srcIdx];
                    } else {
                        frame_out[dstIdx] = 0.0;
                    }
                }
            }
            out_vx.write(frame_out.data(), H5::PredType::NATIVE_FLOAT, memSpaceOut, outSpace);

            // Read/Write VY
            ds_in_vy.read(frame_in.data(), H5::PredType::NATIVE_DOUBLE, memSpaceIn, inSpace);
            for(int i = 0; i < gx; ++i) {
                for(int j = 0; j < gy; ++j) {
                    int src_i = i + ox;
                    int src_j = j + oy;
                    size_t dstIdx = j + (size_t)i * gy;
                    if(src_i >= 0 && src_i < widthIn && src_j >= 0 && src_j < heightIn) {
                         size_t srcIdx = src_i * heightIn + src_j;
                         frame_out[dstIdx] = frame_in[srcIdx];
                    } else {
                        frame_out[dstIdx] = 0.0;
                    }
                }
            }
            out_vy.write(frame_out.data(), H5::PredType::NATIVE_FLOAT, memSpaceOut, outSpace);
        }

        inFile.close();
        outFile.close();
        spdlog::info("Wind data added successfully");

    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 Error in AddWindToFlowFieldHDF5: " + e.getDetailMsg());
    }
}




void FlowFieldGenerator::GenerateFluentFlow(const ParameterParser& params)
{
    spdlog::info("FlowFieldGenerator::GenerateFluentFlow starting");
    spdlog::info("  ConfigDirectory: {}", params.ConfigFileDirectory);
    spdlog::info("  CAS file: {}", params.InputFluentCAS);
    spdlog::info("  DAT file: {}", params.InputFluentDAT);
    spdlog::info("  SVG file: {}", params.SVG);
    spdlog::info("  RectanglePathID: {}", params.RectanglePathID);
    spdlog::info("  FluentPathID: {}", params.FluentPathID);

    // Note: The FLUENT raster must match the initialization image dimensions exactly
    // so that it aligns pixel-for-pixel with the land/ice/color masks
    // The image dimensions are the ground truth from the loaded images (PrepareGridAndPoints)

    spdlog::info("InitializationImageSize: {}x{}", imageWidth, imageHeight);

    // Import FLUENT flow field - rasterize to match initialization image dimensions
    FluentFlowImporter importer;
    importer.Import(params.ConfigFileDirectory,
                   params.InputFluentCAS,
                   params.InputFluentDAT,
                   params.SVG,
                   params.RectanglePathID,
                   params.FluentPathID,
                   imageWidth, imageHeight,  // Must match the initialization image dimensions
                   params.VelocityMultiplier);  // Apply velocity multiplier

    // Get the actual rasterized dimensions and velocity data
    int width = importer.image_width;
    int height = importer.image_height;
    std::vector<double> vx_full = importer.vx_data;
    std::vector<double> vy_full = importer.vy_data;

    spdlog::info("FLUENT import completed: {}x{} grid", width, height);

    // Verify that FLUENT raster matches initialization image dimensions
    if (width != imageWidth || height != imageHeight) {
        throw std::runtime_error(fmt::format("FLUENT raster dimensions ({}x{}) do not match initialization image ({}x{})",
                                            width, height, imageWidth, imageHeight));
    }

    // Extract modeled region from full raster
    // Full raster is column-major (i + width*j) and matches image dimensions
    // Output needs to be row-major (j + i*gy) per WriteFlowFieldToHDF5 convention

    spdlog::info("Extracting region: grid={}x{}, offset=({}, {})", gx, gy, ox, oy);
    spdlog::info("  Full raster size: {}x{}", width, height);
    spdlog::info("  Modeled region size: {}x{}", gx, gy);

    std::vector<double> vx_data(gx * gy, 0.0);
    std::vector<double> vy_data(gx * gy, 0.0);

    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            // Source: full raster grid (image coordinates) with column-major indexing
            // The modeled region has offset (ox, oy) from the image origin
            int src_i = i + ox;
            int src_j = j + oy;

            // Verify within bounds
            if (src_i < 0 || src_i >= width || src_j < 0 || src_j >= height) {
                spdlog::warn("Region index ({}, {}) out of raster bounds ({}x{})",
                           src_i, src_j, width, height);
                vx_data[j + (size_t)i * gy] = 0.0;
                vy_data[j + (size_t)i * gy] = 0.0;
                continue;
            }

            int src_idx = src_i + width * src_j;  // Column-major indexing in raster
            int dst_idx = j + (size_t)i * gy;    // Row-major indexing in output

            vx_data[dst_idx] = vx_full[src_idx];
            vy_data[dst_idx] = vy_full[src_idx];
        }
    }

    spdlog::info("Region extraction completed");

    // Package into frame containers (single frame for static FLUENT flow)
    std::vector<std::vector<double>> vx_frames = {vx_data};
    std::vector<std::vector<double>> vy_frames = {vy_data};

    // Write to HDF5 with static flow parameters
    double time_interval = 0.0;  // Static flow (single frame, no interpolation)
    int loop_mode = 1;           // Hold last frame (irrelevant for single frame)

    WriteFlowFieldToHDF5(params.FlowType, 1, time_interval, loop_mode, params.CompressFlow,
                        vx_frames, vy_frames);

    spdlog::info("GenerateFluentFlow completed");
}

void FlowFieldGenerator::GenerateFluentTransient(const ParameterParser& params)
{
    spdlog::info("FlowFieldGenerator::GenerateFluentTransient starting");
    spdlog::info("  Range: Frame {} to {}", params.FLUENTFirstFrame, params.FLUENTLastFrame);
    spdlog::info("  Time Interval: {}", params.FLUENTTimeInterval);
    spdlog::info("  DAT Prefix: {}", params.InputFluentDAT);

    int startFrame = params.FLUENTFirstFrame;
    int endFrame = params.FLUENTLastFrame;
    int num_frames = endFrame - startFrame + 1;

    if (num_frames <= 0) {
        throw std::runtime_error("Invalid FLUENT frame range: FirstFrame > LastFrame");
    }

    // Initialize HDF5 file structure
    // loop_mode = 0 (periodic) for transient
    CreateFlowFieldHDF5(params.FlowType, num_frames, params.FLUENTTimeInterval, 0, params.CompressFlow);

    // Loop over frames
    FluentFlowImporter importer; // Re-use importer instance (though Import() might be stateless)

    for (int i = 0; i < num_frames; ++i) {
        int frameID = startFrame + i;
        
        // Construct DAT filename
        // Assumes Prefix ends with '-' or correct base, appends 00575.dat.h5
        std::string datFile = fmt::format("{}{:05d}.dat.h5", params.InputFluentDAT, frameID);
        
        spdlog::info("Processing frame {}/{}: {}", i + 1, num_frames, datFile);

        if (!std::filesystem::exists(datFile)) {
             throw std::runtime_error("FLUENT DAT file not found: " + datFile);
        }

        // Import
        // Note: This re-parses CAS and SVG every time. 
        // Optimization possible but keeping it simple as requested.
        importer.Import(params.ConfigFileDirectory,
                        params.InputFluentCAS,
                        datFile,
                        params.SVG,
                        params.RectanglePathID,
                        params.FluentPathID,
                        imageWidth, imageHeight,
                        params.VelocityMultiplier);

        // Verify dimensions (paranoid check)
        if (importer.image_width != imageWidth || importer.image_height != imageHeight) {
            throw std::runtime_error("Imported dimensions mismatch initialization image");
        }

        // Extract Region and Populate buffers
        #pragma omp parallel for
        for (int gy_idx = 0; gy_idx < gy; ++gy_idx) {
            for (int gx_idx = 0; gx_idx < gx; ++gx_idx) {
                 int src_i = gx_idx + ox;
                 int src_j = gy_idx + oy;
                 size_t dst_idx = gy_idx + (size_t)gx_idx * gy;
                 
                 // Valid check - import should ensure correct size, but boundary check is safe
                 if (src_i >= 0 && src_i < imageWidth && src_j >= 0 && src_j < imageHeight) {
                     size_t src_idx = src_i + imageWidth * src_j;
                     vx_frame[dst_idx] = (float)importer.vx_data[src_idx];
                     vy_frame[dst_idx] = (float)importer.vy_data[src_idx];
                 } else {
                     vx_frame[dst_idx] = 0.0f;
                     vy_frame[dst_idx] = 0.0f;
                 }
            }
        }

        // Write Frame
        WriteFrameToHDF5(i);
    }

    spdlog::info("GenerateFluentTransient completed");
}


void FlowFieldGenerator::CreateFlowFieldHDF5(const std::string& flowType, int num_frames, double time_interval,
                                            int loop_mode,
                                            bool compressFlow)
{
    spdlog::info("CreateFlowFieldHDF5: flowType={}, gx={}, gy={}, num_frames={}, compress={}", flowType, gx, gy, num_frames, compressFlow);

    std::string flowFilePath = projectDirectory + "/grid_flow.h5";
    H5::H5File file(flowFilePath, H5F_ACC_TRUNC);

    // Create datasets for vx and vy (3D: [num_frames, gx, gy])
    // Data is stored in column-major order (row-minor): gx is the major axis
    hsize_t dims[3] = {static_cast<hsize_t>(num_frames), static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};
    H5::DataSpace dataspace(3, dims);

    H5::DSetCreatPropList props;
    if (compressFlow) {
        hsize_t chunks[3] = {1, std::min<hsize_t>(gx, 64), std::min<hsize_t>(gy, 64)};
        props.setChunk(3, chunks);
        props.setDeflate(6);
    }

    H5::DataSet ds_vx = file.createDataSet("water_current_vx", H5::PredType::NATIVE_FLOAT, dataspace, props);
    H5::DataSet ds_vy = file.createDataSet("water_current_vy", H5::PredType::NATIVE_FLOAT, dataspace, props);

    H5::DataSpace att_space(H5S_SCALAR);

    // Set flow_type attribute on the root group
    H5::Group root = file.openGroup("/");
    H5::StrType str_type(H5::PredType::C_S1, flowType.size() + 1);
    root.createAttribute("flow_type", str_type, att_space).write(str_type, flowType.c_str());

    ds_vx.createAttribute("time_interval", H5::PredType::NATIVE_DOUBLE, att_space)
        .write(H5::PredType::NATIVE_DOUBLE, &time_interval);
    ds_vx.createAttribute("loop_mode", H5::PredType::NATIVE_INT, att_space)
        .write(H5::PredType::NATIVE_INT, &loop_mode);
    ds_vx.createAttribute("num_frames", H5::PredType::NATIVE_INT, att_space)
        .write(H5::PredType::NATIVE_INT, &num_frames);

    file.close();

    spdlog::info("CreateFlowFieldHDF5 completed: file created at {}", flowFilePath);
}

void FlowFieldGenerator::WriteFrameToHDF5(int frame_index)
{
    std::string flowFilePath = projectDirectory + "/grid_flow.h5";
    H5::H5File file(flowFilePath, H5F_ACC_RDWR);

    // Open existing datasets
    H5::DataSet ds_vx = file.openDataSet("water_current_vx");
    H5::DataSet ds_vy = file.openDataSet("water_current_vy");

    // Get dataspace and select hyperslab for this frame
    H5::DataSpace dataspace = ds_vx.getSpace();
    hsize_t frame_offset[3] = {static_cast<hsize_t>(frame_index), 0, 0};
    hsize_t frame_dims[3] = {1, static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};
    dataspace.selectHyperslab(H5S_SELECT_SET, frame_dims, frame_offset);

    // Create memory dataspace for single frame
    H5::DataSpace frame_space(3, frame_dims);

    // Write vx, vy data for this frame
    ds_vx.write(vx_frame.data(), H5::PredType::NATIVE_FLOAT, frame_space, dataspace);
    ds_vy.write(vy_frame.data(), H5::PredType::NATIVE_FLOAT, frame_space, dataspace);

    file.close();
}

void FlowFieldGenerator::WriteFlowFieldToHDF5(const std::string& flowType, int num_frames, double time_interval,
                                             int loop_mode,
                                             bool compressFlow,
                                             const std::vector<std::vector<double>>& vx_frames,
                                             const std::vector<std::vector<double>>& vy_frames)
{
    // Legacy wrapper: Create file structure then write all frames
    CreateFlowFieldHDF5(flowType, num_frames, time_interval, loop_mode, compressFlow);

    for (int frame = 0; frame < num_frames; frame++) {
        // Copy data to member buffers
        vx_frame.resize(gx * gy);
        vy_frame.resize(gx * gy);
        for(size_t i=0; i < vx_frames[frame].size(); ++i) {
            vx_frame[i] = (float)vx_frames[frame][i];
            vy_frame[i] = (float)vy_frames[frame][i];
        }
        WriteFrameToHDF5(frame);
    }

    spdlog::info("WriteFlowFieldToHDF5 completed: saved {} frames to {}/grid_flow.h5", num_frames, projectDirectory);
}











