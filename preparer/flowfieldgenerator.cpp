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

void FlowFieldGenerator::GenerateConstantFlow(int gx, int gy, double cellsize, int ox, int oy,
                                             double flowBearing, double flowSpeed,
                                             const std::string& projectDirectory, bool compressFlow)
{
    spdlog::info("FlowFieldGenerator::GenerateConstantFlow: bearing={}, speed={}", flowBearing, flowSpeed);

    // Convert bearing angle to radians (0° = north = +y direction)
    double bearing_rad = flowBearing * M_PI / 180.0;
    double vx = flowSpeed * std::sin(bearing_rad);  // east component
    double vy = flowSpeed * std::cos(bearing_rad);  // north component

    spdlog::info("Constant flow: vx={}, vy={}", vx, vy);

    // Create single frame
    std::vector<double> vx_data(gx * gy, vx);
    std::vector<double> vy_data(gx * gy, vy);

    std::vector<std::vector<double>> vx_frames = {vx_data};
    std::vector<std::vector<double>> vy_frames = {vy_data};

    double time_interval = 0.0;  // Static flow
    int loop_mode = 1;           // Hold last frame (irrelevant for single frame)

    WriteFlowFieldToHDF5(gx, gy, 1, time_interval, loop_mode, projectDirectory, compressFlow,
                        vx_frames, vy_frames);

    spdlog::info("GenerateConstantFlow completed");
}


void FlowFieldGenerator::GenerateWaveFlow(int gx, int gy, double cellsize, int ox, int oy,
                                         double flowBearing, double waveAmplitude, double waveLength, double phaseSpeed,
                                         int nFrames, const std::string& projectDirectory, bool compressFlow)
{
    spdlog::info("FlowFieldGenerator::GenerateWaveFlow: bearing={}, amplitude={}, waveLength={}, phaseSpeed={}, nFrames={}",
                 flowBearing, waveAmplitude, waveLength, phaseSpeed, nFrames);

    // Wave parameters
    double k = 2.0 * M_PI / waveLength;  // wavenumber
    double omega = k * phaseSpeed;        // angular frequency
    double amplitude_omega_sq = waveAmplitude * std::pow(omega, 2.0);  // A*ω²

    // Convert bearing to wave direction
    double bearing_rad = flowBearing * M_PI / 180.0;
    double wave_dir_x = std::sin(bearing_rad);   // wave propagates east
    double wave_dir_y = std::cos(bearing_rad);   // wave propagates north

    spdlog::info("Wave parameters: k={}, omega={}, A*omega²={}", k, omega, amplitude_omega_sq);
    spdlog::info("Wave direction: ({}, {})", wave_dir_x, wave_dir_y);

    // Calculate time interval
    double T = 2.0 * M_PI / omega;
    double time_interval = T / nFrames;

    spdlog::info("Wave period: {}, time_interval: {}, num_frames: {}", T, time_interval, nFrames);

    // Generate frames one at a time to manage RAM
    std::vector<std::vector<double>> vx_frames;
    std::vector<std::vector<double>> vy_frames;

    for (int frame = 0; frame < nFrames; frame++) {
        double t = frame * time_interval;

        std::vector<double> vx_frame(gx * gy);
        std::vector<double> vy_frame(gx * gy);

        for (int i = 0; i < gx; i++) {
            for (int j = 0; j < gy; j++) {
                // Physical coordinates (cell centers)
                double x = (i + ox) * cellsize;
                double y = (j + oy) * cellsize;

                // Project onto wave direction: phase = k*(x*wave_dir_x + y*wave_dir_y) - omega*t
                double phase = k * (x * wave_dir_x + y * wave_dir_y) - omega * t;

                // Velocity magnitude: u' = A*ω²*sin(phase)
                double vel_mag = amplitude_omega_sq * std::sin(phase);

                // Decompose velocity into x and y components
                size_t idx = j + (size_t)i * gy;
                vx_frame[idx] = vel_mag * wave_dir_x;
                vy_frame[idx] = vel_mag * wave_dir_y;
            }
        }

        vx_frames.push_back(vx_frame);
        vy_frames.push_back(vy_frame);
    }

    double loop_mode = 0;  // Periodic for waves

    WriteFlowFieldToHDF5(gx, gy, nFrames, time_interval, loop_mode, projectDirectory, compressFlow,
                        vx_frames, vy_frames);

    spdlog::info("GenerateWaveFlow completed: saved {} frames", nFrames);
}


void FlowFieldGenerator::GenerateFlow(const ParameterParser& params,
                                     int gx, int gy, double cellsize, int ox, int oy,
                                     int imageWidth, int imageHeight,
                                     const std::string& projectDirectory)
{
    spdlog::info("FlowFieldGenerator::GenerateFlow: FlowType='{}'", params.FlowType);

    if (params.FlowType == "constant") {
        GenerateConstantFlow(gx, gy, cellsize, ox, oy,
                            params.FlowBearing, params.FlowSpeed,
                            projectDirectory, params.CompressFlow);
    }
    else if (params.FlowType == "wave") {
        GenerateWaveFlow(gx, gy, cellsize, ox, oy,
                        params.FlowBearing, params.WaveAmplitude, params.WaveLength, params.PhaseSpeed,
                        params.NFrames, projectDirectory, params.CompressFlow);
    }
    else if (params.FlowType == "FLUENT-static") {
        // For FLUENT, we need to pass grid parameters and image dimensions
        GenerateFluentFlow(params, gx, gy, ox, oy, imageWidth, imageHeight, projectDirectory);
    }
    else if (params.FlowType.empty()) {
        spdlog::info("No flow field requested (FlowType is empty)");
    }
    else {
        throw std::runtime_error("Unknown FlowType: " + params.FlowType);
    }

    spdlog::info("FlowFieldGenerator::GenerateFlow completed");
}


void FlowFieldGenerator::GenerateFluentFlow(const ParameterParser& params,
                                           int gx, int gy, int ox, int oy,
                                           int imageWidth, int imageHeight,
                                           const std::string& projectDirectory)
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

    WriteFlowFieldToHDF5(gx, gy, 1, time_interval, loop_mode, projectDirectory, params.CompressFlow,
                        vx_frames, vy_frames);

    spdlog::info("GenerateFluentFlow completed");
}


void FlowFieldGenerator::WriteFlowFieldToHDF5(int gx, int gy, int num_frames, double time_interval,
                                             int loop_mode, const std::string& projectDirectory,
                                             bool compressFlow,
                                             const std::vector<std::vector<double>>& vx_frames,
                                             const std::vector<std::vector<double>>& vy_frames)
{
    spdlog::info("WriteFlowFieldToHDF5: gx={}, gy={}, num_frames={}, compress={}", gx, gy, num_frames, compressFlow);

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

    H5::DataSet ds_vx = file.createDataSet("water_current_vx", H5::PredType::NATIVE_DOUBLE, dataspace, props);
    H5::DataSet ds_vy = file.createDataSet("water_current_vy", H5::PredType::NATIVE_DOUBLE, dataspace, props);

    // Write all frames
    // Data is stored in column-major order: idx = j + i*gy
    // HDF5 dataset is [frames, gx, gy] which matches the column-major layout when linearized
    for (int frame = 0; frame < num_frames; frame++) {
        hsize_t frame_offset[3] = {static_cast<hsize_t>(frame), 0, 0};
        hsize_t frame_dims[3] = {1, static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};
        H5::DataSpace frame_space(3, frame_dims);
        dataspace.selectHyperslab(H5S_SELECT_SET, frame_dims, frame_offset);

        ds_vx.write(vx_frames[frame].data(), H5::PredType::NATIVE_DOUBLE, frame_space, dataspace);
        ds_vy.write(vy_frames[frame].data(), H5::PredType::NATIVE_DOUBLE, frame_space, dataspace);
    }

    // Write metadata attributes
    H5::DataSpace att_space(H5S_SCALAR);

    ds_vx.createAttribute("time_interval", H5::PredType::NATIVE_DOUBLE, att_space)
        .write(H5::PredType::NATIVE_DOUBLE, &time_interval);
    ds_vx.createAttribute("loop_mode", H5::PredType::NATIVE_INT, att_space)
        .write(H5::PredType::NATIVE_INT, &loop_mode);
    ds_vx.createAttribute("num_frames", H5::PredType::NATIVE_INT, att_space)
        .write(H5::PredType::NATIVE_INT, &num_frames);

    file.close();

    spdlog::info("WriteFlowFieldToHDF5 completed: saved to {}", flowFilePath);
}
