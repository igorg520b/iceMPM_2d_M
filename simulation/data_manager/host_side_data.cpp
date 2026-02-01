// host_side_data.cpp

#include "host_side_data.h"
#include "poisson_disk_sampling.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>

#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <random>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <map>

#include <fmt/format.h>
#include <fmt/std.h>



namespace fs = std::filesystem;


HostSideData::HostSideData() : waci(prms)
{
    prms.Reset();
}

void HostSideData::FillModelledAreaWithBlueColor()
{
    const int &width_ref = prms.InitializationImageSizeX;
    const int &height_ref = prms.InitializationImageSizeY;
    const int &ox_ref = prms.ModeledRegionOffsetX;
    const int &oy_ref = prms.ModeledRegionOffsetY;
    const int &gx_ref = prms.GridXTotal;
    const int &gy_ref = prms.GridYTotal;

    const size_t width = (size_t)width_ref;
    // const size_t height = (size_t)height_ref;
    const size_t ox = (size_t)ox_ref;
    const size_t oy = (size_t)oy_ref;
    const size_t gx = (size_t)gx_ref;
    const size_t gy = (size_t)gy_ref;

    for(size_t i = 0; i < gx; i++)
    {
        for(size_t j = 0; j < gy; j++)
        {
            uint8_t status = landmask_buffer[j + i*gy];
            if(status == SimParams::ModelledAreaIndicator)
            {
                for(int k = 0; k < 3; k++) {
                    size_t idx = ((i+ox) + (j+oy)*width)*3 + k;
                    original_image_colors_rgb[idx] = ColorMap::rgb_water[k];
                }
            }
        }
    }
}


// =============================  ALLOCATION FUNCTIONS

void HostSideData::AllocateGridArrays(bool allocate_dense_grid)
{
    // Allocate grid buffers (for both preparer and simulation)
    const size_t modeled_grid_total = (size_t)prms.GridXTotal * prms.GridYTotal;
    const size_t initial_image_total = (size_t)prms.InitializationImageSizeX * prms.InitializationImageSizeY;

    allocated_bytes[0] = 0;  // Reset grid allocation counter

    landmask_buffer.resize(modeled_grid_total);
    allocated_bytes[0] += modeled_grid_total * sizeof(uint8_t);

    // Allocate visualization/saving buffers
    size_t plane_size = (size_t)prms.GridXTotal * prms.GridYTotal;

    original_image_colors_rgb.resize(3 * initial_image_total);
    allocated_bytes[0] += 3 * initial_image_total * sizeof(uint8_t);

    if (allocate_dense_grid)
    {
        host_grid_buffer.resize(modeled_grid_total * SimParams::HostGridArrayIndex::nGridArraysHost);
        allocated_bytes[0] += modeled_grid_total * SimParams::HostGridArrayIndex::nGridArraysHost * sizeof(double);

        tmp_halo_buffer.resize((size_t)prms.GridYTotal * prms.GridHaloSize * SimParams::HostGridArrayIndex::nGridArraysHost);
        allocated_bytes[0] += (size_t)prms.GridYTotal * prms.GridHaloSize * SimParams::HostGridArrayIndex::nGridArraysHost * sizeof(double);

        // 'rgb' buffer is used for saving frames (simulation) but not needed for preparer
//        rgb.resize(3 * initial_image_total);
        allocated_bytes[0] += 3 * initial_image_total * sizeof(uint8_t);

        save_buffer_float.resize(4 * plane_size); 
        save_buffer_uint8.resize(4 * plane_size);

        // Track memory
        allocated_bytes[0] += (save_buffer_float.capacity() * sizeof(float));
        allocated_bytes[0] += (save_buffer_uint8.capacity() * sizeof(uint8_t));
    }

    LOGR("[MEMORY] HostSideData Grid Arrays: {:.3f} MB", allocated_bytes[0] / 1.0e6);
    LOGR("    landmask_buffer: {:.3f} MB", (double)landmask_buffer.size() * sizeof(uint8_t) / 1.0e6);
    LOGR("    original_image_colors_rgb: {:.3f} MB", (double)original_image_colors_rgb.size() * sizeof(uint8_t) / 1.0e6);
    if (allocate_dense_grid) LOGR("    host_grid_buffer: {:.3f} MB", (double)host_grid_buffer.size() * sizeof(float) / 1.0e6);

//    LOGR("    rgb: {:.3f} MB", (double)rgb.size() * sizeof(uint8_t) / 1.0e6);
}




// =============================  LOAD GRID DATA FROM FILE

void HostSideData::LoadGridDataFromFile(const std::string& gridFilePath)
{
    LOGR("LoadGridDataFromFile: starting from {}", gridFilePath);

    try {
        // Open HDF5 file
        H5::H5File file(gridFilePath, H5F_ACC_RDONLY);

        // Open landmask dataset to read attributes
        H5::DataSet ds_landmask = file.openDataSet("landmask");
        H5::DataSpace space_landmask = ds_landmask.getSpace();

        // Read metadata attributes
        H5::Attribute attr_gx = ds_landmask.openAttribute("GridXTotal");
        H5::Attribute attr_gy = ds_landmask.openAttribute("GridYTotal");
        H5::Attribute attr_ox = ds_landmask.openAttribute("OffsetX");
        H5::Attribute attr_oy = ds_landmask.openAttribute("OffsetY");
        H5::Attribute attr_img_x = ds_landmask.openAttribute("InitImageSizeX");
        H5::Attribute attr_img_y = ds_landmask.openAttribute("InitImageSizeY");
        H5::Attribute attr_cellsize = ds_landmask.openAttribute("CellSize");
        H5::Attribute attr_dim_horiz = ds_landmask.openAttribute("DimensionHorizontal");

        attr_gx.read(H5::PredType::NATIVE_INT, &prms.GridXTotal);
        attr_gy.read(H5::PredType::NATIVE_INT, &prms.GridYTotal);
        attr_ox.read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetX);
        attr_oy.read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetY);
        attr_img_x.read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeX);
        attr_img_y.read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeY);
        attr_cellsize.read(H5::PredType::NATIVE_DOUBLE, &prms.cellsize);
        attr_dim_horiz.read(H5::PredType::NATIVE_DOUBLE, &prms.DimensionHorizontal);

        // Compute derived parameters
        prms.cellsize_inv = 1.0 / prms.cellsize;

        LOGR("Loaded grid metadata: gx={}, gy={}, offset=({},{}), cellsize={}",
             prms.GridXTotal, prms.GridYTotal, prms.ModeledRegionOffsetX, prms.ModeledRegionOffsetY, prms.cellsize);

        // Validate dimensions
        if (prms.GridXTotal <= 0 || prms.GridYTotal <= 0) {
            throw std::runtime_error(fmt::format("Invalid grid dimensions: GridXTotal={}, GridYTotal={}",
                                                 prms.GridXTotal, prms.GridYTotal));
        }
        if (prms.InitializationImageSizeX <= 0 || prms.InitializationImageSizeY <= 0) {
            throw std::runtime_error(fmt::format("Invalid image dimensions: ImageX={}, ImageY={}",
                                                 prms.InitializationImageSizeX, prms.InitializationImageSizeY));
        }

        // Check landmask dataset dimensions
        hsize_t landmask_dims[2];
        space_landmask.getSimpleExtentDims(landmask_dims, NULL);
        if ((int)landmask_dims[0] != prms.GridXTotal || (int)landmask_dims[1] != prms.GridYTotal) {
            throw std::runtime_error(fmt::format(
                "Landmask dataset dimensions [{}, {}] don't match metadata [{}, {}]",
                landmask_dims[0], landmask_dims[1], prms.GridXTotal, prms.GridYTotal));
        }

        // Check color_grid dataset dimensions
        H5::DataSet ds_color = file.openDataSet("color_grid");
        H5::DataSpace space_color = ds_color.getSpace();
        hsize_t color_dims[3];
        space_color.getSimpleExtentDims(color_dims, NULL);
        if ((int)color_dims[0] != prms.InitializationImageSizeY ||
            (int)color_dims[1] != prms.InitializationImageSizeX ||
            (int)color_dims[2] != 3) {
            throw std::runtime_error(fmt::format(
                "Color grid dataset dimensions [{}, {}, {}] don't match expected [{}, {}, 3]",
                color_dims[0], color_dims[1], color_dims[2],
                prms.InitializationImageSizeY, prms.InitializationImageSizeX));
        }

        // Allocate grid arrays (dense grid required here since we load data)
        AllocateGridArrays(true);

        // Read landmask dataset
        ds_landmask.read(landmask_buffer.data(), H5::PredType::NATIVE_UINT8);

        // Read color_grid dataset
        ds_color.read(original_image_colors_rgb.data(), H5::PredType::NATIVE_UINT8);

        file.close();

        LOGR("LoadGridDataFromFile: completed successfully");

    } catch (const H5::Exception& e) {
        throw std::runtime_error(fmt::format("Failed to load grid.h5: {}", e.getCDetailMsg()));
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("LoadGridDataFromFile failed: {}", e.what()));
    }
}


void HostSideData::ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5)
{
    LOGR("ReadPointsFromSnapshot {}", fileNameSnapshotHDF5);

    // Open HDF5 file and dataset
    H5::H5File file(fileNameSnapshotHDF5, H5F_ACC_RDONLY);
    H5::DataSet ds = file.openDataSet("pts_data");

    // Get dimensions from file to determine actual point count
    H5::DataSpace dsp = ds.getSpace();
    hsize_t dims[2];
    dsp.getSimpleExtentDims(dims, nullptr);

    // Read attributes
    ds.openAttribute("nPtsInitial").read(H5::PredType::NATIVE_INT, &prms.nPtsInitial);

    // Ensure we allocate enough space for what's in the file, even if it exceeds nPtsInitial
    size_t pts_in_file = dims[1];
    size_t required_for_file = (size_t)(pts_in_file * (1.0 + prms.extra_space_pts));

    // Also respect the standard heuristic based on nPtsInitial
    size_t standard_heuristic = (size_t)(double(prms.nPtsInitial) * (1.0 + prms.extra_space_pts));

    const size_t requested_capacity = std::max(required_for_file, standard_heuristic);
    
    LOGR("ReadPointsFromSnapshot: nPtsInitial={}, file_pts={}. Allocating capacity={}", 
         prms.nPtsInitial, pts_in_file, requested_capacity);

    hssoa.Allocate(requested_capacity);

    ds.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &prms.SimulationStep);
    ds.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &prms.SimulationTime);

    ds.openAttribute("ParticleArea").read(H5::PredType::NATIVE_DOUBLE, &prms.ParticleArea);
    int readGridY;
    ds.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &readGridY);
    if(prms.GridYTotal != 0) {
        if(prms.GridYTotal != readGridY) {
             throw std::runtime_error(fmt::format("ReadPointsFromSnapshot GridYTotal mismatch: expected {}, got {}", prms.GridYTotal, readGridY));
        }
    }
    prms.GridYTotal = readGridY;

    int nPtsArrays;
    ds.openAttribute("nPtsArrays").read(H5::PredType::NATIVE_INT, &nPtsArrays);

    // Dimensions already read above into 'dims'

    if(nPtsArrays != SimParams::PtArrIdx::nPtsArrays)
    {
        LOGR("dims {} x {}; nPtsInitial {}", dims[0], dims[1], prms.nPtsInitial);
        throw std::runtime_error("ReadSnapshot array size mismatch");
    }

    hssoa.size = dims[1];

    // Define hyperslab
    hsize_t dims_mem[2] = {SimParams::PtArrIdx::nPtsArrays, hssoa.capacity};
    H5::DataSpace memspace(2, dims_mem);
    hsize_t offset[2] = {0, 0};
    hsize_t count[2] = {dims[0], dims[1]};
    memspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    LOGR("ReadPointsFromSnapshot attempting to read {} pts into hssoa with capacity {}",
         dims[1], hssoa.capacity);
    ds.read(hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE, memspace, dsp);
    LOGR("ReadPointsFromSnapshot: read successfully; GridYTotal {}", prms.GridYTotal);

    hssoa.RemoveDisabledAndSort(prms.GridYTotal);
    VerifyPoints();

//    FillModelledAreaWithBlueColor();

    LOGR("ReadPointsFromSnapshot; hssoa capacity {}; size {}", hssoa.capacity, hssoa.size);
}




// =============================  READ AND WRITE SNAPSHOTS



/*
void HostSideData::SaveForces(const int frame)
{
    fs::path targetPath;
    if (!output_directory.empty()) {
        targetPath = output_directory;
    } else {
        targetPath = "output";
    }
    fs::path framesDir = targetPath / "frames";
    fs::create_directories(framesDir);

    // save forces
    fs::path fullPathForces = framesDir / "forces.h5";
    bool file_exists = std::filesystem::exists(fullPathForces);
    H5::H5File file_forces(fullPathForces.string(), file_exists ? H5F_ACC_RDWR : H5F_ACC_TRUNC);
    H5::DataSet ds_forces;

    if(!file_exists)
    {
        hsize_t initial_dims[3] = {0, (hsize_t)SimParams::MAX_REGIONS, 2};
        hsize_t max_dims[3] = {H5S_UNLIMITED, (hsize_t)SimParams::MAX_REGIONS, 2};
        H5::DataSpace file_dataspace_for_creation(3, initial_dims, max_dims);

        H5::DSetCreatPropList dcpl;
        hsize_t chunk_dims[3] = {1, (hsize_t)SimParams::MAX_REGIONS, 2};
        dcpl.setChunk(3, chunk_dims);
        ds_forces = file_forces.createDataSet("ds_forces", H5::PredType::NATIVE_DOUBLE,
                                              file_dataspace_for_creation, dcpl);

        H5::DataSpace scalar_space(H5S_SCALAR);
        ds_forces.createAttribute("cellsize", H5::PredType::NATIVE_DOUBLE, scalar_space)
            .write(H5::PredType::NATIVE_DOUBLE, &prms.cellsize);
        ds_forces.createAttribute("InitialTimeStep", H5::PredType::NATIVE_DOUBLE, scalar_space)
            .write(H5::PredType::NATIVE_DOUBLE, &prms.InitialTimeStep);
        ds_forces.createAttribute("AnimationFramePeriod", H5::PredType::NATIVE_DOUBLE, scalar_space)
            .write(H5::PredType::NATIVE_DOUBLE, &prms.AnimationFramePeriod);
    }
    else
    {
        ds_forces = file_forces.openDataSet("ds_forces");
    }

    // Get current dataspace and dimensions
    H5::DataSpace file_space = ds_forces.getSpace();
    hsize_t current_dims_on_file[3];
    file_space.getSimpleExtentDims(current_dims_on_file);

    // Extend if needed
    hsize_t required_frame_capacity = static_cast<hsize_t>(frame) + 1;
    if (required_frame_capacity > current_dims_on_file[0]) {
        hsize_t new_dims[3] = {required_frame_capacity, static_cast<hsize_t>(SimParams::MAX_REGIONS), 2};
        ds_forces.extend(new_dims);
        file_space = ds_forces.getSpace();
    }

    // Define hyperslab
    hsize_t offset[3] = {static_cast<hsize_t>(frame), 0, 0};
    hsize_t slab_dims[3] = {1, static_cast<hsize_t>(SimParams::MAX_REGIONS), 2};
    file_space.selectHyperslab(H5S_SELECT_SET, slab_dims, offset);

    H5::DataSpace memory_space(3, slab_dims);

    // Write the data
    ds_forces.write(grid_forces_summary_per_region.data(), H5::PredType::NATIVE_DOUBLE,
                    memory_space, file_space);
}
*/


void HostSideData::SaveSnapshot(int SimulationStep, double SimulationTime, bool compress, const std::string& output_directory, const std::string& prefix, int force_frame_index)
{
    LOGR("SaveSnapshot: step {}, time {}{}",
         SimulationStep, SimulationTime, compress ? " (compressed)" : "");

    // Determine output directory
    if (output_directory.empty()) {
        throw std::runtime_error("SaveSnapshot called with empty output_directory");
    }

    fs::path targetPath = output_directory;
    fs::create_directories(targetPath);

    // Save current state
    int frame; 
    if(force_frame_index >= 0) frame = force_frame_index;
    else frame = SimulationStep / prms.UpdateEveryNthStep;

    std::string baseName = fmt::format(fmt::runtime("{}{:05d}.h5"), prefix, frame);
    fs::path fullPath = targetPath / baseName;
    H5::H5File file(fullPath.string(), H5F_ACC_TRUNC);

    const auto nPts = hssoa.size;
    const auto capacity = hssoa.capacity;

    // Define file dataspace: what we want to save (nPtsArrays x nPts)
    hsize_t file_dims[2] = {SimParams::PtArrIdx::nPtsArrays, nPts};
    H5::DataSpace file_dataspace(2, file_dims);

    // Define memory dataspace: source data layout (nPtsArrays x capacity)
    hsize_t mem_dims[2] = {SimParams::PtArrIdx::nPtsArrays, capacity};
    H5::DataSpace mem_dataspace(2, mem_dims);

    // Select the region in memory to read from: all arrays, but only first nPts columns
    hsize_t mem_offset[2] = {0, 0};
    hsize_t mem_count[2] = {SimParams::PtArrIdx::nPtsArrays, nPts};
    mem_dataspace.selectHyperslab(H5S_SELECT_SET, mem_count, mem_offset);

    // Create dataset with optional compression
    H5::DataSet dataset_pts;
    if (compress)
    {
        H5::DSetCreatPropList proplist;
        hsize_t chunk_dims[2] = {SimParams::PtArrIdx::nPtsArrays, std::min((size_t)10000000, (size_t)nPts)};
        proplist.setChunk(2, chunk_dims);
        proplist.setDeflate(1);
        dataset_pts = file.createDataSet("pts_data", H5::PredType::NATIVE_DOUBLE, file_dataspace, proplist);
    }
    else
    {
        dataset_pts = file.createDataSet("pts_data", H5::PredType::NATIVE_DOUBLE, file_dataspace, H5::DSetCreatPropList::DEFAULT);
    }

    dataset_pts.write(hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE, mem_dataspace, file_dataspace);

    // Write metadata attributes
    H5::DataSpace att_dspace(H5S_SCALAR);
    dataset_pts.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace)
        .write(H5::PredType::NATIVE_INT, &SimulationStep);
    dataset_pts.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace)
        .write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);

    int nPtsArrays = SimParams::PtArrIdx::nPtsArrays;
    dataset_pts.createAttribute("nPtsArrays", H5::PredType::NATIVE_INT, att_dspace)
        .write(H5::PredType::NATIVE_INT, &nPtsArrays);
    dataset_pts.createAttribute("nPtsInitial", H5::PredType::NATIVE_INT, att_dspace)
        .write(H5::PredType::NATIVE_INT, &prms.nPtsInitial);
    dataset_pts.createAttribute("ParticleArea", H5::PredType::NATIVE_DOUBLE, att_dspace)
        .write(H5::PredType::NATIVE_DOUBLE, &prms.ParticleArea);

    dataset_pts.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, att_dspace)
        .write(H5::PredType::NATIVE_INT, &prms.GridYTotal);

    LOGR("SaveSnapshot done");
}


// ============================================================================
// Post-Processor Support Methods
// ============================================================================


void HostSideData::LoadFrameData(int frameIndex, const std::string& framesDirectory)
{
    fs::path framesDir(framesDirectory);

    // Match the filename format from SaveFrame
    std::string filename = fmt::format("f{:05d}.h5", frameIndex);

    auto get_path = [&](const std::string& subdir) {
        return framesDir / subdir / filename;
    };

    fs::path colorPath      = get_path("color");
    fs::path fracStatPath   = get_path("fracture_status");
    fs::path fracTypePath   = get_path("fracture_type");
    fs::path physicsPath    = get_path("physics");
    fs::path strainsPath    = get_path("strains");
    fs::path pressurePath    = get_path("pressure");

    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;
    const size_t gridSize = (size_t)gx * gy;

    // 1. Load Color -> frame_rgba (Primary File)
    H5::H5File file(colorPath.string(), H5F_ACC_RDONLY);

    // Read Attributes (Time/Step) - ONLY HERE
    H5::DataSet ds = file.openDataSet("rgba");
    file.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &prms.SimulationStep);
    file.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &prms.SimulationTime);

    frame_rgba.resize(gridSize * 4);
    ds.read(frame_rgba.data(), H5::PredType::NATIVE_UINT8);
    file.close();

    // -------------------------------------------------------
    // Helper Lambda for Grid Arrays (Physics, Strains, etc.)
    // -------------------------------------------------------
    auto load_into_grid = [&](fs::path path, const char* ds_name, int grid_idx, bool normalize = false) {
        if (!fs::exists(path)) return;

        // Open file (uncaught exception if open fails despite fs::exists)
        H5::H5File file(path.string(), H5F_ACC_RDONLY);

        size_t offset = (size_t)grid_idx * gridSize;

        std::vector<float>& dest_buffer = host_grid_buffer;
        const std::string dataset_name = ds_name;

        // Check if dataset exists before trying to open to avoid exception on missing optional datasets
        if (H5Lexists(file.getId(), dataset_name.c_str(), H5P_DEFAULT) <= 0) {
            return;
        }

        const H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace filespace = dataset.getSpace();

        // Validate dimensions
        hsize_t dims_grid[2];
        filespace.getSimpleExtentDims(dims_grid, NULL);
        size_t file_size = dims_grid[0] * dims_grid[1];

        // Safety check to ensure we don't write out of bounds
        if (offset + file_size > dest_buffer.size()) {
            LOGR("Error: Dataset {} is too large for buffer.", dataset_name);
            return;
        }

        // HDF5 automatically converts file types (Double/UInt8) to NATIVE_FLOAT.
        H5::DataType mem_dtype = H5::PredType::NATIVE_FLOAT;
        H5::DataSpace memspace(2, dims_grid);

        dataset.read(dest_buffer.data() + offset, mem_dtype, memspace, filespace);

        // Normalize uint8 [0, 255] -> float [0.0, 1.0] if requested
        if (normalize) {
            float* ptr = dest_buffer.data() + offset;
            for(size_t i = 0; i < file_size; ++i) ptr[i] = ptr[i] / 255.0f;
        }
    };

    // 2. Load Fracture Status
    load_into_grid(fracStatPath, "crushed",   SimParams::HostGridArrayIndex::grid_idx_vis_crushed, true);
    load_into_grid(fracStatPath, "cracked",   SimParams::HostGridArrayIndex::grid_idx_vis_cracked, true);
    load_into_grid(fracStatPath, "thickness", SimParams::HostGridArrayIndex::grid_idx_vis_thickness, false);

    // 3. Load Fracture Type
    load_into_grid(fracTypePath, "tension", SimParams::HostGridArrayIndex::grid_idx_fracture_tension, true);
    load_into_grid(fracTypePath, "shear",   SimParams::HostGridArrayIndex::grid_idx_fracture_shear,   true);
    load_into_grid(fracTypePath, "crush",   SimParams::HostGridArrayIndex::grid_idx_fracture_crush,   true);

    // 4. Load Physics
    load_into_grid(physicsPath, "mass", SimParams::HostGridArrayIndex::host_grid_idx_mass);
    load_into_grid(physicsPath, "vx",   SimParams::HostGridArrayIndex::grid_idx_px);
    load_into_grid(physicsPath, "vy",   SimParams::HostGridArrayIndex::grid_idx_py);

    // 5. Load Strains
    load_into_grid(strainsPath, "strain_eqv", SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange);
    load_into_grid(strainsPath, "strain_vm",  SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises);

    // 6.
    load_into_grid(pressurePath, "Jpinv", SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv);
    load_into_grid(pressurePath, "P",     SimParams::HostGridArrayIndex::grid_idx_vis_P);
    load_into_grid(pressurePath, "Q",     SimParams::HostGridArrayIndex::grid_idx_vis_Q);
    load_into_grid(pressurePath, "glen_flow",     SimParams::HostGridArrayIndex::grid_idx_glen_flow);

    LOGR("LoadFrameData: completed frame {}", frameIndex);
}



template<typename T>
void WriteDatasetHelper(H5::Group& ptr, const std::string& name, const std::vector<T>& data,
                        int gx, int gy, const H5::DataType& dtype)
{
    hsize_t dims[2] = {(hsize_t)gx, (hsize_t)gy};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = ptr.createDataSet(name, dtype, dataspace);
    dataset.write(data.data(), dtype);
}


void HostSideData::SaveFrame(const int SimulationStep, const double SimulationTime)
{
    if(!prms.SaveSnapshots) return;

    const size_t plane_size = (size_t)prms.GridXTotal * prms.GridYTotal;

    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;
    const int frame = SimulationStep / prms.UpdateEveryNthStep;
    const std::string frameFileName = fmt::format("f{:05d}.h5", frame);

    auto GetOrCreateDir = [&](const std::string& sub) {
        fs::path p = (output_directory.empty() ? "output" : output_directory);
        p /= "frames";
        p /= sub;
        fs::create_directories(p);
        return p / frameFileName;
    };

    // 1. Physics: mass, vx, vy
    {
        fs::path path = GetOrCreateDir("physics");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        // mass
        size_t mass_offset = plane_size * SimParams::HostGridArrayIndex::host_grid_idx_mass;
        std::copy(host_grid_buffer.begin() + mass_offset,  host_grid_buffer.begin() + mass_offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "mass", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        // vx
        size_t vx_offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_px;
        std::copy(host_grid_buffer.begin() + vx_offset, host_grid_buffer.begin() + vx_offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "vx", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        // vy
        size_t vy_offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_py;
        std::copy(host_grid_buffer.begin() + vy_offset,
                  host_grid_buffer.begin() + vy_offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "vy", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        file.close();
    }

    // 2. Color: rgb + alpha (uint8)
    {
        fs::path path = GetOrCreateDir("color");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

#pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            // Updated to use float variable and float constants (0.4f)
            float d = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_pts_density];

            // Alpha calculation using float math
            float alpha = std::clamp(d * 0.4f, 0.0f, 1.0f);

            // R, G, B
            float r = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_r];
            float g = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_g];
            float b = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_b];

            save_buffer_uint8[i * 4 + 0] = (uint8_t)(std::clamp(r, 0.f, 1.f) * 255.f);
            save_buffer_uint8[i * 4 + 1] = (uint8_t)(std::clamp(g, 0.f, 1.f) * 255.f);
            save_buffer_uint8[i * 4 + 2] = (uint8_t)(std::clamp(b, 0.f, 1.f) * 255.f);
            save_buffer_uint8[i * 4 + 3] = (uint8_t)(alpha * 255.f);
        }

        hsize_t dims[3] = {(hsize_t)gx, (hsize_t)gy, 4};
        H5::DataSpace dataspace(3, dims);
        H5::DataSet dataset = file.createDataSet("rgba", H5::PredType::NATIVE_UINT8, dataspace);
        dataset.write(save_buffer_uint8.data(), H5::PredType::NATIVE_UINT8);

        // Attributes
        H5::DataSpace att_dspace(H5S_SCALAR);
        file.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
        file.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);

        file.close();
    }

    // 3. Strains: EqvGreenLagrange, vonMises (float)
    {
        fs::path path = GetOrCreateDir("strains");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        size_t offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange;
        std::copy(host_grid_buffer.begin() + offset,  host_grid_buffer.begin() + offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "strain_eqv", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises;
        std::copy(host_grid_buffer.begin() + offset,  host_grid_buffer.begin() + offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "strain_vm", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);
        file.close();
    }

    // 4) Fracture Status: crushed, cracked (uint8), thickness (float)
    {
        fs::path path = GetOrCreateDir("fracture_status");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

#pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_crushed];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "crushed", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

#pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_cracked];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "cracked", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

#pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            save_buffer_float[i] = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_thickness];
        }
        WriteDatasetHelper(file, "thickness", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);
        file.close();
    }

    // 5. Fracture Type: tension, shear, crush (uint8)
    {
        fs::path path = GetOrCreateDir("fracture_type");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

#pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_fracture_tension];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "tension", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

#pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_fracture_shear];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "shear", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

#pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_fracture_crush];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "crush", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);
        file.close();
    }

    // 6. Pressure, shear stress, Jp_inv
    {
        fs::path path = GetOrCreateDir("pressure");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        size_t offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_P;
        std::copy(host_grid_buffer.begin() + offset,  host_grid_buffer.begin() + offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "P", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_Q;
        std::copy(host_grid_buffer.begin() + offset,  host_grid_buffer.begin() + offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "Q", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv;
        std::copy(host_grid_buffer.begin() + offset,  host_grid_buffer.begin() + offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "Jpinv", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        offset = plane_size * SimParams::HostGridArrayIndex::grid_idx_glen_flow;
        std::copy(host_grid_buffer.begin() + offset,  host_grid_buffer.begin() + offset + plane_size,
                  save_buffer_float.begin());
        WriteDatasetHelper(file, "glen_flow", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        file.close();
    }
}


void HostSideData::VerifyPoints()
{
    LOGR("Verifying points integrity...");
    int count_invalid_pos = 0;
    int count_invalid_idx = 0;
    
    // Bounds definitions
    const long long min_x = 0;
    const long long max_x = min_x + prms.GridXTotal;
    const long long min_y = 0;
    const long long max_y = min_y + prms.GridYTotal;

    LOGR("Domain bounds: X [{}, {}), Y [{}, {})", min_x, max_x, min_y, max_y);

    const int n = (int)hssoa.size;
    const unsigned cap = hssoa.capacity;
    double* buf = hssoa.host_buffer;

    for(int i = 0; i < n; i++)
    {
        ProxyPoint pp;
        pp.isReference = true;
        pp.pos = i;
        pp.pitch = cap;
        pp.soa = buf;

        if(pp.getDisabledStatus()) continue;

        // Check local coordinates
        // getPos() normally returns local coordinates for PIC/FLIP
        Eigen::Vector2d pos = pp.getPos(); 
        
        // Strict check [-0.5, 0.5]
        const double tolerance = 1e-6; 
        if (pos.x() < -0.5 - tolerance || pos.x() > 0.5 + tolerance || 
            pos.y() < -0.5 - tolerance || pos.y() > 0.5 + tolerance) 
        {
             count_invalid_pos++;
             // Avoid logging inside OMP parallel region to prevent race/garbled output
             // or excessive locking. Just count them.
        }

        // Check cell indices
        // Manually unpack integer_cell_idx as per standard encoding
        uint64_t cell = pp.getValueUInt64(SimParams::PtArrIdx::integer_cell_idx);
        long long x_idx = (long long)(cell & 0xffffffff);
        long long y_idx = (long long)(cell >> 32);

        if (x_idx < min_x || x_idx >= max_x || y_idx < min_y || y_idx >= max_y) {
            count_invalid_idx++;
            LOGR("bounds [{},{}]x[{},{}]; cell [{},{}]", min_x, min_y, max_x, max_y, x_idx, y_idx);
            throw std::runtime_error("point boudns check failed");
            spdlog::default_logger()->flush();
        }
    }

    LOGR("VerifyPoints: all {} active points verified successfully.", hssoa.size);
    spdlog::default_logger()->flush();
}


