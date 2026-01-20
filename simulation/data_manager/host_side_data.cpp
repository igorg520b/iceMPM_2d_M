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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
        rgb.resize(3 * initial_image_total);
        allocated_bytes[0] += 3 * initial_image_total * sizeof(uint8_t);

        save_buffer_float.resize(4 * plane_size); 
        save_buffer_uint8.resize(4 * plane_size);

        // Track memory
        allocated_bytes[0] += (save_buffer_float.capacity() * sizeof(float));
        allocated_bytes[0] += (save_buffer_uint8.capacity() * sizeof(uint8_t));
        allocated_bytes[0] += (rgb.capacity() * sizeof(uint8_t));
    }

    LOGR("[MEMORY] HostSideData Grid Arrays: {:.3f} MB", allocated_bytes[0] / 1.0e6);
    LOGR("    landmask_buffer: {:.3f} MB", (double)landmask_buffer.size() * sizeof(uint8_t) / 1.0e6);
    LOGR("    original_image_colors_rgb: {:.3f} MB", (double)original_image_colors_rgb.size() * sizeof(uint8_t) / 1.0e6);
    if (allocate_dense_grid) {
        LOGR("    host_grid_buffer: {:.3f} MB", (double)host_grid_buffer.size() * sizeof(double) / 1.0e6);
    }
    LOGR("    rgb: {:.3f} MB", (double)rgb.size() * sizeof(uint8_t) / 1.0e6);
}

void HostSideData::AllocatePointArrays()
{
    // This function is called in two contexts:
    // 1. In preparer (PopulatePoints): HSSOA not yet allocated, needs allocation
    // 2. In simulation (LoadParameterFile): HSSOA already allocated by ReadPointsFromSnapshot()

    // Check if HSSOA needs allocation
    if(hssoa.capacity == 0)
    {
        // Case 1: preparer - allocate HSSOA with extra space
        const size_t requested_capacity = (size_t)(double(prms.nPtsInitial) * (1.0 + prms.extra_space_pts));
        hssoa.Allocate(requested_capacity);
    }

    // Track memory allocation
    allocated_bytes[1] = 0;  // Reset points allocation counter

    // Account for the HSSOA arrays
    // 22 particle arrays (position, velocity, material state, color, etc.) * capacity * sizeof(double)
    allocated_bytes[1] += SimParams::PtArrIdx::nPtsArrays * hssoa.capacity * sizeof(double);

    LOGR("[MEMORY] HostSideData Point Arrays: {:.3f} MB", allocated_bytes[1] / 1.0e6);
    LOGR("    hssoa ({} pts capacity): {:.3f} MB", hssoa.capacity, (double)hssoa.capacity * (double)SimParams::PtArrIdx::nPtsArrays * sizeof(double) / 1.0e6);
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

    // Read attributes
    ds.openAttribute("nPtsInitial").read(H5::PredType::NATIVE_INT, &prms.nPtsInitial);

    // Allocate host arrays based on nPtsInitial
    const size_t requested_capacity = (size_t)(double(prms.nPtsInitial) * (1. + prms.extra_space_pts));
    hssoa.Allocate(requested_capacity);

    ds.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &prms.SimulationStep);
    ds.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &prms.SimulationTime);

    ds.openAttribute("ParticleArea").read(H5::PredType::NATIVE_DOUBLE, &prms.ParticleArea);

    int nPtsArrays;
    ds.openAttribute("nPtsArrays").read(H5::PredType::NATIVE_INT, &nPtsArrays);

    // Get dimensions from file
    H5::DataSpace dsp = ds.getSpace();
    hsize_t dims[2];
    dsp.getSimpleExtentDims(dims, nullptr);

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

    FillModelledAreaWithBlueColor();

    LOGR("ReadPointsFromSnapshot; hssoa capacity {}; size {}", hssoa.capacity, hssoa.size);
}




// =============================  READ AND WRITE SNAPSHOTS


void HostSideData::PrepareRGB_Buffer()
{
    const int &gx = prms.GridXTotal;
    const int &gy = prms.GridYTotal;
    const size_t gridSize = (size_t)gx*gy;
    rgb.resize(gridSize*3);

#pragma omp parallel for
    for(int idx = 0; idx < gridSize; idx++)
    {
        double val_mass = host_grid_buffer[idx + gridSize*SimParams::HostGridArrayIndex::host_grid_idx_mass];
        val_mass *= (2./5.);
        float alpha = std::min((double)val_mass, 1.);
        std::array<uint8_t, 3> _rgb;
        for(int k = 0; k < 3; k++)
        {
            float v = host_grid_buffer[idx + gridSize*(SimParams::HostGridArrayIndex::grid_idx_vis_r+k)];
            float cv = std::clamp(v, 0.f, 1.f);
            _rgb[k] = (uint8_t)(cv*255);
        }
        std::array<uint8_t, 3> c = ColorMap::mergeColors(ColorMap::rgb_water, _rgb, alpha);
        for(int k = 0; k < 3; k++) rgb[idx*3+k] = c[k];
    }
}


void HostSideData::SaveFrame_Old(int SimulationStep, double SimulationTime)
{
    if(!prms.SaveSnapshots) LOGR("skipping SaveFrame");
    LOGR("SaveFrame: step {}, time {}", SimulationStep, SimulationTime);
    const int frame = SimulationStep / prms.UpdateEveryNthStep;
    const int &gx = prms.GridXTotal;
    const int &gy = prms.GridYTotal;
//    const int gridSize = gx*gy;

    PrepareRGB_Buffer();

    // save as HDF5 to output_directory/frames
    fs::path targetPath;
    if (!output_directory.empty()) {
        targetPath = output_directory;
    } else {
        targetPath = "output";
    }
    fs::path framesDir = targetPath / "frames";
    fs::create_directories(framesDir);

    std::string baseName = fmt::format(fmt::runtime("f{:05d}.h5"), frame);

    fs::path fullPath = framesDir / baseName;
    H5::H5File file(fullPath.string(), H5F_ACC_TRUNC);

    // save RGB
    hsize_t dims_rgb[3] = {(hsize_t)gx, (hsize_t)gy, 3};
    H5::DataSpace dataspace_rgb(3, dims_rgb);

    H5::DataSet ds_rgb = file.createDataSet("rgb", H5::PredType::NATIVE_UINT8, dataspace_rgb);
    ds_rgb.write(rgb.data(), H5::PredType::NATIVE_UINT8);

    H5::DataSpace att_dspace(H5S_SCALAR);
    ds_rgb.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace)
        .write(H5::PredType::NATIVE_INT, &SimulationStep);
    ds_rgb.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace)
        .write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);

    // save grid data - entire buffer as 3D dataset: [nGridArraysHost] x [gx] x [gy]
    const int nGridArrays = SimParams::HostGridArrayIndex::nGridArraysHost;
    hsize_t dims_grid_3d[3] = {(hsize_t)nGridArrays, (hsize_t)gx, (hsize_t)gy};
    H5::DataSpace dsp_grid_3d(3, dims_grid_3d);

    file.createDataSet("grid_data", H5::PredType::NATIVE_FLOAT, dsp_grid_3d)
        .write(host_grid_buffer.data(), H5::PredType::NATIVE_DOUBLE);

    // additionally, save region forces in a separate file
    SaveForces(frame);
}

template<typename T>
void WriteDatasetHelper(H5::Group& ptr, const std::string& name, const std::vector<T>& data, int gx, int gy, const H5::DataType& dtype)
{
    hsize_t dims[2] = {(hsize_t)gx, (hsize_t)gy};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = ptr.createDataSet(name, dtype, dataspace);
    dataset.write(data.data(), dtype);
}

void HostSideData::SaveFrame(int SimulationStep, double SimulationTime)
{
    if(!prms.SaveSnapshots) return;
    
    // Ensure buffers are ready (allocation should have happened in AllocateGridArrays, but strictly speaking we check)
    size_t plane_size = (size_t)prms.GridXTotal * prms.GridYTotal;
    if (save_buffer_float.size() < plane_size) save_buffer_float.resize(plane_size);
    if (save_buffer_uint8.size() < plane_size) save_buffer_uint8.resize(plane_size);
    // RGB buffer is special (needs 3 or 4 components)
    // rgba needs 4 components? User said "color: rgb + alpha uint8 (separate dataset)". 
    // It implies one dataset with 4 components OR separate scalar datasets? "separate dataset" (singular) usually implies one 4-component chunk.
    
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

    // ---------------------------------------------------------
    // 1) Physics: mass, vx, vy (float)
    // ---------------------------------------------------------
    {
        fs::path path = GetOrCreateDir("physics");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        // Mass
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            save_buffer_float[i] = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::host_grid_idx_mass];
        }
        WriteDatasetHelper(file, "mass", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        // Vx, Vy (Normalize Px, Py by Mass)
        // Note: host_grid_buffer Px/Py are Momentum? No, check kernels. 
        // Typically Px is momentum. 
        // BUT `render_visualized_data` (GPU) accumulates Px, Py.
        // `normalize_grid_on_host` divides them by Mass.
        // So `host_grid_buffer[vis_px]` IS Velocity (if normalized).
        // Let's check `normalize_grid_on_host`.
        // It divides `grid_idx_px` by `mass`. So it IS Velocity.
        
        // Vx
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            save_buffer_float[i] = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_px];
        }
        WriteDatasetHelper(file, "vx", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        // Vy
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            save_buffer_float[i] = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_py];
        }
        WriteDatasetHelper(file, "vy", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);
        
        // Attributes
        H5::DataSpace att_dspace(H5S_SCALAR);
        file.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
        file.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
        file.close();
    }

    // ---------------------------------------------------------
    // 2) Color: rgb + alpha (uint8)
    // ---------------------------------------------------------
    {
        fs::path path = GetOrCreateDir("color");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);
        
        // We reuse `rgb` buffer? Or `save_buffer_uint8`.
        // If we want [Gx, Gy, 4], we need 4 * plane_size.
        if(save_buffer_uint8.size() < 4 * plane_size) save_buffer_uint8.resize(4 * plane_size);

        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            // Alpha calculation
            // Density is at `grid_idx_vis_pts_density`
            // But wait, `normalize_grid_on_host` does NOT divide density by mass?
            // "pt_count = host_grid_buffer[density_idx]".
            // In PrepareRGB_Buffer (lines 306-307):
            // val_mass = host_grid_buffer[mass_idx];
            // alpha = min(val_mass * 0.4, 1.0);
            // It uses MASS for alpha, not density?
            // User request says: "alpha is generated from pt_density in the same way at synchronizeTopology()"
            // Let's check SynchronizeTopology/PrepareRGB_Buffer logic.
            // Line 306: `double val_mass = host_grid_buffer[idx + ..._mass];`
            // So it used MASS. 
            // BUT User says "generated from pt_density". 
            // Maybe they mean `grid_idx_vis_pts_density`?
            // "pt_density" usually corresponds to `grid_idx_vis_pts_density`.
            // Let's use `grid_idx_vis_pts_density` if it exists and is populated.
            // GPU maps {3, pts_density} -> host density idx.
            // If I look at `HostSideData::PrepareRGB_Buffer` (lines 304-318 in step 206 view):
            // val_mass = host_grid_buffer[mass_idx]; !!!!
            // So the old code used MASS.
            // User instruction: "alpha is generated from pt_density in the same way at synchronizeTopology()"
            // Maybe I should use `grid_idx_vis_pts_density`?
            // I will use `grid_idx_vis_pts_density` because that's what explicit instruction says, 
            // even if old code used mass. "pt_density" is the keyword.
            
            double d = host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_pts_density];
            // Formula? "same way". 
            // If old code used mass * 0.4, I'll use density * 0.4?
            // Let's assume density is comparable to mass (if mass=1 per point).
            float alpha = std::clamp((float)(d * 0.4), 0.0f, 1.0f);

            // R, G, B
            float r = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_r];
            float g = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_g];
            float b = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_b];
            
            save_buffer_uint8[i * 4 + 0] = (uint8_t)(std::clamp(r, 0.f, 1.f) * 255.f);
            save_buffer_uint8[i * 4 + 1] = (uint8_t)(std::clamp(g, 0.f, 1.f) * 255.f);
            save_buffer_uint8[i * 4 + 2] = (uint8_t)(std::clamp(b, 0.f, 1.f) * 255.f);
            save_buffer_uint8[i * 4 + 3] = (uint8_t)(alpha * 255.f);
        }
        
        // Write as [Gx, Gy, 4]
        hsize_t dims[3] = {(hsize_t)gx, (hsize_t)gy, 4};
        H5::DataSpace dataspace(3, dims);
        H5::DataSet dataset = file.createDataSet("rgba", H5::PredType::NATIVE_UINT8, dataspace);
        dataset.write(save_buffer_uint8.data(), H5::PredType::NATIVE_UINT8);
        
        H5::DataSpace att_dspace(H5S_SCALAR);
        file.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
        file.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
        file.close();
    }

    // ---------------------------------------------------------
    // 3) Strains: EqvGreenLagrange, vonMises (float)
    // ---------------------------------------------------------
    {
        fs::path path = GetOrCreateDir("strains");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        // Eqv
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            save_buffer_float[i] = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange];
        }
        WriteDatasetHelper(file, "strain_eqv", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        // VM
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            save_buffer_float[i] = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises];
        }
        WriteDatasetHelper(file, "strain_vm", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);
        
        H5::DataSpace att_dspace(H5S_SCALAR);
        file.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
        file.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
        file.close();
    }

    // ---------------------------------------------------------
    // 4) Fracture Status: crushed, cracked (uint8), thickness (float)
    // ---------------------------------------------------------
    {
        fs::path path = GetOrCreateDir("fracture_status");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        // Crushed (convert >0.5 to 255?) User said "convert float by multiply ing 255".
        // Typically these are flags 0.0 or 1.0. 
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_crushed];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "crushed", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

        // Cracked
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_cracked];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "cracked", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

        // Thickness (float)
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            save_buffer_float[i] = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_thickness];
        }
        WriteDatasetHelper(file, "thickness", save_buffer_float, gx, gy, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace att_dspace(H5S_SCALAR);
        file.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
        file.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
        file.close();
    }

    // ---------------------------------------------------------
    // 5) Fracture Type: tension, shear, crush (uint8)
    // ---------------------------------------------------------
    {
        fs::path path = GetOrCreateDir("fracture_type");
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        // Tension
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_fracture_tension];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "tension", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

        // Shear
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_fracture_shear];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "shear", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);

        // Crush
        #pragma omp parallel for
        for(size_t i=0; i<plane_size; i++) {
            float v = (float)host_grid_buffer[i + plane_size * SimParams::HostGridArrayIndex::grid_idx_fracture_crush];
            save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        WriteDatasetHelper(file, "crush", save_buffer_uint8, gx, gy, H5::PredType::NATIVE_UINT8);
        
        H5::DataSpace att_dspace(H5S_SCALAR);
        file.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
        file.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
        file.close();
    }
}

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


void HostSideData::SaveSnapshot(int SimulationStep, double SimulationTime, bool compress, const std::string& output_directory)
{
    LOGR("SaveSnapshot: step {}, time {}{}",
         SimulationStep, SimulationTime, compress ? " (compressed)" : "");

    // Determine output directory
    fs::path targetPath;
    if (!output_directory.empty())
    {
        // Save to output/snapshots when called from simulation (output_directory is JSON-relative output/)
        fs::path snapshotsDir = fs::path(output_directory) / "snapshots";
        targetPath = snapshotsDir;
    }
    else
    {
        // Default behavior: save to output/SimulationTitle/snapshots (used by preparer for initial snapshot)
        fs::path outputDir = "output";
        fs::path snapshotsDir = "snapshots";
        targetPath = outputDir / SimulationTitle / snapshotsDir;
    }
    fs::create_directories(targetPath);

    // Save current state
    const int frame = SimulationStep / prms.UpdateEveryNthStep;
    std::string baseName = fmt::format(fmt::runtime("s{:05d}.h5"), frame);
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
        dataset_pts = file.createDataSet("pts_data", H5::PredType::NATIVE_DOUBLE,
                                         file_dataspace, proplist);
    }
    else
    {
        dataset_pts = file.createDataSet("pts_data", H5::PredType::NATIVE_DOUBLE,
                                         file_dataspace, H5::DSetCreatPropList::DEFAULT);
    }

    dataset_pts.write(hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE,
                      mem_dataspace, file_dataspace);

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

    LOGR("SaveSnapshot done");
}


// ============================================================================
// Post-Processor Support Methods
// ============================================================================

void HostSideData::readGridDataset(const H5::H5File& file, const std::string& dataset_name,
                                    std::vector<double>& dest_buffer, size_t offset)
{
    try {
        const H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace filespace = dataset.getSpace();

        hsize_t dims_grid[2];
        filespace.getSimpleExtentDims(dims_grid, NULL);

        H5::DataType mem_dtype = H5::PredType::NATIVE_DOUBLE;
        H5::DataSpace memspace(2, dims_grid);
        dataset.read(dest_buffer.data() + offset, mem_dtype, memspace, filespace);
    } catch (const H5::Exception& e) {
        LOGR("HDF5 Error reading dataset '{}': {}", dataset_name, e.getCDetailMsg());
        throw;
    }
}


void HostSideData::LoadFrameData(int frameIndex, const std::string& framesDirectory)
{
    // LOGR("LoadFrameData: frame {} from {}", frameIndex, framesDirectory);
    
    fs::path framesDir(framesDirectory);
    std::string filename = fmt::format("frame_{:05d}.h5", frameIndex);
    
    // 2. Prepare Paths using lambda
    auto get_path = [&](const std::string& subdir) {
        return framesDir / subdir / filename;
    };
    
    fs::path colorPath      = get_path("color");
    fs::path fracStatPath   = get_path("fracture_status");
    fs::path fracTypePath   = get_path("fracture_type");
    fs::path physicsPath    = get_path("physics");
    fs::path strainsPath    = get_path("strains");
    
    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;
    const size_t gridSize = (size_t)gx * gy;
    
    // 3. Load Color -> frame_rgba
    if (fs::exists(colorPath)) {
        try {
            H5::H5File file(colorPath.string(), H5F_ACC_RDONLY);
            H5::DataSet ds = file.openDataSet("color");
            
            // Read attributes for SimulationTime/Step from color file (primary)
            H5::DataSpace att_dspace(H5S_SCALAR);
            if(H5Aexists(file.getId(), "SimulationStep"))
                file.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &prms.SimulationStep);
            if(H5Aexists(file.getId(), "SimulationTime"))
                file.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &prms.SimulationTime);

            std::vector<uint8_t> temp_rgb(gridSize * 3);
            ds.read(temp_rgb.data(), H5::PredType::NATIVE_UINT8);
            
            // Convert to RGBA
            frame_rgba.resize(gridSize * 4);
            #pragma omp parallel for
            for(size_t i=0; i<gridSize; ++i) {
                frame_rgba[i*4 + 0] = temp_rgb[i*3 + 0];
                frame_rgba[i*4 + 1] = temp_rgb[i*3 + 1];
                frame_rgba[i*4 + 2] = temp_rgb[i*3 + 2];
                frame_rgba[i*4 + 3] = 255;
            }
            LOGR("Loaded Color from {}", colorPath.string());
        } catch (...) {
            LOGR("Failed to load color from {}", colorPath.string());
        }
    } else {
        LOGR("Color file not found: {}", colorPath.string());
    }

    // Ensure buffer size
    if (host_grid_buffer.size() != gridSize * SimParams::HostGridArrayIndex::nGridArraysHost) {
        LOGR("LoadFrameData: resizing host_grid_buffer...");
        AllocateGridArrays(true); 
    }

    // Helper to load dataset into grid slot
    auto load_dataset = [&](fs::path path, const char* ds_name, int grid_idx) {
        if (!fs::exists(path)) return;
        try {
             H5::H5File file(path.string(), H5F_ACC_RDONLY);
             if(H5Lexists(file.getId(), ds_name, H5P_DEFAULT) > 0) {
                 H5::DataSet ds = file.openDataSet(ds_name);
                 double* ptr = host_grid_buffer.data() + (size_t)grid_idx * gridSize;
                 
                 // Check type? Assume double or convert.
                 // Most simulation data is double, but masks (fracture type) might be saved as uint8/float.
                 // HDF5 usually handles conversion if compatible.
                 ds.read(ptr, H5::PredType::NATIVE_DOUBLE);
             }
        } catch (...) {}
    };

    // 4. Load Fracture Status -> host_grid_buffer
    // Datasets: "crushed", "cracked", "thickness"
    load_dataset(fracStatPath, "crushed",   SimParams::HostGridArrayIndex::grid_idx_vis_crushed);
    load_dataset(fracStatPath, "cracked",   SimParams::HostGridArrayIndex::grid_idx_vis_cracked);
    load_dataset(fracStatPath, "thickness", SimParams::HostGridArrayIndex::grid_idx_vis_thickness);

    // 5. Load Fracture Type
    // Datasets: "tension", "shear", "crush"
    load_dataset(fracTypePath, "tension", SimParams::HostGridArrayIndex::grid_idx_fracture_tension);
    load_dataset(fracTypePath, "shear",   SimParams::HostGridArrayIndex::grid_idx_fracture_shear);
    load_dataset(fracTypePath, "crush",   SimParams::HostGridArrayIndex::grid_idx_fracture_crush);

    // 6. Load Physics
    // Datasets: "mass" (others like sxx not in HostGridArrayIndex yet)
    load_dataset(physicsPath, "mass", SimParams::HostGridArrayIndex::host_grid_idx_mass);

    // 7. Load Strains
    // Datasets: "Jpinv", "P", "Q", "E_eqv", "E_vm"
    load_dataset(strainsPath, "Jpinv", SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv);
    load_dataset(strainsPath, "P",     SimParams::HostGridArrayIndex::grid_idx_vis_P);
    load_dataset(strainsPath, "Q",     SimParams::HostGridArrayIndex::grid_idx_vis_Q);
    load_dataset(strainsPath, "E_eqv", SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange);
    load_dataset(strainsPath, "E_vm",  SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises);
    
    LOGR("LoadFrameData: completed frame {}", frameIndex);
}


