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
    const int &width = prms.InitializationImageSizeX;
    const int &height = prms.InitializationImageSizeY;
    const int &ox = prms.ModeledRegionOffsetX;
    const int &oy = prms.ModeledRegionOffsetY;
    const int &gx = prms.GridXTotal;
    const int &gy = prms.GridYTotal;

    for(int i = 0; i < gx; i++)
        for(int j = 0; j < gy; j++)
        {
            uint8_t status = landmask_buffer[j + i*gy];
            if(status == SimParams::ModelledAreaIndicator)
            {
                for(int k = 0; k < 3; k++)
                    original_image_colors_rgb[((i+ox)+(j+oy)*width)*3+k] = ColorMap::rgb_water[k];
            }
        }
}


// =============================  ALLOCATION FUNCTIONS

void HostSideData::AllocateGridArrays()
{
    // Allocate grid buffers (for both preparer and simulation)
    const int modeled_grid_total = prms.GridXTotal * prms.GridYTotal;
    const int initial_image_total = prms.InitializationImageSizeX * prms.InitializationImageSizeY;

    allocated_bytes[0] = 0;  // Reset grid allocation counter

    landmask_buffer.resize(modeled_grid_total);
    allocated_bytes[0] += modeled_grid_total * sizeof(uint8_t);

    original_image_colors_rgb.resize(3 * initial_image_total);
    allocated_bytes[0] += 3 * initial_image_total * sizeof(uint8_t);

    host_grid_buffer.resize(modeled_grid_total * SimParams::HostGridArrayIndex::nGridArraysHost);
    allocated_bytes[0] += modeled_grid_total * SimParams::HostGridArrayIndex::nGridArraysHost * sizeof(double);

    tmp_halo_buffer.resize(prms.GridYTotal * prms.GridHaloSize * SimParams::HostGridArrayIndex::nGridArraysHost);
    allocated_bytes[0] += prms.GridYTotal * prms.GridHaloSize * SimParams::HostGridArrayIndex::nGridArraysHost * sizeof(double);

    rgb.resize(3 * initial_image_total);
    allocated_bytes[0] += 3 * initial_image_total * sizeof(uint8_t);

    LOGR("AllocateGridArrays: Grid memory: {:.3f} GB", allocated_bytes[0] / 1e9);
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

    // Ensure point_partitions is allocated
    if(point_partitions.size() < prms.nPtsInitial)
    {
        point_partitions.resize(prms.nPtsInitial);
    }
    allocated_bytes[1] += prms.nPtsInitial * sizeof(uint8_t);

    LOGR("AllocatePointArrays: Points memory: {:.3f} GB", allocated_bytes[1] / 1e9);
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

        // Allocate grid arrays
        AllocateGridArrays();

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


// =============================  UNIFIED GRID AND POINTS PREPARATION

void HostSideData::PrepareGridAndPoints(std::string fileNameLandMask, std::string fileNameColor,
                                       std::string fileNameIceMask, std::string fileNameCrushedMask,
                                       std::string fileNameCrackedMask,
                                       std::string projectDirectory, double dimensionHorizontal, int pointsPerCell,
                                       double thicknessFrom, double thicknessTo, std::string fileNameThicknessMask)
{
    LOGR("PrepareGridAndPoints: starting");

    // Store the data directory for later use when saving snapshots
    data_directory = projectDirectory;

    // Load ImageColor first to determine image dimensions (ImageColor is mandatory)
    int channels_color, width, height;
    unsigned char *color_raw = stbi_load(fileNameColor.c_str(), &width, &height, &channels_color, 3);
    if (!color_raw) {
        throw std::runtime_error("Failed to load ImageColor: " + fileNameColor);
    }
    std::vector<uint8_t> color_vec(color_raw, color_raw + width * height * 3);
    stbi_image_free(color_raw);

    // Load or create ImageLandMask (optional)
    std::vector<uint8_t> landmask_vec;
    if (!fileNameLandMask.empty()) {
        int channels_land, width_land, height_land;
        unsigned char *landmask_raw = stbi_load(fileNameLandMask.c_str(), &width_land, &height_land, &channels_land, 1);
        if (!landmask_raw) {
            throw std::runtime_error("Failed to load ImageLandMask: " + fileNameLandMask);
        }
        if (width_land != width || height_land != height) {
            stbi_image_free(landmask_raw);
            throw std::runtime_error(
                fmt::format("ImageLandMask dimension mismatch: expected {}x{}, got {}x{}",
                           width, height, width_land, height_land)
            );
        }
        landmask_vec.assign(landmask_raw, landmask_raw + width * height);
        stbi_image_free(landmask_raw);
        LOGR("Loaded ImageLandMask from file: {}x{}", width, height);
    } else {
        landmask_vec.assign(width * height, 0); // Default to all water
        LOGR("No ImageLandMask provided - creating zero buffer (entire domain is modeled area): {}x{}", width, height);
    }

    // Load ImageIceMask (mandatory)
    int channels_ice, width_ice, height_ice;
    unsigned char *icemask_raw = stbi_load(fileNameIceMask.c_str(), &width_ice, &height_ice, &channels_ice, 1);
    if (!icemask_raw) {
        throw std::runtime_error("Failed to load ImageIceMask: " + fileNameIceMask);
    }
    if (width_ice != width || height_ice != height) {
        stbi_image_free(icemask_raw);
        throw std::runtime_error(
            fmt::format("ImageIceMask dimension mismatch: expected {}x{}, got {}x{}",
                       width, height, width_ice, height_ice)
        );
    }
    std::vector<uint8_t> icemask_vec(icemask_raw, icemask_raw + width * height);
    stbi_image_free(icemask_raw);

    // Load crushed mask if provided (optional)
    std::vector<uint8_t> crushed_vec(width * height, 255); // Default to white (not crushed)
    bool has_crushed_mask = false;
    if (!fileNameCrushedMask.empty()) {
        int channels_crushed, width_crushed, height_crushed;
        unsigned char *crushed_raw = stbi_load(fileNameCrushedMask.c_str(), &width_crushed, &height_crushed, &channels_crushed, 1);
        if (!crushed_raw) {
            throw std::runtime_error("Failed to load ImageCrushedMask: " + fileNameCrushedMask);
        }
        if (width_crushed != width || height_crushed != height) {
            stbi_image_free(crushed_raw);
            throw std::runtime_error(
                fmt::format("ImageCrushedMask dimension mismatch: expected {}x{}, got {}x{}",
                           width, height, width_crushed, height_crushed)
            );
        }
        crushed_vec.assign(crushed_raw, crushed_raw + width * height);
        stbi_image_free(crushed_raw);
        has_crushed_mask = true;
    }

    // Load cracked mask if provided (optional)
    std::vector<uint8_t> cracked_vec(width * height, 255); // Default to white (not cracked)
    if (!fileNameCrackedMask.empty()) {
        int channels_cracked, width_cracked, height_cracked;
        unsigned char *cracked_raw = stbi_load(fileNameCrackedMask.c_str(), &width_cracked, &height_cracked, &channels_cracked, 1);
        if (!cracked_raw) {
            throw std::runtime_error("Failed to load ImageCrackedMask: " + fileNameCrackedMask);
        }
        if (width_cracked != width || height_cracked != height) {
            stbi_image_free(cracked_raw);
            throw std::runtime_error(
                fmt::format("ImageCrackedMask dimension mismatch: expected {}x{}, got {}x{}",
                           width, height, width_cracked, height_cracked)
            );
        }
        cracked_vec.assign(cracked_raw, cracked_raw + width * height);
        stbi_image_free(cracked_raw);
    }

    // Load or generate thickness mask (optional)
    std::vector<uint8_t> thickness_vec;
    bool has_thickness_mask = false;

    if (!fileNameThicknessMask.empty()) {
        int channels_thickness, width_thickness, height_thickness;
        unsigned char *thickness_raw = stbi_load(fileNameThicknessMask.c_str(), &width_thickness, &height_thickness, &channels_thickness, 1);
        if (!thickness_raw) {
            throw std::runtime_error("Failed to load ImageThicknessMask: " + fileNameThicknessMask);
        }
        if (width_thickness != width || height_thickness != height) {
            stbi_image_free(thickness_raw);
            throw std::runtime_error(
                fmt::format("ImageThicknessMask dimension mismatch: expected {}x{}, got {}x{}",
                           width, height, width_thickness, height_thickness)
            );
        }
        thickness_vec.assign(thickness_raw, thickness_raw + width * height);
        stbi_image_free(thickness_raw);
        has_thickness_mask = true;
    } else {
        thickness_vec.resize(width * height);
        std::mt19937 gen(1337);
        std::normal_distribution<double> dist(127.5, 255.0 / 4.0);
        for (int i = 0; i < width * height; ++i) {
            thickness_vec[i] = (uint8_t)std::clamp(dist(gen), 0.0, 255.0);
        }
        has_thickness_mask = true;
        LOGR("No ImageThicknessMask provided - initialized with random values (Normal distribution, mean 127.5, 4-sigma range)");
    }

    // Flip all images vertically (PNG uses top-left origin, simulation uses bottom-left)
    std::vector<uint8_t> landmask_flipped(width * height);
    std::vector<uint8_t> color_flipped(width * height * 3);
    std::vector<uint8_t> icemask_flipped(width * height);
    std::vector<uint8_t> crushed_flipped(width * height);
    std::vector<uint8_t> cracked_flipped(width * height);
    std::vector<uint8_t> thickness_flipped(width * height);

    for (int y = 0; y < height; y++) {
        int y_flipped = height - y - 1;
        for (int x = 0; x < width; x++) {
            landmask_flipped[x + y * width] = landmask_vec[x + y_flipped * width];
            icemask_flipped[x  + y * width] = icemask_vec[x  + y_flipped * width];
            crushed_flipped[x  + y * width] = crushed_vec[x  + y_flipped * width];
            cracked_flipped[x  + y * width] = cracked_vec[x  + y_flipped * width];
            thickness_flipped[x + y * width] = thickness_vec[x + y_flipped * width];
            for (int c = 0; c < 3; c++) {
                color_flipped[(x + y * width) * 3 + c] = color_vec[(x + y_flipped * width) * 3 + c];
            }
        }
    }

    LOGR("All images flipped");

    // Save original colors before PrepareGrid paints water blue (needed for point colors)
    std::vector<uint8_t> original_colors_copy = color_flipped;

    // Process grid using flipped images (this will modify color_flipped by painting water blue)
    PrepareGrid(landmask_flipped, color_flipped, width, height, projectDirectory, dimensionHorizontal);

    // Process points using original colors (before blue painting)
    PopulatePoints(icemask_flipped, crushed_flipped, cracked_flipped, original_colors_copy, width, height, pointsPerCell,
                   thicknessFrom, thicknessTo, thickness_flipped);

    LOGR("PrepareGridAndPoints: completed");
}

// =============================  PREPARE GRID

void HostSideData::PrepareGrid(const std::vector<uint8_t> &landmask, std::vector<uint8_t> &color,
                               int imgWidth, int imgHeight, std::string projectDirectory, double dimensionHorizontal)
{
    LOGR("PrepareGrid: starting");

    prms.InitializationImageSizeX = imgWidth;
    prms.InitializationImageSizeY = imgHeight;

    // (1) Find cropped area (bounding box of non-water pixels in flipped landmask)
    // Black pixels (low values) = water = modeled area
    // White pixels (high values) = land
    int xmin = imgWidth;
    int xmax = -1;
    int ymin = imgHeight;
    int ymax = -1;

    for (int i = 0; i < imgWidth; i++) {
        for (int j = 0; j < imgHeight; j++) {
            uint8_t pixel = landmask[i + j * imgWidth];
            // Threshold: if pixel < 128, consider it water (modeled area)
            if (pixel < 128) {
                xmin = std::min(xmin, i);
                xmax = std::max(xmax, i);
                ymin = std::min(ymin, j);
                ymax = std::max(ymax, j);
            }
        }
    }

    // Pad by ±2 cells and clamp
    xmin = std::max(0, xmin - 2);
    xmax = std::min(imgWidth - 1, xmax + 2);
    ymin = std::max(0, ymin - 2);
    ymax = std::min(imgHeight - 1, ymax + 2);

    prms.ModeledRegionOffsetX = xmin;
    prms.ModeledRegionOffsetY = ymin;
    prms.GridXTotal = xmax - xmin + 1;
    prms.GridYTotal = ymax - ymin + 1;

    if (prms.GridXTotal <= 0 || prms.GridYTotal <= 0) {
        throw std::runtime_error("Modeled area not found in landmask image");
    }

    LOGR("Initialization image: {} x {}", prms.InitializationImageSizeX, prms.InitializationImageSizeY);
    LOGR("Grid size: {} x {}", prms.GridXTotal, prms.GridYTotal);
    LOGR("Modeled area offset: {}, {}", prms.ModeledRegionOffsetX, prms.ModeledRegionOffsetY);

    // (2) Calculate physical parameters
    prms.DimensionHorizontal = dimensionHorizontal;
    prms.cellsize = prms.DimensionHorizontal / (prms.InitializationImageSizeX - 1);
    prms.cellsize_inv = 1.0 / prms.cellsize;

    LOGR("Cell size: {}, DimensionHorizontal: {}", prms.cellsize, prms.DimensionHorizontal);

    // (3) Allocate grid arrays
    AllocateGridArrays();

    // (4) Build landmask_buffer (cropped region, column-major: idx = j + i*GridY)
    // Images are already flipped, so no additional Y-flip needed
    for (int i = 0; i < prms.GridXTotal; i++) {
        for (int j = 0; j < prms.GridYTotal; j++) {
            int img_x = i + prms.ModeledRegionOffsetX;
            int img_y = j + prms.ModeledRegionOffsetY;
            uint8_t pixel = landmask[img_x + img_y * imgWidth];

            // Black (pixel < 128) = water = ModelledAreaIndicator, White (pixel >= 128) = land = 0
            uint8_t status = (pixel < 128) ? SimParams::ModelledAreaIndicator : 0;

            size_t idx = j + (size_t)i * prms.GridYTotal;  // column-major
            landmask_buffer[idx] = status;
        }
    }

    // (5) Store original colors (images already flipped, just copy)
    for (int i = 0; i < imgWidth; i++) {
        for (int j = 0; j < imgHeight; j++) {
            for (int k = 0; k < 3; k++) {
                original_image_colors_rgb[(i + j * imgWidth) * 3 + k] = color[(i + j * imgWidth) * 3 + k];
            }
        }
    }

    // (6) Fill water areas with blue color for visualization
    FillModelledAreaWithBlueColor();

    // (7) Save grid HDF5 file
    std::string gridFilePath = projectDirectory + "/grid.h5";
    H5::H5File file(gridFilePath, H5F_ACC_TRUNC);

    // Save landmask as 2D dataset (column-major format: {GridX, GridY})
    hsize_t landmask_dims[2] = {static_cast<hsize_t>(prms.GridXTotal), static_cast<hsize_t>(prms.GridYTotal)};
    H5::DataSpace landmask_space(2, landmask_dims);

    H5::DSetCreatPropList landmask_props;
    hsize_t landmask_chunks[2] = {std::min<hsize_t>(prms.GridXTotal, 64), std::min<hsize_t>(prms.GridYTotal, 64)};
    landmask_props.setChunk(2, landmask_chunks);
    landmask_props.setDeflate(6);

    H5::DataSet dataset_landmask = file.createDataSet("landmask", H5::PredType::NATIVE_UINT8,
                                                      landmask_space, landmask_props);
    dataset_landmask.write(landmask_buffer.data(), H5::PredType::NATIVE_UINT8);

    // Save color as 3D dataset [Height][Width][3] (row-major)
    hsize_t color_dims[3] = {static_cast<hsize_t>(prms.InitializationImageSizeY),
                              static_cast<hsize_t>(prms.InitializationImageSizeX), 3};
    H5::DataSpace color_space(3, color_dims);

    H5::DSetCreatPropList color_props;
    hsize_t color_chunks[3] = {std::min<hsize_t>(prms.InitializationImageSizeY, 64),
                                std::min<hsize_t>(prms.InitializationImageSizeX, 64), 3};
    color_props.setChunk(3, color_chunks);
    color_props.setDeflate(6);

    H5::DataSet dataset_color = file.createDataSet("color_grid", H5::PredType::NATIVE_UINT8,
                                                   color_space, color_props);
    dataset_color.write(original_image_colors_rgb.data(), H5::PredType::NATIVE_UINT8);

    // Write metadata attributes
    H5::DataSpace att_space(H5S_SCALAR);

    int gx = prms.GridXTotal, gy = prms.GridYTotal;
    int ox = prms.ModeledRegionOffsetX, oy = prms.ModeledRegionOffsetY;
    int img_x = prms.InitializationImageSizeX, img_y = prms.InitializationImageSizeY;

    dataset_landmask.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &gx);
    dataset_landmask.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &gy);
    dataset_landmask.createAttribute("OffsetX", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &ox);
    dataset_landmask.createAttribute("OffsetY", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &oy);
    dataset_landmask.createAttribute("InitImageSizeX", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &img_x);
    dataset_landmask.createAttribute("InitImageSizeY", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &img_y);

    double cellsize = prms.cellsize;
    double dim_horiz = prms.DimensionHorizontal;
    dataset_landmask.createAttribute("CellSize", H5::PredType::NATIVE_DOUBLE, att_space).write(H5::PredType::NATIVE_DOUBLE, &cellsize);
    dataset_landmask.createAttribute("DimensionHorizontal", H5::PredType::NATIVE_DOUBLE, att_space).write(H5::PredType::NATIVE_DOUBLE, &dim_horiz);

    file.close();

    LOGR("PrepareGrid completed");
}

// =============================  POISSON POINT GENERATION HELPERS

std::string HostSideData::prepare_cache_filename(int gx, int gy, int ppc)
{
    return fmt::format("_data/poisson_cache/points_{}x{}_{:d}.h5", gx, gy, ppc);
}

bool HostSideData::attempt_to_fill_from_cache(int gx, int gy, int ppc, std::vector<std::array<float, 2>> &buffer)
{
    std::string cache_file = prepare_cache_filename(gx, gy, ppc);
    std::filesystem::path cache_path(cache_file);

    if (!std::filesystem::exists(cache_path)) {
        LOGR("Poisson cache file does not exist");
        return false;
    }

    LOGR("Attempting to load Poisson points from cache: {}", cache_file);
    try {
        H5::H5File cache_hfile(cache_file, H5F_ACC_RDONLY);
        H5::DataSet cache_dataset = cache_hfile.openDataSet("coords");

        hsize_t dims[2];
        cache_dataset.getSpace().getSimpleExtentDims(dims, NULL);
        if (dims[1] != 2) {
            LOGR("Cache file has invalid format");
            return false;
        }

        buffer.resize(dims[0]);
        cache_dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
        cache_hfile.close();

        LOGR("Loaded {} points from cache", buffer.size());
        return true;
    } catch (const H5::Exception &e) {
        LOGR("Failed to read cache: {}", e.getCDetailMsg());
        buffer.clear();
        return false;
    }
}

void HostSideData::generate_and_save_poisson(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer)
{
    const float dy = (float)gy / gx;

    LOGR("Generating Poisson points with radius-based parameters");

    const std::array<float, 2> x_min{0, 0};
    const std::array<float, 2> x_max{1, dy};
    constexpr float magic_constant = 0.6;
    const float radius = std::sqrt(magic_constant / (points_per_cell * gx * gx));

    LOGR("Poisson parameters: grid={}x{}, dy={}, radius={}, target_ppc={}", gx, gy, dy, radius, points_per_cell);

    buffer = thinks::PoissonDiskSampling(radius, x_min, x_max);
    LOGR("Generated {} raw Poisson points", buffer.size());

    // Log the actual points per cell achieved before scaling
    const float raw_ppc = (float)buffer.size() / (gx * gy);
    LOGR("Raw achieved ppc: {:.4f} (target was {:.4f})", raw_ppc, points_per_cell);

    // Scale points to achieve target ppc
    const float scale = std::sqrt(raw_ppc / (points_per_cell * 1.0005f));
    if (scale < 1.0f) {
        LOGR("requested ppc {}; generated ppc {}", points_per_cell, raw_ppc);
        throw std::runtime_error("point generation error: achieved ppc is lower than requested");
    }

    LOGR("requested ppc {}; generated ppc {}; scale {}%", points_per_cell, raw_ppc, 100.0f * (scale - 1.0f));

    // Scale all points
    for (auto &pt : buffer) {
        pt[0] *= scale;
        pt[1] *= scale;
    }

    // Remove out-of-bounds points after scaling
    auto result_it = std::remove_if(buffer.begin(), buffer.end(),
                                    [&](const std::array<float, 2> &pt) {
                                        return (pt[0] > 1.0f || pt[1] > dy || pt[0] < 0.0f || pt[1] < 0.0f);
                                    });
    buffer.erase(result_it, buffer.end());

    // Log final ppc after cropping
    const float final_ppc = (float)buffer.size() / (gx * gy);
    LOGR("grid: {}x{}; final pts {:>8}; final_ppc {:.4f}", gx, gy, buffer.size(), final_ppc);

    // Save to cache
    std::string cache_file = prepare_cache_filename(gx, gy, (int)points_per_cell);
    std::filesystem::create_directories("_data/poisson_cache");
    try {
        H5::H5File cache_hfile(cache_file, H5F_ACC_TRUNC);

        hsize_t dims[2] = {buffer.size(), 2};
        H5::DataSpace cache_space(2, dims);

        H5::DataSet cache_dataset = cache_hfile.createDataSet("coords", H5::PredType::NATIVE_FLOAT, cache_space);
        cache_dataset.write(buffer.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSpace att_space(H5S_SCALAR);
        int gx_attr = gx, gy_attr = gy, ppc_attr = (int)points_per_cell;
        cache_dataset.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &gx_attr);
        cache_dataset.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &gy_attr);
        cache_dataset.createAttribute("PointsPerCell", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &ppc_attr);

        cache_hfile.close();
        LOGR("Saved Poisson points to cache: {}", cache_file);
    } catch (const H5::Exception &e) {
        LOGR("Warning: Failed to save cache: {}", e.getCDetailMsg());
    }
}


// =============================  POPULATE POINTS

void HostSideData::PopulatePoints(const std::vector<uint8_t> &icemask, const std::vector<uint8_t> &crushed,
                                  const std::vector<uint8_t> &cracked,
                                  const std::vector<uint8_t> &original_colors, int imgWidth, int imgHeight, int pointsPerCell,
                                  double thicknessFrom, double thicknessTo, const std::vector<uint8_t> &thicknessMask)
{
    LOGR("PopulatePoints: starting");
    LOGR("  Ice thickness scaling: [{}, {}]", thicknessFrom, thicknessTo);

    // (3) Generate or load Poisson points
    std::vector<std::array<float, 2>> pt_buffer;
    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;
    const float points_per_cell = (float)pointsPerCell;

    // Try to load from cache, or generate if not found
    if (!attempt_to_fill_from_cache(gx, gy, pointsPerCell, pt_buffer)) {
        generate_and_save_poisson(gx, gy, points_per_cell, pt_buffer);
    }

    if (pt_buffer.empty()) {
        throw std::runtime_error("No Poisson points generated or loaded");
    }

    // (4) Calculate ParticleArea BEFORE filtering
    const double h = prms.cellsize;
    prms.ParticleArea = (h * h * gx * gy) / (double)pt_buffer.size();
    LOGR("ParticleArea (before filtering): {}", prms.ParticleArea);

    // (5) Helper lambda to convert normalized point coords to grid indices
    auto idxPt = [&](const std::array<float, 2> &pt) -> std::pair<int, int> {
        const double scale = gx - 1;
        return {(int)(pt[0] * scale + 0.5), (int)(pt[1] * scale + 0.5)};
    };

    // (6) Filter points: remove if on boundary, on land, or not ice
    auto shouldRemove = [&](const std::array<float, 2> &pt) -> bool {
        auto [i, j] = idxPt(pt);

        // Boundary check
        if (i <= 1 || j <= 1 || i >= (gx - 2) || j >= (gy - 2)) {
            return true;
        }

        // Land check
        int img_x = i + prms.ModeledRegionOffsetX;
        int img_y = j + prms.ModeledRegionOffsetY;
        uint8_t land_status = landmask_buffer[j + (size_t)i * gy];
        if (land_status != SimParams::ModelledAreaIndicator) {
            return true;  // Is land, remove
        }

        // Ice check (white=ice, black=no-ice; threshold at 128)
        // Images are already flipped, so no additional Y-flip needed
        uint8_t ice_pixel = icemask[img_x + img_y * imgWidth];
        if (ice_pixel < 128) {
            return true;  // Not ice, remove
        }

        return false;  // Keep this point
    };

    std::erase_if(pt_buffer, shouldRemove);
    prms.nPtsInitial = pt_buffer.size();
    LOGR("After filtering: {} points remaining", prms.nPtsInitial);

    if (prms.nPtsInitial == 0) {
        throw std::runtime_error("All points were filtered out");
    }

    // (7) Allocate HostSideSOA
    AllocatePointArrays();
    hssoa.size = prms.nPtsInitial;

    // (8) Transfer points to HostSideSOA
    const double pointScale = (gx - 1) * h;
    const int ox = prms.ModeledRegionOffsetX;
    const int oy = prms.ModeledRegionOffsetY;
    const int width = prms.InitializationImageSizeX;

    for (size_t k = 0; k < pt_buffer.size(); k++) {
        std::array<float, 2> &pt = pt_buffer[k];
        auto [i, j] = idxPt(pt);

        SOAIterator it = hssoa.begin() + k;
        ProxyPoint &p = *it;

        // Position (scaled to physical coordinates)
        p.setValue(SimParams::PtArrIdx::posx, pt[0] * pointScale);
        p.setValue(SimParams::PtArrIdx::posx + 1, pt[1] * pointScale);

        // Point index
        // p.setValueInt(SimParams::PtArrIdx::integer_point_idx, k);

        // Color from image (using original colors, not the blue-painted version)
        const size_t idx_in_image = ((i + ox) + (j + oy) * width) * 3;
        uint32_t r = original_colors[idx_in_image + 0];
        uint32_t g = original_colors[idx_in_image + 1];
        uint32_t b = original_colors[idx_in_image + 2];

        // RGB stored in utility_data now


        // Determine thickness: use thickness mask if provided, otherwise use crushed mask
        // Images are already flipped, so no additional Y-flip needed
        float thickness;
        bool is_crushed = false;

        if (!thicknessMask.empty()) {
            // Use thickness mask to determine thickness
            // White (255) = ThicknessTo, Black (0) = ThicknessFrom, Gray = interpolation
            uint8_t thickness_pixel = thicknessMask[(i + ox) + (j + oy) * width];
            thickness = (float)thickness_pixel / 255.0f;  // Scale to [0, 1]
            // Scale thickness from [0, 1] range to [ThicknessFrom, ThicknessTo]
            thickness = (float)(thicknessFrom + thickness * (thicknessTo - thicknessFrom));

            // Determine crushed status from the thickness mask if crushed mask is not available
            // If both masks are present, the crushed mask is purely for the crushed flag
            if (!crushed.empty()) {
                uint8_t crushed_pixel = crushed[(i + ox) + (j + oy) * width];
                is_crushed = (crushed_pixel < 255);
            }
        } else {
            // Use crushed mask (fallback when no thickness mask provided)
            uint8_t crushed_pixel = crushed[(i + ox) + (j + oy) * width];

            if (crushed_pixel == 255) {
                // White = not crushed
                thickness = 1.0f;
                is_crushed = false;
            } else {
                // Gray = crushed with thickness = grayscale / 255
                thickness = (float)crushed_pixel / 255.0f;
                is_crushed = true;
            }

            // Scale thickness from [0, 1] range to [ThicknessFrom, ThicknessTo]
            // Formula: scaled = ThicknessFrom + thickness * (ThicknessTo - ThicknessFrom)
            thickness = (float)(thicknessFrom + thickness * (thicknessTo - thicknessFrom));
        }

        p.setValue(SimParams::PtArrIdx::idx_thickness, thickness);

        // Pack RGB into utility_data (starting at bit 24)
        uint64_t utility_data = 0;
        
        // R: 24-31, G: 32-39, B: 40-47
        utility_data |= ((uint64_t)r << 24);
        utility_data |= ((uint64_t)g << 32);
        utility_data |= ((uint64_t)b << 40);

        if (is_crushed) {
            utility_data |= SimParams::status_crushed;  // Crushed flag (bit 16)
        }
        
        // Determine cracked status
        if (!cracked.empty()) {
            // White (255) = Not cracked
            // Anything else = Cracked
            uint8_t cracked_pixel = cracked[(i + ox) + (j + oy) * width];
            if (cracked_pixel < 255) {
                utility_data |= SimParams::status_cracked;
            }
        }
        
        // Initialize partition index (bits 0-15) to 0 (will be set later)
        
        p.setValueUInt64(SimParams::PtArrIdx::idx_utility_data, utility_data);

        // Initialize other fields
        p.setValue(SimParams::PtArrIdx::idx_P, 0.0);
        p.setValue(SimParams::PtArrIdx::idx_Q, 0.0);
        p.setValue(SimParams::PtArrIdx::idx_Jp_inv, 1.0);

        // Identity matrix for Fe
        for (int idx = 0; idx < SimParams::dim; idx++) {
            p.setValue(SimParams::PtArrIdx::Fe00 + idx * 2 + idx, 1.0);
        }

        // Zero velocity
        p.setValue(SimParams::PtArrIdx::velx + 0, 0.0);
        p.setValue(SimParams::PtArrIdx::velx + 1, 0.0);
    }

    // (9) Convert to cell-based local coordinates
    hssoa.convertToIntegerCellFormat(h);

    // (10) Save point snapshot
    prms.nPtsInitial = pt_buffer.size();  // Update in case it changed
    SaveSnapshot(0, 0.0, false, data_directory);  // Save to same directory as grid.h5 and grid_flow.h5

    LOGR("PopulatePoints completed");

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
    point_partitions.resize(prms.nPtsInitial);

    ds.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &prms.SimulationStep);
    ds.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &prms.SimulationTime);

    unsigned hssoa_size;
    ds.openAttribute("HSSOA_size").read(H5::PredType::NATIVE_UINT, &hssoa_size);

    ds.openAttribute("ParticleArea").read(H5::PredType::NATIVE_DOUBLE, &prms.ParticleArea);

    int nPtsArrays;
    ds.openAttribute("nPtsArrays").read(H5::PredType::NATIVE_INT, &nPtsArrays);

    // Get dimensions from file
    H5::DataSpace dsp = ds.getSpace();
    hsize_t dims[2];
    dsp.getSimpleExtentDims(dims, nullptr);

    if(dims[0] != SimParams::PtArrIdx::nPtsArrays ||
        dims[1] != hssoa_size ||
        nPtsArrays != SimParams::PtArrIdx::nPtsArrays)
    {
        LOGR("dims {} x {}; hssoa_size {}; nPtsInitial {}",
             dims[0], dims[1], hssoa_size, prms.nPtsInitial);
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
    const int gridSize = gx*gy;
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


void HostSideData::SaveFrame(int SimulationStep, double SimulationTime)
{
    if(!prms.SaveSnapshots) LOGR("skipping SaveFrame");
    LOGR("SaveFrame: step {}, time {}", SimulationStep, SimulationTime);
    const int frame = SimulationStep / prms.UpdateEveryNthStep;
    const int &gx = prms.GridXTotal;
    const int &gy = prms.GridYTotal;
    const int gridSize = gx*gy;

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

    LOGR("SaveFrame done; step {}, time {}", SimulationStep, SimulationTime);
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
        hsize_t chunk_dims[2] = {SimParams::PtArrIdx::nPtsArrays, 100000};
        proplist.setChunk(2, chunk_dims);
        proplist.setDeflate(6);
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
    dataset_pts.createAttribute("HSSOA_size", H5::PredType::NATIVE_UINT, att_dspace)
        .write(H5::PredType::NATIVE_UINT, &nPts);

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


void HostSideData::LoadFrameData(const std::string& framePath)
{
    LOGR("LoadFrameData: Loading frame from {}", framePath);

    namespace fs = std::filesystem;
    if (!fs::exists(framePath)) {
        LOGR("Error: Frame file does not exist: {}", framePath);
        throw std::runtime_error(fmt::format("Frame file not found: {}", framePath));
    }

    try {
        const H5::H5File file(framePath, H5F_ACC_RDONLY);

        // 1. Read Frame Attributes from the "rgb" dataset
        const H5::DataSet attr_dset = file.openDataSet("rgb");
        int simulationStep = 0;
        double simulationTime = 0.0;
        attr_dset.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &simulationStep);
        attr_dset.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &simulationTime);

        prms.SimulationStep = simulationStep;
        prms.SimulationTime = simulationTime;

        // 2. Prepare Memory Buffers
        const auto gx = prms.GridXTotal;
        const auto gy = prms.GridYTotal;
        const size_t gridSize = (size_t)gx * gy;

        rgb.resize(gridSize * 3);
        host_grid_buffer.assign(gridSize * SimParams::HostGridArrayIndex::nGridArraysHost, 0.0);

        // 3. Load the pre-rendered RGB data into the rgb buffer
        try {
            attr_dset.read(rgb.data(), H5::PredType::NATIVE_UINT8);
        } catch (const H5::Exception& e) {
            LOGR("HDF5 Error reading 'rgb' dataset: {}", e.getCDetailMsg());
            throw;
        }

        // 4. De-interleave and transpose RGB data into host_grid_buffer
        // The HDF5 data has dims (gx, gy, 3) in row-major (x varies fastest)
        // Our internal grid buffers are column-major (y varies fastest)
        const size_t r_offset = gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_r;
        const size_t g_offset = gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_g;
        const size_t b_offset = gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_b;

        for (int i = 0; i < gx; ++i) {
            for (int j = 0; j < gy; ++j) {
                const size_t src_idx = ((size_t)i * gy + j) * 3;
                const size_t dst_idx = (size_t)j + (size_t)i * gy;

                host_grid_buffer[r_offset + dst_idx] = static_cast<double>(rgb[src_idx + 0]) / 255.0;
                host_grid_buffer[g_offset + dst_idx] = static_cast<double>(rgb[src_idx + 1]) / 255.0;
                host_grid_buffer[b_offset + dst_idx] = static_cast<double>(rgb[src_idx + 2]) / 255.0;
            }
        }

        // 5. Load grid_data: 3D array [nGridArraysHost] x [gx] x [gy]
        const H5::DataSet grid_dset = file.openDataSet("grid_data");
        grid_dset.read(host_grid_buffer.data(), H5::PredType::NATIVE_DOUBLE);

        LOGR("Successfully loaded frame from {}", framePath);
    } catch (const H5::Exception& e) {
        LOGR("Critical HDF5 Error loading frame: {}", e.getCDetailMsg());
        throw;
    }
}


void HostSideData::LoadParametersFromConfigFile(const std::string& parameterFile,
                                                 const std::string& mapFile,
                                                 const std::string& pngImageFile)
{
    LOGR("LoadParametersFromConfigFile: {}", parameterFile);
    LOGR("  Parameter file: {}", parameterFile);
    LOGR("  Map file: {}", mapFile);
    LOGR("  PNG file: {}", pngImageFile.empty() ? "(not provided)" : pngImageFile);

    namespace fs = std::filesystem;

    // Verify map file exists
    if (!fs::exists(mapFile)) {
        LOGR("ERROR: Map file does not exist: {}", mapFile);
        throw std::runtime_error(fmt::format("Map file not found: {}", mapFile));
    }

    // 1. Load Grid Parameters from HDF5 Map File
    std::vector<int> path_indices;
    {
        LOGR("Loading grid metadata from map file: {}", mapFile);
        H5::H5File file(mapFile, H5F_ACC_RDONLY);
        H5::DataSet ds_path_indices = file.openDataSet("path_indices");

        ds_path_indices.openAttribute("width").read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeX);
        ds_path_indices.openAttribute("height").read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeY);
        ds_path_indices.openAttribute("ModeledRegionOffsetX").read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetX);
        ds_path_indices.openAttribute("ModeledRegionOffsetY").read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetY);
        ds_path_indices.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &prms.GridXTotal);
        ds_path_indices.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &prms.GridYTotal);

        prms.cellsize = prms.DimensionHorizontal / (prms.InitializationImageSizeX - 1);
        prms.cellsize_inv = 1.0 / prms.cellsize;

        path_indices.resize(prms.InitializationImageSizeX * prms.InitializationImageSizeY);
        ds_path_indices.read(path_indices.data(), H5::PredType::NATIVE_INT);
    } // file closed automatically

    // 2. Load PNG Image (optional - only if provided)
    if (!pngImageFile.empty()) {
        if (!fs::exists(pngImageFile)) {
            LOGR("ERROR: PNG file does not exist: {}", pngImageFile);
            throw std::runtime_error(fmt::format("PNG file not found: {}", pngImageFile));
        }

        LOGR("Loading PNG background image: {}", pngImageFile);
        int channels, imgx, imgy;
        unsigned char* png_data = stbi_load(pngImageFile.c_str(), &imgx, &imgy, &channels, 3);
        if (!png_data || channels != 3 || imgx != prms.InitializationImageSizeX || imgy != prms.InitializationImageSizeY) {
            LOGR("Fatal Error: PNG file '{}' could not be loaded or has incorrect dimensions.", pngImageFile);
            throw std::runtime_error("Failed to load PNG image for background.");
        }

        // 3. Populate original_image_colors_rgb
        original_image_colors_rgb.resize(prms.InitializationImageSizeX * prms.InitializationImageSizeY * 3);
        auto idxInPng = [&](int i, int j) -> int {
            return 3 * ((prms.InitializationImageSizeY - j - 1) * prms.InitializationImageSizeX + i);
        };

        for (int j = 0; j < prms.InitializationImageSizeY; j++) {
            for (int i = 0; i < prms.InitializationImageSizeX; i++) {
                for (int k = 0; k < 3; k++) {
                    original_image_colors_rgb[((j * prms.InitializationImageSizeX) + i) * 3 + k] =
                        png_data[idxInPng(i, j) + k];
                }
            }
        }
        stbi_image_free(png_data);
    } else {
        LOGR("PNG file not provided - skipping background image loading (post-processor mode)");
    }

    // 4. Allocate host_grid_buffer
    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;
    const size_t gridSize = (size_t)gx * gy;
    host_grid_buffer.assign(gridSize * SimParams::HostGridArrayIndex::nGridArraysHost, 0.0);

    // 5. Generate grid_status_buffer (landmask_buffer for post-processor)
    landmask_buffer.resize((size_t)gx * gy);

    auto transformPathIdx = [](const int& idx) -> uint8_t {
        if (idx < 1000) return (uint8_t)(idx + 1);
        else if (idx == 1000) return (uint8_t)(100); // 1000 -> 100 (modeled area)
        else return (uint8_t)(idx - 1000 + 1);
    };

    const int ox = prms.ModeledRegionOffsetX;
    const int oy = prms.ModeledRegionOffsetY;

    for (int i = 0; i < gx; i++) {
        for (int j = 0; j < gy; j++) {
            // Index in the full-size image/path_indices map (row-major)
            size_t image_map_idx = (size_t)(i + ox) + (size_t)(j + oy) * prms.InitializationImageSizeX;
            // Index in the grid-sized status buffer (column-major)
            size_t grid_buffer_idx = (size_t)j + (size_t)i * gy;

            landmask_buffer[grid_buffer_idx] = transformPathIdx(path_indices[image_map_idx]);
        }
    }

    LOGR("LoadParametersFromConfigFile: Done");
}
