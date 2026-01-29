#include "data_preparer.h"
#include "poisson_disk_sampling.h"
#include "optimized_poisson_disk_sampling.h"
#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <fmt/format.h>
#include <fmt/std.h>

// Helper includes
#include <png.h>


// Static helper for PNG loading
bool DataPreparer::LoadPng(const std::string& filename, int& w, int& h, int& channels, std::vector<uint8_t>& data)
{
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        LOGR("LoadPng: Failed to open file {}", filename);
        return false;
    }

    // verify signature
    png_byte header[8];
    if (fread(header, 1, 8, fp) != 8) {
        LOGR("LoadPng: Failed to read signature from {}", filename);
        fclose(fp);
        return false;
    }
    if (png_sig_cmp(header, 0, 8)) {
        LOGR("LoadPng: File {} is not a PNG", filename);
        fclose(fp);
        return false;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        LOGR("LoadPng: png_create_read_struct failed");
        fclose(fp);
        return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        LOGR("LoadPng: png_create_info_struct failed");
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        LOGR("LoadPng: Error during init_io");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    w = png_get_image_width(png_ptr, info_ptr);
    h = png_get_image_height(png_ptr, info_ptr);
    auto color_type = png_get_color_type(png_ptr, info_ptr);
    auto bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Standardize to RGBA 8-bit
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    // Expand gray to RGB
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    // Add alpha if missing
    if (!(color_type & PNG_COLOR_MASK_ALPHA))
         png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);

    png_read_update_info(png_ptr, info_ptr);

    // We effectively forced it to RGBA 8-bit, so channels = 4
    channels = 4;
    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    
    // row_bytes should be w * 4
    if (row_bytes != w * 4) {
        LOGR("LoadPng: Unexpected row_bytes {} for width {}", row_bytes, w);
    }
    
    data.resize((size_t)h * row_bytes);

    std::vector<png_bytep> row_pointers(h);
    for (int y = 0; y < h; y++) {
        row_pointers[y] = &data[y * row_bytes];
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        LOGR("LoadPng: Error during read_image");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }

    png_read_image(png_ptr, row_pointers.data());

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    return true;
}




namespace fs = std::filesystem;

DataPreparer::DataPreparer(HostSideData& hsd) : hsd(hsd)
{
}

std::string DataPreparer::prepare_cache_filename(int gx, int gy, int ppc)
{
    return fmt::format("_data/poisson_cache/points_{}x{}_{:d}.h5", gx, gy, ppc);
}

bool DataPreparer::attempt_to_fill_from_cache(int gx, int gy, int ppc, std::vector<std::array<float, 2>> &buffer)
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



void DataPreparer::generate_and_save_poisson(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer)
{
    // Use (gy-1)/(gx-1) to ensure that isotropic scaling (by gx-1) maps exactly to [0, gy-1]
    const float dy = (float)(gy - 1) / (gx - 1);

    LOGR("Generating Poisson points with radius-based parameters (OPTIMIZED)");

    const std::array<float, 2> x_min{0, 0};
    const std::array<float, 2> x_max{1, dy};
    constexpr float magic_constant = 0.6;
    const float radius = std::sqrt(magic_constant / (points_per_cell * gx * gx));

    LOGR("Poisson parameters: grid={}x{}, dy={}, radius={}, target_ppc={}", gx, gy, dy, radius, points_per_cell);

    // MASK LAMBDA for optimization
    // Maps local point p [0, 1]x[0, dy] to global image coordinates and checks ICE mask.
    auto mask_func = [&](const std::array<float, 2>& pt) -> bool {
        // Map to integer grid coordinates relative to Ice Region
        // pt[0] is in [0, 1], pt[1] is in [0, dy]
        // gx is IceRegionWidth, gy is IceRegionHeight
        
        // Map pt[0] from [0, 1] to [0, gx-1]
        int i = (int)(pt[0] * (gx - 1) + 0.5f);
        // Map pt[1] from [0, dy] to [0, gy-1]. 
        // Since dy = (gy-1)/(gx-1), pt[1] * (gx-1) maps [0, dy] -> [0, gy-1]
        int j = (int)(pt[1] * (gx - 1) + 0.5f); 
        
        // Bounds check (local to IceRegion)
        if (i < 0 || i >= gx || j < 0 || j >= gy) return false;

        // Global check
        int img_x = i + this->IceRegionOffsetX;
        int img_y = j + this->IceRegionOffsetY;
        
        // Safety check for image bounds
        if (img_x < 0 || img_x >= m_width || img_y < 0 || img_y >= m_height) return false;
        
        uint8_t flags = m_flags[(size_t)img_y * m_width + img_x];
        bool is_ice = (flags & FLAG_ICE) && (flags & FLAG_WATER);
        return is_ice;
    };

    // Use the optimized version with Mask
    buffer = thinks::PoissonDiskSampling(radius, x_min, x_max, mask_func);
    LOGR("Generated {} raw Poisson points", buffer.size());

    // Calculate actual Ice Area (count of valid cells) for correct PPC calculation
    long long count_ice_cells = 0;
    for(int j=0; j<gy; ++j) {
        for(int i=0; i<gx; ++i) {
            int img_x = i + this->IceRegionOffsetX;
            int img_y = j + this->IceRegionOffsetY;
            
            if (img_x >= 0 && img_x < m_width && img_y >= 0 && img_y < m_height) {
                uint8_t flags = m_flags[(size_t)img_y * m_width + img_x];
                if ((flags & FLAG_ICE) && (flags & FLAG_WATER)) {
                    count_ice_cells++;
                }
            }
        }
    }

    // Log the actual points per cell achieved based on ACTIVE area
    double active_area_fraction = (double)count_ice_cells / (double)(gx * gy);
    float raw_ppc = 0.0f;
    if (count_ice_cells > 0) {
        raw_ppc = (float)buffer.size() / (float)count_ice_cells;
    }

    LOGR("Raw achieved ppc: {:.4f} (target {:.4f}) | IceCells: {}/{} ({:.1f}%)", 
         raw_ppc, points_per_cell, count_ice_cells, gx*gy, active_area_fraction*100.0);

    // Note: We deliberately DO NOT scale the points. 
    // Scaling shifts coordinates relative to the underlying Mask (IceRegion), causing data misalignment.
    // Small variations in PPC are handled by the ParticleArea normalization in PopulatePoints.
    
    // Log final ppc
    LOGR("grid: {}x{}; final pts {:>8}; final_ppc {:.4f}", gx, gy, buffer.size(), raw_ppc);

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



bool DataPreparer::ProcessMaskLayer(const std::string& filename, uint8_t flag, bool invert, int threshold)
{
    if (filename.empty()) return false;

    int w, h, c;
    std::vector<uint8_t> raw;
    if (!LoadPng(filename, w, h, c, raw)) {
        throw std::runtime_error("Failed to load mask: " + filename);
    }

    if (w != m_width || h != m_height) {
        throw std::runtime_error(fmt::format("Dimension mismatch for {}: expected {}x{}, got {}x{}", 
                                             filename, m_width, m_height, w, h));
    }

    // raw is RGBA (4 bytes per pixel)
    for (int y = 0; y < h; ++y) {
        // LoadPng returns top-left origin, same as stbi.
        // We flip Y to match simulation coordinates (bottom-left origin).
        int src_y = h - 1 - y; 
        for (int x = 0; x < w; ++x) {
            size_t idx = ((size_t)src_y * w + x) * 4;
            // Use Red channel (index 0) as the value
            uint8_t val = raw[idx];
            bool condition = (val < threshold); // Default: black (<128) matches condition
            if (invert) condition = !condition;

            if (condition) {
                m_flags[(size_t)y * w + x] |= flag;
            }
        }
    }

    LOGR("Loaded mask {} -> flag {}", filename, flag);
    return true;
}

void DataPreparer::PrepareGridAndPoints(std::string fileNameLandMask, std::string fileNameColor,
                                       std::string fileNameIceMask, std::string fileNameCrushedMask,
                                       std::string fileNameCrackedMask,
                                       std::string projectDirectory, double dimensionHorizontal, int pointsPerCell,
                                       double thicknessFrom, double thicknessTo,
                                       double probCracked, double stdDevThickness,
                                       std::string fileNameThicknessMask,
                                       bool allocate_dense_grid,
                                       bool reload_after_save)
{
    LOGR("PrepareGridAndPoints: starting optimization run");

    hsd.data_directory = projectDirectory;

    // 1. Load Color Image (Master Dimensions)
    // 1. Load Color Image (Master Dimensions)
    int c;
    std::vector<uint8_t> color_raw;
    if (!LoadPng(fileNameColor, m_width, m_height, c, color_raw)) {
        throw std::runtime_error("Failed to load ImageColor: " + fileNameColor);
    }
    
    // Store in m_color (RGB) and flip vertically
    m_color.resize((size_t)m_width * m_height * 3);
    
    for (int y = 0; y < m_height; ++y) {
        int src_y = m_height - 1 - y; // Flip Y
        for (int x = 0; x < m_width; ++x) {
            size_t src_idx = ((size_t)src_y * m_width + x) * 4; // RGBA
            size_t dst_idx = ((size_t)y * m_width + x) * 3;     // RGB
            m_color[dst_idx + 0] = color_raw[src_idx + 0];
            m_color[dst_idx + 1] = color_raw[src_idx + 1];
            m_color[dst_idx + 2] = color_raw[src_idx + 2];
        }
    }
    
    // Clear raw buffer
    color_raw.clear(); color_raw.shrink_to_fit();

    LOGR("[MEMORY] DataPreparer: m_color allocated: {:.3f} MB", (double)m_color.size() * sizeof(uint8_t) / 1.0e6);

    // Initialize buffers
    m_flags.assign((size_t)m_width * m_height, 0); // All zero initially
    m_thickness.resize((size_t)m_width * m_height);
    LOGR("[MEMORY] DataPreparer: m_flags allocated: {:.3f} MB", (double)m_flags.size() * sizeof(uint8_t) / 1.0e6);
    LOGR("[MEMORY] DataPreparer: m_thickness allocated: {:.3f} MB", (double)m_thickness.size() * sizeof(uint8_t) / 1.0e6);

    // 2. Load Masks and merge into m_flags
    // Landmask: Black (<128) = Water (Modeled Area), White = Land
    // So we set FLAG_WATER if pixel < 128
    if (!fileNameLandMask.empty()) {
        ProcessMaskLayer(fileNameLandMask, FLAG_WATER, false); 
    } else {
        // No landmask -> All water
        std::fill(m_flags.begin(), m_flags.end(), FLAG_WATER);
        LOGR("No ImageLandMask -> Interpreting entire domain as WATER");
    }

    // IceMask: White (>=128) = Ice, Black = No Ice
    // Default ProcessMask logic is "val < threshold" (Black).
    // So for Ice (White), we need to invert the condition logic? 
    // Wait, let's look at `ProcessMaskLayer`. 
    // condition = (val < 128).
    // If we want White to set the flag, we need `condition = (val >= 128)`.
    // This is equivalent to `invert=true`.
    ProcessMaskLayer(fileNameIceMask, FLAG_ICE, true); 

    // CrushedMask: Black (<128) = Crushed.
    if (!fileNameCrushedMask.empty()) {
        ProcessMaskLayer(fileNameCrushedMask, FLAG_CRUSHED, false);
    }

    // CrackedMask: Black (<128) = Cracked.
    if (!fileNameCrackedMask.empty()) {
        ProcessMaskLayer(fileNameCrackedMask, FLAG_CRACKED, false);
    }

    // 3. Load Thickness Mask (Grayscale) or Generate
    if (!fileNameThicknessMask.empty()) {
         int tw, th, tc;
         std::vector<uint8_t> traw;
         if (!LoadPng(fileNameThicknessMask, tw, th, tc, traw)) 
              throw std::runtime_error("Failed to load ThicknessMask");
         
         if (tw != m_width || th != m_height) {
             throw std::runtime_error("ThicknessMask dimension mismatch");
         }
         
         // Flip and store (traw is RGBA)
         for(int y=0; y<th; ++y) {
             int src_y = th - 1 - y;
             for (int x=0; x<tw; ++x) {
                  m_thickness[(size_t)y*tw + x] = traw[(src_y*(size_t)tw + x)*4]; // Red channel
             }
         }
    } else {
        // Generate random thickness
        std::mt19937 gen(1337);
        std::normal_distribution<double> dist(127.5, 255.0 / 4.0);
        for (auto& val : m_thickness) {
            val = (uint8_t)std::clamp(dist(gen), 0.0, 255.0);
        }
        LOGR("Generated random thickness field");
    }

    DetermineExtents(dimensionHorizontal);


    // 5. Populate Points (Using m_flags, m_thickness, and m_color)
//    PopulatePoints(pointsPerCell, thicknessFrom, thicknessTo, probCracked, stdDevThickness);
    PopulatePoints_RAM_Optimized(pointsPerCell, thicknessFrom, thicknessTo, probCracked, stdDevThickness);

    // 4. Process Grid (Using m_flags and m_color)
    PrepareGrid(projectDirectory, dimensionHorizontal, allocate_dense_grid);

    // Cleanup member buffers to release memory
    m_color.clear(); m_color.shrink_to_fit();
    m_flags.clear(); m_flags.shrink_to_fit();
    m_thickness.clear(); m_thickness.shrink_to_fit();

    LOGR("PrepareGridAndPoints: completed");
}


void DataPreparer::DetermineExtents(double dimensionHorizontal)
{
    hsd.prms.InitializationImageSizeX = m_width;
    hsd.prms.InitializationImageSizeY = m_height;

    // (1) Find cropped area (bounding box of WATER pixels)
    int xmin = m_width, xmax = -1;
    int ymin = m_height, ymax = -1;

    // Ice bounding box
    int xmin_ice = m_width, xmax_ice = -1;
    int ymin_ice = m_height, ymax_ice = -1;
    bool found_ice = false;

    bool found_water = false;
    for (int j = 0; j < m_height; j++) {
        for (int i = 0; i < m_width; i++) {
            uint8_t flags = m_flags[(size_t)j * m_width + i];
            if (flags & FLAG_WATER) {
                xmin = std::min(xmin, i);
                xmax = std::max(xmax, i);
                ymin = std::min(ymin, j);
                ymax = std::max(ymax, j);
                found_water = true;
            }
            if (flags & FLAG_ICE) {
                xmin_ice = std::min(xmin_ice, i);
                xmax_ice = std::max(xmax_ice, i);
                ymin_ice = std::min(ymin_ice, j);
                ymax_ice = std::max(ymax_ice, j);
                found_ice = true;
            }
        }
    }

    if (!found_water) {
        throw std::runtime_error("No modeled area (WATER) found in landmask!");
    }

    // Pad by ±2 cells and clamp (Water)
    xmin = std::max(0, xmin - 2);
    xmax = std::min(m_width - 1, xmax + 2);
    ymin = std::max(0, ymin - 2);
    ymax = std::min(m_height - 1, ymax + 2);

    hsd.prms.ModeledRegionOffsetX = xmin;
    hsd.prms.ModeledRegionOffsetY = ymin;
    hsd.prms.GridXTotal = xmax - xmin + 1;
    hsd.prms.GridYTotal = ymax - ymin + 1;

    // Process Ice Region
    if (found_ice) {
        // Pad Ice Region as well for safety
        xmin_ice = std::max(0, xmin_ice - 2);
        xmax_ice = std::min(m_width - 1, xmax_ice + 2);
        ymin_ice = std::max(0, ymin_ice - 2);
        ymax_ice = std::min(m_height - 1, ymax_ice + 2);

        IceRegionOffsetX = xmin_ice;
        IceRegionOffsetY = ymin_ice;
        IceRegionWidth = xmax_ice - xmin_ice + 1;
        IceRegionHeight = ymax_ice - ymin_ice + 1;
        LOGR("Ice Region: offset({}, {}), size {}x{}", xmin_ice, ymin_ice, IceRegionWidth, IceRegionHeight);
    } else {
        // No ice found? (Shouldn't happen if simulation expects ice, but handle gracefully)
        LOGR("No ICE found!");
        IceRegionOffsetX = 0;
        IceRegionOffsetY = 0;
        IceRegionWidth = 0;
        IceRegionHeight = 0;
    }

    LOGR("Grid size: {} x {}", hsd.prms.GridXTotal, hsd.prms.GridYTotal);

    // (2) Calculate physical parameters
    hsd.prms.DimensionHorizontal = dimensionHorizontal;
    hsd.prms.cellsize = hsd.prms.DimensionHorizontal / (hsd.prms.InitializationImageSizeX - 1);
    hsd.prms.cellsize_inv = 1.0 / hsd.prms.cellsize;

}


void DataPreparer::PrepareGrid(std::string projectDirectory, double dimensionHorizontal, bool allocate_dense_grid)
{
    LOGR("PrepareGrid: starting");


    // (3) Allocate grid arrays
    hsd.AllocateGridArrays(allocate_dense_grid);

    // (4) Build landmask_buffer
    LOGR("PrepareGrid: building landmask_buffer...");
    for (int i = 0; i < hsd.prms.GridXTotal; i++) {
        for (int j = 0; j < hsd.prms.GridYTotal; j++) {
            int img_x = i + hsd.prms.ModeledRegionOffsetX;
            int img_y = j + hsd.prms.ModeledRegionOffsetY;
            
            bool is_water = (m_flags[(size_t)img_y * m_width + img_x] & FLAG_WATER);
            uint8_t status = is_water ? SimParams::ModelledAreaIndicator : 0;

            size_t idx = j + (size_t)i * hsd.prms.GridYTotal;
            hsd.landmask_buffer[idx] = status;
        }
    }

    // (5) Store original colors (copy from m_color)
    LOGR("PrepareGrid: copying color buffer...");
    hsd.original_image_colors_rgb = m_color; 
    // This is the COPY the user might be worried about, but HSD needs it for visualization.
    // m_color will be freed after this whole function ends.

    // (6) Fill water areas with blue color for visualization in HSD
    LOGR("PrepareGrid: filling modelled area with blue...");
    hsd.FillModelledAreaWithBlueColor();

    // (7) Save grid HDF5 file
    LOGR("PrepareGrid: saving grid.h5...");
    std::string gridFilePath = projectDirectory + "/grid.h5";
    H5::H5File file(gridFilePath, H5F_ACC_TRUNC);

    // Save landmask
    hsize_t landmask_dims[2] = {static_cast<hsize_t>(hsd.prms.GridXTotal), static_cast<hsize_t>(hsd.prms.GridYTotal)};
    H5::DataSpace landmask_space(2, landmask_dims);
    H5::DSetCreatPropList landmask_props;
    hsize_t chunks[2] = {std::min<hsize_t>(hsd.prms.GridXTotal, 64), std::min<hsize_t>(hsd.prms.GridYTotal, 64)};
    landmask_props.setChunk(2, chunks);
    landmask_props.setDeflate(1);
    file.createDataSet("landmask", H5::PredType::NATIVE_UINT8, landmask_space, landmask_props)
        .write(hsd.landmask_buffer.data(), H5::PredType::NATIVE_UINT8);

    // Save color
    hsize_t color_dims[3] = {static_cast<hsize_t>(m_height), static_cast<hsize_t>(m_width), 3};
    H5::DataSpace color_space(3, color_dims);
    H5::DSetCreatPropList color_props;
    hsize_t cchunks[3] = {std::min<hsize_t>(m_height, 64), std::min<hsize_t>(m_width, 64), 3};
    color_props.setChunk(3, cchunks);
    color_props.setDeflate(6);
    file.createDataSet("color_grid", H5::PredType::NATIVE_UINT8, color_space, color_props)
        .write(hsd.original_image_colors_rgb.data(), H5::PredType::NATIVE_UINT8);

    // Attributes
    int gx = hsd.prms.GridXTotal, gy = hsd.prms.GridYTotal;
    int ox = hsd.prms.ModeledRegionOffsetX, oy = hsd.prms.ModeledRegionOffsetY;
    H5::DataSpace scalar(H5S_SCALAR);
    auto writeAttr = [&](auto& ds, const char* name, auto type, const void* val) {
        ds.createAttribute(name, type, scalar).write(type, val);
    };
    
    // We need a dataset to write attributes to. `landmask` works.
    auto ds = file.openDataSet("landmask");
    writeAttr(ds, "GridXTotal", H5::PredType::NATIVE_INT, &gx);
    writeAttr(ds, "GridYTotal", H5::PredType::NATIVE_INT, &gy);
    writeAttr(ds, "OffsetX", H5::PredType::NATIVE_INT, &ox);
    writeAttr(ds, "OffsetY", H5::PredType::NATIVE_INT, &oy);
    writeAttr(ds, "InitImageSizeX", H5::PredType::NATIVE_INT, &m_width);
    writeAttr(ds, "InitImageSizeY", H5::PredType::NATIVE_INT, &m_height);
    writeAttr(ds, "CellSize", H5::PredType::NATIVE_DOUBLE, &hsd.prms.cellsize);
    writeAttr(ds, "DimensionHorizontal", H5::PredType::NATIVE_DOUBLE, &hsd.prms.DimensionHorizontal);

    file.close();
    LOGR("PrepareGrid completed");
}

void DataPreparer::PopulatePoints_RAM_Optimized(int pointsPerCell, double thicknessFrom, double thicknessTo,
                                     double probCracked, double stdDevThickness, bool compress)
{
    LOGR("PopulatePoints_RAM_Optimized (New): starting");

    // (1) Load points from cache
    LOGR("PopulatePoints_RAM_Optimized: preparing point buffer...");
    std::vector<std::array<float, 2>> pt_buffer;
    
    // START OPTIMIZATION: Use Ice Region for point generation
    if (this->IceRegionWidth <= 0 || this->IceRegionHeight <= 0) {
        LOGR("PopulatePoints: IceRegion is empty. Skipping point generation.");
        throw std::runtime_error("no ice");
        return; 
    }

    const int gx = this->IceRegionWidth;
    const int gy = this->IceRegionHeight;
    const int ox = this->IceRegionOffsetX;
    const int oy = this->IceRegionOffsetY;

    // Relative offset from Simulation Grid Origin (ModeledRegionOffset) to Ice Region Origin
    const int rel_ox = ox - hsd.prms.ModeledRegionOffsetX;
    const int rel_oy = oy - hsd.prms.ModeledRegionOffsetY;

    LOGR("PopulatePoints: Generating in Ice Region {}x{}, offset ({},{}), relative ({},{})", 
         gx, gy, ox, oy, rel_ox, rel_oy);

    // Try cache or generate
    if (!attempt_to_fill_from_cache(gx, gy, pointsPerCell, pt_buffer)) {
        generate_and_save_poisson(gx, gy, (float)pointsPerCell, pt_buffer);
    }

    if (pt_buffer.empty()) throw std::runtime_error("No Poisson points generated");

    // (3) Filter points 
    LOGR("PopulatePoints_RAM_Optimized: filtering points...");
    
    // Consistent dy definition
    const float dy = (float)(gy - 1) / (gx - 1);
    
    auto idxPt = [&](const std::array<float, 2> &pt) -> std::pair<int, int> {
        // Correct isotropic mapping matching generate_and_save_poisson mask logic
        int i = (int)(pt[0] * (gx - 1) + 0.5f);
        int j = (int)(pt[1] * (gx - 1) + 0.5f);
        return {i, j};
    };

    auto shouldRemove = [&](const std::array<float, 2> &pt) -> bool {
        auto [i, j] = idxPt(pt);
        if (i <= 1 || j <= 1 || i >= (gx - 2) || j >= (gy - 2)) return true;

        // ox/oy are IceRegionOffsets here
        int img_x = i + ox;
        int img_y = j + oy;
        
        uint8_t flags = m_flags[(size_t)img_y * m_width + img_x];
        if (!(flags & FLAG_WATER)) return true;
        if (!(flags & FLAG_ICE)) return true;
        return false;
    };

    std::erase_if(pt_buffer, shouldRemove);
    pt_buffer.shrink_to_fit();

    hsd.prms.nPtsInitial = pt_buffer.size();
    if (hsd.prms.nPtsInitial == 0) throw std::runtime_error("All points filtered out!");

    LOGR("PopulatePoints_RAM_Optimized: {} points remaining", hsd.prms.nPtsInitial);

    // Calculate ParticleArea using ICE AREA (not bounding box area)
    long long count_ice_cells = 0;
    for(int j=0; j<gy; ++j) {
        for(int i=0; i<gx; ++i) {
            int img_x = i + ox;
            int img_y = j + oy;
            if (img_x >= 0 && img_x < m_width && img_y >= 0 && img_y < m_height) {
                uint8_t f = m_flags[(size_t)img_y * m_width + img_x];
                if ((f & FLAG_ICE) && (f & FLAG_WATER)) {
                    count_ice_cells++;
                }
            }
        }
    }
    
    const double h = hsd.prms.cellsize;
    double total_ice_area = (double)count_ice_cells * h * h;
    hsd.prms.ParticleArea = total_ice_area / (double)hsd.prms.nPtsInitial;
    
    LOGR("ParticleArea Calc: IceCells={}, TotalArea={:.4e}, nPts={}, PtArea={:.4e}", 
         count_ice_cells, total_ice_area, hsd.prms.nPtsInitial, hsd.prms.ParticleArea);

    // Initial random generators
    std::mt19937 rng(12345);
    std::normal_distribution<double> thickness_dist((double)thicknessFrom, (double)stdDevThickness); // check types
    std::bernoulli_distribution cracked_dist(probCracked);

    // Checks for generic badness (NaNs, Infs)
    for (size_t k = 0; k < pt_buffer.size(); ++k) {
        if (!std::isfinite(pt_buffer[k][0]) || !std::isfinite(pt_buffer[k][1])) {
             throw std::runtime_error(fmt::format("PopulatePoints: Point {} is NaN/Inf: ({}, {})", k, pt_buffer[k][0], pt_buffer[k][1]));
        }
    }

    // --- Transform and Sort pt_buffer ---
    LOGR("PopulatePoints: Transforming and Sorting points in RAM...");
    
    // Scale factor for transforming normalized coordinates to grid indices.
    // User requested equal scaling for both directions.
    // pt[0] is in [0, 1], mapped to [0, gx-1]
    const double scale = (double)(gx - 1);
    
    // Transform to Global Grid Continuous Coordinates
    for(size_t k = 0; k < pt_buffer.size(); ++k) {
        std::array<float, 2> &pt = pt_buffer[k];
        // pt[0] -> u_global
        pt[0] = (float)(pt[0] * scale + rel_ox);
        // pt[1] -> v_global
        pt[1] = (float)(pt[1] * scale + rel_oy);
    }
    
    // Sort by Cell Index (j + i * GridYTotal)
    // Note: Use a stable sort or regular sort? Regular is fine.
    const int GridYTotal = hsd.prms.GridYTotal;
    std::sort(pt_buffer.begin(), pt_buffer.end(), [GridYTotal](const std::array<float, 2>& a, const std::array<float, 2>& b){
        int ia = (int)(a[0] + 0.5f);
        int ja = (int)(a[1] + 0.5f);
        int ib = (int)(b[0] + 0.5f);
        int jb = (int)(b[1] + 0.5f);
        
        long long idx_a = ja + (long long)ia * GridYTotal;
        long long idx_b = jb + (long long)ib * GridYTotal;
        return idx_a < idx_b;
    });

    // --- Allocate HSSOA ---
    hsd.hssoa.Allocate(hsd.prms.nPtsInitial);
    hsd.hssoa.size = hsd.prms.nPtsInitial;

    unsigned capacity = hsd.hssoa.capacity;
    double* host_buffer = hsd.hssoa.host_buffer;

    // Pointers to arrays
    double* ptr_utility = host_buffer + SimParams::PtArrIdx::idx_utility_data * capacity;
    double* ptr_cell    = host_buffer + SimParams::PtArrIdx::integer_cell_idx * capacity;
    double* ptr_posx    = host_buffer + SimParams::PtArrIdx::posx * capacity;
    double* ptr_posy    = host_buffer + SimParams::PtArrIdx::posy * capacity;
    double* ptr_velx    = host_buffer + SimParams::PtArrIdx::velx * capacity;
    double* ptr_vely    = host_buffer + SimParams::PtArrIdx::vely * capacity;
    double* ptr_thick   = host_buffer + SimParams::PtArrIdx::idx_thickness * capacity;
    double* ptr_Jpinv   = host_buffer + SimParams::PtArrIdx::idx_Jp_inv * capacity;
    double* ptr_Fe00    = host_buffer + SimParams::PtArrIdx::Fe00 * capacity;
    double* ptr_Fe11    = host_buffer + (SimParams::PtArrIdx::Fe00 + SimParams::dim + 1) * capacity;

    LOGR("PopulatePoints: Filling HSSOA...");

    for (size_t k = 0; k < hsd.prms.nPtsInitial; ++k) {
        std::array<float, 2> &pt = pt_buffer[k];
        
        // Points are already in Global Continuous Grid Coordinates
        double u_global = pt[0];
        double v_global = pt[1];

        // Global Integer Cell Index
        int i_global = (int)(u_global + 0.5f);
        int j_global = (int)(v_global + 0.5f);

        if (i_global < 0 || i_global >= hsd.prms.GridXTotal || 
            j_global < 0 || j_global >= hsd.prms.GridYTotal) {
            LOGR("FATAL: Point {} out of bounds! Global: ({:.3f}, {:.3f}) -> Int: ({}, {}) Grid: {}x{}",
                 k, u_global, v_global, i_global, j_global, hsd.prms.GridXTotal, hsd.prms.GridYTotal);
            throw std::runtime_error("Point Global Index Out of Bounds");
        }
        
        // Image Index for property lookup
        // i_global is relative to ModeledRegionOffset
        // img_x is relative to Image Origin (0,0)
        int img_x = i_global + hsd.prms.ModeledRegionOffsetX;
        int img_y = j_global + hsd.prms.ModeledRegionOffsetY;
        
        if (img_x < 0) img_x = 0; if (img_x >= m_width) img_x = m_width - 1;
        if (img_y < 0) img_y = 0; if (img_y >= m_height) img_y = m_height - 1;
        
        size_t img_idx = (size_t)img_y * m_width + img_x;

        // Utility Data (Color, Flags)
        uint32_t r = m_color[img_idx * 3 + 0];
        uint32_t g = m_color[img_idx * 3 + 1];
        uint32_t b = m_color[img_idx * 3 + 2];

        uint64_t utility = 0;
        utility |= ((uint64_t)r << 24);
        utility |= ((uint64_t)g << 32);
        utility |= ((uint64_t)b << 40);

        uint8_t flags = m_flags[img_idx];
        if (flags & FLAG_CRUSHED) utility |= SimParams::status_crushed;
        if (flags & FLAG_CRACKED) utility |= SimParams::status_cracked;
        if (probCracked > 0.0 && cracked_dist(rng)) utility |= SimParams::status_cracked;

        ptr_utility[k] = *reinterpret_cast<double*>(&utility);

        // Global Cell Indices
        uint64_t x_idx_global = (uint64_t)i_global;
        uint64_t y_idx_global = (uint64_t)j_global;
        uint64_t cell = (y_idx_global << 32) | x_idx_global;
        ptr_cell[k] = *reinterpret_cast<double*>(&cell);

        // Normalized Local Position [-0.5, 0.5)
        double x_local_norm = u_global - (double)i_global; 
        double y_local_norm = v_global - (double)j_global;
        
        ptr_posx[k] = x_local_norm;
        ptr_posy[k] = y_local_norm;

        // Velocities
        ptr_velx[k] = 0.0;
        ptr_vely[k] = 0.0;

        // Thickness
        // m_thickness is 0-255 map.
        uint8_t t_val = m_thickness[img_idx];
        float t_norm = (float)t_val / 255.0f;
        // Linear mix
        double thick_val = thicknessFrom + t_norm * (thicknessTo - thicknessFrom);
        
        if (stdDevThickness > 0.0) {
            double noise = thickness_dist(rng);
             thick_val += noise;
             thick_val = std::clamp(thick_val, (double)thicknessFrom, (double)thicknessTo);
        }
        ptr_thick[k] = thick_val;
        
        // Jp_inv
        ptr_Jpinv[k] = 1.0;
        
        // Fe identity
        ptr_Fe00[k] = 1.0;
        ptr_Fe11[k] = 1.0;
    }

    // Clear buffer!
    LOGR("PopulatePoints: Cleaning up RAM buffer...");
    pt_buffer.clear(); 
    pt_buffer.shrink_to_fit();

    // Sort
    LOGR("PopulatePoints: Sorting points...");
    hsd.hssoa.RemoveDisabledAndSort(hsd.prms.GridYTotal);

    // Save
    LOGR("PopulatePoints: Saving s00000.h5 via HSSOA...");
    std::string snapDir = (std::filesystem::path(hsd.data_directory) / "snapshots").string();
    hsd.SaveSnapshot(0, 0.0, true, snapDir);

    LOGR("PopulatePoints_RAM_Optimized (New) completed");
}
