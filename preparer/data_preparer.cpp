#include "data_preparer.h"
#include "poisson_disk_sampling.h"
#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <fmt/format.h>
#include <fmt/std.h>

// Helper includes
#include "stb_image.h"
#include "stb_image_write.h"

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

// Helper to flip an image vertically in place
static void FlipImageVertically(std::vector<uint8_t>& data, int width, int height, int channels)
{
    int row_size = width * channels;
    std::vector<uint8_t> temp_row(row_size);
    for (int y = 0; y < height / 2; ++y) {
        int top_idx = y * row_size;
        int bottom_idx = (height - 1 - y) * row_size;
        
        std::memcpy(temp_row.data(), &data[top_idx], row_size);
        std::memcpy(&data[top_idx], &data[bottom_idx], row_size);
        std::memcpy(&data[bottom_idx], temp_row.data(), row_size);
    }
}

bool DataPreparer::ProcessMaskLayer(const std::string& filename, uint8_t flag, bool invert, int threshold)
{
    if (filename.empty()) return false;

    int w, h, c;
    unsigned char* raw = stbi_load(filename.c_str(), &w, &h, &c, 1);
    if (!raw) {
        throw std::runtime_error("Failed to load mask: " + filename);
    }

    if (w != m_width || h != m_height) {
        stbi_image_free(raw);
        throw std::runtime_error(fmt::format("Dimension mismatch for {}: expected {}x{}, got {}x{}", 
                                             filename, m_width, m_height, w, h));
    }

    // Apply to bitmask (handling vertical flip implicitly by reading bottom-up or flipping first)
    // Let's flip the raw buffer first to match our coordinate system (bottom-left origin)
    // Actually, `FlipImageVertically` is easy to write.
    // Or we can just iterate appropriately. Let's iterate.
    // Our m_flags is flat [width * height]. 
    // Image is top-left origin. m_flags matches simulation (bottom-left origin).
    
    for (int y = 0; y < h; ++y) {
        int src_y = h - 1 - y; // Flip Y
        for (int x = 0; x < w; ++x) {
            uint8_t val = raw[src_y * w + x];
            bool condition = (val < threshold); // Default: black (<128) matches condition
            if (invert) condition = !condition;

            if (condition) {
                m_flags[y * w + x] |= flag;
            } else {
                // Ensure specific flag is CLEARED if it was set (though we usually start with 0)
                // m_flags[y * w + x] &= ~flag; 
            }
        }
    }

    stbi_image_free(raw);
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
                                       bool allocate_dense_grid)
{
    LOGR("PrepareGridAndPoints: starting optimization run");

    hsd.data_directory = projectDirectory;

    // 1. Load Color Image (Master Dimensions)
    int c;
    unsigned char* color_raw = stbi_load(fileNameColor.c_str(), &m_width, &m_height, &c, 3);
    if (!color_raw) {
        throw std::runtime_error("Failed to load ImageColor: " + fileNameColor);
    }
    
    // Store in m_color and flip vertically
    m_color.assign(color_raw, color_raw + m_width * m_height * 3);
    stbi_image_free(color_raw);
    FlipImageVertically(m_color, m_width, m_height, 3);
    LOGR("[MEMORY] DataPreparer: m_color allocated: {:.3f} MB", (double)m_color.size() * sizeof(uint8_t) / 1.0e6);

    // Initialize buffers
    m_flags.assign(m_width * m_height, 0); // All zero initially
    m_thickness.resize(m_width * m_height);
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
         unsigned char* traw = stbi_load(fileNameThicknessMask.c_str(), &tw, &th, &tc, 1);
         if (!traw) throw std::runtime_error("Failed to load ThicknessMask");
         if (tw != m_width || th != m_height) {
             stbi_image_free(traw);
             throw std::runtime_error("ThicknessMask dimension mismatch");
         }
         
         // Flip and store
         for(int y=0; y<th; ++y) {
             int src_y = th - 1 - y;
             std::memcpy(&m_thickness[y*tw], &traw[src_y*tw], tw);
         }
         stbi_image_free(traw);
    } else {
        // Generate random thickness
        std::mt19937 gen(1337);
        std::normal_distribution<double> dist(127.5, 255.0 / 4.0);
        for (auto& val : m_thickness) {
            val = (uint8_t)std::clamp(dist(gen), 0.0, 255.0);
        }
        LOGR("Generated random thickness field");
    }

    // 4. Process Grid (Using m_flags and m_color)
    PrepareGrid(projectDirectory, dimensionHorizontal, allocate_dense_grid);

    // 5. Populate Points (Using m_flags, m_thickness, and m_color)
    PopulatePoints(pointsPerCell, thicknessFrom, thicknessTo, probCracked, stdDevThickness);
    
    // Cleanup member buffers to release memory
    m_color.clear(); m_color.shrink_to_fit();
    m_flags.clear(); m_flags.shrink_to_fit();
    m_thickness.clear(); m_thickness.shrink_to_fit();

    LOGR("PrepareGridAndPoints: completed");
}

void DataPreparer::PrepareGrid(std::string projectDirectory, double dimensionHorizontal, bool allocate_dense_grid)
{
    LOGR("PrepareGrid: starting");

    hsd.prms.InitializationImageSizeX = m_width;
    hsd.prms.InitializationImageSizeY = m_height;

    // (1) Find cropped area (bounding box of WATER pixels)
    int xmin = m_width, xmax = -1;
    int ymin = m_height, ymax = -1;
    
    bool found_water = false;
    for (int j = 0; j < m_height; j++) {
        for (int i = 0; i < m_width; i++) {
            if (m_flags[j * m_width + i] & FLAG_WATER) {
                xmin = std::min(xmin, i);
                xmax = std::max(xmax, i);
                ymin = std::min(ymin, j);
                ymax = std::max(ymax, j);
                found_water = true;
            }
        }
    }

    if (!found_water) {
         throw std::runtime_error("No modeled area (WATER) found in landmask!");
    }

    // Pad by ±2 cells and clamp
    xmin = std::max(0, xmin - 2);
    xmax = std::min(m_width - 1, xmax + 2);
    ymin = std::max(0, ymin - 2);
    ymax = std::min(m_height - 1, ymax + 2);

    hsd.prms.ModeledRegionOffsetX = xmin;
    hsd.prms.ModeledRegionOffsetY = ymin;
    hsd.prms.GridXTotal = xmax - xmin + 1;
    hsd.prms.GridYTotal = ymax - ymin + 1;

    LOGR("Grid size: {} x {}", hsd.prms.GridXTotal, hsd.prms.GridYTotal);

    // (2) Calculate physical parameters
    hsd.prms.DimensionHorizontal = dimensionHorizontal;
    hsd.prms.cellsize = hsd.prms.DimensionHorizontal / (hsd.prms.InitializationImageSizeX - 1);
    hsd.prms.cellsize_inv = 1.0 / hsd.prms.cellsize;

    // (3) Allocate grid arrays
    hsd.AllocateGridArrays(allocate_dense_grid);

    // (4) Build landmask_buffer
    for (int i = 0; i < hsd.prms.GridXTotal; i++) {
        for (int j = 0; j < hsd.prms.GridYTotal; j++) {
            int img_x = i + hsd.prms.ModeledRegionOffsetX;
            int img_y = j + hsd.prms.ModeledRegionOffsetY;
            
            bool is_water = (m_flags[img_y * m_width + img_x] & FLAG_WATER);
            uint8_t status = is_water ? SimParams::ModelledAreaIndicator : 0;

            size_t idx = j + (size_t)i * hsd.prms.GridYTotal;
            hsd.landmask_buffer[idx] = status;
        }
    }

    // (5) Store original colors (copy from m_color)
    hsd.original_image_colors_rgb = m_color; 
    // This is the COPY the user might be worried about, but HSD needs it for visualization.
    // m_color will be freed after this whole function ends.

    // (6) Fill water areas with blue color for visualization in HSD
    hsd.FillModelledAreaWithBlueColor();

    // (7) Save grid HDF5 file
    std::string gridFilePath = projectDirectory + "/grid.h5";
    H5::H5File file(gridFilePath, H5F_ACC_TRUNC);

    // Save landmask
    hsize_t landmask_dims[2] = {static_cast<hsize_t>(hsd.prms.GridXTotal), static_cast<hsize_t>(hsd.prms.GridYTotal)};
    H5::DataSpace landmask_space(2, landmask_dims);
    H5::DSetCreatPropList landmask_props;
    hsize_t chunks[2] = {std::min<hsize_t>(hsd.prms.GridXTotal, 64), std::min<hsize_t>(hsd.prms.GridYTotal, 64)};
    landmask_props.setChunk(2, chunks);
    landmask_props.setDeflate(6);
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

void DataPreparer::PopulatePoints(int pointsPerCell, double thicknessFrom, double thicknessTo,
                                 double probCracked, double stdDevThickness)
{
    LOGR("PopulatePoints: starting");
    
    // (3) Generate or load Poisson points
    std::vector<std::array<float, 2>> pt_buffer;
    const int gx = hsd.prms.GridXTotal;
    const int gy = hsd.prms.GridYTotal;

    if (!attempt_to_fill_from_cache(gx, gy, pointsPerCell, pt_buffer)) {
        generate_and_save_poisson(gx, gy, (float)pointsPerCell, pt_buffer);
    }

    if (pt_buffer.empty()) throw std::runtime_error("No Poisson points generated");

    // (4) Calculate ParticleArea
    const double h = hsd.prms.cellsize;
    hsd.prms.ParticleArea = (h * h * gx * gy) / (double)pt_buffer.size();

    // (5) Filter points
    auto idxPt = [&](const std::array<float, 2> &pt) -> std::pair<int, int> {
        const double scale = gx - 1;
        return {(int)(pt[0] * scale + 0.5), (int)(pt[1] * scale + 0.5)};
    };

    auto shouldRemove = [&](const std::array<float, 2> &pt) -> bool {
        auto [i, j] = idxPt(pt);
        if (i <= 1 || j <= 1 || i >= (gx - 2) || j >= (gy - 2)) return true;

        int img_x = i + hsd.prms.ModeledRegionOffsetX;
        int img_y = j + hsd.prms.ModeledRegionOffsetY;
        
        // Use m_flags
        uint8_t flags = m_flags[img_y * m_width + img_x];
        
        // Must be WATER
        if (!(flags & FLAG_WATER)) return true;
        
        // Must be ICE
        if (!(flags & FLAG_ICE)) return true;

        return false;
    };

    std::erase_if(pt_buffer, shouldRemove);
    hsd.prms.nPtsInitial = pt_buffer.size();
    
    if (hsd.prms.nPtsInitial == 0) throw std::runtime_error("All points filtered out!");

    // (7) Allocate HSSOA
    hsd.AllocatePointArrays();
    hsd.hssoa.size = hsd.prms.nPtsInitial;

    // (8) Transfer
    const double pointScale = (gx - 1) * h;
    const int ox = hsd.prms.ModeledRegionOffsetX;
    const int oy = hsd.prms.ModeledRegionOffsetY;
    
    std::mt19937 rng(12345);
    std::normal_distribution<float> thickness_dist(0.0f, (float)stdDevThickness);
    std::bernoulli_distribution cracked_dist(probCracked);

    for (size_t k = 0; k < pt_buffer.size(); k++) {
        std::array<float, 2> &pt = pt_buffer[k];
        auto [i, j] = idxPt(pt);
        int img_idx = (j + oy) * m_width + (i + ox);

        SOAIterator it = hsd.hssoa.begin() + k;
        ProxyPoint &p = *it;

        p.setValue(SimParams::PtArrIdx::posx, pt[0] * pointScale);
        p.setValue(SimParams::PtArrIdx::posx + 1, pt[1] * pointScale);

        // Color: We MUST use m_color because hsd.original_image_colors_rgb has been painted blue in water!
        // This justifies keeping m_color alive until here.
        uint32_t r = m_color[img_idx * 3 + 0];
        uint32_t g = m_color[img_idx * 3 + 1];
        uint32_t b = m_color[img_idx * 3 + 2];

        // Thickness
        uint8_t t_val = m_thickness[img_idx];
        float t_norm = (float)t_val / 255.0f;
        float thickness = (float)(thicknessFrom + t_norm * (thicknessTo - thicknessFrom));

        if (stdDevThickness > 0.0) {
            thickness += thickness_dist(rng);
            thickness = std::clamp(thickness, (float)thicknessFrom, (float)thicknessTo);
        }
        p.setValue(SimParams::PtArrIdx::idx_thickness, thickness);

        // Flags
        uint64_t utility = 0;
        utility |= ((uint64_t)r << 24);
        utility |= ((uint64_t)g << 32);
        utility |= ((uint64_t)b << 40);

        uint8_t flags = m_flags[img_idx];
        if (flags & FLAG_CRUSHED) utility |= SimParams::status_crushed;
        if (flags & FLAG_CRACKED) utility |= SimParams::status_cracked;
        if (probCracked > 0.0 && cracked_dist(rng)) utility |= SimParams::status_cracked;

        p.setValueUInt64(SimParams::PtArrIdx::idx_utility_data, utility);
        p.setValue(SimParams::PtArrIdx::idx_Jp_inv, 1.0);
        // Identity matrix for Fe
        p.setValue(SimParams::PtArrIdx::Fe00, 1.0);
        p.setValue(SimParams::PtArrIdx::Fe00 + SimParams::dim * 1 + 1, 1.0);

        p.setValue(SimParams::PtArrIdx::velx, 0.0);
        p.setValue(SimParams::PtArrIdx::velx + 1, 0.0);
    }

    hsd.hssoa.convertToIntegerCellFormat(h);
    hsd.SaveSnapshot(0, 0.0, true, hsd.data_directory);
    LOGR("PopulatePoints completed");
}
