// generalgriddata.cpp

#include "generalgriddata.h"

#include <map>
#include <regex>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


void GeneralGridData::ReadParameterFile(std::string parameterFileName)
{
    // --- 1. Parse Parameter File ---
    std::map<std::string, std::string> additionalFiles = prms.ParseFile(parameterFileName);
    std::string pngFile = additionalFiles["InputPNG"];
    std::string mapFile = additionalFiles["InputMap"];

    // --- 2. Load Grid Parameters and Path Indices from HDF5 Map File ---



    std::vector<int> path_indices;

    // Scoped H5File object ensures it's closed automatically
    {
        LOGV("Loading path_indices");
        H5::H5File file(mapFile, H5F_ACC_RDONLY);
        H5::DataSet ds_path_indices = file.openDataSet("path_indices");

        ds_path_indices.openAttribute("width").read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeX);
        ds_path_indices.openAttribute("height").read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeY);

        ds_path_indices.openAttribute("ModeledRegionOffsetX").read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetX);
        ds_path_indices.openAttribute("ModeledRegionOffsetY").read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetY);
        ds_path_indices.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &prms.GridXTotal);
        ds_path_indices.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &prms.GridYTotal);
        prms.cellsize = prms.DimensionHorizontal / (prms.InitializationImageSizeX-1);
        prms.cellsize_inv = 1.0/prms.cellsize;

        path_indices.resize(prms.InitializationImageSizeX*prms.InitializationImageSizeY);

        LOGV("Loading path_indices - dataset opened");
        // Parameters are already loaded by ParseFile, so we just need the dataset
        ds_path_indices.read(path_indices.data(), H5::PredType::NATIVE_INT);
        LOGV("Loading path_indices - done");
    } // file is closed here


    const int width = prms.InitializationImageSizeX;
    const int height = prms.InitializationImageSizeY;
    const int ox = prms.ModeledRegionOffsetX;
    const int oy = prms.ModeledRegionOffsetY;
    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;

    // --- 3. Load PNG Image ---
    int channels, imgx, imgy;
    unsigned char* png_data = stbi_load(pngFile.c_str(), &imgx, &imgy, &channels, 3);
    if (!png_data || channels != 3 || imgx != width || imgy != height) {
        LOGR("Fatal Error: PNG file '{}' could not be loaded or has incorrect dimensions.", pngFile);
        throw std::runtime_error("Failed to load PNG for background image.");
    }

    // --- 4. Populate `original_image_colors_rgb` ---
    original_image_colors_rgb.resize(width * height * 3);
    auto idxInPng = [&](int i, int j) -> int { return 3 * ((height - j - 1) * width + i); };

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            for (int k = 0; k < 3; k++) {
                original_image_colors_rgb[((j * width) + i) * 3 + k] = png_data[idxInPng(i, j) + k];
            }
        }
    }
    stbi_image_free(png_data);

    // --- 5. Generate `grid_status_buffer` (The Core Logic) ---
    // This replicates the logic from SnapshotManager::PrepareGrid
    grid_status_buffer.resize((size_t)gx * gy);

    auto transformPathIdx = [](const int &idx) -> uint8_t {
        if (idx < 1000) return (uint8_t)(idx + 1);
        else if (idx == 1000) return (uint8_t)(100); // 1000 -> 100 (modeled area)
        else return (uint8_t)(idx - 1000 + 1);
    };

    for (int i = 0; i < gx; i++) {
        for (int j = 0; j < gy; j++) {
            // Index in the full-size image/path_indices map (row-major)
            size_t image_map_idx = (size_t)(i + ox) + (size_t)(j + oy) * width;
            // Index in the grid-sized status buffer (column-major)
            size_t grid_buffer_idx = (size_t)j + (size_t)i * gy;

            grid_status_buffer[grid_buffer_idx] = transformPathIdx(path_indices[image_map_idx]);
        }
    }

    LOGV("GeneralGridData::ReadParameterFile: Done.");
}



void GeneralGridData::ScanDirectory(std::string directoryName)
{
    frameDirectory = directoryName;
    LOGR("GeneralGridData::ScanDirectory scanning: {}", directoryName);

    // --- Start of rewritten logic ---
    this->countFrames = 0; // Initialize/reset the count

    const std::filesystem::path dirPath(directoryName);

    // 2. Define the regex pattern for files like f00001.h5
    //    ^      - start of the string
    //    f      - literal 'f'
    //    \d+    - one or more digits (use \d{5} if it MUST be exactly 5 digits)
    //    \.     - literal '.' (must escape the dot)
    //    h5     - literal 'h5'
    //    $      - end of the string
    //    R"(...)" - Raw string literal avoids needing to double-escape backslashes
    const std::regex filePattern(R"(^f\d+\.h5$)");
    // If exactly 5 digits are required:
    // const std::regex filePattern(R"(^f\d{5}\.h5$)");

    int foundCount = 0;

    // 3. Iterate through the directory entries
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        // 4. Check if it's a regular file (not a directory, symlink, etc.)
        if (entry.is_regular_file()) {
            // 5. Get the filename part of the path
            const std::string filename = entry.path().filename().string();

            // 6. Check if the filename matches the pattern
            if (std::regex_match(filename, filePattern)) {
                foundCount++;
            }
        }
    }

    // 7. Store the result in the member variable
    this->countFrames = foundCount;
    LOGR("Found {} matching frame files in directory {}.", this->countFrames, directoryName);
    LOGV("GeneralGridData::ScanDirectory done");
}
