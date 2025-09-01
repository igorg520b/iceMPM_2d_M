// generalgriddata.cpp

#include "generalgriddata.h"

#include <map>
#include <regex>
#include <filesystem>

//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


void GeneralGridData::ReadParameterFile(std::string parameterFileName)
{
    std::map<std::string,std::string> additionalFiles = prms.ParseFile(parameterFileName);

    std::string pngFile = additionalFiles["InputPNG"];
    std::string mapFile = additionalFiles["InputMap"];


    // load HDF5
    H5::H5File file(mapFile, H5F_ACC_RDONLY);

    H5::DataSet ds_path_indices = file.openDataSet("path_indices");
    ds_path_indices.openAttribute("width").read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeX);
    ds_path_indices.openAttribute("height").read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeY);

    ds_path_indices.openAttribute("ModeledRegionOffsetX").read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetX);
    ds_path_indices.openAttribute("ModeledRegionOffsetY").read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetY);
    ds_path_indices.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &prms.GridXTotal);
    ds_path_indices.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &prms.GridYTotal);

    const int &width = prms.InitializationImageSizeX;
    const int &height = prms.InitializationImageSizeY;
    prms.cellsize = prms.DimensionHorizontal / (prms.InitializationImageSizeX-1);
    prms.cellsize_inv = 1.0/prms.cellsize;

    path_indices.resize(width*height);
    ds_path_indices.read(path_indices.data(), H5::PredType::NATIVE_INT);
    file.close();




    // load image data
    // (3) Load PNG image (only for rendering)
    int channels, imgx, imgy;
    unsigned char *png_data = stbi_load(pngFile.c_str(), &imgx, &imgy, &channels, 3); // expect 3 channels - RGB
    if(!png_data || channels != 3 || imgx != width || imgy != height)
    {
        LOGR("filename {} not loaded; channels {}", pngFile, channels);
        LOGR("png expected size {} x {}; actual size {} x {}", width, height, imgx, imgy);
        throw std::runtime_error("png1 not loaded");
    }

    // function for obtaining index in png_data from the pixel's 2D index (i,j)
    auto idxInPng = [&](int i, int j) -> int { return 3*((height - j - 1)*width + i); };

    // save original colors for the whole image
    constexpr uint8_t waterColor[3] = {0x15, 0x1f, 0x2f};
    original_image_colors_rgb.resize(imgx*imgy*3);
    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
            for(int k=0;k<3;k++)
            {
                if(path_indices[i+j*width] == 1000)
                {
                    original_image_colors_rgb[(i+j*width)*3+k] = waterColor[k];
                }
                else
                {
                    original_image_colors_rgb[(i+j*width)*3+k] = png_data[idxInPng(i, j)+k];
                }
            }

    stbi_image_free(png_data);

    LOGV("GeneralGridData::ReadParameterFile: done");
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
