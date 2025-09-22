#include "mainimageimporter.h"
#include <stack>
#include <algorithm>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

MainImageImporter::MainImageImporter() {}


void MainImageImporter::IdentifyIceThickness(const ParameterParser &params)
{
    this->colordata_OpenWater = params.colordata_OpenWater;
    this->colordata_Solid = params.colordata_Solid;
    this->colordata_Crushed = params.colordata_Crushed;
    precomputeThicknessRange();

    // --- Pre-conditions (Assumed true, checks removed) ---
    // 1. width, height > 0
    // 2. pngData is populated and size = width * height * 3
    // 3. colordata_OpenWater and colordata_Solid are populated.

    const size_t numPixels = static_cast<size_t>(width) * height;

    iceStatus.resize(numPixels);
    iceThickness.resize(numPixels);

    const float inv255 = 1.0f / 255.0f;

    const float thickness_range = solid_thickness - crushed_thickness;

    // Iterate through pixels using COLUMN-MAJOR indexing
    // x = column index, y = row index (from bottom-left)
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {

            const int idx = i+width*j;
            int path_val = sip.path_indices[i + width*j];
//            if(path_val != 1000)
            if(path_val == -1 || path_val > 1000)
            {
                iceStatus[idx] = 3;
                iceThickness[idx] = 0;
                continue;
            }

            const size_t png_byte_idx = (i+width*j)*3;
            // Extract RGB, convert to float [0, 1]
            const Eigen::Vector3f rgb(
                static_cast<float>(pngData[png_byte_idx]) * inv255,     // R
                static_cast<float>(pngData[png_byte_idx + 1]) * inv255, // G
                static_cast<float>(pngData[png_byte_idx + 2]) * inv255  // B
                );

            // Categorize the color using the helper function
            // Assumes categorizeColor returns:
            // {0, 0.0} for water
            // {1, 0.0} for crushed
            // {2, proj_pos} for intact (proj_pos is 0-1 along solid curve)
            auto [status, raw_value] = categorizeColor(rgb);
            if(params.MakeAllIceSolid && status==1) status = 2;

            // Store status
            iceStatus[idx] = status;

            // Assign thickness based on status and new rules
            switch (status) {
            case 0: // Open Water
                iceThickness[idx] = 0.0f;
                break;
            case 1: // Crushed Ice
                iceThickness[idx] = crushed_thickness; // Constant 0.8f
                break;
            case 2: // Intact Ice
                // Map raw_value (0-1) to the range [crushed_thickness, solid_thickness]
                // Formula: min + t * (max - min)
                if(!params.MakeAllIceSolid)
                {
                    iceThickness[idx] = crushed_thickness + raw_value * thickness_range;
                    if(iceThickness[idx] < (crushed_thickness+solid_thickness)/2)
                    {
                        iceStatus[idx] = 1;
                    }
                }
                else
                    iceThickness[idx] = solid_thickness;
                    // Optional: Clamp just in case raw_value is slightly outside [0,1] due to float errors
                    // iceThickness[idx] = std::clamp(iceThickness[idx], crushed_thickness, solid_thickness);
                break;
            default:
                // Handle unexpected status? Assign a default?
                iceThickness[idx] = 0.0f; // Or crushed_thickness?
                break;
            }
        }
    }
}



void MainImageImporter::LoadImage(std::string fileNamePNG, int expectedWidth, int expectedHeigth)
{
    int channels, imgx, imgy;
    unsigned char *png_data = stbi_load(fileNamePNG.c_str(), &imgx, &imgy, &channels, 3);
    if(!png_data || channels != 3 || imgx != expectedWidth || imgy != expectedHeigth)
    {
        std::cerr << "png not loaded" << std::endl;
        throw std::runtime_error("png not loaded");
    }
    width = imgx;
    height = imgy;

    size_t size_elems = imgx*imgy*3;
    pngData.resize(size_elems);

    for(int i=0;i<imgx;i++)
        for(int j=0;j<imgy;j++)
            for(int k=0;k<3;k++)
                pngData[(i+imgx*j)*3+k] = png_data[(i+imgx*(imgy-1-j))*3+k];

//    pngData.assign(png_data, png_data + size_elems);
    stbi_image_free(png_data);


    imageData->SetDimensions(imgx, imgy, 1); // 2D image, depth = 1
    imageData->SetSpacing(1.0, 1.0, 1.0);      // Pixel spacing
    imageData->SetOrigin(0.0, 0.0, 0.0);       // Origin at (0,0,0)


    scalars->SetNumberOfComponents(3);          // RGB has 3 components
    scalars->SetArray(pngData.data(), pngData.size(), 1);
    imageData->GetPointData()->SetScalars(scalars);

    plane->SetOrigin(0.0, 0.0, -1.0);           // Bottom-left corner
    plane->SetPoint1(imgx, 0.0, -1.0);         // Bottom-right (x-axis)
    plane->SetPoint2(0.0, imgy, -1.0);        // Top-left (y-axis)
    plane->SetNormal(0.0, 0.0, 1.0);           // Normal along z-axis (facing forward)

    texture->SetInputData(imageData);
    texture->InterpolateOff(); // Smooth texture rendering

    mapper->SetInputConnection(plane->GetOutputPort());

    actor->SetMapper(mapper);
    actor->SetTexture(texture);
}



void MainImageImporter::getColorFromValue(float val, unsigned char& r, unsigned char& g, unsigned char& b)
{
    // --- Optional Clamping: Ensure val is within the expected range ---
    val = std::max(-1.0f, std::min(val, 1.0f));

    float r_f = 0.0f;
    float g_f = 0.0f;
    float b_f = 0.0f;

    if (val <= 0.0f)
    {
        // Blue (-1.0) to White (0.0) segment
        // Interpolation factor goes from 0 (at val=-1) to 1 (at val=0)
        float factor = val + 1.0f; // Range [0, 1] for val in [-1, 0]

        // Blue stays at max (255)
        b_f = 255.0f;
        // Red and Green interpolate from 0 to 255
        r_f = factor * 255.0f;
        g_f = factor * 255.0f;
    }
    else // val > 0.0f
    {
        // White (0.0) to Red (1.0) segment
        // Interpolation factor goes from 0 (at val=0) to 1 (at val=1)
        float factor = val; // Range (0, 1] for val in (0, 1]

        // Red stays at max (255)
        r_f = 255.0f;
        // Green and Blue interpolate from 255 down to 0
        g_f = (1.0f - factor) * 255.0f;
        b_f = (1.0f - factor) * 255.0f;
    }

    // Convert float [0, 255] to unsigned char [0, 255]
    // Direct casting truncates, which is generally acceptable here.
    // Using std::round might be slightly more accurate but adds dependency/overhead.
    r = static_cast<unsigned char>(r_f);
    g = static_cast<unsigned char>(g_f);
    b = static_cast<unsigned char>(b_f);
}




// =============================== rendering

void MainImageImporter::Render(bool renderBC, bool renderCurrents)
{
    std::cout << "MainImageImporter::Render; renderBC " << renderBC << std::endl;

    renderedImage.assign(pngData.begin(),pngData.end());

    if(renderBC)
    {
        for(int i=0;i<width;i++)
            for(int j=0;j<height;j++)
            {
                int idx_rgb = (i+width*j)*3;
                int path_val = sip.path_indices[i + width*j];
                if(path_val != -1)
                {
                    SatelliteImageProcessor::getColorFromIndex(path_val,
                                                               renderedImage[idx_rgb+0],
                                                               renderedImage[idx_rgb+1],
                                                               renderedImage[idx_rgb+2]);
                }
            }
    }

    scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);

    scalars->Modified();
    imageData->Modified();
    texture->Modified();
}


void MainImageImporter::lerpColor(float t,
                                  unsigned char r1, unsigned char g1, unsigned char b1, // Color at t=0
                                  unsigned char r2, unsigned char g2, unsigned char b2, // Color at t=1
                                  unsigned char& out_r, unsigned char& out_g, unsigned char& out_b) {
    // Clamp t to [0, 1]
    t = std::max(0.0f, std::min(1.0f, t));
    // C++20 provides std::lerp, otherwise manual implementation
    out_r = static_cast<unsigned char>(static_cast<float>(r1) * (1.0f - t) + static_cast<float>(r2) * t);
    out_g = static_cast<unsigned char>(static_cast<float>(g1) * (1.0f - t) + static_cast<float>(g2) * t);
    out_b = static_cast<unsigned char>(static_cast<float>(b1) * (1.0f - t) + static_cast<float>(b2) * t);
}



void MainImageImporter::RenderV_n()
{
    const float scale = 0.2;
    std::cout << "MainImageImporter::RenderV_n()" << std::endl;
    renderedImage.assign(pngData.begin(),pngData.end());

    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            int idx_rgb = (i+width*j)*3;
            int path_val = sip.path_indices[i + width*j];
            if(path_val == 1000)
            {
                float val = fdp.velocity_field[i + width*j].norm();

                getColorFromValue(val*scale,
                                  renderedImage[idx_rgb+0],
                                  renderedImage[idx_rgb+1],
                                  renderedImage[idx_rgb+2]);

            }
            else if(path_val != -1)
            {
                SatelliteImageProcessor::getColorFromIndex(path_val,
                                                           renderedImage[idx_rgb+0],
                                                           renderedImage[idx_rgb+1],
                                                           renderedImage[idx_rgb+2]);
            }
        }

    scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);

    scalars->Modified();
    imageData->Modified();
    texture->Modified();

}



void MainImageImporter::RenderV_x()
{
    std::cout << "MainImageImporter::RenderV_x()" << std::endl;
    renderedImage.assign(pngData.begin(),pngData.end());

    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            int idx_rgb = (i+width*j)*3;
            int path_val = sip.path_indices[i + width*j];
            if(path_val != -1)
            {
                float val = fdp.velocity_field[i + width*j].x()/0.25;

                getColorFromValue(val,
                                  renderedImage[idx_rgb+0],
                                  renderedImage[idx_rgb+1],
                                  renderedImage[idx_rgb+2]);

            }
        }

    scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);

    scalars->Modified();
    imageData->Modified();
    texture->Modified();

}

void MainImageImporter::RenderV_y()
{
    std::cout << "MainImageImporter::RenderV_x()" << std::endl;
    renderedImage.assign(pngData.begin(),pngData.end());

    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            int idx_rgb = (i+width*j)*3;
            int path_val = sip.path_indices[i + width*j];
            if(path_val != -1)
            {
                float val = fdp.velocity_field[i + width*j].y()/0.25;

                getColorFromValue(val,
                                  renderedImage[idx_rgb+0],
                                  renderedImage[idx_rgb+1],
                                  renderedImage[idx_rgb+2]);

            }
        }

    scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);

    scalars->Modified();
    imageData->Modified();
    texture->Modified();

}


void MainImageImporter::RenderIceThickness()
{
    renderedImage.assign(pngData.begin(),pngData.end());


    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            int idx_rgb = (i+width*j)*3;
            int path_val = sip.path_indices[i + width*j];
            int idx = i + width*j;
            if(path_val == 1000)
            {
                uint8_t status = iceStatus[idx];
                float thickness = iceThickness[idx]; // Value is 0-1 for intact ice
                unsigned char r = 0, g = 0, b = 0;

                if(status == 0)
                {
                    // open water
                    r = 0; g = 0; b = 139; // Deep Blue
                }
                else if(status == 1)
                {
                    // crushed ice
                    r = 0; g = 128; b = 0; // Green
                }
                else if(status == 2)
                {
                    // Interpolate between Red (at thickness 0.8) and White (at thickness 1.0)
                    const float min_thick = crushed_thickness;
                    const float max_thick = solid_thickness;

                    // Clamp thickness to the interpolation range for safety
                    float clamped_thick = std::max(min_thick, std::min(max_thick, thickness));

                    // Calculate interpolation factor 't' (0 at min_thick, 1 at max_thick)
                    // Avoid division by zero if min_thick == max_thick
                    float t = 0.0f;
                    if (max_thick > min_thick) {
                        t = (clamped_thick - min_thick) / (max_thick - min_thick);
                    } else if (clamped_thick >= max_thick) {
                        t = 1.0f; // Assign white if thickness is at or above max_thick
                    }

                    // Define target colors
                    const unsigned char red_r = 255, red_g = 0, red_b = 0;
                    const unsigned char white_r = 255, white_g = 255, white_b = 255;

                    // Interpolate (Requires lerpColor helper function to be defined elsewhere)
                    lerpColor(t, red_r, red_g, red_b, white_r, white_g, white_b, r, g, b);
                }
                renderedImage[idx_rgb + 0] = r;
                renderedImage[idx_rgb + 1] = g;
                renderedImage[idx_rgb + 2] = b;
            }
        }
    scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);

    scalars->Modified();
    imageData->Modified();
    texture->Modified();
}


void MainImageImporter::RenderOriginal()
{
    renderedImage.assign(pngData.begin(),pngData.end());
    scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);
    scalars->Modified();
    imageData->Modified();
    texture->Modified();
}




void MainImageImporter::SaveAsHDF5(std::string ProjectDirectory)
{
    H5::H5File file((ProjectDirectory + std::string("/map.h5")), H5F_ACC_TRUNC);

    // Define data spaces (2D: width x height now)
    hsize_t dims[2] = {static_cast<hsize_t>(height), static_cast<hsize_t>(width)};
    H5::DataSpace dataspace(2, dims);

    // Set chunking and compression properties
    H5::DSetCreatPropList props;
    hsize_t chunk_dims[2] = {std::min<hsize_t>(height, 64), std::min<hsize_t>(width, 64)}; // Swapped: width x height
    props.setChunk(2, chunk_dims);
    props.setDeflate(6);


    H5::DataSet dataset_status = file.createDataSet("iceStatus", H5::PredType::NATIVE_UINT8,
                                                     dataspace, props);
    dataset_status.write(iceStatus.data(), H5::PredType::NATIVE_UINT8);


    H5::DataSet dataset_thickness = file.createDataSet("iceThickness", H5::PredType::NATIVE_FLOAT,
                                                    dataspace, props);
    dataset_thickness.write(iceThickness.data(), H5::PredType::NATIVE_FLOAT);

    sip.saveToHDF5(file);
}



//==================================  CLASSIFICATION ALGORITHM

// Pre-computes the min/max grayscale range from the solid ice samples.
// CALL THIS ONCE after loading the color data.
void MainImageImporter::precomputeThicknessRange()
{
    // Reset to defaults
    minSolidGray = 0.0f;
    maxSolidGray = 1.0f;

    if (colordata_Solid.empty()) {
        std::cerr << "Warning: colordata_Solid is empty. Using default thickness range [0, 1]." << std::endl;
        return;
    }

    minSolidGray = std::numeric_limits<float>::max();
    maxSolidGray = std::numeric_limits<float>::lowest();

    for (const Eigen::Vector3f& color : colordata_Solid) {
        float gray = toGrayscale(color);
        if (gray < minSolidGray) minSolidGray = gray;
        if (gray > maxSolidGray) maxSolidGray = gray;
    }

    // Edge case: if all sample colors have the same luminance, avoid division by zero.
    // We create a tiny, non-zero range.
    if (maxSolidGray - minSolidGray < 1e-6f) {
        maxSolidGray = minSolidGray + 1e-6f;
    }

    std::cout << "INFO: Calibrated solid ice thickness range from grayscale values ["
              << minSolidGray << ", " << maxSolidGray << "]" << std::endl;
}


// Helper function to find the minimum distance from a point to a "cloud" of other points.
// This replaces the incorrect projectPointOntoCurve logic for this problem.
float MainImageImporter::findClosestDistanceInCloud(
    const Eigen::Vector3f& point,
    const std::vector<Eigen::Vector3f>& cloud) const
{
    if (cloud.empty()) {
        return std::numeric_limits<float>::max(); // Return "infinity" if the category has no samples
    }

    float minDistanceSq = std::numeric_limits<float>::max();

    // Iterate through every sample point in the cloud
    for (const Eigen::Vector3f& cloudPoint : cloud) {
        // Calculate squared distance (faster than regular distance as it avoids sqrt)
        float distSq = (point - cloudPoint).squaredNorm();
        if (distSq < minDistanceSq) {
            minDistanceSq = distSq;
        }
    }

    // Return the actual distance
    return std::sqrt(minDistanceSq);
}


// Categorizes based on color by finding the closest sample point in any category's point cloud.
// Status: 0=water, 1=crushed, 2=solid
// Value: Normalized thickness (0-1) for solid ice, 0 otherwise.
std::pair<uint8_t, float> MainImageImporter::categorizeColor(const Eigen::Vector3f& rgb) const
{
    // Ensure all three color data clouds are loaded before proceeding.
    if (colordata_OpenWater.empty() || colordata_Solid.empty() || colordata_Crushed.empty()) {
        std::cerr << "Warning: One or more color data sets are empty. Cannot categorize pixel." << std::endl;
        return {1, 0.0f}; // Default to crushed ice with zero thickness
    }

    // 1. Find the minimum distance to each of the three color clouds.
    float dist_water = findClosestDistanceInCloud(rgb, colordata_OpenWater);
    float dist_solid = findClosestDistanceInCloud(rgb, colordata_Solid);
    float dist_crushed = findClosestDistanceInCloud(rgb, colordata_Crushed);

    // 2. Compare the distances to find the minimum, which determines the class.
    if (dist_water <= dist_solid && dist_water <= dist_crushed) {
        // The color's nearest neighbor is in the Open Water set.
        return {0, 0.0f}; // Status 0: Water, Thickness 0.0
    }
    else if (dist_solid <= dist_crushed) {
        // The color's nearest neighbor is in the Solid Ice set.
        // Convert the pixel's color to grayscale.
        float currentGray = toGrayscale(rgb);

        // Linearly rescale the grayscale value from the calibrated range to the [0, 1] thickness range.
        float thickness = (currentGray - minSolidGray) / (maxSolidGray - minSolidGray);

        // Clamp the result to ensure it's always valid, even if a pixel is darker/brighter
        // than any of the samples.
        return {2, std::clamp(thickness, 0.0f, 1.0f)};
    }
    else {
        // The color's nearest neighbor is in the Crushed Ice set.
        return {1, 0.0f}; // Status 1: Crushed Ice, Thickness 0.0
    }
}



// mark pier regions

/**
 * @brief Identifies contiguous regions of a specific color in the source image and merges them
 * into the SatelliteImageProcessor's path_indices grid.
 *
 * @param pierColor The RGB color of the pier regions to be identified.
 */
void MainImageImporter::ProcessBridgePiers(const Eigen::Vector3i& pierColor)
{
    std::cout << "INFO: Starting bridge pier processing..." << std::endl;

    // --- 1. Sanity Checks ---
    if (pngData.empty() || sip.path_indices.empty()) {
        spdlog::error("Cannot process bridge piers: Image data or SIP path_indices not loaded.");
        throw std::runtime_error("Prerequisite data not loaded for pier processing.");
    }
    if (sip.path_indices.size() != static_cast<size_t>(width * height)) {
        spdlog::error("Mismatch between image size and path_indices size.");
        throw std::runtime_error("Dimension mismatch in pier processing.");
    }

    // --- 2. Find the next available index to use for the new regions ---
    // This prevents overwriting the indices from the SVG file.
    int max_existing_index = 0;
    if (!sip.path_indices.empty()) {
        max_existing_index = *std::max_element(sip.path_indices.begin(), sip.path_indices.end());
    }
    const int next_available_index = max_existing_index + 1;
    std::cout << "INFO: Next available path index: " << next_available_index << std::endl;


    // --- 3. Find and label all contiguous pier regions ---
    std::vector<int> pier_labels(width * height, 0); // Temp storage for labeled regions (1, 2, 3...)
    int regionCount = 0;
    findConnectedRegions(pierColor, pier_labels, regionCount);

    if (regionCount == 0) {
        std::cout << "WARN: No regions with the specified pier color were found." << std::endl;
        return;
    }
    std::cout << "INFO: Found " << regionCount << " distinct pier regions." << std::endl;

    // --- 4. Merge the newly found regions into sip.path_indices ---
    int pixels_updated = 0;
    for (size_t i = 0; i < pier_labels.size(); ++i) {
        // A label > 0 means this pixel belongs to a pier region.
        if (pier_labels[i] > 0) {
            // The label in pier_labels is a temporary ID (1, 2, 3...).
            // We map it to a globally unique ID using our starting offset.
            // Example: region 1 becomes ID `next_available_index`
            //          region 2 becomes ID `next_available_index + 1`
            sip.path_indices[i] = next_available_index + (pier_labels[i] - 1);
            pixels_updated++;
        }
    }
    std::cout << "INFO: Merged pier regions into path_indices. Updated " << pixels_updated << " pixels." << std::endl;
}


/**
 * @brief Scans the entire image to find all distinct, contiguous regions of a target color.
 *
 * @param targetColor The RGB color to search for.
 * @param labels      Output vector where each region will be marked with a unique integer label (1, 2, 3...).
 * @param regionCount Output parameter that will hold the total number of distinct regions found.
 */
void MainImageImporter::findConnectedRegions(const Eigen::Vector3i& targetColor,
                                             std::vector<int>& labels,
                                             int& regionCount) const
{
    std::vector<bool> visited(width * height, false);
    regionCount = 0;

    // Iterate through every pixel of the image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * width + x;

            // If we've already processed this pixel, skip it
            if (visited[index]) {
                continue;
            }

            // Get the color of the current pixel (assuming 3 channels: R, G, B)
            const unsigned char r = pngData[index * 3 + 0];
            const unsigned char g = pngData[index * 3 + 1];
            const unsigned char b = pngData[index * 3 + 2];

            // Check if the color matches our target
            if (r == targetColor.x() && g == targetColor.y() && b == targetColor.z()) {
                // This is the start of a new, unvisited region
                regionCount++;
                // Use flood fill to find all connected pixels in this region and label them
                floodFill(x, y, regionCount, targetColor, labels, visited);
            }
        }
    }
}


/**
 * @brief Performs a flood-fill algorithm to find all connected pixels of a target color,
 * starting from a given seed point. This is an iterative implementation to avoid stack overflow.
 *
 * @param startX      The x-coordinate of the starting pixel.
 * @param startY      The y-coordinate of the starting pixel.
 * @param label       The integer label to assign to all pixels in the found region.
 * @param targetColor The color of the pixels that constitute the region.
 * @param labels      The output vector where labeled pixels are stored.
 * @param visited     A boolean vector to keep track of already-processed pixels.
 */
void MainImageImporter::floodFill(int startX, int startY, int label,
                                  const Eigen::Vector3i& targetColor,
                                  std::vector<int>& labels,
                                  std::vector<bool>& visited) const
{
    std::stack<std::pair<int, int>> pixels_to_visit;
    pixels_to_visit.push({startX, startY});

    // Mark the starting pixel as visited immediately to avoid re-processing
    visited[static_cast<size_t>(startY) * width + startX] = true;

    while (!pixels_to_visit.empty()) {
        std::pair<int, int> current = pixels_to_visit.top();
        pixels_to_visit.pop();

        int x = current.first;
        int y = current.second;

        size_t index = static_cast<size_t>(y) * width + x;
        labels[index] = label; // Assign the region label

        // --- Check 4-way neighbors (up, down, left, right) ---
        int dx[] = {0, 0, 1, -1};
        int dy[] = {1, -1, 0, 0};

        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];

            // Check if neighbor is within image bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                size_t neighbor_index = static_cast<size_t>(ny) * width + nx;

                // If neighbor has been visited, skip
                if (visited[neighbor_index]) {
                    continue;
                }

                // Check if neighbor has the target color
                const unsigned char r = pngData[neighbor_index * 3 + 0];
                const unsigned char g = pngData[neighbor_index * 3 + 1];
                const unsigned char b = pngData[neighbor_index * 3 + 2];

                if (r == targetColor.x() && g == targetColor.y() && b == targetColor.z()) {
                    visited[neighbor_index] = true; // Mark as visited
                    pixels_to_visit.push({nx, ny}); // Add to the stack for processing
                }
            }
        }
    }
}
