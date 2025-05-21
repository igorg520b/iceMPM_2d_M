#include "mainimageimporter.h"



#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

MainImageImporter::MainImageImporter() {}


void MainImageImporter::IdentifyIceThickness()
{
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
            const auto [status, raw_value] = categorizeColor(rgb);

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
                iceThickness[idx] = crushed_thickness + raw_value * thickness_range;
                if(iceThickness[idx] < (crushed_thickness+solid_thickness)/2)
                {
                    iceStatus[idx] = 1;
                }
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
        spdlog::error("filename {}", fileNamePNG);
        spdlog::error("png_data {}", (void*)png_data);
        spdlog::error("channels {}", channels);
        spdlog::error("image dimensions {} x {}", imgx, imgy);
        spdlog::error("png not loaded");
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

    // --- Alternative using std::round (include <cmath>) ---
    // r = static_cast<unsigned char>(std::round(r_f));
    // g = static_cast<unsigned char>(std::round(g_f));
    // b = static_cast<unsigned char>(std::round(b_f));
}



MainImageImporter::ProjectionResult MainImageImporter::projectPointOntoCurve(
    const Eigen::Vector3f& point,
    const std::vector<Eigen::Vector3f>& curvePoints)
{
    ProjectionResult bestResult;
    float accumulatedLength = 0.0f;
    std::vector<float> segmentLengths;
    float totalLength = 0.0f;

    if (curvePoints.size() < 2) return bestResult; // Need at least 2 points

    // Pre-calculate segment lengths and total length
    for (size_t i = 0; i < curvePoints.size() - 1; ++i) {
        const Eigen::Vector3f& p0 = curvePoints[i];
        const Eigen::Vector3f& p1 = curvePoints[i + 1];
        float len = (p1 - p0).norm();
        segmentLengths.push_back(len);
        totalLength += len;
    }

    // Handle degenerate curve (all points coincident or only one point)
    if (totalLength <= 1e-9f) {
        if (!curvePoints.empty()) {
            bestResult.distance = (point - curvePoints[0]).norm();
            bestResult.position = 0.0f; // Or arguably undefined?
        }
        return bestResult;
    }


    // Iterate through segments to find the closest point
    for (size_t i = 0; i < curvePoints.size() - 1; ++i) {
        const Eigen::Vector3f& p0 = curvePoints[i];
        const Eigen::Vector3f& p1 = curvePoints[i + 1];
        const Eigen::Vector3f segmentVec = p1 - p0;
        const float segmentLengthSq = segmentVec.squaredNorm();
        const float currentSegmentLength = segmentLengths[i];

        Eigen::Vector3f closestPointOnSegment;
        float t = 0.0f; // Parameter along the segment [0, 1]

        if (segmentLengthSq > 1e-9f) {
            const Eigen::Vector3f pointVec = point - p0;
            t = pointVec.dot(segmentVec) / segmentLengthSq;
            t = std::clamp(t, 0.0f, 1.0f); // Clamp t to [0, 1]
            closestPointOnSegment = p0 + t * segmentVec;
        } else {
            closestPointOnSegment = p0; // Segment is effectively a point
            t = 0.0f;
        }

        float dist = (point - closestPointOnSegment).norm();

        if (dist < bestResult.distance) {
            bestResult.distance = dist;
            // Calculate normalized position along the *entire* curve
            bestResult.position = (accumulatedLength + t * currentSegmentLength) / totalLength;
                // Ensure position doesn't slightly exceed 1 due to float errors
            bestResult.position = std::clamp(bestResult.position, 0.0f, 1.0f);
        }
        accumulatedLength += currentSegmentLength;
    }

    return bestResult;
}


// Categorizes based on color, returns status and thickness/value
// Accesses member color data vectors.
// Status: 0=water, 1=crushed, 2=intact
// Value: Normalized thickness (0-1) for intact, 0 otherwise
std::pair<uint8_t, float> MainImageImporter::categorizeColor(const Eigen::Vector3f& rgb) const // Mark const
{
    // Ensure color data is loaded
    if (colordata_OpenWater.empty() || colordata_Solid.empty()) {
        // Maybe throw an exception or return a default?
        std::cerr << "Warning: Color data not loaded, cannot categorize pixel." << std::endl;
        return {1, 0.0f}; // Default to crushed? Or maybe water?
    }

    // 1. Check for Open Water
    ProjectionResult waterProj = projectPointOntoCurve(rgb, colordata_OpenWater);
    if (waterProj.distance < openWaterThreshold) {
        return {0, 0.0f}; // Status 0: Open Water, Thickness 0
    }

    // 2. Check for Solid/Intact Ice
    ProjectionResult solidProj = projectPointOntoCurve(rgb, colordata_Solid);

    // If close enough to the solid curve, classify as Intact
    if (solidProj.distance < intactIceThreshold) {
        // Status 2: Intact Ice, Value is the normalized position
        return {2, solidProj.position};
    }

    // 3. Otherwise, classify as Crushed Ice
    return {1, 0.0f}; // Status 1: Crushed Ice, Thickness 0
}




// =============================== rendering

void MainImageImporter::Render(bool renderBC, bool renderCurrents)
{
    spdlog::info("MainImageImporter::Render; renderBC {}", renderBC);

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
    spdlog::info("MainImageImporter::RenderV_n()");
    spdlog::info("velocity field size {}; expected size {}", fdp.velocity_field.size(), width*height);
    renderedImage.assign(pngData.begin(),pngData.end());

    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            int idx_rgb = (i+width*j)*3;
            int path_val = sip.path_indices[i + width*j];
            if(path_val == 1000)
            {
                float val = fdp.velocity_field[i + width*j].norm();

                getColorFromValue(val*4,
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
    spdlog::info("MainImageImporter::RenderV_x()");
    spdlog::info("velocity field size {}; expected size {}", fdp.velocity_field.size(), width*height);
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
    spdlog::info("MainImageImporter::RenderV_x()");
    spdlog::info("velocity field size {}; expected size {}", fdp.velocity_field.size(), width*height);
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
