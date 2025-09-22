#include "satelliteimageprocessor.h"

// #include <spdlog/spdlog.h> // Removed spdlog include
#include <iostream> // Added for std::cout and std::cerr
#include <algorithm>
#include <filesystem>
#include <unordered_set>

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h" // Assuming nanosvg.h is in the same directory or include path

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Assuming stb_image_write.h is in the same directory or include path


void SatelliteImageProcessor::LoadSVG(std::string fileName, int rasterWidth, int rasterHeight,
                                      std::string MainPathID, std::string RectanglePathID, std::string FluentPathID,
                                      const std::vector<std::string>& rasterizedPathIDs,
                                      std::string ProjectDirectory)
{
    std::cout << "INFO: SatelliteImageProcessor::LoadSVG" << std::endl;
    this->ProjectDirectory = ProjectDirectory;
    this->width = rasterWidth;
    this->height = rasterHeight;

    // --- 1. Extract ALL Bezier paths from the SVG file ---
    // This initial step loads everything so we can validate against the full list.
    extractBezierCurvesFromSVG(fileName, RectanglePathID, MainPathID, FluentPathID);

    // --- 2. Validate that all requested paths exist in the SVG ---
    std::unordered_set<std::string> foundPathIDs;
    for (const auto& path : bezierPaths) {
        if (!path.name.empty()) {
            foundPathIDs.insert(path.name);
        }
    }

    // Check for MainPathID
    if (foundPathIDs.find(MainPathID) == foundPathIDs.end()) {
        throw std::runtime_error("Validation failed: The required MainPathID '" + MainPathID + "' was not found in the SVG file.");
    }

    // Check for all other requested paths
    for (const auto& requiredID : rasterizedPathIDs) {
        if (foundPathIDs.find(requiredID) == foundPathIDs.end()) {
            throw std::runtime_error("Validation failed: The requested path '" + requiredID + "' was not found in the SVG file.");
        }
    }
    std::cout << "INFO: All requested paths were successfully validated." << std::endl;


    // --- 3. Filter the paths to keep only the ones we need to rasterize ---
    std::unordered_set<std::string> pathsToKeep(rasterizedPathIDs.begin(), rasterizedPathIDs.end());
    pathsToKeep.insert(MainPathID); // Ensure MainPathID is always included

    std::vector<BezierPath> filteredPaths;
    for (const auto& path : bezierPaths) {
        if (pathsToKeep.count(path.name)) {
            filteredPaths.push_back(path);
        }
    }
    // Replace the original full list with our new filtered list
    bezierPaths = std::move(filteredPaths);
    std::cout << "INFO: Filtered SVG paths. Kept " << bezierPaths.size() << " paths for processing." << std::endl;


    // --- 4. Process the filtered list of paths ---

    // Keeps only closed paths from our filtered list
    bezierPaths.erase(std::remove_if(bezierPaths.begin(), bezierPaths.end(),
                                     [](const BezierPath& path) { return !path.closed; }),
                      bezierPaths.end());

    // Move main path to the top of the filtered list
    auto it = std::find_if(bezierPaths.begin(), bezierPaths.end(),
                           [&](const BezierPath& path) {
                               return path.name == MainPathID;
                           });

    // Our earlier validation guarantees that 'it' will be valid, but we keep the check for safety.
    if (it != bezierPaths.end()) {
        if (it != bezierPaths.begin()) { // Avoid swapping with itself
            std::swap(*bezierPaths.begin(), *it);
        }
        bezierPaths[0].isMain = true;
        std::cout << "INFO: Moved '" << MainPathID << "' to the top of the processing list." << std::endl;
    } else {
        // This case should not be reachable due to the validation step above.
        throw std::runtime_error("Logic error: MainPathID '" + MainPathID + "' was lost after filtering. It might be an open path.");
    }


    // --- 5. Apply coordinate transformation to the final list of paths ---
    std::cout << "INFO: Applying transformation to align SVG coordinates with raster image." << std::endl;

    Eigen::Vector2f svg_min = extents[0];
    Eigen::Vector2f svg_max = extents[1];
    Eigen::Vector2f svg_dims = svg_max - svg_min;

    if (svg_dims.x() <= 0 || svg_dims.y() <= 0) {
        throw std::runtime_error("Invalid extents for RectanglePathID. Dimensions must be positive.");
    }

    float scale = static_cast<float>(width) / svg_dims.x();
    Eigen::Vector2f translation = svg_min;

    float svg_aspect_ratio = svg_dims.x() / svg_dims.y();
    float raster_aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    if (std::abs(svg_aspect_ratio - raster_aspect_ratio) > 1e-3) {
        std::cerr << "WARN: SVG rectangle aspect ratio (" << svg_aspect_ratio
                  << ") differs from raster image aspect ratio (" << raster_aspect_ratio
                  << "). This may cause vertical stretching." << std::endl;
    }
    std::cout << "INFO: Calculated transform: scale = " << scale << ", translation = (" << translation.x() << ", " << translation.y() << ")" << std::endl;

    // Apply the transformation to all points in every *filtered* bezier path
    for (auto& path : bezierPaths)
    {
        for (auto& segment : path.segments)
        {
            Eigen::Vector2f p0(segment.x0, segment.y0);
            p0 = (p0 - translation) * scale;
            segment.x0 = p0.x();
            segment.y0 = p0.y();

            Eigen::Vector2f p1(segment.x1, segment.y1);
            p1 = (p1 - translation) * scale;
            segment.x1 = p1.x();
            segment.y1 = p1.y();

            Eigen::Vector2f p2(segment.x2, segment.y2);
            p2 = (p2 - translation) * scale;
            segment.x2 = p2.x();
            segment.y2 = p2.y();

            Eigen::Vector2f p3(segment.x3, segment.y3);
            p3 = (p3 - translation) * scale;
            segment.x3 = p3.x();
            segment.y3 = p3.y();
        }
    }

    // Apply the transformation to the fluent extents
    extents_fluent[0] = (extents_fluent[0] - translation) * scale;
    extents_fluent[1] = (extents_fluent[1] - translation) * scale;

    const float old_y_min = extents_fluent[0].y();
    const float old_y_max = extents_fluent[1].y();
    extents_fluent[0].y() = static_cast<float>(height) - old_y_max;
    extents_fluent[1].y() = static_cast<float>(height) - old_y_min;
    std::cout << "INFO: Transformed fluent extents: (" << extents_fluent[0].x() << ", " << extents_fluent[0].y()
              << ") - (" << extents_fluent[1].x() << ", " << extents_fluent[1].y() << ")" << std::endl;

    // Update the main extents to the new raster coordinate system
    extents[0] = Eigen::Vector2f(0.0f, 0.0f);
    extents[1] = Eigen::Vector2f(static_cast<float>(width), static_cast<float>(height));

    std::cout << "INFO: done SatelliteImageProcessor::LoadSVG" << std::endl;
}


void SatelliteImageProcessor::RasterizeSVG()
{
    // rasterize
    normals.assign(width * height, Eigen::Vector2f::Zero());
    path_indices.assign(width * height, -1);
    for (size_t pathIdx = 0; pathIdx < bezierPaths.size(); ++pathIdx) {
        auto& path = bezierPaths[pathIdx];
        path.render(normals, path_indices, width, height, static_cast<int>(pathIdx));
    }


    for (size_t pathIdx = 0; pathIdx < bezierPaths.size(); ++pathIdx) {
        auto& path = bezierPaths[pathIdx];
        path.renderFilled(path_indices, width, height, pathIdx);
    }


    DetermineSimulationSubregion();

    //    renderNormalsToPNG();

    // write path IDs into a file - if needed for subsequent analysis
    std::ofstream csvFile(ProjectDirectory + "/path_ids.csv");
    for(int k=0; k<bezierPaths.size();k++)
    {
        auto& path = bezierPaths[k];
        csvFile << k <<','<<path.name << '\n';
    }
    csvFile.close();
}


void SatelliteImageProcessor::renderNormalsToPNG()
{
    // Validate inputs
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions: width and height must be positive");
    }
    if (normals.size() != static_cast<size_t>(width * height)) {
        throw std::runtime_error("Normals array size does not match width * height");
    }

    // Lambda for index computation (sideways and vertically flipped)
    auto computeIndex = [&](int i, int j) -> size_t {
        return (height - 1 - j)*width + i;
    };

    // Allocate RGB raster arrays (3 bytes per pixel: R, G, B)
    std::vector<unsigned char> nx_raster(width * height * 3, 0);
    std::vector<unsigned char> ny_raster(width * height * 3, 0);
    std::vector<unsigned char> indices_raster(width * height * 3, 0);

    // Render normals into RGB rasters
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = computeIndex(x, y);         // Use lambda for sideways indexing
            size_t rgb_idx = (x + width*y) * 3;                // Index in RGB raster (3 bytes per pixel)

            // Get normal components (nx, ny range from -1 to 1)
            if (normals[idx].norm() > 0.1) {
                float nx = normals[idx].x();
                float ny = normals[idx].y();

                // Map nx and ny from [-1, 1] to [0, 255] for RGB
                // For nx: Red channel, zero = 128 (gray), positive = red, negative = blue
                if (nx > 0) nx_raster[rgb_idx + 0] = static_cast<unsigned char>(nx * 255.f);
                else if (nx <= 0) nx_raster[rgb_idx + 2] = static_cast<unsigned char>(-nx * 255.f);

                if (ny > 0) ny_raster[rgb_idx + 0] = static_cast<unsigned char>(ny * 255.f);
                else if (ny <= 0) ny_raster[rgb_idx + 2] = static_cast<unsigned char>(-ny * 255.f);
            }

            // Indices
            int path_idx = path_indices[idx];
            if (path_idx == -1)
            {
                indices_raster[rgb_idx + 0] = 40;
                indices_raster[rgb_idx + 1] = 40;
            }
            else if (path_idx < 1000)
            {
                // boundary
                getColorFromIndex(path_idx, indices_raster[rgb_idx + 0], indices_raster[rgb_idx + 1], indices_raster[rgb_idx + 2]);
            } else
            {
                getColorFromIndex(path_idx+10, indices_raster[rgb_idx + 0], indices_raster[rgb_idx + 1], indices_raster[rgb_idx + 2]);
            }
        }
    }

    // Write nx raster to PNG
    if (!stbi_write_png("normals_nx.png", width, height, 3, nx_raster.data(), width * 3)) {
        throw std::runtime_error("Failed to write nx output to normals_nx.png");
    }

    // Write ny raster to PNG
    if (!stbi_write_png("normals_ny.png", width, height, 3, ny_raster.data(), width * 3)) {
        throw std::runtime_error("Failed to write ny output to normals_ny.png");
    }

    // Write indices raster to PNG
    if (!stbi_write_png("indices.png", width, height, 3, indices_raster.data(), width * 3)) {
        throw std::runtime_error("Failed to write indices output to indices.png");
    }
}



void SatelliteImageProcessor::extractBezierCurvesFromSVG(const std::string& filename,
                                                         const std::string& RectanglePathID,
                                                         const std::string& MainPathID,
                                                         const std::string& FluentPathID)
{
    bezierPaths.clear();

    // Load and parse the SVG file
    NSVGimage* image = nsvgParseFromFile(filename.c_str(), "px", 96);
    if (!image) throw std::runtime_error("Failed to parse SVG file: " + filename);


    // Iterate over all shapes in the SVG
    for (NSVGshape* shape = image->shapes; shape != nullptr; shape = shape->next)
    {
        // RectanglePathID should be ignored as BezierPath
        if (shape->id && RectanglePathID == std::string(shape->id))
        {
            // Derive bounding box from shape extents
            extents[0] = Eigen::Vector2f(shape->bounds[0], shape->bounds[1]); // min x,y
            extents[1] = Eigen::Vector2f(shape->bounds[2], shape->bounds[3]); // max x,y
            continue; // skip adding it as a BezierPath
        }
        else if (shape->id && FluentPathID == std::string(shape->id))
        {
            // Derive bounding box from shape extents
            extents_fluent[0] = Eigen::Vector2f(shape->bounds[0], shape->bounds[1]); // min x,y
            extents_fluent[1] = Eigen::Vector2f(shape->bounds[2], shape->bounds[3]); // max x,y
            std::cout << "fluent extents\n" << extents_fluent[0] << std::endl << extents_fluent[1] << std::endl;
            continue; // skip adding it as a BezierPath
        }

        // Iterate over all paths in the shape
        for (NSVGpath* path = shape->paths; path != nullptr; path = path->next)
        {
            BezierPath bezierPath;
            bezierPath.closed = (path->closed != 0);
            bezierPath.name   = std::string(shape->id ? shape->id : "");
            std::cout << "bezierPath.name = " << bezierPath.name << std::endl;

            if(!bezierPath.closed && bezierPath.name == MainPathID)
            {
                std::cerr << "ERROR: MainPathID is open" << std::endl;
                throw std::runtime_error("MainPathID is open");
            }

            int npts = path->npts;
            float* pts = path->pts;

            int numSegments = (npts - 1) / 3;
            if (npts < 4 || (npts - 1) % 3 != 0)
            {
                std::cerr << "Warning: Invalid point count " << npts << " for path, skipping\n";
                continue;
            }

            for (int i = 0; i < numSegments * 6; i += 6)
            {
                if (i + 6 >= npts * 2) break;
                BezierPath::BezierSegment segment(
                    pts[i],     pts[i + 1],
                    pts[i + 2], pts[i + 3],
                    pts[i + 4], pts[i + 5],
                    pts[i + 6], pts[i + 7]
                    );
                bezierPath.segments.push_back(segment);
            }

            bezierPaths.push_back(bezierPath);
        }
    }

    nsvgDelete(image);
    std::cout << "INFO: extractBezierCurvesFromSVG done" << std::endl;
}





void SatelliteImageProcessor::getColorFromIndex(int index, unsigned char& r, unsigned char& g, unsigned char& b)
{
    if(index >=1000) index+=5;
    // Table of 20 distinct, nice colors (RGB values from 0-255)
    static const struct { unsigned char r, g, b; } colors[20] = {
        {255, 87, 51},   // Coral
        {46, 204, 113},  // Emerald
        {52, 152, 219},  // Sky Blue
        {155, 89, 182},  // Amethyst
        {241, 196, 15},  // Sunflower
        {231, 76, 60},   // Alizarin
        {26, 188, 156},  // Turquoise
        {52, 73, 94},    // Wet Asphalt
        {46, 204, 64},   // Green Sea
        {142, 68, 173},  // Wisteria
        {230, 126, 34},  // Carrot
        {44, 62, 80},    // Midnight Blue
        {39, 174, 96},   // Nephritis
        {192, 57, 43},   // Pomegranate
        {41, 128, 185},  // Belize Hole
        {243, 156, 18},  // Orange
        {22, 160, 133},  // Peter River
        {189, 195, 199}, // Silver
        {211, 84, 0},    // Pumpkin
        {127, 140, 141}  // Concrete
    };

    // Clamp index to 0-19 and assign color
    index = index % 20;
    r = colors[index].r;
    g = colors[index].g;
    b = colors[index].b;
}

void SatelliteImageProcessor::saveToHDF5(H5::H5File &file) const
{
    // Define data spaces (2D: width x height now)
    hsize_t dims[2] = {static_cast<hsize_t>(height), static_cast<hsize_t>(width)};
    H5::DataSpace dataspace(2, dims);

    // Set chunking and compression properties
    H5::DSetCreatPropList props;
    hsize_t chunk_dims[2] = {std::min<hsize_t>(height, 64), std::min<hsize_t>(width, 64)};
    props.setChunk(2, chunk_dims);
    props.setDeflate(6);

    // Save path_indices
    H5::DataSet dataset_indices = file.createDataSet("path_indices", H5::PredType::NATIVE_INT,
                                                     dataspace, props);
    dataset_indices.write(path_indices.data(), H5::PredType::NATIVE_INT);

    // Save normals (3D: width x height x 2)
    hsize_t dims_normals[3] = {static_cast<hsize_t>(height), static_cast<hsize_t>(width), 2};
    H5::DataSpace dataspace_normals(3, dims_normals);
    hsize_t chunk_dims_normals[3] = {std::min<hsize_t>(height, 64), std::min<hsize_t>(width, 64), 2};
    props.setChunk(3, chunk_dims_normals);
    H5::DataSet dataset_normals = file.createDataSet("normals", H5::PredType::NATIVE_FLOAT,
                                                     dataspace_normals, props);
    dataset_normals.write(normals.data(), H5::PredType::NATIVE_FLOAT);


    // save the number of paths
    int nPaths = bezierPaths.size();
    H5::DataSpace att_dspace(H5S_SCALAR);
    dataset_indices.createAttribute("nPaths", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &nPaths);
    dataset_indices.createAttribute("width", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &width);
    dataset_indices.createAttribute("height", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &height);

    dataset_indices.createAttribute("ModeledRegionOffsetX", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &ModeledRegionOffsetX);
    dataset_indices.createAttribute("ModeledRegionOffsetY", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &ModeledRegionOffsetY);
    dataset_indices.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &GridXTotal);
    dataset_indices.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &GridYTotal);
}


void SatelliteImageProcessor::DetermineSimulationSubregion()
{
    // (2) determine the extent and offset of the highlighted (modeled) region
    int xmin = width;
    int xmax = -1;
    int ymin = height;
    int ymax = -1;

    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            if(path_indices[i + j*width] != -1)  // is modeled area
            {
                xmin = std::min(xmin, i);
                xmax = std::max(xmax, i);
                ymin = std::min(ymin, j);
                ymax = std::max(ymax, j);
            }
        }

    xmin = std::max(0, xmin-2);
    xmax = std::min(width-1, xmax+2);
    ymin = std::max(0, ymin-2);
    ymax = std::min(height-1, ymax+2);

    ModeledRegionOffsetX = xmin;
    ModeledRegionOffsetY = ymin;
    GridXTotal = xmax-xmin+1;
    GridYTotal = ymax-ymin+1;
    if(GridXTotal <= 0 || GridYTotal <= 0 )
    {
        std::cout << "modeled area not found" << std::endl;
        throw std::runtime_error("modeled area not found");
    }
}
