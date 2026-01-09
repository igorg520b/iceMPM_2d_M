#include "fluentflowimporter.h"

#include "simulation/data_manager/fluent_importer/vtkFLUENTCFFCustomReader.h"
#include "simulation/parameters_sim.h"

#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkImageData.h>
#include <vtkProbeFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkCellDataToPointData.h>
#include <vtkMultiBlockDataSet.h>

#include <Eigen/Core>

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <iostream>

FluentFlowImporter::FluentFlowImporter() {
    // Initialize VTK objects
    reader = vtkFLUENTCFFCustomReader::New();
    transform = vtkTransform::New();
    transformFilter = vtkTransformFilter::New();
    imageData = vtkImageData::New();
    probeFilter = vtkProbeFilter::New();
    spdlog::info("FluentFlowImporter constructed");
}

FluentFlowImporter::~FluentFlowImporter() {
    // Cleanup VTK objects (reference counted)
    if(reader) reader->Delete();
    if(transform) transform->Delete();
    if(transformFilter) transformFilter->Delete();
    if(imageData) imageData->Delete();
    if(probeFilter) probeFilter->Delete();
    spdlog::info("FluentFlowImporter destructed");
}

void FluentFlowImporter::Import(const std::string& configDirectory,
                                const std::string& casFile,
                                const std::string& datFile,
                                const std::string& svgFile,
                                const std::string& rectanglePathID,
                                const std::string& fluentPathID,
                                int imageWidth,
                                int imageHeight,
                                double velocityMultiplier) {
    spdlog::info("FluentFlowImporter::Import starting");

    image_width = imageWidth;
    image_height = imageHeight;
    velocity_multiplier = velocityMultiplier;

    // Step 1: Extract SVG geometry (bounding boxes in SVG space)
    std::string fullSvgPath;
    if (std::filesystem::path(svgFile).is_absolute()) {
        fullSvgPath = svgFile;
    } else {
        fullSvgPath = configDirectory + "/" + svgFile;
    }
    LoadSVGGeometry(fullSvgPath, rectanglePathID, fluentPathID);

    // Step 2: Load FLUENT CAS/DAT files into VTK grid
    std::string fullCasPath;
    if (std::filesystem::path(casFile).is_absolute()) {
        fullCasPath = casFile;
    } else {
        fullCasPath = configDirectory + "/" + casFile;
    }

    std::string fullDatPath;
    if (std::filesystem::path(datFile).is_absolute()) {
        fullDatPath = datFile;
    } else {
        fullDatPath = configDirectory + "/" + datFile;
    }
    LoadFluentGrid(fullCasPath, fullDatPath);

    // Step 3: Transform FLUENT grid using extents_fluent bounds
    // (Maps FLUENT grid space → image pixel space with Y-flip)
    TransformFluentGrid();

    // Step 4: Rasterize transformed grid to regular image grid
    // (Probes VTK grid at each image pixel location)
    RasterizeToImageGrid(imageWidth, imageHeight);

    spdlog::info("FluentFlowImporter::Import completed");
}

void FluentFlowImporter::LoadSVGGeometry(const std::string& svgFile,
                                        const std::string& rectanglePathID,
                                        const std::string& fluentPathID) {
    spdlog::info("FluentFlowImporter::LoadSVGGeometry from: {}", svgFile);

    // Algorithm from SatelliteImageProcessor::extractBezierCurvesFromSVG
    // We only need bounding boxes, not full path extraction

    NSVGimage* svgImage = nsvgParseFromFile(svgFile.c_str(), "px", 96);
    if (!svgImage) {
        throw std::runtime_error("Failed to parse SVG file: " + svgFile);
    }

    bool found_rectangle = false;
    bool found_fluent = false;

    Eigen::Vector2f rect_min_raw, rect_max_raw;
    Eigen::Vector2f fluent_min_raw, fluent_max_raw;

    // Iterate through SVG shapes to find rectangle and fluent paths
    for (NSVGshape* shape = svgImage->shapes; shape != nullptr; shape = shape->next) {
        if (!shape->id) continue;

        std::string shapeId(shape->id);

        if (shapeId == rectanglePathID) {
            // Rectangle defines full image bounds in SVG space
            // bounds[0]=xmin, bounds[1]=ymin, bounds[2]=xmax, bounds[3]=ymax
            rect_min_raw = Eigen::Vector2f(shape->bounds[0], shape->bounds[1]);
            rect_max_raw = Eigen::Vector2f(shape->bounds[2], shape->bounds[3]);
            spdlog::info("Rectangle bounds (SVG, raw): ({}, {}) to ({}, {})",
                        rect_min_raw.x(), rect_min_raw.y(),
                        rect_max_raw.x(), rect_max_raw.y());
            found_rectangle = true;
        }
        else if (shapeId == fluentPathID) {
            // FLUENT path defines where FLUENT grid sits in SVG space
            fluent_min_raw = Eigen::Vector2f(shape->bounds[0], shape->bounds[1]);
            fluent_max_raw = Eigen::Vector2f(shape->bounds[2], shape->bounds[3]);
            spdlog::info("FLUENT path bounds (SVG, raw): ({}, {}) to ({}, {})",
                        fluent_min_raw.x(), fluent_min_raw.y(),
                        fluent_max_raw.x(), fluent_max_raw.y());
            found_fluent = true;
        }
    }

    nsvgDelete(svgImage);

    if (!found_rectangle) {
        throw std::runtime_error("Rectangle path ID '" + rectanglePathID + "' not found in SVG");
    }
    if (!found_fluent) {
        throw std::runtime_error("FLUENT path ID '" + fluentPathID + "' not found in SVG");
    }

    // Y-flip all SVG coordinates immediately
    // SVG has Y pointing down, but we want bottom-left origin like VTK
    // When we negate Y, the min and max swap:
    // old_min → new_max, old_max → new_min
    float rect_min_y_flipped = -rect_max_raw.y();
    float rect_max_y_flipped = -rect_min_raw.y();
    float fluent_min_y_flipped = -fluent_max_raw.y();
    float fluent_max_y_flipped = -fluent_min_raw.y();

    // Ensure min < max after flip
    extents_rectangle[0] = Eigen::Vector2f(rect_min_raw.x(), std::min(rect_min_y_flipped, rect_max_y_flipped));
    extents_rectangle[1] = Eigen::Vector2f(rect_max_raw.x(), std::max(rect_min_y_flipped, rect_max_y_flipped));
    extents_fluent[0] = Eigen::Vector2f(fluent_min_raw.x(), std::min(fluent_min_y_flipped, fluent_max_y_flipped));
    extents_fluent[1] = Eigen::Vector2f(fluent_max_raw.x(), std::max(fluent_min_y_flipped, fluent_max_y_flipped));

    spdlog::info("Rectangle bounds (SVG, after Y-flip): ({}, {}) to ({}, {})",
                extents_rectangle[0].x(), extents_rectangle[0].y(),
                extents_rectangle[1].x(), extents_rectangle[1].y());
    spdlog::info("FLUENT path bounds (SVG, after Y-flip): ({}, {}) to ({}, {})",
                extents_fluent[0].x(), extents_fluent[0].y(),
                extents_fluent[1].x(), extents_fluent[1].y());

    spdlog::info("FluentFlowImporter::LoadSVGGeometry completed");
}

void FluentFlowImporter::LoadFluentGrid(const std::string& casFile,
                                        const std::string& datFile) {
    spdlog::info("FluentFlowImporter::LoadFluentGrid");
    spdlog::info("  CAS file: {}", casFile);
    spdlog::info("  DAT file: {}", datFile);

    // Algorithm from FlowDataProcessor::LoadFluentResult
    reader->SetDataFileName(datFile.c_str());    // Velocity data
    reader->SetFileName(casFile.c_str());        // Mesh data
    reader->Update();

    spdlog::info("FLUENT grid loaded successfully");
}

void FluentFlowImporter::TransformFluentGrid() {
    spdlog::info("FluentFlowImporter::TransformFluentGrid");

    // Algorithm:
    // 1. Load SVG bounds (already Y-flipped in LoadSVGGeometry)
    // 2. Calculate transformation that maps rectangle to [0,0] → [width, height]
    // 3. Apply same transformation to FLUENT path bounds
    // 4. Transform FLUENT grid to match those bounds

    // Get the unstructured grid from reader
    vtkUnstructuredGrid* grid = vtkUnstructuredGrid::SafeDownCast(
        reader->GetOutput()->GetBlock(0));
    if (!grid) {
        throw std::runtime_error("Failed to get FLUENT grid from reader");
    }

    // 1. Get original grid bounds (source space - FLUENT native coordinates)
    double original_bounds[6];
    grid->GetBounds(original_bounds);

    Eigen::Vector2f source_min(original_bounds[0], original_bounds[2]);
    Eigen::Vector2f source_max(original_bounds[1], original_bounds[3]);
    Eigen::Vector2f source_dims = source_max - source_min;

    if (source_dims.x() <= 0 || source_dims.y() <= 0) {
        throw std::runtime_error("FLUENT grid has invalid dimensions");
    }

    spdlog::info("--- FLUENT Grid Original Bounds (source space) ---");
    spdlog::info("  X range: {} to {}", source_min.x(), source_max.x());
    spdlog::info("  Y range: {} to {}", source_min.y(), source_max.y());
    spdlog::info("  Dimensions: {} x {}", source_dims.x(), source_dims.y());

    // 2. Get SVG bounds (already Y-flipped)
    Eigen::Vector2f svg_rect_min = extents_rectangle[0];
    Eigen::Vector2f svg_rect_max = extents_rectangle[1];
    Eigen::Vector2f svg_rect_dims = svg_rect_max - svg_rect_min;

    if (svg_rect_dims.x() <= 0 || svg_rect_dims.y() <= 0) {
        throw std::runtime_error("Invalid SVG rectangle dimensions");
    }

    spdlog::info("--- SVG Rectangle (full image reference, after Y-flip) ---");
    spdlog::info("  X range: {} to {}", svg_rect_min.x(), svg_rect_max.x());
    spdlog::info("  Y range: {} to {}", svg_rect_min.y(), svg_rect_max.y());
    spdlog::info("  Dimensions: {} x {}", svg_rect_dims.x(), svg_rect_dims.y());

    // 3. Calculate SVG to image space transformation
    // The rectangle defines the full image, so we map it to [0, 0] → [image_width, image_height]
    // Use the SAME scale for both X and Y to preserve aspect ratio
    // Scale is calculated from the X dimension to match the image width
    float svg_scale = (float)image_width / svg_rect_dims.x();

    // The Y scale should be the same to preserve aspect ratio
    float svg_scale_y = (float)image_height / svg_rect_dims.y();

    Eigen::Vector2f svg_offset = svg_rect_min;

    spdlog::info("--- SVG to Image Space Transformation ---");
    spdlog::info("  Image dimensions: {} x {}", image_width, image_height);
    spdlog::info("  SVG scale factor (X): {}", svg_scale);
    spdlog::info("  SVG scale factor (Y): {}", svg_scale_y);
    spdlog::info("  SVG offset (min corner): ({}, {})", svg_offset.x(), svg_offset.y());

    // Verify rectangle maps correctly to [0, width] x [0, height]
    Eigen::Vector2f rect_check_min(
        (svg_rect_min.x() - svg_offset.x()) * svg_scale,
        (svg_rect_min.y() - svg_offset.y()) * svg_scale_y
    );
    Eigen::Vector2f rect_check_max(
        (svg_rect_max.x() - svg_offset.x()) * svg_scale,
        (svg_rect_max.y() - svg_offset.y()) * svg_scale_y
    );
    spdlog::info("  Rectangle check (should be [0,{}] x [0,{}]): ({}, {}) to ({}, {})",
                 image_width, image_height,
                 rect_check_min.x(), rect_check_min.y(),
                 rect_check_max.x(), rect_check_max.y());

    // 4. Transform the FLUENT path bounds using the same SVG→image transformation
    Eigen::Vector2f fluent_svg_min = extents_fluent[0];
    Eigen::Vector2f fluent_svg_max = extents_fluent[1];
    Eigen::Vector2f fluent_svg_dims = fluent_svg_max - fluent_svg_min;

    spdlog::info("--- FLUENT Path in SVG Space (after Y-flip) ---");
    spdlog::info("  X range: {} to {}", fluent_svg_min.x(), fluent_svg_max.x());
    spdlog::info("  Y range: {} to {}", fluent_svg_min.y(), fluent_svg_max.y());
    spdlog::info("  Dimensions: {} x {}", fluent_svg_dims.x(), fluent_svg_dims.y());

    // Apply SVG→image transformation: (point - offset) * scale
    // This should map FLUENT path to fall within or overlap with [0, image_width] x [0, image_height]
    Eigen::Vector2f target_min(
        (fluent_svg_min.x() - svg_offset.x()) * svg_scale,
        (fluent_svg_min.y() - svg_offset.y()) * svg_scale_y
    );
    Eigen::Vector2f target_max(
        (fluent_svg_max.x() - svg_offset.x()) * svg_scale,
        (fluent_svg_max.y() - svg_offset.y()) * svg_scale_y
    );
    Eigen::Vector2f target_dims = target_max - target_min;

    spdlog::info("--- FLUENT Path Placeholder in Image Space ---");
    spdlog::info("  X range: {} to {}", target_min.x(), target_max.x());
    spdlog::info("  Y range: {} to {}", target_min.y(), target_max.y());
    spdlog::info("  Dimensions: {} x {}", target_dims.x(), target_dims.y());

    if (target_dims.x() <= 0 || target_dims.y() <= 0) {
        throw std::runtime_error("Target extents have zero or negative dimensions");
    }

    // 5. Calculate the uniform scale from source (FLUENT native) to target (image space)
    double scale = target_dims.x() / source_dims.x();

    float source_aspect = source_dims.x() / source_dims.y();
    float target_aspect = target_dims.x() / target_dims.y();

    spdlog::info("--- Scale Calculation (FLUENT native → image space) ---");
    spdlog::info("  Source aspect ratio: {}", source_aspect);
    spdlog::info("  Target aspect ratio: {}", target_aspect);
    spdlog::info("  Calculated uniform scale: {}", scale);

    if (std::abs(source_aspect - target_aspect) > 1e-3) {
        spdlog::warn("Source grid aspect ratio ({}) differs from target extents aspect ratio ({}). "
                     "This may cause distortion.", source_aspect, target_aspect);
    }

    // 6. Set up the transformation
    // The correct order to map one box to another is:
    // 1. Translate the source to the origin: -source_min
    // 2. Scale it: * scale
    // 3. Translate it to the target's final position: + target_min
    // VTK applies these in reverse order of declaration.
    transform->Identity();
    transform->Translate(target_min.x(), target_min.y(), 0.0);
    transform->Scale(scale, scale, 1.0);
    transform->Translate(-source_min.x(), -source_min.y(), 0.0);
    transform->Update();

    // 7. Apply the transform
    transformFilter->SetTransform(transform);
    transformFilter->SetInputData(grid);
    transformFilter->Update();

    // 8. Verify the results
    vtkUnstructuredGrid* transformed = transformFilter->GetUnstructuredGridOutput();
    if (!transformed) {
        throw std::runtime_error("Transform filter failed to produce output");
    }

    // Deep copy the transformed grid for later visualization
    transformed_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    transformed_grid->ShallowCopy(transformed);

    double transformed_bounds[6];
    transformed->GetBounds(transformed_bounds);

    Eigen::Vector2f actual_min(transformed_bounds[0], transformed_bounds[2]);
    Eigen::Vector2f actual_max(transformed_bounds[1], transformed_bounds[3]);
    Eigen::Vector2f actual_dims = actual_max - actual_min;

    spdlog::info("--- Transformed Grid Bounds (actual result) ---");
    spdlog::info("  X range: {} to {}", actual_min.x(), actual_max.x());
    spdlog::info("  Y range: {} to {}", actual_min.y(), actual_max.y());
    spdlog::info("  Dimensions: {} x {}", actual_dims.x(), actual_dims.y());

    // Compare expected vs actual
    spdlog::info("--- Verification ---");
    spdlog::info("  Expected target dims: {} x {}", target_dims.x(), target_dims.y());
    spdlog::info("  Actual transformed dims: {} x {}", actual_dims.x(), actual_dims.y());

    float dim_diff_x = std::abs(target_dims.x() - actual_dims.x());
    float dim_diff_y = std::abs(target_dims.y() - actual_dims.y());

    if (dim_diff_x < 1e-3 && dim_diff_y < 1e-3) {
        spdlog::info("  ✓ Dimensions match!");
    } else {
        spdlog::warn("  ✗ Dimension mismatch: X diff={}, Y diff={}", dim_diff_x, dim_diff_y);
    }

    spdlog::info("FLUENT grid transformation complete");
}

void FluentFlowImporter::RasterizeToImageGrid(int imageWidth, int imageHeight) {
    LOGR("FluentFlowImporter::RasterizeToImageGrid with velocity_multiplier={}", velocity_multiplier);

    // Store the dimensions for later access
    image_width = imageWidth;
    image_height = imageHeight;

    // Algorithm from FlowDataProcessor::Rasterize

    // Convert cell data to point data
    vtkNew<vtkCellDataToPointData> cellToPoint;
    cellToPoint->SetInputData(transformFilter->GetUnstructuredGridOutput());
    cellToPoint->Update();

    vtkUnstructuredGrid* ug = cellToPoint->GetUnstructuredGridOutput();
    if (!ug) {
        throw std::runtime_error("Cell to point filter failed");
    }

    if (!ug->GetPointData()->HasArray("SV_V")) {
        throw std::runtime_error("FLUENT grid missing SV_V (Y-velocity) array");
    }
    if (!ug->GetPointData()->HasArray("SV_U")) {
        throw std::runtime_error("FLUENT grid missing SV_U (X-velocity) array");
    }

    // Create regular image grid (pixel locations)
    imageData->SetDimensions(imageWidth, imageHeight, 1);
    imageData->SetSpacing(1.0, 1.0, 1.0);

    // Probe VTK grid at image pixel locations
    probeFilter->SetInputData(imageData);
    probeFilter->SetSourceData(cellToPoint->GetOutput());
    probeFilter->PassPointArraysOn();
    probeFilter->Update();

    vtkImageData* probedData = vtkImageData::SafeDownCast(
        probeFilter->GetOutput());
    if (!probedData) {
        throw std::runtime_error("Probe filter failed");
    }

    // Extract velocity components (following old code pattern)
    vtkDoubleArray* dataArray = vtkDoubleArray::SafeDownCast(
        probedData->GetPointData()->GetScalars("SV_V"));

    if (!dataArray) {
        spdlog::error("Failed to extract SV_V from probedData");
        throw std::runtime_error("Failed to extract SV_V from probedData");
    }

    // Resize output arrays
    vx_data.resize(imageWidth * imageHeight);
    vy_data.resize(imageWidth * imageHeight);

    // Fill velocity field from probed data - query each element individually
    // Index mapping: pixel (i,j) → linear index i + width*j (column-major)
    int total_count = imageWidth * imageHeight;

    for (int i = 0; i < imageWidth; ++i) {
        for (int j = 0; j < imageHeight; ++j) {
            int idx = i + imageWidth * j;
            double val_v = dataArray->GetValue(idx);
            vy_data[idx] = val_v * velocity_multiplier;
        }
    }


    // Extract SV_U (X velocity)
    dataArray = vtkDoubleArray::SafeDownCast(
        probedData->GetPointData()->GetScalars("SV_U"));

    if (!dataArray) {
        spdlog::error("Failed to extract SV_U from probedData");
        throw std::runtime_error("Failed to extract SV_U from probedData");
    }

    for (int i = 0; i < imageWidth; ++i) {
        for (int j = 0; j < imageHeight; ++j) {
            int idx = i + imageWidth * j;
            double val_u = dataArray->GetValue(idx);
            vx_data[idx] = val_u * velocity_multiplier;
        }
    }

    // Compute statistics
    double sum_vx = 0.0;
    double sum_vy = 0.0;
    for (int i = 0; i < total_count; ++i) {
        sum_vx += vx_data[i];
        sum_vy += vy_data[i];
    }
    double avg_vx = sum_vx / total_count;
    double avg_vy = sum_vy / total_count;

    spdlog::info("Rasterization complete: {} x {} grid", imageWidth, imageHeight);
    spdlog::info("Average velocity: vx={}, vy={}", avg_vx, avg_vy);
}
