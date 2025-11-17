#ifndef FLUENTFLOWIMPORTER_H
#define FLUENTFLOWIMPORTER_H

#include <string>
#include <vector>
#include <Eigen/Core>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

// Forward declarations to avoid including heavy VTK headers
class vtkFLUENTCFFCustomReader;
class vtkTransform;
class vtkTransformFilter;
class vtkImageData;
class vtkProbeFilter;

class FluentFlowImporter
{
public:
    FluentFlowImporter();
    ~FluentFlowImporter();

    // Main workflow: orchestrate SVG + FLUENT loading and rasterization
    void Import(const std::string& configDirectory,
                const std::string& casFile,
                const std::string& datFile,
                const std::string& svgFile,
                const std::string& rectanglePathID,
                const std::string& fluentPathID,
                int imageWidth,
                int imageHeight,
                double velocityMultiplier = 1.0);

    // Output data: actual rasterized dimensions and velocity fields
    int image_width = 0;
    int image_height = 0;
    std::vector<double> vx_data;
    std::vector<double> vy_data;

    // Visualization: transformed FLUENT grid for debugging
    vtkSmartPointer<vtkUnstructuredGrid> transformed_grid;

private:
    // Step 1: Extract SVG geometry (bounding boxes only)
    void LoadSVGGeometry(const std::string& svgFile,
                         const std::string& rectanglePathID,
                         const std::string& fluentPathID);

    // Step 2: Load FLUENT grid from CAS/DAT files
    void LoadFluentGrid(const std::string& casFile,
                        const std::string& datFile);

    // Step 3: Transform FLUENT grid to image coordinate space
    // (follows FlowDataProcessor::ApplyTransform pattern)
    void TransformFluentGrid();

    // Step 4: Rasterize transformed grid to image pixels
    // (follows FlowDataProcessor::Rasterize pattern)
    void RasterizeToImageGrid(int imageWidth, int imageHeight);

    // SVG-derived extents (in SVG coordinate space initially, then transformed to image space)
    Eigen::Vector2f extents_rectangle[2];  // Rectangle bounds (image in SVG coords)
    Eigen::Vector2f extents_fluent[2];    // FLUENT grid bounds (in image coordinate space after transform)

    // Velocity multiplier (applied during rasterization)
    double velocity_multiplier = 1.0;

    // VTK objects for grid transformation and rasterization
    vtkFLUENTCFFCustomReader* reader = nullptr;
    vtkTransform* transform = nullptr;
    vtkTransformFilter* transformFilter = nullptr;
    vtkImageData* imageData = nullptr;
    vtkProbeFilter* probeFilter = nullptr;
};

#endif // FLUENTFLOWIMPORTER_H
