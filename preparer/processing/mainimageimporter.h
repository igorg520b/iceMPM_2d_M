#ifndef MAINIMAGEIMPORTER_H
#define MAINIMAGEIMPORTER_H

#include <string>
#include <vector>
#include <spdlog/spdlog.h>

#include <vtkNew.h>
#include <vtkUniformGrid.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkImageData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkImageMapper.h>
#include <vtkActor2D.h>
#include <vtkPointData.h>
#include <vtkPlaneSource.h>
#include <vtkTexture.h>
#include <vtkPolyDataMapper.h>

#include "satelliteimageprocessor.h"
#include "flowdataprocessor.h"
#include "parameterparser.h"


class MainImageImporter
{
public:
    SatelliteImageProcessor sip;
    FlowDataProcessor fdp;

    MainImageImporter();
    void LoadImage(std::string fileNamePNG, int expectedWidth, int expectedHeigth);
    void ProcessBridgePiers(const Eigen::Vector3i& pierColor);

    void IdentifyIceThickness(const ParameterParser &params);

    void Render(bool renderBC, bool renderCurrents);
    void RenderV_n();
    void RenderV_x();
    void RenderV_y();
    void RenderIceThickness();
    void RenderOriginal();

    void SaveAsHDF5(std::string ProjectDirectory);

    vtkNew<vtkActor> actor;

private:
    int width, height;
    std::vector<unsigned char> pngData;         // background png (dont' change it)
    std::vector<unsigned char> renderedImage;  // what we see on the screen


    // for pier marking
    void findConnectedRegions(const Eigen::Vector3i& targetColor,
                              std::vector<int>& labels,
                              int& regionCount) const;

    void floodFill(int startX, int startY, int label,
                   const Eigen::Vector3i& targetColor,
                   std::vector<int>& labels,
                   std::vector<bool>& visited) const;


    // VTK
    vtkNew<vtkImageData> imageData;
    vtkNew<vtkUnsignedCharArray> scalars;

    vtkNew<vtkPlaneSource> plane;
    vtkNew<vtkTexture> texture;

    vtkNew<vtkPolyDataMapper> mapper;


    // --- Helper Struct for Projection Results ---
    // Needed because classification requires distance, not just relative position
    static constexpr float crushed_thickness = 0.f;
    static constexpr float solid_thickness = 1.f; // Max thickness for intact ice

    struct ProjectionResult {
        float position = 0.0f; // Normalized position along the entire curve (0-1)
        float distance = std::numeric_limits<float>::max(); // Min distance to the curve
    };

    static void getColorFromValue(float val, unsigned char& r, unsigned char& g, unsigned char& b);
    static ProjectionResult projectPointOntoCurve(const Eigen::Vector3f& point, const std::vector<Eigen::Vector3f>& curvePoints);
    std::pair<uint8_t, float> categorizeColor(const Eigen::Vector3f& rgb) const;

    void lerpColor(float t,
                   unsigned char r1, unsigned char g1, unsigned char b1, // Color at t=0
                   unsigned char r2, unsigned char g2, unsigned char b2, // Color at t=1
                   unsigned char& out_r, unsigned char& out_g, unsigned char& out_b);

    // ice thickness analysis
    // ice variability array
    std::vector<float> iceThickness;    // quantitive measure of "thickness"
    std::vector<uint8_t> iceStatus;     // 0 for open water, 1 for crushed, 2 for "intact"


    // --- Thresholds for classification ---
    static constexpr float openWaterThreshold = 0.05f; // Max distance to water curve
    static constexpr float intactIceThreshold = 0.1f;  // Max distance to solid curve for intact ice

    // additional function for classifier
    std::vector<Eigen::Vector3f> colordata_OpenWater, colordata_Solid, colordata_Crushed;
    float findClosestDistanceInCloud(const Eigen::Vector3f& point, const std::vector<Eigen::Vector3f>& cloud) const;
    void precomputeThicknessRange();
    static float toGrayscale(const Eigen::Vector3f& rgb) {return 0.299f * rgb.x() + 0.587f * rgb.y() + 0.114f * rgb.z();}

    float minSolidGray = 0.0f;
    float maxSolidGray = 1.0f;
};

#endif // MAINIMAGEIMPORTER_H
