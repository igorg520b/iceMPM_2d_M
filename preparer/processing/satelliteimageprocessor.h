#ifndef SATELLITEIMAGEPROCESSOR_H
#define SATELLITEIMAGEPROCESSOR_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <H5Cpp.h>

#include "bezierpath.h"


class SatelliteImageProcessor
{
public:
    SatelliteImageProcessor();

    void LoadSVG(std::string fileName, int rasterWidth, int rasterHeight,
                 std::string MainPathID, std::string ProjectDirectory);

    int width = 0;
    int height = 0;

    std::vector<Eigen::Vector2f> normals;
    std::vector<int> path_indices;
    std::vector<BezierPath> bezierPaths;

    static void getColorFromIndex(int index, unsigned char& r, unsigned char& g, unsigned char& b);

    void saveToHDF5(H5::H5File &file) const;
private:
    void extractBezierCurvesFromSVG(const std::string& filename);
    void renderNormalsToPNG();

    void DetermineSimulationSubregion();

    // rectangular sub-region to which the simulation is confined
    int ModeledRegionOffsetX, ModeledRegionOffsetY;
    int GridXTotal, GridYTotal;

};

#endif // SATELLITEIMAGEPROCESSOR_H
