// snapshotmanager.h

#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <filesystem>
#include <thread>
#include <future>
#include <chrono>

#include <H5Cpp.h>
#include <Eigen/Core>

#include "gui/colormap.h"
#include "parameters_sim.h"

namespace icy {class SnapshotManager; class Model;}

class icy::SnapshotManager
{
public:
    SnapshotManager();
    SnapshotManager(const icy::SnapshotManager &other) {};
    ~SnapshotManager();

    icy::Model *model;
    std::string SimulationTitle;
    int SimulationStep;    // only used when reading frame from file
    double SimulationTime; // only used when reading frame from file
    int FrameNumber = -1;       // used when loading frame
    std::atomic<bool> data_ready_flag_; // for postprocessor

    void PrepareGrid(std::string fileNamePNG, std::string fileNameModelledAreaHDF5);
    void PopulatePoints(std::string fileNameModelledAreaHDF5, bool onlyGenerateCache);
    void ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5);
    void SplitIntoPartitionsAndTransferToDevice();
    void SaveSnapshot(int SimulationStep, double SimulationTime);

    void SaveFrame(int SimulationStep, double SimulationTime);

//    void LoadWindData(std::string fileName);    // netCDF4 data

    std::vector<uint8_t> rgb;   // for saving/visualization

private:
    ColorMap colormap;
    std::vector<uint8_t> count;   // used for counting points per cell and image generation

    void CalculateWeightCoeffs(const PointVector2r &pos, PointArray2r ww[3]);
    constexpr static std::string_view pts_cache_path = "_data/point_cache";

    static std::string prepare_file_name(int gx, int gy);
    static bool attempt_to_fill_from_cache(int gx, int gy, std::vector<std::array<float, 2>> &buffer);
    static void generate_and_save(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer);
    static void generate_points(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer);

    void FillModelledAreaWithBlueColor();
    void PrepareRGB_Buffer(); // invoked from SaveFrame
    void SaveForces(const int frame);
};

#endif // SNAPSHOTWRITER_H
