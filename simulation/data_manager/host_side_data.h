// host_side_data.h

#ifndef HOST_SIDE_DATA_H
#define HOST_SIDE_DATA_H

#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <filesystem>
#include <memory>

#include <Eigen/Core>

// Forward decl
namespace H5 { class H5File; }

#include "gui/colormap.h"
#include "parameters_sim.h"
#include "host_side_soa.h"
#include "windandcurrentinterpolator.h"


class HostSideData
{
public:
    HostSideData();
    ~HostSideData() = default;

    SimParams prms;
    HostSideSOA hssoa;                        // host-side points
    std::vector<double> host_grid_buffer;     // host-side grid
    WindAndCurrentInterpolator waci;          // water current and wind interpolator

    std::string SimulationTitle;
    std::string data_directory;  // directory where grid.h5, grid_flow.h5, and initial snapshot are located
    std::string output_directory;  // directory where frames and snapshots will be saved

    // host-side simulation data
    std::vector<uint8_t> point_partitions;      // stores which point belongs to which partition for visualization
    std::vector<uint8_t> landmask_buffer;       // land (0), modeled area (255), cropped region only
    std::vector<uint8_t> original_image_colors_rgb;     // 3-component original image for background coloring (full image)
    std::vector<double> tmp_halo_buffer;        // temporary buffer for GPU halo communication
    std::array<double, 2*SimParams::MAX_REGIONS> grid_forces_summary_per_region;

    std::vector<uint8_t> rgb;   // for saving/visualization frame

    // Memory tracking: [0]=grid bytes, [1]=points bytes
    size_t allocated_bytes[2] = {0, 0};

    void AllocateGridArrays(bool allocate_dense_grid = true);
    void AllocatePointArrays();

    void LoadGridDataFromFile(const std::string& gridFilePath);

    void PrepareGridAndPoints(std::string fileNameLandMask, std::string fileNameColor,
                              std::string fileNameIceMask, std::string fileNameCrushedMask,
                              std::string fileNameCrackedMask,
                              std::string projectDirectory, double dimensionHorizontal, int pointsPerCell,
                              double thicknessFrom, double thicknessTo,
                              double probCracked, double stdDevThickness,
                              std::string fileNameThicknessMask,
                              bool allocate_dense_grid);

    void ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5);
    void SaveSnapshot(int SimulationStep, double SimulationTime, bool compress, const std::string& output_directory = "");
    void SaveFrame(int SimulationStep, double SimulationTime);

    // Post-processor support: Load frame data from saved simulation output
    void LoadFrameData(const std::string& framePath);

    // Post-processor support: Load parameters and grid metadata from configuration
    void LoadParametersFromConfigFile(const std::string& parameterFile,
                                      const std::string& mapFile,
                                      const std::string& pngImageFile);

private:
    ColorMap colormap;

    constexpr static std::string_view pts_cache_path = "_data/poisson_cache";

    // Helper method for reading grid datasets from HDF5 files
    static void readGridDataset(const H5::H5File& file, const std::string& dataset_name,
                                std::vector<double>& dest_buffer, size_t offset);

    // Preparation helper methods (internal use only)
    void PrepareGrid(const std::vector<uint8_t> &landmask, std::vector<uint8_t> &color,
                     int imgWidth, int imgHeight, std::string projectDirectory, double dimensionHorizontal,
                     bool allocate_dense_grid);
    void PopulatePoints(const std::vector<uint8_t> &icemask, const std::vector<uint8_t> &crushed,
                        const std::vector<uint8_t> &cracked,
                        const std::vector<uint8_t> &original_colors, int imgWidth, int imgHeight, int pointsPerCell,
                        double thicknessFrom, double thicknessTo,
                        double probCracked, double stdDevThickness,
                        const std::vector<uint8_t> &thicknessMask);

    // Poisson point generation helpers
    static std::string prepare_cache_filename(int gx, int gy, int ppc);
    static bool attempt_to_fill_from_cache(int gx, int gy, int ppc, std::vector<std::array<float, 2>> &buffer);
    static void generate_and_save_poisson(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer);

    void FillModelledAreaWithBlueColor();
    void PrepareRGB_Buffer(); // invoked from SaveFrame
    void SaveForces(const int frame);
};

#endif
