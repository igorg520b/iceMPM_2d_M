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
    std::vector<uint8_t> landmask_buffer;       // land (0), modeled area (255), cropped region only
    std::vector<uint8_t> original_image_colors_rgb;     // 3-component original image for background coloring (full image)
    std::vector<double> tmp_halo_buffer;        // temporary buffer for GPU halo communication
    std::array<double, 2*SimParams::MAX_REGIONS> grid_forces_summary_per_region;

    std::vector<uint8_t> rgb;   // for saving/visualization frame (RGB 3 bytes)
    std::vector<uint8_t> frame_rgba; // NEW: for loading/visualization (RGBA 4 bytes)

    // Memory tracking: [0]=grid bytes, [1]=points bytes
    size_t allocated_bytes[2] = {0, 0};

    void AllocateGridArrays(bool allocate_dense_grid = true);
    void AllocatePointArrays();

    void LoadGridDataFromFile(const std::string& gridFilePath);

    void FillModelledAreaWithBlueColor();

    void ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5);
    void SaveSnapshot(int SimulationStep, double SimulationTime, bool compress, const std::string& output_directory = "");
    void SaveFrame_Old(int SimulationStep, double SimulationTime);
    void SaveFrame(int SimulationStep, double SimulationTime); // New split-saving implementation

    // Reusable buffers for saving (allocated once to avoid reallocation)
    std::vector<float> save_buffer_float; 
    std::vector<uint8_t> save_buffer_uint8; 

    // Post-processor support: Load frame data from saved simulation output
    void LoadFrameData(int frameIndex, const std::string& framesDirectory);

private:
    ColorMap colormap;

    constexpr static std::string_view pts_cache_path = "_data/poisson_cache";

    // Helper method for reading grid datasets from HDF5 files
    static void readGridDataset(const H5::H5File& file, const std::string& dataset_name,
                                std::vector<double>& dest_buffer, size_t offset);


    void PrepareRGB_Buffer(); // invoked from SaveFrame
    void SaveForces(const int frame);
};

#endif
