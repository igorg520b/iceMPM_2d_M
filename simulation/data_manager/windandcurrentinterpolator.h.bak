#ifndef WINDINTERPOLATOR_H
#define WINDINTERPOLATOR_H

#include <string_view>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <memory>

#include "parameters_sim.h"

// Forward declaration to avoid including heavy/conflicting HDF5 headers in .h
namespace H5 { class H5File; }


class WindAndCurrentInterpolator
{
public:
    explicit WindAndCurrentInterpolator(SimParams& params);
    ~WindAndCurrentInterpolator();

    SimParams &prms;

    // Set HDF5 file path (must be called before SetTime)
    void SetHDF5Path(const std::string& filePath);
    
    // Set ERA5 file path (must be called before SetTime)
    void SetEra5Path(const std::string& filePath);

    // Set current time; return {ocean_changed, wind_changed}
    std::pair<bool, bool> SetTime(double t);

    // Get interpolated velocity at grid cell (i,j)
    // Assumes SetTime(t) has been called beforehand to set up frame buffers
    // Returns (vx, vy) pair using current_alpha for interpolation
    std::pair<double, double> GetOceanValue(int i, int j) const;

    // Get interpolated wind velocity at grid cell (i,j)
    // Assumes SetTime(t) has been called beforehand
    // Returns (vx, vy) pair using current_wind_alpha for interpolation
    std::pair<double, double> GetWindValue(int i, int j) const;

    // Get Latitude and Longitude at grid cell (i,j)
    // Returns (lat, lon) pair in degrees
    std::pair<double, double> GetLatLon(int i, int j) const;



    // GPU-accessible buffers (only 2 frames in RAM at a time)
    std::vector<float> vx_frame_buffer[2];  // frame data for GPU upload
    std::vector<float> vy_frame_buffer[2];
    
    // Wind buffers
    std::vector<float> wind_vx_frame_buffer[2];
    std::vector<float> wind_vy_frame_buffer[2];


    // Interpolation parameter for temporal interpolation between frame buffers
    // Range: [0.0, 1.0] where 0.0 = at first frame, 1.0 = at second frame
    // Computed during SetTime() to indicate position between current_first_idx and current_second_idx
    double current_alpha = 0.0;
    double current_wind_alpha = 0.0;

private:
    std::string hdf5_path;              // path to flow field HDF5 file
    std::unique_ptr<H5::H5File> file_flow;   // HDF5 file handle (opened lazily)
    
    std::string era5_path;
    std::unique_ptr<H5::H5File> file_wind;

    // Metadata from HDF5 (Flow)
    double time_interval = 0.0;         // time between frames
    int num_frames = 0;                 // total number of frames
    int loop_mode = 0;                  // 0 = periodic, 1 = hold last frame
    int gx = 0, gy = 0;                 // grid dimensions
    
    // Metadata from ERA5
    std::vector<double> era5_lats;      // 1D latitude array (descending)
    std::vector<double> era5_lons;      // 1D longitude array (ascending)
    std::vector<long long> era5_times;  // Linux timestamps
    long long era5_start_time = 0;
    int era5_num_frames = 0;

    // Current state (Flow)
    int current_first_idx = -1;         // index of first cached frame
    int current_second_idx = -1;        // index of second cached frame
    
    // Current state (Wind)
    int current_wind_first_idx = -1;
    int current_wind_second_idx = -1;

    // Flow descriptor (read from HDF5 "/" group)
    std::string flow_type_id = "";

    // Helper methods
    void LoadHDF5Metadata();
    void LoadFrame(int frameIdx, int bufferSlot);

    void LoadEra5Metadata();
    void LoadWindFrame(int frameIdx, int bufferSlot);
    
    // Projection Helpers
    struct RotMat { double ex, ey, nx, ny; };
    struct LatLon { double lat_deg, lon_deg; bool valid; };
    
    LatLon ProjectPixel(int x, int y) const;
    RotMat ComputeRotation(double lat_rad, double lon_rad) const;
};

#endif // WINDINTERPOLATOR_H
