#ifndef WINDINTERPOLATOR_H
#define WINDINTERPOLATOR_H

#include <string_view>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <memory>
#include <functional>
#include <future>

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

    // Set GLO12 NetCDF file path
    // Set GLO12 NetCDF file path
    void SetGLO12Path(const std::string& filePath);
    // Set GLO12 Tidal Currents file path
    void SetGLO12TidesPath(const std::string& filePath);

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

    // Get Data Pointer for GPU transfer logic
    // logicalFrame: 0 for the "first" frame (t), 1 for the "second" frame (t+dt)
    // component: 0 for X, 1 for Y
    const float* GetOceanDataPointer(int logicalFrame, int component) const;
    const float* GetWindDataPointer(int logicalFrame, int component) const;

    // Get Latitude and Longitude at grid cell (i,j)
    // Returns (lat, lon) pair in degrees
    std::pair<double, double> GetLatLon(int i, int j) const;



    // GPU-accessible buffers (3 frames in RAM for ring buffering)
    std::vector<float> ocean_vx_frame_buffer[3];
    std::vector<float> ocean_vy_frame_buffer[3];
    
    // Wind buffers
    std::vector<float> wind_vx_frame_buffer[3];
    std::vector<float> wind_vy_frame_buffer[3];


    // Interpolation parameter for temporal interpolation between frame buffers
    // Range: [0.0, 1.0] where 0.0 = at first frame, 1.0 = at second frame
    // Computed during SetTime() to indicate position between current_first_idx and current_second_idx
    double current_ocean_alpha = 0.0;
    double current_wind_alpha = 0.0;

private:
    std::string hdf5_path;              // path to flow field HDF5 file
    std::unique_ptr<H5::H5File> file_flow;   // HDF5 file handle (opened lazily)
    
    std::string era5_path;
    std::unique_ptr<H5::H5File> file_wind;

    std::string glo12_path;
    std::unique_ptr<H5::H5File> file_glo12;

    std::string glo12_tides_path;
    std::unique_ptr<H5::H5File> file_glo12_tides;

    // Metadata from HDF5 (Flow)
    double time_interval = 0.0;         // time between frames
    int num_frames = 0;                 // total number of frames
    int loop_mode = 0;                  // 0 = periodic, 1 = hold last frame
    
    // Metadata from ERA5
    std::vector<double> era5_lats;      // 1D latitude array (descending)
    std::vector<double> era5_lons;      // 1D longitude array (ascending)
    std::vector<long long> era5_times;  // Linux timestamps
    long long era5_start_time = 0;
    int era5_num_frames = 0;

    // Metadata from GLO12
    std::vector<double> glo12_lats;      // 1D latitude array (descending usually, but we check)
    std::vector<double> glo12_lons;      // 1D longitude array
    std::vector<long long> glo12_times;  // Linux timestamps (or hours since epoch converted)
    long long glo12_start_time = 0;
    int glo12_num_frames = 0;

    // Current state (Flow)
    int current_ocean_first_idx = -1;         // index of first cached frame (logical)
    int current_ocean_second_idx = -1;        // index of second cached frame (logical)
    
    // Slot Management (Ocean)
    static constexpr int NUM_SLOTS = 3;
    int ocean_slot_frames[NUM_SLOTS] = {-1, -1, -1}; // Which source frame is in each physical slot?
    int current_ocean_active_slots[2] = {0, 0};      // Which physical slot corresponds to logical frame 0 and 1?

    // Current state (Wind)
    int current_wind_first_idx = -1;
    int current_wind_second_idx = -1;
    
    // Slot Management (Wind)
    int wind_slot_frames[NUM_SLOTS] = {-1, -1, -1};
    int current_wind_active_slots[2] = {0, 0};

    // Flow descriptor (read from HDF5 "/" group)
    std::string flow_type_id = "";

    // Helper methods
    void LoadHDF5Metadata();
    void LoadOceanFrame(int frameIdx, int bufferSlot);

    void LoadEra5Metadata();
    void LoadWindFrame(int frameIdx, int bufferSlot);

    void LoadGLO12Metadata();
    void LoadGLO12Frame(int frameIdx, int bufferSlot);

    // Async preloading
    std::future<void> ocean_preload_future;
    std::future<void> wind_preload_future;

    // Smart Ring Buffer Update Logic

    // Smart Ring Buffer Update Logic
    // needed_f0, needed_f1: the two logical frames required by the current time
    // slot_frames: array of size NUM_SLOTS (3) tracking which source frame is in each physical slot
    // active_slots: array of size 2, outputting which physical slot index corresponds to f0 and f1
    // load_func: callback to load a specific frame into a specific physical slot
    void UpdateRingBufferSlots(int needed_f0, int needed_f1, int* slot_frames, int* active_slots, const std::function<void(int frame, int slot)>& load_func);
    
    // Projection Helpers
    struct RotMat { double ex, ey, nx, ny; };
    struct LatLon { double lat_deg, lon_deg; bool valid; };
    
    LatLon ProjectPixel(int x, int y) const;
    RotMat ComputeRotation(double lat_rad, double lon_rad) const;
};

#endif // WINDINTERPOLATOR_H
