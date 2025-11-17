#ifndef WINDINTERPOLATOR_H
#define WINDINTERPOLATOR_H

#include <string_view>
#include <string>
#include <utility>
#include <vector>
#include <H5Cpp.h>
#include <Eigen/Core>
#include <memory>

#include "parameters_sim.h"


class WindAndCurrentInterpolator
{
public:
    explicit WindAndCurrentInterpolator(SimParams& params);
    ~WindAndCurrentInterpolator() = default;

    SimParams &prms;

    // Set HDF5 file path (must be called before SetTime)
    void SetHDF5Path(const std::string& filePath);

    // Set current time; return whether frame indices changed (data needs GPU upload)
    bool SetTime(double t);

    // Get interpolated velocity at grid cell (i,j) and time t
    // Assumes SetTime(t) has been called beforehand to set up frame buffers
    // Returns (vx, vy) pair
    std::pair<double, double> GetInterpolatedValue(int i, int j, double t) const;

    // GPU-accessible buffers (only 2 frames in RAM at a time)
    std::vector<double> vx_frame_buffer[2];  // frame data for GPU upload
    std::vector<double> vy_frame_buffer[2];

    // Interpolation parameter for temporal interpolation between frame buffers
    // Range: [0.0, 1.0] where 0.0 = at first frame, 1.0 = at second frame
    // Computed during SetTime() to indicate position between current_first_idx and current_second_idx
    double current_alpha = 0.0;

private:
    std::string hdf5_path;              // path to flow field HDF5 file
    std::unique_ptr<H5::H5File> file;   // HDF5 file handle (opened lazily)

    // Metadata from HDF5
    double time_interval = 0.0;         // time between frames
    int num_frames = 0;                 // total number of frames
    int loop_mode = 0;                  // 0 = periodic, 1 = hold last frame
    int gx = 0, gy = 0;                 // grid dimensions

    // Current state
    int current_first_idx = -1;         // index of first cached frame
    int current_second_idx = -1;        // index of second cached frame

    // Helper methods
    void LoadHDF5Metadata();
    void LoadFrame(int frameIdx, int bufferSlot);
};


#endif // WINDINTERPOLATOR_H
