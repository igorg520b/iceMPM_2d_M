#include "windandcurrentinterpolator.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <filesystem>
#include <fmt/format.h>


WindAndCurrentInterpolator::WindAndCurrentInterpolator(SimParams& params) : prms(params)
{
    // Initialize with empty/zero state
    vx_frame_buffer[0].clear();
    vx_frame_buffer[1].clear();
    vy_frame_buffer[0].clear();
    vy_frame_buffer[1].clear();
}


void WindAndCurrentInterpolator::SetHDF5Path(const std::string& filePath)
{
    // Check file exists before attempting to open
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(fmt::format("Flow field file not found: {}", filePath));
    }

    hdf5_path = filePath;
    LOGR("WindAndCurrentInterpolator::SetHDF5Path: {}", hdf5_path);

    // Lazily open file and load metadata
    LoadHDF5Metadata();
}


void WindAndCurrentInterpolator::LoadHDF5Metadata()
{
    if (hdf5_path.empty()) {
        LOGR("WindAndCurrentInterpolator: HDF5 path not set, no flow field available");
        return;
    }

    try {
        file = std::make_unique<H5::H5File>(hdf5_path, H5F_ACC_RDONLY);

        // Open dataset to get dimensions
        H5::DataSet ds_vx = file->openDataSet("water_current_vx");
        H5::DataSpace space = ds_vx.getSpace();

        hsize_t dims[3];
        space.getSimpleExtentDims(dims, NULL);

        num_frames = static_cast<int>(dims[0]);
        gx = static_cast<int>(dims[1]);
        gy = static_cast<int>(dims[2]);

        // Read attributes
        H5::Attribute attr_time = ds_vx.openAttribute("time_interval");
        attr_time.read(H5::PredType::NATIVE_DOUBLE, &time_interval);

        H5::Attribute attr_loop = ds_vx.openAttribute("loop_mode");
        attr_loop.read(H5::PredType::NATIVE_INT, &loop_mode);

        LOGR("WindAndCurrentInterpolator: Loaded metadata - num_frames={}, gx={}, gy={}, time_interval={}, loop_mode={}",
             num_frames, gx, gy, time_interval, loop_mode);

    } catch (const H5::Exception& e) {
        LOGR("WindAndCurrentInterpolator: Failed to load HDF5 metadata: {}", e.getCDetailMsg());
        throw std::runtime_error("Failed to load flow field HDF5 metadata");
    }
}


void WindAndCurrentInterpolator::LoadFrame(int frameIdx, int bufferSlot)
{
    if (!file) {
        LOGR("WindAndCurrentInterpolator::LoadFrame: File not open, cannot load frame");
        return;
    }

    // Clamp frame index based on loop mode
    int actualIdx = frameIdx;
    if (loop_mode == 0) {
        // Periodic: wrap around
        actualIdx = frameIdx % num_frames;
        if (actualIdx < 0) actualIdx += num_frames;
    } else {
        // Hold last frame
        if (actualIdx >= num_frames) actualIdx = num_frames - 1;
        if (actualIdx < 0) actualIdx = 0;
    }

    int gridSize = gx * gy;
    vx_frame_buffer[bufferSlot].resize(gridSize);
    vy_frame_buffer[bufferSlot].resize(gridSize);

    try {
        H5::DataSet ds_vx = file->openDataSet("water_current_vx");
        H5::DataSet ds_vy = file->openDataSet("water_current_vy");

        // Select hyperslab for this frame
        // HDF5 stores [frames, gx, gy] in column-major order (gx is major axis)
        hsize_t offset[3] = {static_cast<hsize_t>(actualIdx), 0, 0};
        hsize_t dims[3] = {1, static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};

        H5::DataSpace space_vx = ds_vx.getSpace();
        space_vx.selectHyperslab(H5S_SELECT_SET, dims, offset);

        H5::DataSpace mem_space(3, dims);

        // Read directly into frame buffer (data is already in column-major format)
        ds_vx.read(vx_frame_buffer[bufferSlot].data(), H5::PredType::NATIVE_DOUBLE, mem_space, space_vx);
        ds_vy.read(vy_frame_buffer[bufferSlot].data(), H5::PredType::NATIVE_DOUBLE, mem_space, space_vx);

        LOGR("WindAndCurrentInterpolator: Loaded frame {} (actual idx {})", frameIdx, actualIdx);

    } catch (const H5::Exception& e) {
        LOGR("WindAndCurrentInterpolator::LoadFrame: Failed to read frame: {}", e.getCDetailMsg());
        throw std::runtime_error("Failed to load flow field frame from HDF5");
    }
}


bool WindAndCurrentInterpolator::SetTime(double t)
{
//    LOGR("WindAndCurrentInterpolator::SetTime: t={}, hdf5_path.empty()={}, num_frames={}", t, hdf5_path.empty(), num_frames);

    // Calculate frame index from time
    double frame_idx_f;
    if (time_interval > 0.0) {
        frame_idx_f = std::fmod(t / time_interval, static_cast<double>(num_frames));
    } else {
        // Static flow (time_interval == 0), always use frame 0
        frame_idx_f = 0.0;
    }

    // Handle negative modulo
    if (frame_idx_f < 0.0) frame_idx_f += num_frames;

    int first_idx = static_cast<int>(std::floor(frame_idx_f));
    int second_idx = (first_idx + 1) % num_frames;  // wraps automatically
    double alpha = frame_idx_f - first_idx;

//    LOGR("WindAndCurrentInterpolator::SetTime: frame_idx_f={}, first_idx={}, second_idx={}, alpha={}",
//         frame_idx_f, first_idx, second_idx, alpha);

    // Handle non-periodic case: if at or beyond last frame, don't interpolate
    if (loop_mode == 1 && first_idx >= num_frames - 1) {
        second_idx = first_idx;
        alpha = 0.0;
//        LOGR("WindAndCurrentInterpolator::SetTime: Non-periodic, at/beyond last frame");
    }

    // Check if frames changed
    if (first_idx != current_first_idx || second_idx != current_second_idx) {
  //      LOGR("WindAndCurrentInterpolator::SetTime: Loading frames {} and {}", first_idx, second_idx);
        LoadFrame(first_idx, 0);
        LoadFrame(second_idx, 1);
        current_first_idx = first_idx;
        current_second_idx = second_idx;
        current_alpha = alpha;
//        LOGR("WindAndCurrentInterpolator::SetTime: Loaded new frames, buffer[0].size()={}, buffer[1].size()={}",
//             vx_frame_buffer[0].size(), vx_frame_buffer[1].size());
        return true;  // Frames changed, GPU upload needed
    }

    // Update interpolation parameter even if frames didn't change
    current_alpha = alpha;
//    LOGR("WindAndCurrentInterpolator::SetTime: Frames did not change");
    return false;  // Frames didn't change
}


std::pair<double, double> WindAndCurrentInterpolator::GetInterpolatedValue(int i, int j, double t) const
{
    if (hdf5_path.empty() || num_frames == 0) {
        // No flow field, return zero velocity
        return {0.0, 0.0};
    }

    if (i < 0 || i >= gx || j < 0 || j >= gy) {
        // Out of bounds
        return {0.0, 0.0};
    }

    size_t idx = j + static_cast<size_t>(i) * gy;

    if (num_frames == 1) {
        // Special case: constant flow, no interpolation
        return {vx_frame_buffer[0][idx], vy_frame_buffer[0][idx]};
    }

    // General case: linear temporal interpolation
    double vx_first = vx_frame_buffer[0][idx];
    double vx_second = vx_frame_buffer[1][idx];
    double vy_first = vy_frame_buffer[0][idx];
    double vy_second = vy_frame_buffer[1][idx];

    double vx = (1.0 - current_alpha) * vx_first + current_alpha * vx_second;
    double vy = (1.0 - current_alpha) * vy_first + current_alpha * vy_second;

    return {vx, vy};
}
