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

    // Lazily open file and load metadata
    LoadHDF5Metadata();
}


void WindAndCurrentInterpolator::LoadHDF5Metadata()
{
    if (hdf5_path.empty()) {
        return;
    }

    file = std::make_unique<H5::H5File>(hdf5_path, H5F_ACC_RDONLY);

    // 1. Read Flow Type from the root group attribute
    H5::Group root = file->openGroup("/");
    H5::Attribute attr_type = root.openAttribute("flow_type");
    
    // Read string attribute safely
    H5::StrType str_type = attr_type.getStrType();
    size_t len = str_type.getSize();
    std::vector<char> buf(len + 1, '\0');
    attr_type.read(str_type, buf.data());
    flow_type_id = std::string(buf.data());

    // 2. Branch based on Flow Type - Simplified: only standard "water_current_vx" supported now
    // If flow_type_id == "Kelvin_wake" logic is removed. Assuming standard format.

    // Standard wave/current flow
    H5::DataSet ds_vx = file->openDataSet("water_current_vx");
    H5::DataSpace space = ds_vx.getSpace();
    hsize_t dims[3];
    space.getSimpleExtentDims(dims, NULL);
    num_frames = static_cast<int>(dims[0]);
    gx = static_cast<int>(dims[1]);
    gy = static_cast<int>(dims[2]);

    ds_vx.openAttribute("time_interval").read(H5::PredType::NATIVE_DOUBLE, &time_interval);
    ds_vx.openAttribute("loop_mode").read(H5::PredType::NATIVE_INT, &loop_mode);
}


void WindAndCurrentInterpolator::LoadFrame(int frameIdx, int bufferSlot)
{
    if (!file) {
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

    } catch (const H5::Exception& e) {
        throw std::runtime_error("Failed to load flow field frame from HDF5");
    }
}

bool WindAndCurrentInterpolator::SetTime(double t)
{
    // If no flow field is initialized, return false (no change needed)
    if (num_frames == 0) {
        return false;
    }

    // Calculate frame index from time
    double frame_idx_f;
    if (time_interval > 0.0) {
        frame_idx_f = t / time_interval;
    } else {
        // Static flow (time_interval == 0), always use frame 0
        frame_idx_f = 0.0;
    }

    // Handle frame index based on loop mode
    if (loop_mode == 0) {
        // Periodic: wrap around using modulo
        frame_idx_f = std::fmod(frame_idx_f, static_cast<double>(num_frames));
        if (frame_idx_f < 0.0) frame_idx_f += num_frames;
    } else {
        // Non-periodic (hold last frame): clamp to valid range
        if (frame_idx_f > num_frames - 1) {
            frame_idx_f = num_frames - 1;
        }
        if (frame_idx_f < 0.0) frame_idx_f = 0.0;
    }

    int first_idx = static_cast<int>(std::floor(frame_idx_f));
    int second_idx = first_idx;  // Will be set properly below
    double alpha = frame_idx_f - first_idx;

    // Determine second frame index based on loop mode
    if (loop_mode == 0) {
        // Periodic: wrap to next frame
        second_idx = (first_idx + 1) % num_frames;
    } else {
        // Non-periodic: clamp to last frame
        second_idx = std::min(first_idx + 1, num_frames - 1);
    }

    // Validate frame index combinations
    bool valid_combination = false;
    if (first_idx == second_idx && first_idx == num_frames - 1) {
        // Case 1: Both at last frame (happens at end of non-periodic sequence)
        valid_combination = true;
    } else if (first_idx == second_idx - 1) {
        // Case 2: Sequential frames (normal case)
        valid_combination = true;
    } else if (loop_mode == 0 && first_idx == num_frames - 1 && second_idx == 0) {
        // Case 3: Periodic wrap-around at boundary
        valid_combination = true;
    }

    if (!valid_combination) {
        throw std::runtime_error(
            fmt::format("Invalid frame index combination: first_idx={}, second_idx={}, num_frames={}, loop_mode={}",
                        first_idx, second_idx, num_frames, loop_mode));
    }

    // Check if frames changed
    bool frames_changed = (first_idx != current_first_idx || second_idx != current_second_idx);

    if (frames_changed) {
        LoadFrame(first_idx, 0);
        LoadFrame(second_idx, 1);
        current_first_idx = first_idx;
        current_second_idx = second_idx;
    }

    // Always update interpolation parameter (used by GetInterpolatedValue)
    current_alpha = alpha;

    // Return true if frame indices changed (GPU upload needed), false otherwise
    return frames_changed;
}


std::pair<double, double> WindAndCurrentInterpolator::GetInterpolatedValue(int i, int j) const
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

    // General case: linear temporal interpolation using current_alpha
    double vx_first = vx_frame_buffer[0][idx];
    double vx_second = vx_frame_buffer[1][idx];
    double vy_first = vy_frame_buffer[0][idx];
    double vy_second = vy_frame_buffer[1][idx];

    double vx = (1.0 - current_alpha) * vx_first + current_alpha * vx_second;
    double vy = (1.0 - current_alpha) * vy_first + current_alpha * vy_second;

    return {vx, vy};
}


