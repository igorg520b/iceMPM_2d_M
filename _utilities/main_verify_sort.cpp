#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "host_side_soa.h"
#include "parameters_sim.h"
#include <H5Cpp.h>

namespace fs = std::filesystem;

struct PointData {
    uint64_t cell_idx_raw;
    int cell_idx;
};

void verify_sort(const std::string& snapshot_path) {
    fs::path snap_path(snapshot_path);
    if (!fs::exists(snap_path)) {
        throw std::runtime_error("Snapshot file does not exist: " + snapshot_path);
    }
    
    // Assume grid.h5 is in the parent directory of the snapshot file (or parent/parent if in snapshots dir)
    // The user prompts says: "lets' assume we have the gird file 'grid.h5' stored in the parent of the snapshot file."
    // However, typically snapshots are in output/snapshots or similar. Let's try to locate grid.h5
    
    fs::path grid_path;
    if (fs::exists(snap_path.parent_path() / "grid.h5")) {
        grid_path = snap_path.parent_path() / "grid.h5";
    } else if (fs::exists(snap_path.parent_path().parent_path() / "grid.h5")) {
         grid_path = snap_path.parent_path().parent_path() / "grid.h5";
    } else {
        // Fallback: look in current directory or ..
        if (fs::exists("grid.h5")) grid_path = "grid.h5";
        else if (fs::exists("../grid.h5")) grid_path = "../grid.h5";
        else throw std::runtime_error("Could not find grid.h5 in common locations relative to snapshot");
    }
    
    std::cout << "Using grid file: " << grid_path << std::endl;
    std::cout << "Using snapshot file: " << snap_path << std::endl;

    // Read Grid Dimensions
    int GridYTotal = 0;
    int GridXTotal = 0;
    try {
        H5::H5File grid_file(grid_path.string(), H5F_ACC_RDONLY);
        H5::DataSet ds_landmask = grid_file.openDataSet("landmask");
        H5::Attribute attr_gy = ds_landmask.openAttribute("GridYTotal");
        attr_gy.read(H5::PredType::NATIVE_INT, &GridYTotal);

        H5::Attribute attr_gx = ds_landmask.openAttribute("GridXTotal");
        attr_gx.read(H5::PredType::NATIVE_INT, &GridXTotal);

        grid_file.close();
        std::cout << "Read GridXTotal: " << GridXTotal << ", GridYTotal: " << GridYTotal << std::endl;
    } catch (const H5::Exception& e) {
        throw std::runtime_error(std::string("Failed to read GridYTotal from grid.h5: ") + e.getCDetailMsg());
    }

    // Read Points
    std::vector<double> host_buffer;
    int nPts = 0;
    int nPtsArrays = 0;
    
    try {
        H5::H5File snap_file(snap_path.string(), H5F_ACC_RDONLY);
        H5::DataSet ds = snap_file.openDataSet("pts_data");
        H5::DataSpace dsp = ds.getSpace();
        hsize_t dims[2];
        dsp.getSimpleExtentDims(dims, nullptr);
        
        nPtsArrays = dims[0];
        nPts = dims[1];
        
        // Check nPtsArrays logic
        if (nPtsArrays != SimParams::PtArrIdx::nPtsArrays) {
             std::cerr << "Warning: Snapshot nPtsArrays (" << nPtsArrays << ") != compiled SimParams::PtArrIdx::nPtsArrays (" << SimParams::PtArrIdx::nPtsArrays << ")" << std::endl;
        }

        std::cout << "Snapshot Dimensions: nPtsArrays=" << nPtsArrays << ", nPts=" << nPts << std::endl;
        size_t total_elements = (size_t)nPts * (size_t)nPtsArrays;
        std::cout << "Resizing buffer to " << total_elements << " doubles (" << (total_elements * sizeof(double) / 1024.0 / 1024.0) << " MB)" << std::endl;
        
        host_buffer.resize(total_elements);
        ds.read(host_buffer.data(), H5::PredType::NATIVE_DOUBLE);
        snap_file.close();
        std::cout << "Read " << nPts << " points." << std::endl;
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error(std::string("Failed to read snapshot: ") + e.getCDetailMsg());
    }

    // Verify Sort
    // We need to extract the cell index for each point.
    // The integer_cell_idx is at index 1 (SimParams::PtArrIdx::integer_cell_idx)
    // The data structure is SOA: array 0 for all points, then array 1 for all points...
    
    const size_t array_offset = (size_t)nPts; // Offset between arrays in buffer
    
    const size_t cell_idx_array_offset = (size_t)SimParams::PtArrIdx::integer_cell_idx * nPts;
    
    std::cout << "Verifying sort order..." << std::endl;
    
    int64_t previous_cell_idx = -1;
    long long unsorted_count = 0;
    
    for (int i = 0; i < nPts; ++i) {
        // Read uint64_t from double buffer
        // Note: bit casting from double to uint64_t
        double val_dbl = host_buffer[cell_idx_array_offset + i];
        uint64_t cell_data = *reinterpret_cast<uint64_t*>(&val_dbl);
        
        uint64_t x_idx = cell_data & 0xffffffff;
        uint64_t y_idx = (cell_data >> 32);

        if (x_idx < 1 || x_idx > (uint64_t)(GridXTotal - 2) || y_idx < 1 || y_idx > (uint64_t)(GridYTotal - 2)) {
             std::string msg = "Invalid grid index at point " + std::to_string(i) + 
                               ": x=" + std::to_string(x_idx) + ", y=" + std::to_string(y_idx) + 
                               ". Allowed: x[1, " + std::to_string(GridXTotal - 2) + "], y[1, " + std::to_string(GridYTotal - 2) + "]";
             throw std::runtime_error(msg);
        }
        
        int64_t cell_index = (int64_t)x_idx * (int64_t)GridYTotal + (int64_t)y_idx;
        
        if (cell_index < previous_cell_idx) {
             // Decode previous point for debug
             double prev_val_dbl = host_buffer[cell_idx_array_offset + i - 1];
             uint64_t prev_cell_data = *reinterpret_cast<uint64_t*>(&prev_val_dbl);
             uint64_t prev_x = prev_cell_data & 0xffffffff;
             uint64_t prev_y = (prev_cell_data >> 32);

             std::cerr << "Sort violation at point " << i << ": current cell " << cell_index << " < prev " << previous_cell_idx << std::endl;
             std::cerr << "  Prev Point " << i-1 << ": idx=" << previous_cell_idx << " (x=" << prev_x << ", y=" << prev_y << ")" << std::endl;
             std::cerr << "  Curr Point " << i << ": idx=" << cell_index << " (x=" << x_idx << ", y=" << y_idx << ")" << std::endl;
             
             std::cerr << "Stopping after first violation." << std::endl;
             unsorted_count++;
             break;
        }
        previous_cell_idx = cell_index;
    }
    
    if (unsorted_count == 0) {
        std::cout << "SUCCESS: All " << nPts << " points are correctly sorted." << std::endl;
    } else {
        std::cout << "FAILURE: Found " << unsorted_count << " sort violations." << std::endl;
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <snapshot_file.h5>" << std::endl;
        return 1;
    }

    try {
        verify_sort(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
