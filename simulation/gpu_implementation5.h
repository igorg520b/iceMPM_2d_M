#ifndef GPU_IMPLEMENTATION5_H
#define GPU_IMPLEMENTATION5_H


#include "gpu_partition.h"
#include "parameters_sim.h"
#include "host_side_data.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <cuda_runtime.h>

#include <functional>
#include <vector>
#include <array>


class Model;

// contains information relevant to an individual data partition (which corresponds to a GPU device in multi-GPU setup)
class GPU_Implementation5
{
public:
    GPU_Implementation5(HostSideData &_hsd) : hsd(_hsd) {};


    HostSideData &hsd;

    std::vector<GPU_Partition> partitions;
    uint32_t error_code;
    int halo_diffusion; // how far do points diffuse into halo (max value across partitions)

    void allocate_device_arrays();

    void initialize();
    void split_hssoa_into_partitions();     // perform grid and point partitioning
    void transfer_to_device();
    void update_ocean_current_field(const WindAndCurrentInterpolator &wac);
    void update_wind_field(const WindAndCurrentInterpolator &wac);

    void transfer_from_device();
    void transfer_grid_group_to_host(int group);
    void normalize_grid_on_host();

    void render_visualized_data();

    void synchronize(); // call before terminating the main thread
    void update_constants();
    void reset_grid();
    void reset_timings();
    void clear_force_accumulator();

    void p2g();
    void update_nodes(float simulation_time);
    void g2p(const bool recordPQ, const int step);
    void record_timings();

    // specific to multi-gpu implementation
    void point_transfer();

    void SplitIntoPartitionsAndTransferToDevice();  // transfer the data to one or more GPU devices/partitions

private:
    static std::vector<std::pair<int, int>> getGroupSlotMapping(int group);
};

#endif
