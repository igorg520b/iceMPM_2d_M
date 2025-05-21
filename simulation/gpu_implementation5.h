#ifndef GPU_IMPLEMENTATION5_H
#define GPU_IMPLEMENTATION5_H


#include "gpu_partition.h"
#include "parameters_sim.h"
#include "host_side_soa.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace icy { class Model; }


// contains information relevant to an individual data partition (which corresponds to a GPU device in multi-GPU setup)


class GPU_Implementation5
{
public:
    icy::Model *model;
    std::vector<GPU_Partition> partitions;
    HostSideSOA hssoa;  // mainly stores host-side points
    uint32_t error_code;

    std::vector<uint32_t> point_colors_rgb;     // rgb values of original point colors
    std::vector<uint8_t> grid_status_buffer;    // land (0), modeled area (>0)
    std::vector<uint8_t> original_image_colors_rgb;
    std::vector<t_GridReal> grid_boundary_normals, grid_boundary_forces;


    void allocate_host_arrays_grid();
    void allocate_host_arrays_points();
    void allocate_device_arrays();


    std::function<void()> transfer_completion_callback; // currently not used

    void initialize();
    void split_hssoa_into_partitions();     // perform grid and point partitioning
    void transfer_to_device();
    void transfer_wind_and_current_data_to_device();

    void transfer_from_device(const int elapsed_cycles);

    void synchronize(); // call before terminating the main thread
    void update_constants();
    void reset_grid();
    void reset_timings();
    void clear_force_accumulator();

    void p2g();
    void update_nodes(float simulation_time, float windSpeed, float windAngle);
    void g2p(const bool recordPQ, const bool enablePointTransfer, int applyGlensLaw);
    void record_timings(const bool enablePointTransfer);

private:

//    static void CUDART_CB callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif
