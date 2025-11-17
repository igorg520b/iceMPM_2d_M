#ifndef GPU_PARTITION_H
#define GPU_PARTITION_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <spdlog/spdlog.h>

#include <cuda_runtime.h>

#include <functional>
#include <vector>
#include <array>

#include "parameters_sim.h"
#include "host_side_soa.h"
#include "windandcurrentinterpolator.h"
#include "partition_params.h"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                                          \
do {                                                                                          \
        cudaError_t err = call;                                                                   \
        if (err != cudaSuccess) {                                                                 \
            LOGR("CUDA error in {}:{} {} (code {}): {}", __FILE__, __LINE__, #call, static_cast<int>(err), \
                          cudaGetErrorString(err));                                               \
            throw std::runtime_error(std::string("CUDA error in " #call ": ") +                   \
                                     cudaGetErrorString(err));                                    \
    }                                                                                         \
} while (0)


class GPU_Implementation5;

// CUDA declarations - see kernel_declarations.cuh for all CUDA kernel and device function declarations
#include "kernel_declarations.cuh"


struct GPU_Partition
{
    GPU_Partition(const SimParams &params);
    ~GPU_Partition();

    // host-side data
    int Device;
    const SimParams &prms;
    PartitionParams pparams;    // pointers and offsets for current partition
    PartitionParams::PartitionUtilityData *host_pud;     // stores (receives) various utility information from GPU
    uint32_t error_code;             // set by kernels if there is something wrong
    cudaStream_t streamCompute;

    unsigned *host_disabled_points_count;
    unsigned get_disabled_pts() {return *host_disabled_points_count; }
    double *host_grid_forces_summary_per_region;

    // preparation
    void initialize(int device, int partition);
    void allocate(const unsigned n_points_capacity, const unsigned grid_x_capacity);
    void transfer_points_from_soa_to_device(HostSideSOA &hssoa, int point_idx_offset);
    void transfer_grid_data_to_device(GPU_Implementation5* gpu);
    void update_constants();

    void update_current_field(const WindAndCurrentInterpolator &wac);

    void transfer_from_device(HostSideSOA &hssoa, const int point_idx_offset);

    // simulation cycle
    void reset_grid();
    void clear_force_accumulator();
    void p2g();
    void update_nodes(float simulation_time, const double current_alpha);
    void g2p(const bool recordPQ);

    // specific to multi-gpu
    void receive_halos();
    void receive_render_halos();
    void evaluate_halo_diffusion();
    void send_points();
    void receive_points(const unsigned fromLeft, const unsigned fromRight);

    // render visualized data
    void render_visualized_data();
    void normalize_visualized_data();

    // analysis
    void reset_timings();
    void record_timings();
    void normalize_timings(int cycles);

    // frame analysis
    float timing_10_P2GAndHalo;
    float timing_20_acceptHalo;
    float timing_30_updateGrid;
    float timing_40_G2P;
    float timing_60_ptsSent;
    float timing_70_ptsAccepted;
    float timing_stepTotal;

    cudaEvent_t event_10_cycle_start;
    cudaEvent_t event_20_grid_halo_sent;
    cudaEvent_t event_30_halo_accepted;
    cudaEvent_t event_40_grid_updated;
    cudaEvent_t event_50_g2p_completed;
    cudaEvent_t event_70_pts_sent;
    cudaEvent_t event_80_pts_accepted;

private:
    bool initialized = false;
    void check_error_code();
};


#endif // GPU_PARTITION_H
