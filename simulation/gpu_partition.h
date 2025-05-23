#ifndef GPU_PARTITION_H
#define GPU_PARTITION_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <spdlog/spdlog.h>

#include <cuda_runtime.h>

#include <functional>
#include <vector>

#include "parameters_sim.h"
#include "host_side_soa.h"
#include "windandcurrentinterpolator.h"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                                          \
do {                                                                                          \
        cudaError_t err = call;                                                                   \
        if (err != cudaSuccess) {                                                                 \
            spdlog::error("CUDA error in {}:{} {} (code {}): {}", __FILE__, __LINE__, #call, err, \
                          cudaGetErrorString(err));                                               \
            throw std::runtime_error(std::string("CUDA error in " #call ": ") +                   \
                                     cudaGetErrorString(err));                                    \
    }                                                                                         \
} while (0)


class GPU_Implementation5;

// parameters needed by kernels that are unique to each partition
// (for testing, several partitions may reside on the same device)
struct PartitionParams
{
    unsigned PartitionID;

    // device-side arrays
    t_PointReal *buffer_pts;  // *pts_array
    t_GridReal *buffer_grid;  // *grid_array
    uint8_t *buffer_grid_regions;     // grid_status_array

    t_GridReal *halo_transfer_buffer[2];    // computed from *buffer_grid during allocation

    t_PointReal *point_transfer_buffer[4]; // GPU-side buffers to send/receive points between adj. partitions
    size_t point_transfer_buffer_capacity;  // max points it can hold

    size_t pitch_grid, count_pts, pitch_pts;
    size_t partition_gridX, gridX_offset;
};



struct GPU_Partition
{
    GPU_Partition();
    ~GPU_Partition();

    // host-side data
    int Device;
    static SimParams *prms;
    PartitionParams pparams;    // pointers and offsets for current partition
    uint32_t error_code;             // set by kernels if there is something wrong
    cudaStream_t streamCompute;

    // counts how many points are marked as disabled in the partition
    // actual value is stored in disabled_points_count[pparams.PartitionID]
    unsigned disabled_points_count[8];
    unsigned get_disabled_pts() {return disabled_points_count[pparams.PartitionID];}
    t_GridReal *tmp_accumulated_forces;     // buffer for async transfer; later merged into the global grid buffer

    // preparation
    void initialize(int device, int partition);
    void allocate(const unsigned n_points_capacity, const unsigned grid_x_capacity);
    void transfer_points_from_soa_to_device(HostSideSOA &hssoa, int point_idx_offset);
    void transfer_grid_data_to_device(GPU_Implementation5* gpu);
    void update_constants();

    void update_current_field(const WindAndCurrentInterpolator &wac);

    void transfer_from_device(HostSideSOA &hssoa, const int point_idx_offset, std::vector<t_GridReal> &boundary_forces);

    // simulation cycle
    void reset_grid();
    void clear_force_accumulator();
    void p2g();
    void update_nodes(float simulation_time, const GridVector2r vWind, const float interpolation_coeff);
    void g2p(const bool recordPQ, const bool enablePointTransfer, int applyGlensLaw);

    // analysis
    void reset_timings();
    void record_timings(const bool enablePointTransfer);
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



extern __device__ unsigned gpu_disabled_points_count[8];
extern __device__ uint32_t gpu_error_indicator;

// kernels
__global__ void partition_kernel_p2g(const PartitionParams pparams);

__global__ void partition_kernel_update_nodes(const PartitionParams pparams, const t_PointReal simulation_time);

__global__ void partition_kernel_g2p(const PartitionParams pparams, const bool recordPQ);



// helper functions

__device__ void svd2x2(const PointMatrix2r &mA, PointMatrix2r &mU, PointVector2r &mS, PointMatrix2r &mV);

__device__ void Wolper_Drucker_Prager(const t_PointReal &initial_strength,
                                      const t_PointReal &p_tr, const t_PointReal &q_tr, const t_PointReal &Je_tr,
                                      const PointMatrix2r &U, const PointMatrix2r &V, const PointVector2r &vSigmaSquared, const PointVector2r &v_s_hat_tr,
                                      PointMatrix2r &Fe, t_PointReal &Jp_inv);

__device__ void CheckIfPointIsInsideFailureSurface(uint32_t &utility_data, const uint16_t &grain,
                                                   const t_PointReal &p, const t_PointReal &q,
                                                   const t_PointReal &strength);

__device__ void ComputeSVD(const PointMatrix2r &Fe, PointMatrix2r &U, PointVector2r &vSigma, PointMatrix2r &V,
                           PointVector2r &vSigmaSquared, PointVector2r &v_s_hat_tr,
                           const t_PointReal &kappa, const t_PointReal &mu, const t_PointReal &Je_tr);

__device__ void ComputePQ(t_PointReal &Je_tr, t_PointReal &p_tr, t_PointReal &q_tr,
                          const double &kappa, const double &mu, const PointMatrix2r &F);

__device__ void GetParametersForGrain(uint32_t utility_data, t_PointReal &pmin, t_PointReal &pmax, t_PointReal &qmax,
                                      t_PointReal &beta, t_PointReal &mSq, t_PointReal &pmin2);

__device__ PointMatrix2r KirchhoffStress_Wolper(const PointMatrix2r &F);

__device__ PointVector2r dev_d(PointVector2r Adiag);

__device__ PointMatrix2r dev(PointMatrix2r A);

__device__ void CalculateWeightCoeffs(const PointVector2r &pos, PointArray2r ww[3]);

__device__ t_PointReal smoothstep(t_PointReal x);

__device__ GridVector2r get_wind_vector(float lat, float lon, float tb);


#endif // GPU_PARTITION_H
