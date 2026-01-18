#ifndef KERNEL_DECLARATIONS_CUH
#define KERNEL_DECLARATIONS_CUH

#include "parameters_sim.h"
#include "partition_params.h"

// ============================================================================
// CUDA KERNEL DECLARATIONS
// ============================================================================
// This header file declares all CUDA kernels used by the GPU_Partition class.
// Kernel implementations are in kernels.cu.
//
// Notes:
// - All kernels take PartitionParams to access partition-specific data
// - The global SimParams (gprms) is accessed via __constant__ memory
// - gpu_error_indicator is used to track runtime errors
// ============================================================================

// ============================================================================
// PUBLIC COMPUTE INTERFACE - Main MPM Operations
// ============================================================================

// Particle-to-Grid transfer: distributes particle data (mass, momentum) to grid nodes
// using B-spline interpolation
__global__ void partition_kernel_p2g(const PartitionParams pparams);

// Grid node update: computes grid node velocities and applies forces
// Parameters:
//   - simulation_time: current simulation time (used for wind/current interpolation)
//   - current_alpha: temporal interpolation factor for wind/current data (0=frame1, 1=frame2)
__global__ void partition_kernel_update_nodes(const PartitionParams pparams,
                                              const double simulation_time,
                                              const double current_alpha,
                                              const double current_alpha_wind);

// Grid-to-Particle transfer: updates particle velocities and deformation gradients
// from grid velocities; optionally records P (stress) and Q (second stress invariant)
// Parameters:
//   - recordPQ: if true, compute and record stress values for visualization
__global__ void partition_kernel_g2p(const PartitionParams pparams,
                                     const bool recordPQ, const int step);

// ============================================================================
// RENDERING KERNELS - Visualization Data Preparation
// ============================================================================

// Renders visualization data: prepares per-particle values (pressure, stress, etc.)
// for display in the GUI by gathering and formatting data from particle state
// Renders in three groups: (1) mass/momentum/strain, (2) RGB/stress, (3) curvature/rotation
__global__ void partition_kernel_render_results(const PartitionParams pparams, int group);

// Normalizes rendered data: computes final visualization values after halo exchange
// (for multi-GPU partitions, halo data needs to be additively blended)
__global__ void partition_kernel_normalize_render(const PartitionParams pparams);

// Summarizes grid forces: aggregates forces on the grid for visualization/analysis
//__global__ void partition_kernel_summarize_forces(const PartitionParams pparams);

// ============================================================================
// MULTI-GPU COMMUNICATION KERNELS - Halo Exchange and Point Transfer
// ============================================================================

// Receives grid halo data from neighboring partitions and adds it to local grid
// Used in P2G phase for grid/force values computed in neighboring partitions
// Parameters:
//   - transfer_buffer_idx: which buffer contains the incoming halo data (0 or 1)
//   - receive_offset: starting grid X index where halo data should be placed
//   - receive_width: width (in X) of the halo region being received
__global__ void partition_kernel_receive_subgrid(const PartitionParams pparams,
                                                 const size_t transfer_buffer_idx,
                                                 const size_t receive_offset,
                                                 const size_t receive_width);

// Checks which points need to be transferred to neighboring partitions
// Sets flags in partition state (transfer_to_left, transfer_to_right counts)
__global__ void partition_kernel_check_if_transfer_needed(const PartitionParams pparams);

// Transfers points to neighboring partitions: copies particle data that has moved
// outside the partition's X domain to staging buffers for sending to neighbors
__global__ void partition_kernel_point_transfer(const PartitionParams pparams);

// Receives points from neighboring partitions and integrates them into local point array
// Parameters:
//   - nPts: number of points being received
//   - bufferIdx: which receive buffer to use (2=from left, 3=from right)
__global__ void partition_kernel_receive_points(const PartitionParams pparams,
                                                const unsigned nPts,
                                                const unsigned bufferIdx);

// ============================================================================
// DEVICE HELPER FUNCTIONS - Called by kernels for computation
// ============================================================================

// SVD decomposition for 2x2 matrices: mA = mU * diag(mS) * mV^T
// Used for deformation gradient analysis and stress computation
__device__ void svd2x2(const Eigen::Matrix2d &mA, Eigen::Matrix2d &mU, Eigen::Vector2d &mS, Eigen::Matrix2d &mV);

// Computes SVD and derived quantities for constitutive model
// Parameters:
//   - Fe: elastic deformation gradient
//   - kappa, mu: material parameters
//   - Je_tr: elastic volumetric strain (determinant of Fe)
__device__ void ComputeSVD(const Eigen::Matrix2d &Fe, Eigen::Matrix2d &U, Eigen::Vector2d &vSigma, Eigen::Matrix2d &V,
                           Eigen::Vector2d &vSigmaSquared, Eigen::Vector2d &v_s_hat_tr,
                           const double &kappa, const double &mu, const double &Je_tr);

// Computes pressure (p) and second stress invariant (q) from deformation and material properties
// Used for failure surface checks and visualization
__device__ void ComputePQ(double &Je_tr, double &p_tr, double &q_tr, const Eigen::Matrix2d &F);

// Wolper-Drucker-Prager constitutive model: updates Fe and Jp_inv based on stress state
// Implements elastic and plastic deformation for ice material
// Parameters:
//   - initial_strength: ice strength at start of timestep
__device__ void Wolper_Drucker_Prager(const unsigned long long &utility_data,
    const double &initial_strength,
                                      const double &p_tr, const double &q_tr, const double &Je_tr,
                                      const Eigen::Matrix2d &U, const Eigen::Matrix2d &V, const Eigen::Vector2d &vSigmaSquared, const Eigen::Vector2d &v_s_hat_tr,
                                      Eigen::Matrix2d &Fe, double &Jp_inv);

// Glen-Nye flow law for ice rheology: time-dependent plastic strain accumulation
// Updates Fe
__device__ void Glen_Nye_flow_law(const double dt, double &q_tr,
    Eigen::Vector2d &vSigmaSquared,
    const Eigen::Matrix2d &U,
    const Eigen::Matrix2d &V,
    Eigen::Vector2d &v_s_hat_tr,
    Eigen::Matrix2d &Fe, double &track_change);

// Checks if a material point has exceeded the failure surface (yield criterion)
// Sets status flags if failure has occurred
__device__ void CheckIfPointIsInsideFailureSurface(unsigned long long &utility_data,
                                                   const double &p, const double &q,
                                                   const double &strength);

// Retrieves grain-specific material parameters (strength bounds, hardening)
__device__ void GetParametersForGrain(uint32_t utility_data, double &pmin, double &pmax, double &qmax,
                                      double &beta, double &mSq, double &pmin2);

// Computes Kirchhoff stress from deformation gradient using Wolper material model
__device__ Eigen::Matrix2d KirchhoffStress_Wolper(const Eigen::Matrix2d &F, const double &Jp_inv);

// how Jp_inv affects bulk modulus
__device__ double BulkModulusReductionCoeff(const double &Jp_inv);

// Extracts deviatoric (traceless) part of a 2D diagonal matrix
__device__ Eigen::Vector2d dev_d(Eigen::Vector2d Adiag);

// Extracts deviatoric (traceless) part of a 2D matrix
__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A);

// Computes B-spline weight coefficients for particle-grid interpolation
// ww[i+1] contains weights for axis i at positions i-1, i, i+1
__device__ void CalculateWeightCoeffs(const Eigen::Vector2d &pos, Eigen::Array2d ww[3]);

// Retrieves wind vector at given position and time from interpolated wind field data
__device__ Eigen::Vector2d get_wind_vector(float lat, float lon, float tb);

// obtain point's (i,j) cell index from raw double value (stored in points buffer)
__device__ Eigen::Vector2i getIntegerCellIndex(double raw_value);

__device__ void ComputeStressResultants(
    // Inputs
    const Eigen::Matrix2d &kappa_raw,       // Raw curvature (gradient of omega)
    const Eigen::Vector2d &gamma,           // Shear strain
    const Eigen::Matrix2d &Damage,          // Anisotropic Damage Tensor (Eigenvalues 0 to 1)
    const double thickness,
    const double E,                         // Young's Modulus
    const double nu,                        // Poisson's Ratio
    const double mu,                        // Shear Modulus
    // Outputs (by reference)
    Eigen::Matrix2d &Mp_out,
    Eigen::Vector2d &Q_out
    );

__device__ void ComputeElasticForces(
    const Eigen::Matrix2d &kappa_raw,
    const Eigen::Vector2d &gamma,
    const double thickness, const double E, const double nu, const double mu,
    Eigen::Matrix2d &Mp_elastic,
    Eigen::Vector2d &Q_elastic
    );

__device__ void EigenDecomposition2x2(
    const Eigen::Matrix2d &M,
    double &eig1, double &eig2,      // Eigenvalues
    Eigen::Vector2d &v1, Eigen::Vector2d &v2 // Eigenvectors
    );

// ============================================================================
// DEVICE STATE - Accessed by all kernels
// ============================================================================

extern __device__ uint32_t gpu_error_indicator;  // Accumulates error codes during kernel execution
extern __constant__ SimParams gprms;             // Simulation parameters (constant memory)

#endif // KERNEL_DECLARATIONS_CUH
