#ifndef PARTITION_PARAMS_H
#define PARTITION_PARAMS_H

#include <cstdint>
#include <cstddef>

// Parameters needed by kernels that are unique to each partition
// (for testing, several partitions may reside on the same device)
struct PartitionParams
{
    unsigned PartitionID;

    // device-side arrays
    double *buffer_pts;  // *pts_array
    double *buffer_grid;  // *grid_array (simulation grid)
    double *buffer_grid_forcing;  // *grid_forcing_array (forcing frames: vx, vy, eta for 2 frames)
    uint8_t *buffer_grid_regions;     // grid_status_array

    double *point_transfer_buffer[4]; // GPU-side buffers to send/receive points between adj. partitions
    size_t point_transfer_buffer_capacity;  // max points it can hold

    size_t pitch_grid, pitch_grid_forcing, count_pts, pitch_pts;
    size_t partition_gridX, gridX_offset;
    size_t gridX_alloc_capacity;    // max resize capacity (in X-direction) excluding halos

    double *halo_transfer_buffer[2];    // computed from *buffer_grid during allocation
    size_t transfer_buffer_width;

    double* getGridLine(const unsigned line) const { return buffer_grid + line*pitch_grid; }

    // one copy of this structure is stored per partition; modified by kernels; used/cleared on host
    struct PartitionUtilityData
    {
        int diffusion_distance_into_halo;  // how far did pts travel into halo (exceeding halo size crashes simulation)
        unsigned transfer_to_left;              // count pts ready to send left (set by partition_kernel_point_transfer)
        unsigned transfer_to_right;             // count pts ready to send right (set by partition_kernel_point_transfer)
    };

    PartitionUtilityData *pud;          // gpu-side allocation
    unsigned *disabled_points_count;    // how many disabled pts in this partition

    double *grid_forces_summary_per_region;
};

#endif // PARTITION_PARAMS_H
