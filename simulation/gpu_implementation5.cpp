#include "gpu_implementation5.h"
#include "parameters_sim.h"
#include "model.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>

#include <spdlog/spdlog.h>

#include "intinterval.h"

using namespace Eigen;



void GPU_Implementation5::initialize()
{
    const unsigned &nPartitions = hsd.prms.nPartitions;

    // count available GPUs
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("cudaGetDeviceCount error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");
    LOGR("GPU_Implementation5::initialize; devic count {}",deviceCount);

    LOGR("Device Information:");
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        LOGR("  --- Device {}: {} ---", i, deviceProp.name);
        LOGR("      Compute Capability: {}.{}", deviceProp.major, deviceProp.minor);
        // Convert bytes to Megabytes (MB) for readability
        double totalMemMB = static_cast<double>(deviceProp.totalGlobalMem) / (1024.0 * 1024.0);
        LOGR("      Total Global Memory: {:>.2} MB", totalMemMB);
        LOGR("      Clock Rate: {:>.2} GHz", deviceProp.clockRate / (1000.0 * 1000.0)); // Convert kHz to GHz
        LOGR("      Number of SMs: {}\n", deviceProp.multiProcessorCount);
    }
    std::cout << std::endl;

    partitions.clear();
    partitions.reserve(nPartitions);  // Pre-allocate to avoid vector reallocation destroying streams

    // Create GPU_Partition objects with reference to hsd.prms
    for(int i=0; i<nPartitions; i++)
    {
        partitions.emplace_back(hsd.prms);
        GPU_Partition &p = partitions[i];
        p.initialize(i%deviceCount, i);
    }

    if(deviceCount >= 2)
    {
        // enable peer access
        const int nLoop = deviceCount == 2 ? 1 : deviceCount;
        for(int i=0;i<nLoop;i++)
        {
            const int dev_next = (i+1)%deviceCount;
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceEnablePeerAccess(dev_next, 0));
            CUDA_CHECK(cudaSetDevice(dev_next));
            CUDA_CHECK(cudaDeviceEnablePeerAccess(i, 0));
        }
    }
    LOGR("GPU_Implementation5::initialize() completed");
    LOGR("sizeof(PartitionParams) = {}\n", sizeof(PartitionParams));
}

void GPU_Implementation5::split_hssoa_into_partitions()
{
    LOGR("\nsplit_hssoa_into_partitions() start");
    const int &GridXTotal = hsd.prms.GridXTotal;
    const unsigned &nPartitions = hsd.prms.nPartitions;

    unsigned nPointsProcessed = 0;
    partitions[0].pparams.gridX_offset = 0;

    for(unsigned partition_idx=0; partition_idx<nPartitions; partition_idx++)
    {
        GPU_Partition &p = partitions[partition_idx];
        const unsigned nPartitionsRemaining = nPartitions - partition_idx;
        p.pparams.count_pts = (hsd.hssoa.size - nPointsProcessed)/nPartitionsRemaining; // points in this partition

        // find the index of the first point with x-index cellsIdx
        if(partition_idx < nPartitions-1)
        {
            SOAIterator it2 = hsd.hssoa.begin() + (nPointsProcessed + p.pparams.count_pts);
            const unsigned cellsIdx = it2->getCellX();
            p.pparams.partition_gridX = cellsIdx - p.pparams.gridX_offset;
            partitions[partition_idx+1].pparams.gridX_offset = cellsIdx;
        }
        else if(partition_idx == nPartitions-1)
        {
            // the last partition spans the rest of the grid along the x-axis
            p.pparams.partition_gridX = GridXTotal - p.pparams.gridX_offset;
        }

#pragma omp parallel for
        for(int j=nPointsProcessed;j<(nPointsProcessed+p.pparams.count_pts);j++)
        {
            SOAIterator it2 = hsd.hssoa.begin() + (j);
            // Store partition index in the lower 16 bits of utility_data
            uint32_t util = it2->getValueInt(SimParams::PtArrIdx::idx_utility_data);
            util &= 0xFFFF0000; // Clear lower 16 bits
            util |= (uint16_t)partition_idx; // Set partition index
            it2->setValueInt(SimParams::PtArrIdx::idx_utility_data, util);
        }

        nPointsProcessed += p.pparams.count_pts;
    }
    LOGR("split_hssoa_into_partitions() done");
    std::cout << std::endl;

    for(GPU_Partition &p : partitions)
    {
        LOGR("split: P {0:}; grid_offset {1:>7}; grid_size {2:>7}, npts {3:>8}",
             p.pparams.PartitionID, p.pparams.gridX_offset, p.pparams.partition_gridX, p.pparams.count_pts);
    }
    LOGR("GPU_Implementation5::split_hssoa_into_partitions() done");
}




void GPU_Implementation5::reset_grid()
{
    for(GPU_Partition &p : partitions)
    {
        CUDA_CHECK(cudaSetDevice(p.Device));
        CUDA_CHECK(cudaEventRecord(p.event_10_cycle_start, p.streamCompute));
        p.reset_grid();
    }
}

void GPU_Implementation5::clear_force_accumulator()
{
    for(GPU_Partition &p : partitions)
    {
        CUDA_CHECK(cudaSetDevice(p.Device));
        p.clear_force_accumulator();
    }
}


void GPU_Implementation5::p2g()
{
    constexpr unsigned params_to_transfer = SimParams::grid_arrays_to_clear;  // mass, px, py, pz, Lx, Ly, I
    const int &gridY = hsd.prms.GridYTotal;
    const unsigned &halo = hsd.prms.GridHaloSize;

    for(GPU_Partition &p : partitions)
    {
        p.p2g();    // invoke the P2G kernel
    }

    if(partitions.size() == 1)
    {
        // single partition - no need to copy halos
        GPU_Partition &p = partitions.front();
        CUDA_CHECK(cudaEventRecord(p.event_20_grid_halo_sent, p.streamCompute));
        CUDA_CHECK(cudaEventRecord(p.event_30_halo_accepted, p.streamCompute));
        return;
    }


    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        CUDA_CHECK(cudaSetDevice(p.Device));

        const size_t halo_count = 2*gridY*halo*sizeof(double);
        // after P2G step is completed, transfer halos to adjacent partitions
        if(i!=0)
        {
            // send halo to the left
            GPU_Partition &pprev = partitions[i-1];
            for(int j=0;j<params_to_transfer;j++)
            {
                double *src = p.pparams.getGridLine(j);
                double *dst = pprev.pparams.halo_transfer_buffer[1] + j*pprev.pparams.transfer_buffer_width*gridY;
                CUDA_CHECK(cudaMemcpyPeerAsync(dst, pprev.Device, src, p.Device, halo_count, p.streamCompute));
            }
        }

        if(i!=(partitions.size()-1))
        {
            // send halo to the right
            GPU_Partition &pnxt = partitions[i+1];
            for(int j=0;j<params_to_transfer;j++)
            {
                double *src = p.pparams.getGridLine(j) + gridY*p.pparams.partition_gridX;
                double *dst = pnxt.pparams.halo_transfer_buffer[0] + j*pnxt.pparams.transfer_buffer_width*gridY;
                CUDA_CHECK(cudaMemcpyPeerAsync(dst, pnxt.Device, src, p.Device, halo_count, p.streamCompute));
            }
        }
        CUDA_CHECK(cudaEventRecord(p.event_20_grid_halo_sent, p.streamCompute));
    }

    // receive halos
    // wait until all halos are copied to all partitions (!)
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        CUDA_CHECK(cudaSetDevice(p.Device));

        if(i!=0) CUDA_CHECK(cudaStreamWaitEvent(p.streamCompute, partitions[i-1].event_20_grid_halo_sent));
        if(i!=partitions.size()-1) CUDA_CHECK(cudaStreamWaitEvent(p.streamCompute, partitions[i+1].event_20_grid_halo_sent));
    }

    for(GPU_Partition &p : partitions)
    {
        p.receive_halos();
        CUDA_CHECK(cudaEventRecord(p.event_30_halo_accepted, p.streamCompute));
    }
}




void GPU_Implementation5::update_nodes(float simulation_time, float windSpeed, float windAngle)
{
    double current_alpha = hsd.waci.current_alpha;
    double current_alpha_wind = hsd.waci.current_wind_alpha;

    for(GPU_Partition &p : partitions)
    {
        p.update_nodes(simulation_time, current_alpha, current_alpha_wind);
        CUDA_CHECK(cudaEventRecord(p.event_40_grid_updated, p.streamCompute));
    }
}

void GPU_Implementation5::g2p(const bool recordPQ, const int step)
{
    for(GPU_Partition &p : partitions)
    {
        p.g2p(recordPQ, step);
        CUDA_CHECK(cudaEventRecord(p.event_50_g2p_completed, p.streamCompute));
    }
}

void GPU_Implementation5::render_visualized_data()
{
    // Phase 1: Summarize forces from all partitions FIRST
    // This processes accumulated fx, fy into per-region force summary
    // After this, all GPU array slots are free for visualization passes
    for(GPU_Partition &p : partitions)
    {
        p.summarize_forces();
    }

    // Transfer force summary results from GPU to host
    for(GPU_Partition &p : partitions)
    {
        p.transfer_force_summary_from_device();
    }

    // Phase 2: Render visualization data group-by-group and transfer to host
    // Each group reuses the same 10 GPU array slots, so we must transfer before the next group
    // Groups 1 & 2 only now.
    for (int group = 1; group <= 2; ++group)
    {
        // Clear GPU memory and render this group for all partitions
        for(GPU_Partition &p : partitions)
        {
            p.render_visualized_data(group);
        }

        // Transfer this group from all partitions to host (with halo blending)
        transfer_grid_group_to_host(group);
    }

    // Phase 3: Normalize all grid data after all groups have been transferred
    normalize_grid_on_host();
}


void GPU_Implementation5::point_transfer()
{
    if(partitions.size()==1) return;
    // check how far the points diffuse into halo
    for(GPU_Partition &p : partitions)
    {
        p.evaluate_halo_diffusion();
    }

    halo_diffusion = 0;
    for(GPU_Partition &p : partitions)
    {
        CUDA_CHECK(cudaSetDevice(p.Device));
        CUDA_CHECK(cudaStreamSynchronize(p.streamCompute));
        halo_diffusion = std::max(halo_diffusion, p.host_pud->diffusion_distance_into_halo);
    }

    if(halo_diffusion > 0)
    {
        LOGR("GPU_Implementation5::point_transfer(); halo_diffusion {}", halo_diffusion);
        for(int i=0;i<partitions.size();i++)
        {
            GPU_Partition &p = partitions[i];
            LOGR("PID {}; halo diffusion {}", i, p.host_pud->diffusion_distance_into_halo);
        }
        std::cout << std::endl;

        halo_diffusion = 0;
        for(GPU_Partition &p : partitions)
        {
            p.send_points();    // does not actually transfer the data (only prepares); initiates PUD transfer form device
        }
        std::cout << std::endl;


        // after the data is prepared by the kernel, initiate copy
        for(int i=0;i<partitions.size();i++)
        {
            GPU_Partition &p = partitions[i];
            CUDA_CHECK(cudaSetDevice(p.Device));
            CUDA_CHECK(cudaStreamSynchronize(p.streamCompute)); // receive PUD to host

            p.host_pud->transfer_to_left = std::min(p.host_pud->transfer_to_left, (unsigned)p.pparams.point_transfer_buffer_capacity);
            p.host_pud->transfer_to_right = std::min(p.host_pud->transfer_to_right, (unsigned)p.pparams.point_transfer_buffer_capacity);


            if(i!=(partitions.size()-1))
            {
                // send buffer to the right
                GPU_Partition &pnxt = partitions[i+1];
                double* const src_buffer = p.pparams.point_transfer_buffer[1];
                double* const dst_buffer = pnxt.pparams.point_transfer_buffer[2];
                const unsigned &right_buffer_count = p.host_pud->transfer_to_right;
                size_t count = right_buffer_count*sizeof(double)*SimParams::PtArrIdx::nPtsArrays;
                if(count != 0)
                {
                    LOGR("PID {} copying points to the right; npts {}; cap {}", i, right_buffer_count, p.pparams.point_transfer_buffer_capacity);
                    CUDA_CHECK(cudaMemcpyPeerAsync(dst_buffer, pnxt.Device, src_buffer, p.Device, count, p.streamCompute));
                }
            }

            if(i!=0)
            {
                // send buffer to the left
                GPU_Partition &pprev = partitions[i-1];
                double* const src_buffer = p.pparams.point_transfer_buffer[0];
                double* const dst_buffer = pprev.pparams.point_transfer_buffer[3];
                const unsigned &left_buffer_count = p.host_pud->transfer_to_left;
                size_t count = left_buffer_count*sizeof(double)*SimParams::PtArrIdx::nPtsArrays;
                if(count != 0)
                {
                    LOGR("PID {} copying points to the left; npts {}; cap {}", i, left_buffer_count, p.pparams.point_transfer_buffer_capacity);
                    CUDA_CHECK(cudaMemcpyPeerAsync(dst_buffer, pprev.Device, src_buffer, p.Device, count, p.streamCompute));
                }
            }
            CUDA_CHECK(cudaEventRecord(p.event_70_pts_sent, p.streamCompute));
        }

        // wait until data is copied, then invoke kernels to receive points
        for(int i=0;i<partitions.size();i++)
        {
            GPU_Partition &p = partitions[i];
            CUDA_CHECK(cudaSetDevice(p.Device));
            unsigned fromLeft=0, fromRight=0;

            if(i!=0)
            {
                GPU_Partition &pprev = partitions[i-1];
                CUDA_CHECK(cudaStreamWaitEvent(p.streamCompute, pprev.event_70_pts_sent));
                fromLeft = pprev.host_pud->transfer_to_right;
            }

            if(i!=partitions.size()-1)
            {
                GPU_Partition &pnxt = partitions[i+1];
                CUDA_CHECK(cudaStreamWaitEvent(p.streamCompute, pnxt.event_70_pts_sent));
                fromRight = pnxt.host_pud->transfer_to_left;
            }
            LOGR("PID {} receiving points; left {}; right {}", i, fromLeft, fromRight);
            p.receive_points(fromLeft, fromRight);
            CUDA_CHECK(cudaEventRecord(p.event_80_pts_accepted, p.streamCompute));
        }

    }
}


void GPU_Implementation5::record_timings()
{
    for(GPU_Partition &p : partitions) p.record_timings();
}



// ==========================================================================


void GPU_Implementation5::allocate_device_arrays()
{
    LOGR("GPU_Implementation5::allocate_device_arrays()");

    const unsigned &nPts = hsd.hssoa.size;
    const unsigned pts_reserve = (nPts/partitions.size()) * (1. + hsd.prms.extra_space_pts);

    for(GPU_Partition &p : partitions) p.allocate(pts_reserve, hsd.prms.GridXTotal);
    LOGR("allocate_device_arrays done");
    std::cout << std::endl;
}



void GPU_Implementation5::transfer_to_device()
{
    LOGR("GPU_Implementation: transfer_to_device()");

    int points_uploaded = 0;
    for(GPU_Partition &p : partitions)
    {
        LOGR("GPU_Implementation5::transfer_to_device(); PID {}; pts_uploaded {}", p.pparams.PartitionID, points_uploaded);
        p.transfer_points_from_soa_to_device(hsd.hssoa, points_uploaded);
        p.transfer_grid_data_to_device(this);
        points_uploaded += p.pparams.count_pts;
    }
    LOGR("transfer_ponts_to_device() done; transferred points {}", points_uploaded);
}



void GPU_Implementation5::transfer_from_device()
{
    size_t offset_pts = 0;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        const size_t capacity_required = offset_pts + p.pparams.count_pts;
        if(capacity_required > hsd.hssoa.capacity)
        {
            LOGR("transfer_from_device(): capacity {} exceeded ({}) when transferring P {}",
                             hsd.hssoa.capacity, capacity_required, p.pparams.PartitionID);
            throw std::runtime_error("transfer_from_device capacity exceeded");
        }
        p.transfer_from_device(hsd.hssoa, offset_pts);
        offset_pts += p.pparams.count_pts;
    }
    hsd.hssoa.size = (unsigned)offset_pts;

    // Grid data is transferred in render_visualized_data() on a group-by-group basis
    // so we don't call transfer_grid_to_host() here anymore

    // wait until everything is copied to host
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        cudaSetDevice(p.Device);
        cudaStreamSynchronize(p.streamCompute);
        if(p.error_code)
        {
            // throw std::runtime_error("error code");
            this->error_code = p.error_code;
            LOGR("P {}; error code {}; this error code {}", p.pparams.PartitionID, p.error_code, this->error_code);
        }
    }

    hsd.grid_forces_summary_per_region.fill(0.f);
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        for(int k=0;k<SimParams::MAX_REGIONS*2;k++)
            hsd.grid_forces_summary_per_region[k] += p.host_grid_forces_summary_per_region[k];
    }
}


// Helper function to get GPU-to-Host slot mapping for a visualization group
std::vector<std::pair<int, int>> GPU_Implementation5::getGroupSlotMapping(int group)
{
    static const std::map<std::pair<int, int>, int> group_slot_map = {
        // Group 1: Standard Visualizations
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_mass}, SimParams::HostGridArrayIndex::host_grid_idx_mass},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_px}, SimParams::HostGridArrayIndex::grid_idx_px},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_py}, SimParams::HostGridArrayIndex::grid_idx_py},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_r}, SimParams::HostGridArrayIndex::grid_idx_vis_r},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_g}, SimParams::HostGridArrayIndex::grid_idx_vis_g},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_b}, SimParams::HostGridArrayIndex::grid_idx_vis_b},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_Jpinv}, SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_P}, SimParams::HostGridArrayIndex::grid_idx_vis_P},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_Q}, SimParams::HostGridArrayIndex::grid_idx_vis_Q},
        {{1, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_pts_density}, SimParams::HostGridArrayIndex::grid_idx_vis_pts_density},

        // Group 2: Strains (reusing slots)
        {{2, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_strain_EqvGreenLagrange}, SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange},
        {{2, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_strain_vonMises}, SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises},
        {{2, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_crushed}, SimParams::HostGridArrayIndex::grid_idx_vis_crushed},

        {{2, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_cracked}, SimParams::HostGridArrayIndex::grid_idx_vis_cracked},
        {{2, SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_thickness}, SimParams::HostGridArrayIndex::grid_idx_vis_thickness}
    };

    std::vector<std::pair<int, int>> result;
    for (int gpu_slot = 0; gpu_slot < SimParams::GPUGridArrayIndex::nGridArraysGPU; ++gpu_slot) {
        auto it = group_slot_map.find({group, gpu_slot});
        if (it != group_slot_map.end()) {
            result.push_back({gpu_slot, it->second});
        }
    }

    if (result.empty()) {
        throw std::runtime_error("No mapping found for visualization group " + std::to_string(group));
    }

    return result;
}


void GPU_Implementation5::normalize_grid_on_host()
{
    LOGR("Normalizing grid data on host...");

    const int gx = hsd.prms.GridXTotal;
    const int gy = hsd.prms.GridYTotal;
    const double cellsize_sq = hsd.prms.cellsize * hsd.prms.cellsize;
    const size_t grid_plane_size = (size_t)gx * gy;

    // List of grid quantities that need to be normalized by mass
    const size_t planes_to_normalize[] = {
        SimParams::HostGridArrayIndex::grid_idx_px,
        SimParams::HostGridArrayIndex::grid_idx_py,
        SimParams::HostGridArrayIndex::grid_idx_vis_r,
        SimParams::HostGridArrayIndex::grid_idx_vis_g,
        SimParams::HostGridArrayIndex::grid_idx_vis_b,
        SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv,
        SimParams::HostGridArrayIndex::grid_idx_vis_P,
        SimParams::HostGridArrayIndex::grid_idx_vis_Q,
        SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange,
        SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises,
        SimParams::HostGridArrayIndex::grid_idx_vis_crushed,
        SimParams::HostGridArrayIndex::grid_idx_vis_cracked,
        SimParams::HostGridArrayIndex::grid_idx_vis_thickness
    };

    const size_t mass_plane_offset = grid_plane_size * SimParams::HostGridArrayIndex::host_grid_idx_mass;

#pragma omp parallel for
    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            const size_t idx = (size_t)j + (size_t)i * gy;
            const double mass = hsd.host_grid_buffer[mass_plane_offset + idx];
            const double pt_count = hsd.host_grid_buffer[grid_plane_size * SimParams::HostGridArrayIndex::grid_idx_vis_pts_density + idx];

            if (mass == 0) {
                continue;
            }

            for (const auto& plane_idx : planes_to_normalize) {
                hsd.host_grid_buffer[grid_plane_size * plane_idx + idx] /= mass;
            }
            hsd.host_grid_buffer[mass_plane_offset + idx] /= cellsize_sq;
        }
    }
}


void GPU_Implementation5::transfer_grid_group_to_host(int group)
{
    const int gx_total = hsd.prms.GridXTotal;
    const int gy_total = hsd.prms.GridYTotal;
    const int halo = hsd.prms.GridHaloSize;
    const size_t grid_plane_size = (size_t)gx_total * gy_total;
    const int nGridArrays = SimParams::HostGridArrayIndex::nGridArraysHost;

    // Reset host-side grid buffer to zero before the first group
    if (group == 1) {
        hsd.host_grid_buffer.assign(grid_plane_size * nGridArrays, 0.0f);
    }

    // Get the slot mapping for this group (throws if group not found)
    auto slot_mapping = getGroupSlotMapping(group);

    // Phase 1: Copy interior grid data from each partition for this group
    for (int i = 0; i < partitions.size(); i++)
    {
        GPU_Partition &p = partitions[i];
        CUDA_CHECK(cudaSetDevice(p.Device));
        CUDA_CHECK(cudaStreamSynchronize(p.streamCompute)); // Ensure render kernel is finished

        // Transfer each array in this group individually (slice by slice)
        for (auto [gpu_slot, host_slot] : slot_mapping)
        {
            const double* src_dev = p.pparams.buffer_grid + (size_t)gpu_slot * p.pparams.pitch_grid + (size_t)gy_total * halo;
            double* dst_host = hsd.host_grid_buffer.data() + (size_t)host_slot * grid_plane_size + (size_t)gy_total * p.pparams.gridX_offset;

            CUDA_CHECK(cudaMemcpy(
                dst_host,
                src_dev,
                (size_t)p.pparams.partition_gridX * gy_total * sizeof(double),
                cudaMemcpyDeviceToHost
            ));
        }
    }

    // Phase 2: Additively blend halo regions from overlapping partitions
    for (int i = 0; i < partitions.size(); i++)
    {
        GPU_Partition &p = partitions[i];
        CUDA_CHECK(cudaSetDevice(p.Device));

        // Transfer each array in this group individually
        for (auto [gpu_slot, host_slot] : slot_mapping)
        {
            // Left halo
            if (i > 0)
            {
                const double* src_dev = p.pparams.buffer_grid + (size_t)gpu_slot * p.pparams.pitch_grid;
                CUDA_CHECK(cudaMemcpy(
                    hsd.tmp_halo_buffer.data(),
                    src_dev,
                    (size_t)gy_total * halo * sizeof(double),
                    cudaMemcpyDeviceToHost
                ));

                size_t host_x_start = p.pparams.gridX_offset - halo;
                // Additively blend left halo into host buffer
                for (int x = 0; x < halo; x++) {
                    for (int y = 0; y < gy_total; y++) {
                        size_t src_idx = (size_t)x * gy_total + y;
                        size_t dst_idx = (size_t)host_slot * grid_plane_size + (size_t)(host_x_start + x) * gy_total + y;
                        hsd.host_grid_buffer[dst_idx] += hsd.tmp_halo_buffer[src_idx];
                    }
                }
            }

            // Right halo
            if (i < partitions.size() - 1)
            {
                const double* src_dev = p.pparams.buffer_grid + (size_t)gpu_slot * p.pparams.pitch_grid + (size_t)gy_total * (halo + p.pparams.partition_gridX);
                CUDA_CHECK(cudaMemcpy(
                    hsd.tmp_halo_buffer.data(),
                    src_dev,
                    (size_t)gy_total * halo * sizeof(double),
                    cudaMemcpyDeviceToHost
                ));

                size_t host_x_start = p.pparams.gridX_offset + p.pparams.partition_gridX;
                // Additively blend right halo into host buffer
                for (int x = 0; x < halo; x++) {
                    for (int y = 0; y < gy_total; y++) {
                        size_t src_idx = (size_t)x * gy_total + y;
                        size_t dst_idx = (size_t)host_slot * grid_plane_size + (size_t)(host_x_start + x) * gy_total + y;
                        hsd.host_grid_buffer[dst_idx] += hsd.tmp_halo_buffer[src_idx];
                    }
                }
            }
        }
    }
}


void GPU_Implementation5::synchronize()
{
    for(GPU_Partition &p : partitions)
    {
        cudaSetDevice(p.Device);
        cudaDeviceSynchronize();
    }
}

void GPU_Implementation5::update_constants()
{
    error_code = 0;
    for(GPU_Partition &p : partitions) p.update_constants();
}

void GPU_Implementation5::reset_timings()
{
    for(GPU_Partition &p : partitions)
    {
        p.reset_timings();
    }
}



void GPU_Implementation5::update_ocean_current_field(const WindAndCurrentInterpolator &wac)
{
    for(GPU_Partition &p : partitions)
    {
        p.update_ocean_current_field(wac);
    }
}

void GPU_Implementation5::update_wind_field(const WindAndCurrentInterpolator &wac)
{
    for(GPU_Partition &p : partitions)
    {
        p.update_wind_field(wac);
    }
}



void GPU_Implementation5::SplitIntoPartitionsAndTransferToDevice()
{
    // particle volume and mass
    hsd.prms.ComputeHelperVariables();
    hsd.prms.Printout();

    // allocate GPU partitions
    initialize();
    split_hssoa_into_partitions();
    allocate_device_arrays();
    transfer_to_device();
    LOGR("GPU_Implementation5::SplitIntoPartitionsAndTransferToDevice done\n");
}

