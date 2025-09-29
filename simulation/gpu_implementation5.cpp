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
    const unsigned &nPartitions = model->prms.nPartitions;

    // count available GPUs
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("cudaGetDeviceCount error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");
    LOGR("GPU_Implementation5::initialize; devic count {}",deviceCount);

    LOGV("Device Information:");
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
    partitions.resize(nPartitions);

    for(int i=0;i<nPartitions;i++)
    {
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
    LOGV("GPU_Implementation5::initialize() completed");
    LOGR("sizeof(PartitionParams) = {}\n", sizeof(PartitionParams));
}

void GPU_Implementation5::split_hssoa_into_partitions()
{
    LOGV("\nsplit_hssoa_into_partitions() start");
    const int &GridXTotal = model->prms.GridXTotal;
    const unsigned &nPartitions = model->prms.nPartitions;

    unsigned nPointsProcessed = 0;
    partitions[0].pparams.gridX_offset = 0;

    for(unsigned partition_idx=0; partition_idx<nPartitions; partition_idx++)
    {
        GPU_Partition &p = partitions[partition_idx];
        const unsigned nPartitionsRemaining = nPartitions - partition_idx;
        p.pparams.count_pts = (hssoa.size - nPointsProcessed)/nPartitionsRemaining; // points in this partition

        // find the index of the first point with x-index cellsIdx
        if(partition_idx < nPartitions-1)
        {
            SOAIterator it2 = hssoa.begin() + (nPointsProcessed + p.pparams.count_pts);
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
            SOAIterator it2 = hssoa.begin() + (j);
            int pt_idx = it2->getValueInt(SimParams::integer_point_idx);
            point_partitions[pt_idx] = (uint8_t)partition_idx;
        }

        nPointsProcessed += p.pparams.count_pts;
    }
    LOGV("split_hssoa_into_partitions() done");
    std::cout << std::endl;

    for(GPU_Partition &p : partitions)
    {
        LOGR("split: P {0:}; grid_offset {1:>7}; grid_size {2:>7}, npts {3:>8}",
             p.pparams.PartitionID, p.pparams.gridX_offset, p.pparams.partition_gridX, p.pparams.count_pts);
    }
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
    constexpr unsigned params_to_transfer = 3;  // mass, px, py
    const int &gridY = model->prms.GridYTotal;
    const unsigned &halo = model->prms.GridHaloSize;

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

        const size_t halo_count = 2*gridY*halo*sizeof(t_GridReal);
        // after P2G step is completed, transfer halos to adjacent partitions
        if(i!=0)
        {
            // send halo to the left
            GPU_Partition &pprev = partitions[i-1];
            for(int j=0;j<params_to_transfer;j++)
            {
                t_GridReal *src = p.pparams.getGridLine(j);
                t_GridReal *dst = pprev.pparams.halo_transfer_buffer[1] + j*pprev.pparams.transfer_buffer_width*gridY;
                CUDA_CHECK(cudaMemcpyPeerAsync(dst, pprev.Device, src, p.Device, halo_count, p.streamCompute));
            }
        }

        if(i!=(partitions.size()-1))
        {
            // send halo to the right
            GPU_Partition &pnxt = partitions[i+1];
            for(int j=0;j<params_to_transfer;j++)
            {
                t_GridReal *src = p.pparams.getGridLine(j) + gridY*p.pparams.partition_gridX;
                t_GridReal *dst = pnxt.pparams.halo_transfer_buffer[0] + j*pnxt.pparams.transfer_buffer_width*gridY;
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
//    float interpolation_coeff = model->wind_interpolator.interpolationCoeffFromTime(simulation_time);
    float interpolation_coeff_w = 0;

    for(GPU_Partition &p : partitions)
    {
        p.update_nodes(simulation_time, interpolation_coeff_w);
        CUDA_CHECK(cudaEventRecord(p.event_40_grid_updated, p.streamCompute));
    }
}

void GPU_Implementation5::g2p(const bool recordPQ)
{
    for(GPU_Partition &p : partitions)
    {
        p.g2p(recordPQ);
        CUDA_CHECK(cudaEventRecord(p.event_50_g2p_completed, p.streamCompute));
    }
}

void GPU_Implementation5::render_visualized_data()
{
    for(GPU_Partition &p : partitions)
    {
        p.render_visualized_data();
    }
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
                t_PointReal* const src_buffer = p.pparams.point_transfer_buffer[1];
                t_PointReal* const dst_buffer = pnxt.pparams.point_transfer_buffer[2];
                const unsigned &right_buffer_count = p.host_pud->transfer_to_right;
                size_t count = right_buffer_count*sizeof(t_PointReal)*SimParams::nPtsArrays;
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
                t_PointReal* const src_buffer = p.pparams.point_transfer_buffer[0];
                t_PointReal* const dst_buffer = pprev.pparams.point_transfer_buffer[3];
                const unsigned &left_buffer_count = p.host_pud->transfer_to_left;
                size_t count = left_buffer_count*sizeof(t_PointReal)*SimParams::nPtsArrays;
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



void GPU_Implementation5::allocate_host_arrays_grid()
{
    // grid sizes
    const int modeled_grid_total = model->prms.GridXTotal * model->prms.GridYTotal;
    const int initial_image_total = model->prms.InitializationImageSizeX * model->prms.InitializationImageSizeY;

    size_t allocation_estimate = 0;
    allocation_estimate += modeled_grid_total*sizeof(t_GridReal);
    allocation_estimate += 3*initial_image_total*sizeof(t_GridReal);
    allocation_estimate += modeled_grid_total*SimParams::nGridArrays*sizeof(t_GridReal);
    allocation_estimate += model->prms.GridYTotal * model->prms.GridHaloSize * SimParams::nGridArrays * sizeof(t_GridReal);
    LOGR("\nGPU_Implementation5::allocate_host_arrays_grid(); alloc estimate {} GB",(double)allocation_estimate/(1.e9));

    // allocate grid arrays
    grid_status_buffer.resize(modeled_grid_total);
    original_image_colors_rgb.resize(3*initial_image_total);
    host_grid_buffer.resize(modeled_grid_total*SimParams::nGridArrays);
    tmp_halo_buffer.resize(model->prms.GridYTotal * model->prms.GridHaloSize * SimParams::nGridArrays);
    LOGV("GPU_Implementation5::allocate_host_arrays_grid() completed");
}


void GPU_Implementation5::allocate_host_arrays_points()
{
    const size_t requested_capacity = (size_t)(double(model->prms.nPtsInitial)*(1.+model->prms.extra_space_pts));

    size_t allocation_estimate = 0;
    allocation_estimate += 2* sizeof(t_PointReal)*requested_capacity*SimParams::nPtsArrays;
    allocation_estimate += model->prms.nPtsInitial;
    LOGR("\nGPU_Implementation5::allocate_host_arrays_points(); alloc estimate {} GB",(double)allocation_estimate/(1.e9));

    hssoa.Allocate(requested_capacity);
    point_partitions.resize(model->prms.nPtsInitial);
}


void GPU_Implementation5::allocate_device_arrays()
{
    LOGV("GPU_Implementation5::allocate_device_arrays()");

    const unsigned &nPts = model->gpu.hssoa.size;
    const unsigned pts_reserve = (nPts/partitions.size()) * (1. + model->prms.extra_space_pts);

//    auto it = std::max_element(partitions.begin(), partitions.end(),
//                               [](const GPU_Partition &p1, const GPU_Partition &p2)
//                               {return p1.pparams.partition_gridX < p2.pparams.partition_gridX;});
//    size_t max_GridX_size = it->pparams.partition_gridX;
//    size_t GridX_size = std::min((size_t)(max_GridX_size*1.5), (size_t)model->prms.GridXTotal);

    for(GPU_Partition &p : partitions) p.allocate(pts_reserve, model->prms.GridXTotal);
    LOGV("allocate_device_arrays done");
    std::cout << std::endl;
}



void GPU_Implementation5::transfer_to_device()
{
    LOGV("GPU_Implementation: transfer_to_device()");

    int points_uploaded = 0;
    for(GPU_Partition &p : partitions)
    {
        p.transfer_points_from_soa_to_device(hssoa, points_uploaded);
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
        if(capacity_required > hssoa.capacity)
        {
            LOGR("transfer_from_device(): capacity {} exceeded ({}) when transferring P {}",
                             hssoa.capacity, capacity_required, p.pparams.PartitionID);
            throw std::runtime_error("transfer_from_device capacity exceeded");
        }
        p.transfer_from_device(hssoa, offset_pts);
        offset_pts += p.pparams.count_pts;
    }
    hssoa.size = (unsigned)offset_pts;

    transfer_grid_to_host();

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

    grid_forces_summary_per_region.fill(0.f);
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        for(int k=0;k<SimParams::MAX_REGIONS*2;k++)
            grid_forces_summary_per_region[k] += p.host_grid_forces_summary_per_region[k];
    }
}


void GPU_Implementation5::transfer_grid_to_host()
{
    const int gx_total = model->prms.GridXTotal;
    const int gy_total = model->prms.GridYTotal;
    const int halo = model->prms.GridHaloSize;
    const int nGridArrays = SimParams::nGridArrays - 2;
    const size_t grid_plane_size = (size_t)gx_total * gy_total;

    // Clear the host-side buffer.
    host_grid_buffer.assign(grid_plane_size * nGridArrays, 0.0f);

    // --- PHASE 1: Transfer all INTERIOR regions from all partitions ---
    // At the end of this loop, the buffer is fully populated, but the
    // halo overlap regions are not yet summed.
    for (int i = 0; i < partitions.size(); i++)
    {
        GPU_Partition &p = partitions[i];
        CUDA_CHECK(cudaSetDevice(p.Device));

        const t_GridReal* src_dev = p.pparams.buffer_grid + (size_t)gy_total * halo;
        t_GridReal* dst_host = host_grid_buffer.data() + (size_t)gy_total * p.pparams.gridX_offset;

        CUDA_CHECK(cudaMemcpy2D(
            dst_host,
            (size_t)gy_total * gx_total * sizeof(t_GridReal),
            src_dev,
            p.pparams.pitch_grid * sizeof(t_GridReal),
            (size_t)p.pparams.partition_gridX * gy_total * sizeof(t_GridReal),
            nGridArrays,
            cudaMemcpyDeviceToHost
            ));
    }

    // --- PHASE 2: Transfer and ADDITIVELY blend all HALO regions ---

    for (int i = 0; i < partitions.size(); i++)
    {
        GPU_Partition &p = partitions[i];
        CUDA_CHECK(cudaSetDevice(p.Device));

        // --- Transfer and ADD the LEFT halo ---
        // (This halo belongs to partition `i` but is added to the interior of `i-1`)
        if (i > 0)
        {
            const t_GridReal* src_dev = p.pparams.buffer_grid; // Left halo is at the start
            CUDA_CHECK(cudaMemcpy2D(
                tmp_halo_buffer.data(),
                (size_t)gy_total * halo * sizeof(t_GridReal),
                src_dev,
                p.pparams.pitch_grid * sizeof(t_GridReal),
                (size_t)gy_total * halo * sizeof(t_GridReal),
                nGridArrays,
                cudaMemcpyDeviceToHost
                ));

            size_t host_x_start = p.pparams.gridX_offset - halo;
            for (int k = 0; k < nGridArrays; k++) {
                for (int x = 0; x < halo; x++) {
                    for (int y = 0; y < gy_total; y++) {
                        size_t src_idx = (size_t)k * (gy_total * halo) + (size_t)x * gy_total + y;
                        size_t dst_idx = (size_t)k * grid_plane_size + (size_t)(host_x_start + x) * gy_total + y;
                        host_grid_buffer[dst_idx] += tmp_halo_buffer[src_idx];
                    }
                }
            }
        }

        // --- Transfer and ADD the RIGHT halo ---
        // (This halo belongs to partition `i` but is added to the interior of `i+1`)
        if (i < partitions.size() - 1)
        {
            const t_GridReal* src_dev = p.pparams.buffer_grid + (size_t)gy_total * (halo + p.pparams.partition_gridX);
            CUDA_CHECK(cudaMemcpy2D(
                tmp_halo_buffer.data(),
                (size_t)gy_total * halo * sizeof(t_GridReal),
                src_dev,
                p.pparams.pitch_grid * sizeof(t_GridReal),
                (size_t)gy_total * halo * sizeof(t_GridReal),
                nGridArrays,
                cudaMemcpyDeviceToHost
                ));

            size_t host_x_start = p.pparams.gridX_offset + p.pparams.partition_gridX;
            for (int k = 0; k < nGridArrays; k++) {
                for (int x = 0; x < halo; x++) {
                    for (int y = 0; y < gy_total; y++) {
                        size_t src_idx = (size_t)k * (gy_total * halo) + (size_t)x * gy_total + y;
                        size_t dst_idx = (size_t)k * grid_plane_size + (size_t)(host_x_start + x) * gy_total + y;
                        host_grid_buffer[dst_idx] += tmp_halo_buffer[src_idx];
                    }
                }
            }
        }
    }
    NormalizeGridDataOnHost();
}


void GPU_Implementation5::NormalizeGridDataOnHost()
{
    LOGV("Normalizing grid data on host...");

    // --- 1. Get grid dimensions and parameters for clarity ---
    const int gx = model->prms.GridXTotal;
    const int gy = model->prms.GridYTotal;
    const double cellsize_sq = model->prms.cellsize * model->prms.cellsize;
    const size_t grid_plane_size = (size_t)gx * gy;

    // --- 2. Define which grid planes (slices) need to be normalized by mass ---
    // This array makes the code cleaner and easier to maintain.
    const size_t planes_to_normalize[] = {
        SimParams::grid_idx_px,
        SimParams::grid_idx_py,
        SimParams::grid_idx_vis_r,
        SimParams::grid_idx_vis_g,
        SimParams::grid_idx_vis_b,
        SimParams::grid_idx_vis_Jpinv,
        SimParams::grid_idx_vis_P,
        SimParams::grid_idx_vis_Q,
        SimParams::grid_idx_vis_strain_EqvGreenLagrange,
        SimParams::grid_idx_vis_strain_vonMises
    };

    const size_t mass_plane_offset = grid_plane_size * SimParams::grid_idx_mass;

    // --- 3. Iterate over every grid node ---
    // The memory layout of host_grid_buffer is column-major.
#pragma omp parallel for
    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            const size_t idx = (size_t)j + (size_t)i * gy; // Column-major index

            // Get the mass at this node.
            const t_GridReal mass = host_grid_buffer[mass_plane_offset + idx];

            if (mass == 0) {
                continue;
            }

            // --- 4. Normalize all specified planes by the mass ---
            for (const auto& plane_idx : planes_to_normalize) {
                host_grid_buffer[grid_plane_size * plane_idx + idx] /= mass;
            }

            // --- 5. Finally, update the mass plane itself ---
            // This is done last so the original mass value could be used above.
            host_grid_buffer[mass_plane_offset + idx] /= cellsize_sq;
        }
    }
//    LOGV("Grid data normalization complete.");
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



void GPU_Implementation5::transfer_wind_and_current_data_to_device()
{
    LOGV("GPU_Implementation5::transfer_wind_and_current_data_to_device()");
    for(GPU_Partition &p : partitions)
    {
        p.update_current_field(model->wac_interpolator);
    }
}



// ===================================================== multi-gpu



/*

// ========================================= initialization and kernel execution

void CUDART_CB GPU_Implementation5::callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData)
{
    // simulation data was copied to host memory -> proceed with processing of this data
    GPU_Implementation5 *gpu = reinterpret_cast<GPU_Implementation5*>(userData);
    // any additional processing here
    if(gpu->transfer_completion_callback) gpu->transfer_completion_callback();
}
*/
