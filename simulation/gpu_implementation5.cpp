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
}

void GPU_Implementation5::split_hssoa_into_partitions()
{
    LOGV("\nsplit_hssoa_into_partitions() start");
    const int &GridXTotal = model->prms.GridXTotal;
    const unsigned &nPartitions = model->prms.nPartitions;

    unsigned nPointsProcessed = 0;
    partitions[0].GridX_offset = 0;

    for(unsigned partition_idx=0; partition_idx<nPartitions; partition_idx++)
    {
        GPU_Partition &p = partitions[partition_idx];
        p.disabled_points_count = 0;
        const unsigned nPartitionsRemaining = nPartitions - partition_idx;
        p.nPts_partition = (hssoa.size - nPointsProcessed)/nPartitionsRemaining; // points in this partition

        // find the index of the first point with x-index cellsIdx
        if(partition_idx < nPartitions-1)
        {
            SOAIterator it2 = hssoa.begin() + (nPointsProcessed + p.nPts_partition);
            const unsigned cellsIdx = it2->getCellX();
            p.GridX_partition = cellsIdx - p.GridX_offset;
            partitions[partition_idx+1].GridX_offset = cellsIdx;
        }
        else if(partition_idx == nPartitions-1)
        {
            // the last partition spans the rest of the grid along the x-axis
            p.GridX_partition = GridXTotal - p.GridX_offset;
        }

#pragma omp parallel for
        for(int j=nPointsProcessed;j<(nPointsProcessed+p.nPts_partition);j++)
        {
            point_partitions[j] = (uint8_t)partition_idx;
        }

        LOGR("split: P {}; grid_offset {}; grid_size {}, npts {}",
                     partition_idx, p.GridX_offset, p.GridX_partition, p.nPts_partition);
        nPointsProcessed += p.nPts_partition;
    }
    LOGV("split_hssoa_into_partitions() done");
}




void GPU_Implementation5::reset_grid()
{
    for(GPU_Partition &p : partitions)
    {
        cudaError_t err = cudaSetDevice(p.Device);
        if(err != cudaSuccess) throw std::runtime_error("reset_grid set device");
        err = cudaEventRecord(p.event_10_cycle_start, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("reset_grid event");
        p.reset_grid();
    }
}

void GPU_Implementation5::clear_force_accumulator()
{
    for(GPU_Partition &p : partitions)
    {
        cudaError_t err = cudaSetDevice(p.Device);
        if(err != cudaSuccess) throw std::runtime_error("reset_grid set device");
        p.clear_force_accumulator();
    }
}


void GPU_Implementation5::p2g()
{
    cudaError_t err;

    partitions.front().p2g();
    cudaEventRecord(partitions.front().event_20_grid_halo_sent, partitions.front().streamCompute);
}



void GPU_Implementation5::update_nodes(float simulation_time, float windSpeed, float windAngle)
{
//    float interpolation_coeff = model->wind_interpolator.interpolationCoeffFromTime(simulation_time);
    float interpolation_coeff_w = 0;

//    float windSpeed = std::min(0+simulation_time*(40./5.e5), 40.0);
    float alphaRad = (windAngle + 180.0) * M_PI / 180.0;

    // Compute the x and y components
    float vx = windSpeed * std::sin(alphaRad); // Eastward component
    float vy = windSpeed * std::cos(alphaRad); // Northward component

    GridVector2r vWind(vx,vy);

    for(GPU_Partition &p : partitions)
    {
        p.update_nodes(simulation_time, vWind, interpolation_coeff_w);
        cudaError_t err = cudaEventRecord(p.event_40_grid_updated, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("update_nodes cudaEventRecord");
    }
}

void GPU_Implementation5::g2p(const bool recordPQ, const bool enablePointTransfer, int applyGlensLaw)
{
    for(GPU_Partition &p : partitions)
    {
        p.g2p(recordPQ, enablePointTransfer, applyGlensLaw);
        cudaError_t err = cudaEventRecord(p.event_50_g2p_completed, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("g2p cudaEventRecord");
    }
}


void GPU_Implementation5::record_timings(const bool enablePointTransfer)
{
    for(GPU_Partition &p : partitions) p.record_timings(enablePointTransfer);
}



// ==========================================================================



void GPU_Implementation5::allocate_host_arrays_grid()
{
    // grid sizes
    const int modeled_grid_total = model->prms.GridXTotal * model->prms.GridYTotal;
    const int initial_image_total = model->prms.InitializationImageSizeX * model->prms.InitializationImageSizeY;

    // allocate grid arrays
    grid_status_buffer.resize(modeled_grid_total);
    grid_boundary_normals.resize(2*modeled_grid_total);
    original_image_colors_rgb.resize(3*initial_image_total);
    grid_boundary_forces.resize(2*modeled_grid_total);

    LOGV("GPU_Implementation5::allocate_host_arrays_grid() completed");
}


void GPU_Implementation5::allocate_host_arrays_points()
{
    hssoa.Allocate(model->prms.nPtsInitial);
    point_colors_rgb.resize(model->prms.nPtsInitial);
    point_partitions.resize(model->prms.nPtsInitial);
}


void GPU_Implementation5::allocate_device_arrays()
{
    LOGV("GPU_Implementation5::allocate_device_arrays()");

    const unsigned &nPts = model->gpu.hssoa.size;
    const unsigned pts_reserve = (nPts/partitions.size()) * (1. + SimParams::extra_space_pts);

    auto it = std::max_element(partitions.begin(), partitions.end(),
                               [](const GPU_Partition &p1, const GPU_Partition &p2)
                               {return p1.GridX_partition < p2.GridX_partition;});
    unsigned max_GridX_size = it->GridX_partition;
    unsigned GridX_size = std::min((unsigned)(max_GridX_size*1.5), (unsigned)model->prms.GridXTotal);
    for(GPU_Partition &p : partitions) p.allocate(pts_reserve, GridX_size);
}



void GPU_Implementation5::transfer_to_device()
{
    LOGV("GPU_Implementation: transfer_to_device()");

    int points_uploaded = 0;
    for(GPU_Partition &p : partitions)
    {
        p.transfer_points_from_soa_to_device(hssoa, points_uploaded);
        p.transfer_grid_data_to_device(this);
        points_uploaded += p.nPts_partition;
    }
    spdlog::info("transfer_ponts_to_device() done; transferred points {}", points_uploaded);
}



void GPU_Implementation5::transfer_from_device(const int elapsed_cycles)
{
    unsigned offset_pts = 0;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        int capacity_required = offset_pts + p.nPts_partition;
        if(capacity_required > hssoa.capacity)
        {
            LOGR("transfer_from_device(): capacity {} exceeded ({}) when transferring P {}",
                             hssoa.capacity, capacity_required, p.PartitionID);
            throw std::runtime_error("transfer_from_device capacity exceeded");
        }
        p.transfer_from_device(hssoa, offset_pts, grid_boundary_forces);
        offset_pts += p.nPts_partition;
    }
    hssoa.size = offset_pts;

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
            LOGR("P {}; error code {}; this error code {}", p.PartitionID, p.error_code, this->error_code);
        }
    }
#pragma omp parallel for
    for(size_t i=0;i<grid_boundary_forces.size();i++)
        grid_boundary_forces[i]/=(elapsed_cycles*model->prms.InitialTimeStep);

    if(transfer_completion_callback) transfer_completion_callback();
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
