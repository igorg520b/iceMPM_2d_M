#include "gpu_partition.h"
#include "gpu_implementation5.h"
#include <stdio.h>

#include "intinterval.h"


SimParams *GPU_Partition::prms;


GPU_Partition::GPU_Partition()
{
    initialized = false;
    error_code = 0;

    host_disabled_points_count = nullptr;
    pparams.count_pts = 0;
    pparams.partition_gridX = 0;
    pparams.gridX_offset = 0;

    pparams.buffer_grid = nullptr;
    pparams.buffer_pts = nullptr;
    pparams.buffer_grid_regions = nullptr;
    pparams.grid_forces_summary_per_region = nullptr;
    host_pud = nullptr;
    host_grid_forces_summary_per_region = nullptr;
}

GPU_Partition::~GPU_Partition()
{
    cudaSetDevice(Device);

    cudaEventDestroy(event_10_cycle_start);
    cudaEventDestroy(event_20_grid_halo_sent);
    cudaEventDestroy(event_30_halo_accepted);
    cudaEventDestroy(event_40_grid_updated);
    cudaEventDestroy(event_50_g2p_completed);
    cudaEventDestroy(event_70_pts_sent);
    cudaEventDestroy(event_80_pts_accepted);

    cudaStreamDestroy(streamCompute);

    cudaFree(pparams.buffer_grid);
    cudaFree(pparams.buffer_pts);
    cudaFree(pparams.buffer_grid_regions);
    cudaFree(pparams.pud);
    cudaFree(pparams.disabled_points_count);

    cudaFree(pparams.grid_forces_summary_per_region);

    cudaFree(pparams.halo_transfer_buffer[0]);
    cudaFree(pparams.halo_transfer_buffer[1]);

    cudaFreeHost(host_pud);
    cudaFreeHost(host_disabled_points_count);
    LOGR("Destructor invoked; partition {} on device {}", pparams.PartitionID, Device);
}


void GPU_Partition::initialize(int device, int partition)
{
    if(initialized) throw std::runtime_error("GPU_Partition double initialization");
    pparams.PartitionID = partition;
    this->Device = device;
    cudaSetDevice(Device);

    cudaEventCreate(&event_10_cycle_start);
    cudaEventCreate(&event_20_grid_halo_sent);
    cudaEventCreate(&event_30_halo_accepted);
    cudaEventCreate(&event_40_grid_updated);
    cudaEventCreate(&event_50_g2p_completed);
    cudaEventCreate(&event_70_pts_sent);
    cudaEventCreate(&event_80_pts_accepted);

    cudaError_t err = cudaStreamCreate(&streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition initialization failure");
    initialized = true;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    LOGR("Partition {}: initialized dev {}; compute {}.{}", pparams.PartitionID, Device, deviceProp.major, deviceProp.minor);
}


void GPU_Partition::allocate(const unsigned n_points_capacity, const unsigned gx_requested)
{
    CUDA_CHECK(cudaSetDevice(Device));

    const int &gy = prms->GridYTotal;
    const unsigned &halo = prms->GridHaloSize;

    LOGR("alloc P{0:}-{1:} alloc; subgrid {2:}x{3:}; sub-pts {4:}; offsetX {5:}; gridX {6:}",
         pparams.PartitionID, Device, gx_requested, gy, n_points_capacity, pparams.gridX_offset, pparams.partition_gridX);

    // grid
    size_t total_allocated = 0; // count what we allocated

    const size_t grid_requested = sizeof(t_GridReal) * gy * (gx_requested + 2*halo);
    CUDA_CHECK(cudaMallocPitch (&pparams.buffer_grid, &pparams.pitch_grid, grid_requested, SimParams::nGridArrays));
    total_allocated += pparams.pitch_grid * SimParams::nGridArrays;
    if(pparams.pitch_grid % sizeof(t_GridReal) != 0) throw std::runtime_error("pparams.pitch_grid % sizeof(t_GridReal) != 0");
    pparams.pitch_grid /= sizeof(t_GridReal); // assume that this divides without remainder
    pparams.gridX_alloc_capacity = gx_requested;

    // grid regions identifiers/indices
    const size_t grid_regions_size = sizeof(uint8_t) * gy * (gx_requested + 2*halo);
    CUDA_CHECK(cudaMalloc(&pparams.buffer_grid_regions, grid_regions_size));
    total_allocated += grid_regions_size;

    // small array where per-region forces will be accumulated
    CUDA_CHECK(cudaMalloc(&pparams.grid_forces_summary_per_region, sizeof(t_GridReal)*(SimParams::MAX_REGIONS*2)));

    // buffer for force transfer form gird
    CUDA_CHECK(cudaMallocHost(&host_grid_forces_summary_per_region, sizeof(t_GridReal)*SimParams::MAX_REGIONS*2));

    // points
    const size_t pts_buffer_requested = sizeof(t_PointReal) * n_points_capacity;
    CUDA_CHECK(cudaMallocPitch(&pparams.buffer_pts, &pparams.pitch_pts, pts_buffer_requested, SimParams::nPtsArrays));

    total_allocated += pparams.pitch_pts * SimParams::nPtsArrays;
    if(pparams.pitch_pts % sizeof(t_PointReal) != 0) throw std::runtime_error("pparams.pitch_pts % sizeof(t_PointReal) != 0");
    pparams.pitch_pts /= sizeof(t_PointReal);

    // points transfer buffer
    pparams.point_transfer_buffer_capacity = 0;
    if(prms->nPartitions > 1)
    {
        pparams.point_transfer_buffer_capacity = (size_t)(prms->points_transfer_buffer_fraction * n_points_capacity);
        pparams.point_transfer_buffer_capacity = std::max((size_t)100, pparams.point_transfer_buffer_capacity); // at least 100
        const size_t transfer_buffer_alloc_size = sizeof(t_PointReal)*SimParams::nPtsArrays*pparams.point_transfer_buffer_capacity;
        // point transfer buffers
        for(int i=0;i<4;i++)
        {
            CUDA_CHECK(cudaMalloc(&pparams.point_transfer_buffer[i], transfer_buffer_alloc_size));
            total_allocated += transfer_buffer_alloc_size;
        }
    }

    // utility data
    CUDA_CHECK(cudaMalloc(&pparams.pud, sizeof(PartitionParams::PartitionUtilityData)));
    CUDA_CHECK(cudaMemset(pparams.pud, 0, sizeof(PartitionParams::PartitionUtilityData)));

    CUDA_CHECK(cudaMalloc(&pparams.disabled_points_count, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(pparams.disabled_points_count, 0, sizeof(unsigned)));

    CUDA_CHECK(cudaMallocHost(&host_pud, sizeof(PartitionParams::PartitionUtilityData)));
    CUDA_CHECK(cudaMallocHost(&host_disabled_points_count, sizeof(unsigned)));
    *host_disabled_points_count = 0;

    // buffers for transferring gird data
    //pparams.transfer_buffer_width = (prms->GridXTotal + 2*halo);   // for testing
    pparams.transfer_buffer_width = 2*prms->GridHaloSize;
    size_t transfer_buffer_size = 3 * sizeof(t_GridReal) * pparams.transfer_buffer_width * prms->GridYTotal;
    for(int i=0;i<2;i++)
    {
        CUDA_CHECK(cudaMalloc(&pparams.halo_transfer_buffer[i], transfer_buffer_size));
    }
    total_allocated += transfer_buffer_size*2;

    LOGR("allocate: P {}-{}:  requested grid {} x {} = {}; gird pitch {}; Pts-req {}; pts-pitch {}; total alloc {:.2} Mb",
         pparams.PartitionID, Device, gx_requested, gy, gx_requested*gy,
         pparams.pitch_grid, n_points_capacity, pparams.pitch_pts,
         (double)total_allocated/(1024*1024));
}



// =========================================  GPU_Partition class



void GPU_Partition::transfer_from_device(HostSideSOA &hssoa, const int point_idx_offset)
{
    CUDA_CHECK(cudaSetDevice(Device));

    // transfer point data (partition fragment) with cudaMemcpy2DAsync
    const size_t dpitch = hssoa.capacity * sizeof(t_PointReal);
    const size_t spitch = pparams.pitch_pts * sizeof(t_PointReal);
    t_PointReal* const src = pparams.buffer_pts;
    t_PointReal* const dst = hssoa.host_buffer + point_idx_offset;
    const size_t width = pparams.count_pts * sizeof(t_PointReal);
    const size_t height = SimParams::nPtsArrays;
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost, streamCompute);

    // transfer error code
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(error_code), 0,
                                         cudaMemcpyDeviceToHost, streamCompute));

    // transfer the count of disabled points
    CUDA_CHECK(cudaMemcpyAsync(host_disabled_points_count, pparams.disabled_points_count,
                               sizeof(unsigned), cudaMemcpyDeviceToHost, streamCompute));

    // transfer accumulated forces
    const size_t transfer_bytes = sizeof(t_GridReal)*SimParams::MAX_REGIONS*2;
    CUDA_CHECK(cudaMemcpyAsync(host_grid_forces_summary_per_region,
                               pparams.grid_forces_summary_per_region,
                               transfer_bytes, cudaMemcpyDeviceToHost, streamCompute));
}


void GPU_Partition::check_error_code()
{
    CUDA_CHECK(cudaSetDevice(Device));

    // transfer error code
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpyFromSymbol(&error_code, gpu_error_indicator, sizeof(error_code), 0, cudaMemcpyDeviceToHost));
    if(error_code)
    {
        LOGR("error {:#x}", error_code);
        throw std::runtime_error("error code gpu");
    }
}



void GPU_Partition::transfer_points_from_soa_to_device(HostSideSOA &hssoa, int point_idx_offset)
{
    LOGR("PID {}, transfer_points_from_soa_to_device; offset {}", pparams.PartitionID, point_idx_offset);
    CUDA_CHECK(cudaSetDevice(Device));

    // transfer the partition region of points from HSSOA to device
    const size_t spitch = hssoa.capacity * sizeof(t_PointReal);
    const size_t dpitch = pparams.pitch_pts * sizeof(t_PointReal);
    t_PointReal* const dst = pparams.buffer_pts;
    t_PointReal* const src = hssoa.host_buffer + point_idx_offset;
    const size_t width = pparams.count_pts * sizeof(t_PointReal);
    const size_t height = SimParams::nPtsArrays;
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice, streamCompute);


    CUDA_CHECK(cudaMemset(pparams.disabled_points_count, 0, sizeof(unsigned)));
}


void GPU_Partition::transfer_grid_data_to_device(GPU_Implementation5* gpu)
{
    CUDA_CHECK(cudaSetDevice(Device));

    const int &gy = prms->GridYTotal;
    const unsigned &halo = prms->GridHaloSize;
    const int &gx_total = prms->GridXTotal;

    // Clear the entire GPU buffer including halos
    const size_t grid_regions_size = sizeof(uint8_t) * gy * (pparams.gridX_alloc_capacity + 2 * halo);
    CUDA_CHECK(cudaMemsetAsync(pparams.buffer_grid_regions, 0, grid_regions_size, streamCompute));

    IntInterval gpuBufferInterval((int)pparams.gridX_offset - halo,
                                  pparams.gridX_offset + pparams.partition_gridX + halo);

    IntInterval hostInterval(0, gx_total);

    IntInterval gpuInHost = gpuBufferInterval.intersect(hostInterval);
    int transfer_width = gpuInHost.size();
    int offset_gpu = gpuInHost.offset_within(gpuBufferInterval);
    int offset_host = gpuInHost.offset_within(hostInterval);

    const size_t transfer_size = transfer_width * gy * sizeof(uint8_t);

    // Set source and destination pointers
    const uint8_t* src = gpu->grid_status_buffer.data() + gy * offset_host;
    uint8_t* dst = pparams.buffer_grid_regions + gy * offset_gpu;

    CUDA_CHECK(cudaMemcpyAsync(dst, src, transfer_size, cudaMemcpyHostToDevice, streamCompute));

    LOGR("PID {}; transfer_grid_data_to_device; transfer_width {} (src_x {} → dst_x {})",
         pparams.PartitionID, transfer_width, offset_host, offset_gpu);
}


void GPU_Partition::update_current_field(const WindAndCurrentInterpolator &wac)
{
    CUDA_CHECK(cudaSetDevice(Device));

    const int &gy = prms->GridYTotal;
    const int &gx_total = prms->GridXTotal;
    const int halo = prms->GridHaloSize;

    IntInterval gpuBufferInterval((int)pparams.gridX_offset - halo,
                                  pparams.gridX_offset + pparams.partition_gridX + halo);

    IntInterval wacInterval(0, gx_total);

    IntInterval gpuInWAC = gpuBufferInterval.intersect(wacInterval);
    int transfer_width = gpuInWAC.size();
    int offset_gpu = gpuInWAC.offset_within(gpuBufferInterval);
    int offset_wac = gpuInWAC.offset_within(wacInterval);


    const size_t transfer_size = transfer_width * gy * sizeof(t_GridReal);

    // Source pointers (vx, vy are concatenated in the host array)
    const t_GridReal* src_vx = wac.current_flow_data.data() + gy * offset_wac;
    const t_GridReal* src_vy = src_vx + gx_total * gy;

    // Destination pointers in pitched GPU grid buffer
    t_GridReal* dst_vx = pparams.buffer_grid + pparams.pitch_grid * SimParams::grid_idx_current_vx + gy * offset_gpu;
    t_GridReal* dst_vy = pparams.buffer_grid + pparams.pitch_grid * SimParams::grid_idx_current_vy + gy * offset_gpu;

    CUDA_CHECK(cudaMemcpyAsync(dst_vx, src_vx, transfer_size, cudaMemcpyHostToDevice, streamCompute));
    CUDA_CHECK(cudaMemcpyAsync(dst_vy, src_vy, transfer_size, cudaMemcpyHostToDevice, streamCompute));

    LOGR("PID {}; offset{}; size {}; transfer_width {} (src_x_wac {} → dst_x {})",
         pparams.PartitionID, pparams.gridX_offset, pparams.partition_gridX,
         transfer_width, offset_wac, offset_gpu);
}



void GPU_Partition::update_constants()
{
    CUDA_CHECK(cudaSetDevice(Device));

    CUDA_CHECK(cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(error_code)));
    CUDA_CHECK(cudaMemcpyToSymbol(gprms, prms, sizeof(SimParams)));

    LOGR("Constant symbols copied to device {}; partition {}", Device, pparams.PartitionID);
}








// ============================================================= main simulation steps
void GPU_Partition::reset_grid()
{
    CUDA_CHECK(cudaSetDevice(Device));
    const size_t arrays_to_clear = 3;   // mass, px, py
    const size_t gridArraySize = pparams.pitch_grid * arrays_to_clear * sizeof(t_GridReal);
    CUDA_CHECK(cudaMemsetAsync(pparams.buffer_grid, 0, gridArraySize, streamCompute));
}

void GPU_Partition::clear_force_accumulator()
{
    CUDA_CHECK(cudaSetDevice(Device));
    const size_t arrays_to_clear = 2;   // fx, fy
    const size_t bytes_to_clear = pparams.pitch_grid * arrays_to_clear * sizeof(t_GridReal);
    CUDA_CHECK(cudaMemsetAsync(pparams.buffer_grid + pparams.pitch_grid*SimParams::grid_idx_fx, 0, bytes_to_clear, streamCompute));
}


void GPU_Partition::p2g()
{
    CUDA_CHECK(cudaSetDevice(Device));

    const int &n = pparams.count_pts;
    const int &tpb = prms->tpb_P2G;
    const int blocksPerGrid = (n + tpb - 1) / tpb;
    partition_kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>(pparams);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("p2g kernel");
//    check_error_code();
}

void GPU_Partition::update_nodes(float simulation_time, const float interpolation_coeff)
{
    CUDA_CHECK(cudaSetDevice(Device));
    const size_t nGridNodes = prms->GridYTotal * (pparams.partition_gridX + 2*prms->GridHaloSize);
    const int &tpb = prms->tpb_Upd;
    int nBlocks = (nGridNodes + tpb - 1) / tpb;

    partition_kernel_update_nodes<<<nBlocks, tpb, 0, streamCompute>>>(pparams, simulation_time);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("update_nodes");
//    check_error_code();
}

void GPU_Partition::g2p(const bool recordPQ)
{
    CUDA_CHECK(cudaSetDevice(Device));

    const size_t &n = pparams.count_pts;
    const int &tpb = prms->tpb_G2P;
    const int nBlocks = (n + tpb - 1) / tpb;

    partition_kernel_g2p<<<nBlocks, tpb, 0, streamCompute>>>(pparams, recordPQ);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p kernel");

//    check_error_code();
}




void GPU_Partition::record_timings()
{
    CUDA_CHECK(cudaSetDevice(Device));

    float _updateGrid, _gridResetAndHalo, _G2P, _total;
    CUDA_CHECK(cudaStreamSynchronize(streamCompute));

    CUDA_CHECK(cudaEventElapsedTime(&_gridResetAndHalo, event_10_cycle_start, event_20_grid_halo_sent));
    CUDA_CHECK(cudaEventElapsedTime(&_updateGrid, event_20_grid_halo_sent, event_40_grid_updated));
    CUDA_CHECK(cudaEventElapsedTime(&_G2P, event_40_grid_updated, event_50_g2p_completed));
    CUDA_CHECK(cudaEventElapsedTime(&_total, event_10_cycle_start, event_50_g2p_completed));

    timing_10_P2GAndHalo += _gridResetAndHalo;
    timing_30_updateGrid += _updateGrid;
    timing_40_G2P += _G2P;
    timing_stepTotal += _total;
}

void GPU_Partition::reset_timings()
{
    timing_10_P2GAndHalo = 0;
    timing_20_acceptHalo = 0;
    timing_30_updateGrid = 0;
    timing_40_G2P = 0;
    timing_60_ptsSent = 0;
    timing_70_ptsAccepted = 0;
    timing_stepTotal = 0;
}

void GPU_Partition::normalize_timings(int cycles)
{
    float coeff = (float)1000/(float)cycles;
    timing_10_P2GAndHalo *= coeff;
    timing_20_acceptHalo *= coeff;
    timing_30_updateGrid *= coeff;
    timing_40_G2P *= coeff;
    timing_60_ptsSent *= coeff;
    timing_70_ptsAccepted *= coeff;
    timing_stepTotal *= coeff;
}



// ============================================ multi-gpu

void GPU_Partition::receive_halos()
{
    CUDA_CHECK(cudaSetDevice(Device));


    const int &tpb = prms->tpb_Upd;   // threads per block
    const size_t elem_count = 2 * prms->GridHaloSize * prms->GridYTotal;
    const int blocksPerGrid = (elem_count + tpb - 1) / tpb;
    if(pparams.PartitionID != 0)
    partition_kernel_receive_subgrid<<<blocksPerGrid, tpb, 0, streamCompute>>>(pparams, 0,
                                                                               0, 2 * prms->GridHaloSize);
    if(pparams.PartitionID != prms->nPartitions-1)
    partition_kernel_receive_subgrid<<<blocksPerGrid, tpb, 0, streamCompute>>>(pparams, 1,
                                                                               pparams.partition_gridX,
                                                                               2 * prms->GridHaloSize);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) throw std::runtime_error("receive_halos kernel execution");
}


void GPU_Partition::evaluate_halo_diffusion()
{
    CUDA_CHECK(cudaSetDevice(Device));

    const size_t &n = pparams.count_pts;
    const int &tpb = prms->tpb_G2P;
    const int nBlocks = (n + tpb - 1) / tpb;

    CUDA_CHECK(cudaMemsetAsync(pparams.pud, 0, sizeof(PartitionParams::PartitionUtilityData), streamCompute));
    partition_kernel_check_if_transfer_needed<<<nBlocks, tpb, 0, streamCompute>>>(pparams);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("GPU_Partition::evaluate_halo_diffusion()");

    CUDA_CHECK(cudaMemcpyAsync(host_pud, pparams.pud, sizeof(PartitionParams::PartitionUtilityData),
                               cudaMemcpyDeviceToHost, streamCompute));
}


void GPU_Partition::send_points()
{
    CUDA_CHECK(cudaSetDevice(Device));

    const size_t &n = pparams.count_pts;
    const int &tpb = prms->tpb_G2P;
    const int nBlocks = (n + tpb - 1) / tpb;

    CUDA_CHECK(cudaMemsetAsync(pparams.pud, 0, sizeof(PartitionParams::PartitionUtilityData), streamCompute));
    partition_kernel_point_transfer<<<nBlocks, tpb, 0, streamCompute>>>(pparams);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("GPU_Partition::send_points()");

    CUDA_CHECK(cudaMemcpyAsync(host_disabled_points_count, pparams.disabled_points_count, sizeof(unsigned),
                               cudaMemcpyDeviceToHost, streamCompute));
    CUDA_CHECK(cudaMemcpyAsync(host_pud, pparams.pud, sizeof(PartitionParams::PartitionUtilityData),
                               cudaMemcpyDeviceToHost, streamCompute));

}


void GPU_Partition::receive_points(const unsigned fromLeft, const unsigned fromRight)
{
    CUDA_CHECK(cudaSetDevice(Device));

    // check for buffer overflow when receiving points
    if(fromLeft + fromRight + pparams.count_pts > pparams.pitch_pts)
    {
        LOGR("GPU_Partition::receive_points; PID {}; buffer overflow", pparams.PartitionID);
        this->error_code = 0xffff;
        return;
    }

    constexpr int tpb = 64;
    if(fromLeft)
    {
        const unsigned &n = fromLeft;
        const int nBlocks = (n + tpb - 1) / tpb;
        partition_kernel_receive_points<<<nBlocks, tpb, 0, streamCompute>>>(pparams, n, 2); // 2 is dst buffer on the left
        if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("receive_nodes kernel execution left");
        pparams.count_pts += n;     // account for additional incoming points
    }

    if(fromRight)
    {
        const unsigned &n = fromRight;
        const int nBlocks = (n + tpb - 1) / tpb;
        partition_kernel_receive_points<<<nBlocks, tpb, 0, streamCompute>>>(pparams, n, 3); // 3 is dst buffer on the right
        if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("receive_nodes kernel execution right");
        pparams.count_pts += n;
    }
}

// ================================

void GPU_Partition::render_visualized_data()
{
    CUDA_CHECK(cudaSetDevice(Device));

    // clear rendered arrays
    const size_t arrays_to_clear = 12;   // mass,...,grid_idx_vis_pts_density
    const size_t gridArraySize = pparams.pitch_grid * arrays_to_clear * sizeof(t_GridReal);
    //t_GridReal *ptr = pparams.getGridLine(SimParams::grid_idx_vis_r);
    CUDA_CHECK(cudaMemsetAsync(pparams.buffer_grid, 0, gridArraySize, streamCompute));

    // also clear force accumulator
    CUDA_CHECK(cudaMemsetAsync(pparams.grid_forces_summary_per_region, 0, sizeof(t_GridReal)*(SimParams::MAX_REGIONS+1), streamCompute));

    // render
    const int &n = pparams.count_pts;
    int &tpb = prms->tpb_P2G;
    const int blocksPerGrid = (n + tpb - 1) / tpb;
    partition_kernel_render_results<<<blocksPerGrid, tpb, 0, streamCompute>>>(pparams);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("render visualized data");

    // reduction operation on grid forces
    const size_t nGridNodes = prms->GridYTotal * (pparams.partition_gridX + 2*prms->GridHaloSize);
    const int &tpb2 = prms->tpb_Upd;
    const int nBlocks = (nGridNodes + tpb2 - 1) / tpb2;
    partition_kernel_summarize_forces<<<nBlocks, tpb2, 0, streamCompute>>>(pparams);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("partition_kernel_summarize_forces");
}
