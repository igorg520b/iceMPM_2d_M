#include "gpu_partition.h"
#include "gpu_implementation5.h"
#include <stdio.h>



SimParams *GPU_Partition::prms;


GPU_Partition::GPU_Partition()
{
    initialized = false;
    error_code = 0;
    tmp_accumulated_forces = nullptr;

    for(int k=0;k<8;k++) disabled_points_count[k]=0;
    pparams.count_pts = 0;
    pparams.partition_gridX = 0;
    pparams.gridX_offset = 0;

    pparams.buffer_grid = nullptr;
    pparams.buffer_pts = nullptr;
    pparams.buffer_grid_regions = nullptr;
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

    cudaFreeHost(tmp_accumulated_forces);
    LOGR("Destructor invoked; partition {} on device {}", pparams.PartitionID, Device);
}


// =========================================  GPU_Partition class



void GPU_Partition::transfer_from_device(HostSideSOA &hssoa, const int point_idx_offset, std::vector<t_GridReal> &boundary_forces)
{
    CUDA_CHECK(cudaSetDevice(Device));

    for(int j=0;j<SimParams::nPtsArrays;j++)
    {
        if((point_idx_offset + pparams.count_pts) > hssoa.capacity)
            throw std::runtime_error("transfer_from_device() HSSOA capacity");

        t_PointReal* const ptr_src = pparams.buffer_pts + j*pparams.pitch_pts;
        t_PointReal* const ptr_dst = hssoa.getPointerToLine(j)+point_idx_offset;

        CUDA_CHECK(cudaMemcpyAsync(ptr_dst, ptr_src, pparams.count_pts*sizeof(t_PointReal), cudaMemcpyDeviceToHost, streamCompute));
    }

    // transfer error code
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(error_code), 0,
                                         cudaMemcpyDeviceToHost, streamCompute));

    // transfer the count of disabled points
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&disabled_points_count, gpu_disabled_points_count,
                                         sizeof(gpu_disabled_points_count), 0, cudaMemcpyDeviceToHost, streamCompute));

    LOGR("sizeof(gpu_disabled_points_count) = {}",sizeof(gpu_disabled_points_count));

    // transfer grid data (accumulated forces) to temporary buffers
    const size_t tmp_bytes_transfer = 2*sizeof(t_GridReal)*pparams.pitch_grid;
    t_GridReal* const ptr_src = pparams.buffer_grid + pparams.pitch_grid*SimParams::grid_idx_fx;
    CUDA_CHECK(cudaMemcpyAsync(tmp_accumulated_forces, ptr_src, tmp_bytes_transfer, cudaMemcpyDeviceToHost, streamCompute));
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
    CUDA_CHECK(cudaSetDevice(Device));

    // due to the layout of host-side SOA, we transfer the pts arrays one-by-one
    for(int i=0;i<SimParams::nPtsArrays;i++)
    {
        t_PointReal* const ptr_dst = pparams.buffer_pts + i*pparams.pitch_pts;
        t_PointReal* const ptr_src = hssoa.getPointerToLine(i) + point_idx_offset;
        CUDA_CHECK(cudaMemcpyAsync(ptr_dst, ptr_src, pparams.count_pts*sizeof(t_PointReal),
                                   cudaMemcpyHostToDevice, streamCompute));
    }

    // write the disabled points count (a bit overcompicated since we use the array of 8 per device)

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        gpu_disabled_points_count,              // symbol
        &disabled_points_count[pparams.PartitionID],                                 // host pointer
        sizeof(unsigned),                       // size
        pparams.PartitionID * sizeof(unsigned),       // offset in bytes
        cudaMemcpyHostToDevice,                 // direction
        streamCompute                           // optional stream
        ));
}

void GPU_Partition::transfer_grid_data_to_device(GPU_Implementation5* gpu)
{
    CUDA_CHECK(cudaSetDevice(Device));
    const int &gy = prms->GridYTotal;
    const unsigned &halo = prms->GridHaloSize;

    // Clear the device buffer, including halos
    const size_t grid_regions_size = sizeof(uint8_t) * gy * (pparams.gridX_alloc_capacity + 2 * halo);
    CUDA_CHECK(cudaMemsetAsync(pparams.buffer_grid_regions, 0, grid_regions_size, streamCompute));

    // Determine copy parameters
    size_t transfer_width;
    size_t src_offset_x;
    size_t dst_offset_x;

    if (pparams.PartitionID == 0)
    {
        src_offset_x = 0;
        dst_offset_x = halo;

        transfer_width = (prms->nPartitions == 1)
                             ? pparams.partition_gridX
                             : pparams.partition_gridX + halo;
    }
    else
    {
        src_offset_x = pparams.gridX_offset - halo;
        dst_offset_x = 0;

        transfer_width = (pparams.PartitionID == prms->nPartitions - 1)
                             ? pparams.partition_gridX + halo
                             : pparams.partition_gridX + 2 * halo;
    }

    const uint8_t* src = gpu->grid_status_buffer.data() + gy * src_offset_x;
    uint8_t* dst = pparams.buffer_grid_regions + gy * dst_offset_x;

    const size_t transfer_size = transfer_width * gy * sizeof(uint8_t);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, transfer_size, cudaMemcpyHostToDevice, streamCompute));
}


void GPU_Partition::update_current_field(const WindAndCurrentInterpolator &wac)
{
    // transfer current velocity field from wac to device
    CUDA_CHECK(cudaSetDevice(Device));

    const int &gy = prms->GridYTotal;
    const int &gx = prms->GridXTotal;
    const unsigned &halo = prms->GridHaloSize;

    // Compute the transfer region
    size_t transfer_width;
    size_t src_offset_x;
    size_t dst_offset_x;


    if (pparams.PartitionID == 0)
    {
        src_offset_x = 0;
        dst_offset_x = halo;

        transfer_width = (prms->nPartitions == 1)
                             ? pparams.partition_gridX
                             : pparams.partition_gridX + halo;
    }
    else
    {
        src_offset_x = pparams.gridX_offset - halo;
        dst_offset_x = 0;

        transfer_width = (pparams.PartitionID == prms->nPartitions - 1)
                             ? pparams.partition_gridX + halo
                             : pparams.partition_gridX + 2 * halo;
    }

    const size_t transfer_size = transfer_width * gy * sizeof(t_GridReal);

    // Compute source and destination pointers for vx and vy
    const t_GridReal* src_vx = wac.current_flow_data.data() + gy * src_offset_x;
    const t_GridReal* src_vy = wac.current_flow_data.data() + gx * gy + gy * src_offset_x;

    t_GridReal* dst_vx = pparams.buffer_grid + pparams.pitch_grid * SimParams::grid_idx_current_vx + gy * dst_offset_x;
    t_GridReal* dst_vy = pparams.buffer_grid + pparams.pitch_grid * SimParams::grid_idx_current_vy + gy * dst_offset_x;

    CUDA_CHECK(cudaMemcpyAsync(dst_vx, src_vx, transfer_size, cudaMemcpyHostToDevice, streamCompute));
    CUDA_CHECK(cudaMemcpyAsync(dst_vy, src_vy, transfer_size, cudaMemcpyHostToDevice, streamCompute));
}



void GPU_Partition::update_constants()
{
    CUDA_CHECK(cudaSetDevice(Device));

    CUDA_CHECK(cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(error_code)));
    CUDA_CHECK(cudaMemcpyToSymbol(gprms, prms, sizeof(SimParams)));

    LOGR("Constant symbols copied to device {}; partition {}", Device, pparams.PartitionID);
}





void GPU_Partition::initialize(int device, int partition)
{
    if(initialized) throw std::runtime_error("GPU_Partition double initialization");
    pparams.PartitionID = partition;
    this->Device = device;
    disabled_points_count[pparams.PartitionID] = 0;
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
    LOGR("Partition {}: initialized dev {}; compute {}.{}",
         pparams.PartitionID, Device, deviceProp.major, deviceProp.minor);
}


void GPU_Partition::allocate(const unsigned n_points_capacity, const unsigned gx_requested)
{
    CUDA_CHECK(cudaSetDevice(Device));

    const int &gy = prms->GridYTotal;
    const unsigned &halo = prms->GridHaloSize;

    LOGR("P{0:}-{1:} allocate; sub-grid {2:} x {3:}; sub-pts {4:}",
         pparams.PartitionID, Device, gx_requested, gy, n_points_capacity);

    // grid
    size_t total_allocated = 0; // count what we allocated

    const size_t grid_requested = sizeof(t_GridReal) * gy * (gx_requested + 6*halo);
    CUDA_CHECK(cudaMallocPitch (&pparams.buffer_grid, &pparams.pitch_grid, grid_requested, SimParams::nGridArrays));
    total_allocated += pparams.pitch_grid * SimParams::nGridArrays;
    if(pparams.pitch_grid % sizeof(t_GridReal) != 0) throw std::runtime_error("pparams.pitch_grid % sizeof(t_GridReal) != 0");
    pparams.pitch_grid /= sizeof(t_GridReal); // assume that this divides without remainder
    pparams.gridX_alloc_capacity = gx_requested;

    pparams.halo_transfer_buffer[0] = pparams.buffer_grid + gy*(gx_requested+2*halo);
    pparams.halo_transfer_buffer[1] = pparams.buffer_grid + gy*(gx_requested+4*halo);

    // grid regions identifiers/indices
    const size_t grid_regions_size = sizeof(uint8_t) * gy * (gx_requested + 2*halo);
    CUDA_CHECK(cudaMalloc(&pparams.buffer_grid_regions, grid_regions_size));
    total_allocated += grid_regions_size;


    // device-side buffer for force transfer form gird
    // tmp_accumulated_forces
    cudaFreeHost(tmp_accumulated_forces);
    const size_t tmp_alloc_size = 2*sizeof(t_GridReal)*(pparams.pitch_grid);
    CUDA_CHECK(cudaMallocHost(&tmp_accumulated_forces, tmp_alloc_size));


    // points
    const size_t pts_buffer_requested = sizeof(t_PointReal) * n_points_capacity;
    CUDA_CHECK(cudaMallocPitch(&pparams.buffer_pts, &pparams.pitch_pts, pts_buffer_requested, SimParams::nPtsArrays));

    total_allocated += pparams.pitch_pts * SimParams::nPtsArrays;
    if(pparams.pitch_pts % sizeof(t_PointReal) != 0) throw std::runtime_error("pparams.pitch_pts % sizeof(t_PointReal) != 0");
    pparams.pitch_pts /= sizeof(t_PointReal);


    // points transfer buffer
    //points_transfer_buffer_fraction
    pparams.point_transfer_buffer_capacity = (size_t)(SimParams::points_transfer_buffer_fraction * n_points_capacity);
    pparams.point_transfer_buffer_capacity = std::max((size_t)100, pparams.point_transfer_buffer_capacity); // at least 100
    const size_t transfer_buffer_alloc_size = sizeof(t_PointReal)*SimParams::nPtsArrays*pparams.point_transfer_buffer_capacity;

    // point transfer buffers
    for(int i=0;i<4;i++)
    {
        CUDA_CHECK(cudaMalloc(&pparams.point_transfer_buffer[i], transfer_buffer_alloc_size));
        total_allocated += transfer_buffer_alloc_size;
    }

    LOGR("allocate: P {}-{}:  requested grid {} x {} = {}; gird pitch {}; Pts-req {}; pts-pitch {}; total alloc {:.2} Mb",
                 pparams.PartitionID, Device, gx_requested, gy, gx_requested*gy,
                 pparams.pitch_grid, n_points_capacity, pparams.pitch_pts,
                 (double)total_allocated/(1024*1024));
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
    const int gridX = prms->GridXTotal; // todo: change to gridx_partition

    const int &n = pparams.count_pts;
    const int &tpb = prms->tpb_P2G;
    const int blocksPerGrid = (n + tpb - 1) / tpb;
    partition_kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>(pparams);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("p2g kernel");
//    check_error_code();
}

void GPU_Partition::update_nodes(float simulation_time, const GridVector2r vWind, const float interpolation_coeff)
{
    CUDA_CHECK(cudaSetDevice(Device));

    const int &gy = prms->GridYTotal;
    const unsigned &halo = prms->GridHaloSize;
    const size_t nGridNodes = gy * (pparams.partition_gridX + 2*halo);

    int tpb = prms->tpb_Upd;
    int nBlocks = (nGridNodes + tpb - 1) / tpb;

    partition_kernel_update_nodes<<<nBlocks, tpb, 0, streamCompute>>>(pparams, simulation_time);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("update_nodes");
//    check_error_code();
}

void GPU_Partition::g2p(const bool recordPQ, const bool enablePointTransfer, int applyGlensLaw)
{
    CUDA_CHECK(cudaSetDevice(Device));

    const size_t &n = pparams.count_pts;
    const int &tpb = prms->tpb_G2P;
    const int nBlocks = (n + tpb - 1) / tpb;

    partition_kernel_g2p<<<nBlocks, tpb, 0, streamCompute>>>(pparams, recordPQ);

    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p kernel");
//    check_error_code();
}




void GPU_Partition::record_timings(const bool enablePointTransfer)
{
    CUDA_CHECK(cudaSetDevice(Device));

    float _gridResetAndHalo, _acceptHalo=0, _G2P, _total, _ptsSent, _ptsAccepted;
    float _updateGrid;
    cudaError_t err;
    err = cudaStreamSynchronize(streamCompute);
    if(err != cudaSuccess)
    {
        const char *errorString = cudaGetErrorString(err);
        LOGR("error string: {}", errorString);
        throw std::runtime_error("record_timings; cudaStreamSynchronize");
    }

    err = cudaEventElapsedTime(&_gridResetAndHalo, event_10_cycle_start, event_20_grid_halo_sent);
    if(err != cudaSuccess)
    {
        const char *errorString = cudaGetErrorString(err);
        LOGR("error string: {}",errorString);
        throw std::runtime_error("record_timings 1");
    }

    if(false)
    {
        err = cudaEventElapsedTime(&_acceptHalo, event_20_grid_halo_sent, event_30_halo_accepted);
        if(err != cudaSuccess) throw std::runtime_error("record_timings 2");
        err = cudaEventElapsedTime(&_updateGrid, event_30_halo_accepted, event_40_grid_updated);
        if(err != cudaSuccess) throw std::runtime_error("record_timings 3");
    }
    else
    {
        err = cudaEventElapsedTime(&_updateGrid, event_20_grid_halo_sent, event_40_grid_updated);
        if(err != cudaSuccess) throw std::runtime_error("record_timings 3");
    }


    err = cudaEventElapsedTime(&_G2P, event_40_grid_updated, event_50_g2p_completed);
    if(err != cudaSuccess) throw std::runtime_error("record_timings 4");

    if(enablePointTransfer)
    {
        err = cudaEventElapsedTime(&_ptsSent, event_50_g2p_completed, event_70_pts_sent);
        if(err != cudaSuccess) throw std::runtime_error("record_timings 6");
        err = cudaEventElapsedTime(&_ptsAccepted, event_70_pts_sent, event_80_pts_accepted);
        if(err != cudaSuccess) throw std::runtime_error("record_timings 7");

        err = cudaEventElapsedTime(&_total, event_10_cycle_start, event_80_pts_accepted);
        if(err != cudaSuccess) throw std::runtime_error("record_timings pts accepted");
    }
    else
    {
        _ptsSent = 0;
        _ptsAccepted = 0;

        err = cudaEventElapsedTime(&_total, event_10_cycle_start, event_50_g2p_completed);
        if(err != cudaSuccess) throw std::runtime_error("record_timings pts accepted");
    }

    timing_10_P2GAndHalo += _gridResetAndHalo;
    timing_20_acceptHalo += _acceptHalo;
    timing_30_updateGrid += _updateGrid;
    timing_40_G2P += _G2P;
    timing_60_ptsSent += _ptsSent;
    timing_70_ptsAccepted += _ptsAccepted;

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

