#include "gpu_partition.h"
#include "gpu_implementation5.h"
#include "helper_math.cuh"
#include "kernels.cuh"
#include <stdio.h>



SimParams *GPU_Partition::prms;



// =========================================  GPU_Partition class



void GPU_Partition::transfer_from_device(HostSideSOA &hssoa, int point_idx_offset, std::vector<t_GridReal> &boundary_forces)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("transfer_from_device() set");

    for(int j=0;j<SimParams::nPtsArrays;j++)
    {
        if((point_idx_offset + nPts_partition) > hssoa.capacity)
            throw std::runtime_error("transfer_from_device() HSSOA capacity");

        t_PointReal *ptr_src = pts_array + j*nPtsPitch;
        t_PointReal *ptr_dst = hssoa.getPointerToLine(j)+point_idx_offset;

        err = cudaMemcpyAsync(ptr_dst, ptr_src, nPts_partition*sizeof(t_PointReal), cudaMemcpyDeviceToHost, streamCompute);
        if(err != cudaSuccess)
        {
            const char *errorString = cudaGetErrorString(err);
            LOGR("error when copying points: {}", errorString);
            throw std::runtime_error("transfer_from_device() cudaMemcpyAsync points");
        }
    }

    // transfer error code
    err = cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(error_code), 0,
                                    cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("transfer_from_device");

    // transfer the count of disabled points
    err = cudaMemcpyFromSymbolAsync(&disabled_points_count, gpu_disabled_points_count,
                                    sizeof(disabled_points_count), 0, cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("transfer_from_device; disabled_points_count");


    // transfer grid data (accumulated forces)
    const size_t bytes_to_transfer = prms->GridYTotal * GridX_partition * sizeof(t_GridReal);
    for(int j=0;j<2;j++)
    {
        const t_GridReal *ptr_src = grid_array + nGridPitch*(SimParams::grid_idx_fx + j);
        t_GridReal *ptr_dest = boundary_forces.data() + prms->GridYTotal*(j*prms->GridXTotal + GridX_offset);
        err = cudaMemcpyAsync(ptr_dest, ptr_src, bytes_to_transfer, cudaMemcpyDeviceToHost, streamCompute);
    }
}


void GPU_Partition::check_error_code()
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("check_error_code() set");

    // transfer error code
    cudaDeviceSynchronize();
    err = cudaMemcpyFromSymbol(&error_code, gpu_error_indicator, sizeof(error_code), 0, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) throw std::runtime_error("transfer_from_device");
    if(error_code)
    {
        LOGR("error {:#x}", error_code);
        throw std::runtime_error("error code gpu");
    }
}



void GPU_Partition::transfer_points_from_soa_to_device(HostSideSOA &hssoa, int point_idx_offset)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_from_soa_to_device");

    // due to the layout of host-side SOA, we transfer the pts arrays one-by-one
    for(int i=0;i<SimParams::nPtsArrays;i++)
    {
        t_PointReal *ptr_dst = pts_array + i*nPtsPitch;
        t_PointReal *ptr_src = hssoa.getPointerToLine(i)+point_idx_offset;
        err = cudaMemcpyAsync(ptr_dst, ptr_src, nPts_partition*sizeof(t_PointReal), cudaMemcpyHostToDevice, streamCompute);
        if(err != cudaSuccess)
        {
            const char* errorString = cudaGetErrorString(err);
            LOGR("PID {}; line {}; nPts_partition {}, cuda error: {}",PartitionID, i, nPts_partition, errorString);
            throw std::runtime_error("transfer_points_from_soa_to_device");
        }
    }

    err = cudaMemcpyToSymbolAsync(gpu_disabled_points_count, &disabled_points_count,
                                              sizeof(disabled_points_count), 0,
                                              cudaMemcpyHostToDevice,streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("gpu_error_indicator initialization");

}

void GPU_Partition::transfer_grid_data_to_device(GPU_Implementation5* gpu)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("transfer_grid_data_to_device");

    // transfer status array (distinguishes between modelled area and land)
    size_t grid_array_size = prms->GridXTotal * prms->GridYTotal * sizeof(uint8_t);

    err = cudaMemcpyAsync(grid_status_array, gpu->grid_status_buffer.data(),
                          grid_array_size, cudaMemcpyHostToDevice,streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("transfer_grid_data_to_device");


    // transfer boundary normals (for slip collisions)
    const int &gxp = GridX_partition;
    const int &gx = prms->GridXTotal;
    const int &gy = prms->GridYTotal;

    const size_t transfer_size = gxp*gy*sizeof(t_GridReal);
    err = cudaMemcpyAsync(grid_array + nGridPitch * SimParams::grid_idx_bc_normal_nx,
                          gpu->grid_boundary_normals.data(),
                          transfer_size, cudaMemcpyHostToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("transfer_grid_data_to_device bc nx");

    err = cudaMemcpyAsync(grid_array + nGridPitch * SimParams::grid_idx_bc_normal_ny,
                          gpu->grid_boundary_normals.data() + gx*gy,
                          transfer_size, cudaMemcpyHostToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("transfer_grid_data_to_device bc ny");
}


void GPU_Partition::update_constants()
{
    cudaSetDevice(Device);
    cudaError_t err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(error_code));
    if(err != cudaSuccess) throw std::runtime_error("gpu_error_indicator initialization");
    err = cudaMemcpyToSymbol(gprms, prms, sizeof(SimParams));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");

    err = cudaMemcpyToSymbol(partition_buffer_pts, &pts_array, sizeof(t_PointReal*));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants");

    err = cudaMemcpyToSymbol(partition_buffer_grid, &grid_array, sizeof(t_GridReal*));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: grid_array");

    err = cudaMemcpyToSymbol(partition_gridX, &GridX_partition, sizeof(size_t));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: GridX_partition");

    err = cudaMemcpyToSymbol(partition_pitch_grid, &nGridPitch, sizeof(size_t));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: partition_pitch_grid");

    err = cudaMemcpyToSymbol(partition_count_pts, &nPts_partition, sizeof(size_t));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: partition_count_pts");

    err = cudaMemcpyToSymbol(partition_pitch_pts, &nPtsPitch, sizeof(size_t));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: partition_pitch_pts");

    err = cudaMemcpyToSymbol(partition_grid_status, &grid_status_array, sizeof(uint8_t*));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: partition_grid_status");

    LOGR("Constant symbols copied to device {}; partition {}", Device, PartitionID);
}



void GPU_Partition::update_current_field(const WindAndCurrentInterpolator &wac)
{
    // transfer current velocity field from wac to device

    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("update_current_field");

    const int &gxp = GridX_partition;
    const int &gx = prms->GridXTotal;
    const int &gy = prms->GridYTotal;

    const size_t transfer_size = gxp*gy*sizeof(t_GridReal);
    err = cudaMemcpyAsync(grid_array + nGridPitch * SimParams::grid_idx_current_vx,
                          wac.current_flow_data.data(),
                          transfer_size, cudaMemcpyHostToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("update_current_field current vx");

    err = cudaMemcpyAsync(grid_array + nGridPitch * SimParams::grid_idx_current_vy,
                          wac.current_flow_data.data() + gx*gy,
                          transfer_size, cudaMemcpyHostToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("update_current_field current vy");
}



GPU_Partition::GPU_Partition()
{
    initialized = false;
    error_code = 0;
    disabled_points_count = 0;
    nPts_partition = GridX_partition = 0;

    pts_array = nullptr;
    grid_array = nullptr;
    grid_status_array = nullptr;

    GridX_offset = 0;
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

    cudaFree(grid_array);
    cudaFree(pts_array);
    cudaFree(grid_status_array);
    LOGR("Destructor invoked; partition {} on device {}", PartitionID, Device);
}

void GPU_Partition::initialize(int device, int partition)
{
    if(initialized) throw std::runtime_error("GPU_Partition double initialization");
    this->PartitionID = partition;
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
    LOGR("Partition {}: initialized dev {}; compute {}.{}", PartitionID, Device,deviceProp.major, deviceProp.minor);
}


void GPU_Partition::allocate(const int n_points_capacity, const int gx)
{
    cudaError_t err;
    cudaSetDevice(Device);

    const int &gy = prms->GridYTotal;
    LOGR("P{}-{} allocate; sub-grid {} x {}; sub-pts", PartitionID, Device, gx, gy, n_points_capacity);

    // grid
    size_t total_allocated = 0; // count what we allocated
    size_t grid_size_local_requested = sizeof(t_GridReal) * gy * gx;
    err = cudaMallocPitch (&grid_array, &nGridPitch, grid_size_local_requested, SimParams::nGridArrays);
    total_allocated += nGridPitch * SimParams::nGridArrays;
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate grid array");
    nGridPitch /= sizeof(t_GridReal); // assume that this divides without remainder

    // grid status
    err = cudaMalloc(&grid_status_array, gx*gy*sizeof(uint8_t));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate grid status array");
    total_allocated += gx*gy;

    // points
    const size_t pts_buffer_requested = sizeof(t_PointReal) * n_points_capacity;
    err = cudaMallocPitch(&pts_array, &nPtsPitch, pts_buffer_requested, SimParams::nPtsArrays);
    total_allocated += nPtsPitch * SimParams::nPtsArrays;
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    nPtsPitch /= sizeof(t_PointReal);

    LOGR("allocate: P {}-{}:  requested grid {} x {} = {}; gird pitch {}; Pts-req {}; pts-pitch {}; total alloc {:.2} Mb",
                 PartitionID, Device,
                 gx, gy, gx*gy,
                 nGridPitch,
                 n_points_capacity, nPtsPitch,
                 (double)total_allocated/(1024*1024));
}





// ============================================================= main simulation steps
void GPU_Partition::reset_grid()
{
    cudaError_t err;
    cudaSetDevice(Device);
    const size_t arrays_to_clear = 3;   // mass, px, py
    size_t gridArraySize = nGridPitch * arrays_to_clear * sizeof(t_GridReal);
    err = cudaMemsetAsync(grid_array, 0, gridArraySize, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Partition::clear_force_accumulator()
{
    cudaError_t err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition::clear_force_accumulator(): cudaSetDevice");

    const size_t arrays_to_clear = 2;   // fx, fy
    size_t bytes_to_clear = nGridPitch * arrays_to_clear * sizeof(t_GridReal);
    err = cudaMemsetAsync(grid_array + nGridPitch*SimParams::grid_idx_fx, 0, bytes_to_clear, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("clear_force_accumulator() cudaMemsetAsync");
}


void GPU_Partition::p2g()
{
    cudaSetDevice(Device);
    const int gridX = prms->GridXTotal; // todo: change to gridx_partition

    const int &n = nPts_partition;
    const int &tpb = prms->tpb_P2G;
    const int blocksPerGrid = (n + tpb - 1) / tpb;
    partition_kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("p2g kernel");

//    check_error_code();
}

void GPU_Partition::update_nodes(float simulation_time, const GridVector2r vWind, const float interpolation_coeff)
{
    cudaSetDevice(Device);
    const int nGridNodes = prms->GridYTotal * (GridX_partition);

    int tpb = prms->tpb_Upd;
    int nBlocks = (nGridNodes + tpb - 1) / tpb;

    partition_kernel_update_nodes<<<nBlocks, tpb, 0, streamCompute>>>(simulation_time);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("update_nodes");

//    check_error_code();
}

void GPU_Partition::g2p(const bool recordPQ, const bool enablePointTransfer, int applyGlensLaw)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("g2p cudaSetDevice");

    const int &n = nPts_partition;
    const int &tpb = prms->tpb_G2P;
    const int nBlocks = (n + tpb - 1) / tpb;

    partition_kernel_g2p<<<nBlocks, tpb, 0, streamCompute>>>(recordPQ);

    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p kernel");

//    check_error_code();
}




void GPU_Partition::record_timings(const bool enablePointTransfer)
{
    float _gridResetAndHalo, _acceptHalo=0, _G2P, _total, _ptsSent, _ptsAccepted;
    float _updateGrid;
    cudaSetDevice(Device);
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

