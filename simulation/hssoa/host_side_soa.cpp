#include "host_side_soa.h"



HostSideSOA::~HostSideSOA()
{
    delete[] host_buffer;
}



std::pair<Eigen::Vector2d, Eigen::Vector2d> HostSideSOA::getBlockDimensions()
{
    Eigen::Vector2d result[2];
    for(int k=0;k<SimParams::dim;k++)
    {
        std::pair<SOAIterator, SOAIterator> it_res = std::minmax_element(begin(), end(),
                                                                         [k](ProxyPoint p1, ProxyPoint p2)
                                                                         {return p1.getValue(SimParams::PtArrIdx::posx+k)<p2.getValue(SimParams::PtArrIdx::posx+k);});
        result[0][k] = (*it_res.first).getValue(SimParams::PtArrIdx::posx+k);
        result[1][k] = (*it_res.second).getValue(SimParams::PtArrIdx::posx+k);
    }
    return {result[0], result[1]};
}



void HostSideSOA::RemoveDisabledAndSort(int GridY)
{
    LOGR("RemoveDisabledAndSort; nPtsArrays {}", (int)SimParams::PtArrIdx::nPtsArrays);
    uint64_t size_before = size;
    SOAIterator it_result = std::remove_if(begin(), end(), [](ProxyPoint p){return p.getDisabledStatus();});
    size = it_result.pos;
    LOGR("RemoveDisabledAndSort: {} removed; new size {}", size_before-size, size);
    std::sort(begin(), end(),
              [&](ProxyPoint p1, ProxyPoint p2)
              {return p1.getCellIndex(GridY)<p2.getCellIndex(GridY);});

    LOGR("Verifying Sort inside RemoveDisabledAndSort...");
    
    // 1. std::is_sorted check
    bool sorted = std::is_sorted(begin(), end(),
              [&](ProxyPoint p1, ProxyPoint p2)
              {return p1.getCellIndex(GridY) < p2.getCellIndex(GridY);});

    if (!sorted) {
        LOGR("std::is_sorted FAILED");
        throw std::runtime_error("Critical Error: HostSideSOA::RemoveDisabledAndSort failed to sort points.");
    }

        
    LOGR("RemoveDisabledAndSort done");
}



void HostSideSOA::Allocate(uint64_t pts_capacity)
{
    LOGR("HostSideSOA::Allocate; pts {}", pts_capacity);
    capacity = pts_capacity;
    size = 0;
    size_t allocation_size = sizeof(double) * (size_t)capacity * (size_t)SimParams::PtArrIdx::nPtsArrays;
    size_t allocation_elems = (size_t)capacity * (size_t)SimParams::PtArrIdx::nPtsArrays;

    delete[] host_buffer;
    host_buffer = new double[allocation_elems];

    /*
    cudaFreeHost(host_buffer);
    cudaError_t err = cudaMallocHost(&host_buffer, allocation_size);
    if(err != cudaSuccess)
    {
        const char *description = cudaGetErrorString(err);
        LOGR("allocating host buffer of size {}: {}",allocation_size,description);
        LOGR("nPtsArrays {}; sizeof(double) {}", SimParams::PtArrIdx::nPtsArrays, sizeof(double));
        throw std::runtime_error("allocating host buffer for points");
    }
*/

    memset(host_buffer, 0, allocation_size);
    // LOGR("HSSOA allocate capacity {} pt; toal {} GB", capacity, (double)allocation_size/(1024.*1024.*1024.));
}







// ==================================================== SOAIterator

SOAIterator::SOAIterator(uint64_t pos, double *soa_data, uint64_t pitch)
    : pos(pos), soa(soa_data), pitch(pitch)
{
}

SOAIterator::SOAIterator(const SOAIterator& other)
    : pos(other.pos), soa(other.soa), pitch(other.pitch)
{
}

SOAIterator& SOAIterator::operator=(const SOAIterator& other)
{
    pos = other.pos;
    soa = other.soa;
    pitch = other.pitch;
    return *this;
}


