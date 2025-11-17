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
                                                                         [k](ProxyPoint &p1, ProxyPoint &p2)
                                                                         {return p1.getValue(SimParams::PtArrIdx::posx+k)<p2.getValue(SimParams::PtArrIdx::posx+k);});
        result[0][k] = (*it_res.first).getValue(SimParams::PtArrIdx::posx+k);
        result[1][k] = (*it_res.second).getValue(SimParams::PtArrIdx::posx+k);
    }
    return {result[0], result[1]};
}


void HostSideSOA::convertToIntegerCellFormat(double h)
{
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint &p = *it;
        p.ConvertToIntegerCellFormat(h);
    }
}



void HostSideSOA::RemoveDisabledAndSort(int GridY)
{
    LOGR("RemoveDisabledAndSort; nPtsArrays {}", SimParams::PtArrIdx::nPtsArrays);
    unsigned size_before = size;
    SOAIterator it_result = std::remove_if(begin(), end(), [](ProxyPoint &p){return p.getDisabledStatus();});
    size = it_result.m_point.pos;
    LOGR("RemoveDisabledAndSort: {} removed; new size {}", size_before-size, size);
    std::sort(begin(), end(),
              [&](ProxyPoint &p1, ProxyPoint &p2)
              {return p1.getCellIndex(GridY)<p2.getCellIndex(GridY);});
    LOGR("RemoveDisabledAndSort done");
}



void HostSideSOA::Allocate(int pts_capacity)
{
    LOGR("HostSideSOA::Allocate; pts {}", pts_capacity);
    capacity = pts_capacity;
    size = 0;
    size_t allocation_size = sizeof(double)*capacity*SimParams::PtArrIdx::nPtsArrays;
    size_t allocation_elems = capacity*SimParams::PtArrIdx::nPtsArrays;

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
    LOGR("HSSOA allocate capacity {} pt; toal {} Gb", capacity, (double)allocation_size/(1024.*1024.*1024.));
}



void HostSideSOA::PrintOutNearestPoint(double x, double y, double gridSize, int GridY)
{
    // calculate cell index

    const double hinv = 1.0f/gridSize;
    uint32_t x_idx = (uint32_t)(x*hinv + 0.5);
    uint32_t y_idx = (uint32_t)(y*hinv + 0.5);
    int cellIdx = x_idx*GridY + y_idx;

    auto result = std::lower_bound(begin(), end(),cellIdx,
                     [&](ProxyPoint &p1, int cell)
                     {return p1.getCellIndex(GridY)<cell;});

    while(result!=end() && (*result).getCellIndex(GridY) == cellIdx)
    {
        LOGR("Point index {}", (*result).getValueInt(SimParams::PtArrIdx::integer_point_idx));
        LOGR("P: {}", (*result).getValue(SimParams::PtArrIdx::idx_P));

        Eigen::Matrix2d Fe;
        for(int i=0; i<SimParams::dim; i++)
        for(int j=0; j<SimParams::dim; j++)
            Fe(i,j) = (*result).getValue(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j);

        LOGR("Fe: [{},{}],[{},{}]\n",Fe(0,0), Fe(0,1), Fe(1,0), Fe(1,1));
        ++result;
    }
}



// ==================================================== SOAIterator

SOAIterator::SOAIterator(unsigned pos, double *soa_data, unsigned pitch)
{
    m_point.isReference = true;
    m_point.pos = pos;
    m_point.soa = soa_data;
    m_point.pitch = pitch;
}

SOAIterator::SOAIterator(const SOAIterator& other)
{
    m_point.isReference = other.m_point.isReference;
    m_point.pos = other.m_point.pos;
    m_point.soa = other.m_point.soa;
    m_point.pitch = other.m_point.pitch;
}

SOAIterator& SOAIterator::operator=(const SOAIterator& other)
{
    m_point.isReference = other.m_point.isReference;
    m_point.pos = other.m_point.pos;
    m_point.soa = other.m_point.soa;
    m_point.pitch = other.m_point.pitch;
    return *this;
}


