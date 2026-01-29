#ifndef PROXYPOINT2D_H
#define PROXYPOINT2D_H

#include <Eigen/Core>
#include <spdlog/spdlog.h>
#include "parameters_sim.h"

struct ProxyPoint
{
    constexpr static unsigned nArrays = SimParams::PtArrIdx::nPtsArrays;  // count of arrays in SOA
    bool isReference = false;
    unsigned pos, pitch;    // element # and capacity of each array in SOA
    double *soa;            // reference to SOA (assume contiguous space of size nArrays*pitch)
    double data[nArrays];    // local copy of the data when isReference==true

    ProxyPoint() { isReference = false; }

    ProxyPoint(const ProxyPoint &other);
    ProxyPoint& operator=(const ProxyPoint &other);

    // access data
    double getValue(size_t valueIdx);   // valueIdx < nArrays
    void setValue(size_t valueIdx, double value);
    uint32_t getValueInt(size_t valueIdx);
    void setValueInt(size_t valueIdx, uint32_t value);
    uint64_t getValueUInt64(size_t valueIdx);
    void setValueUInt64(size_t valueIdx, uint64_t value);
    Eigen::Matrix2f getTensor(size_t valueIdx);

    Eigen::Vector2d getPos();
    Eigen::Vector2d getPos(double cellsize);
    Eigen::Vector2d getVelocity();
    bool getCrushedStatus();
    bool getCrackedStatus();
    bool getDisabledStatus();
    uint16_t getGrain();

    bool getFractureTension();
    bool getFractureCompressionShear();
    bool getFractureCrush();

    long long getCellIndex(int GridY);  // index of the grid cell at the point's location
    unsigned getCellX();

    // other
    void ConvertToIntegerCellFormat(double h);
};

#endif
