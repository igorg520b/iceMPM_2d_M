#ifndef PROXYPOINT2D_H
#define PROXYPOINT2D_H

#include <Eigen/Core>
#include <spdlog/spdlog.h>
#include "parameters_sim.h"

struct ProxyPoint
{
    constexpr static unsigned nArrays = SimParams::nPtsArrays;  // count of arrays in SOA
    bool isReference = false;
    unsigned pos, pitch;    // element # and capacity of each array in SOA
    t_PointReal *soa;            // reference to SOA (assume contiguous space of size nArrays*pitch)
    t_PointReal data[nArrays];    // local copy of the data when isReference==true

    ProxyPoint() { isReference = false; }

    ProxyPoint(const ProxyPoint &other);
    ProxyPoint& operator=(const ProxyPoint &other);

    // access data
    t_PointReal getValue(size_t valueIdx);   // valueIdx < nArrays
    void setValue(size_t valueIdx, t_PointReal value);
    uint32_t getValueInt(size_t valueIdx);
    void setValueInt(size_t valueIdx, uint32_t value);

    PointVector2r getPos();
    PointVector2r getPos(t_PointReal cellsize);
    PointVector2r getVelocity();
    bool getCrushedStatus();
    bool getDisabledStatus();
    bool getWeakenedStatus();
    uint16_t getGrain();

    int getCellIndex(int GridY);  // index of the grid cell at the point's location

    // other
    void ConvertToIntegerCellFormat(t_PointReal h);
};

#endif
