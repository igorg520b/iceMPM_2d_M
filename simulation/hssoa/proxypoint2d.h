#ifndef PROXYPOINT2D_H
#define PROXYPOINT2D_H

#include <Eigen/Core>
#include <spdlog/spdlog.h>
#include "parameters_sim.h"

struct ProxyPoint
{
    constexpr static unsigned nArrays = SimParams::PtArrIdx::nPtsArrays;  // count of arrays in SOA
    bool isReference = false;
    uint64_t pos, pitch;    // element # and capacity of each array in SOA
    double *soa;            // reference to SOA (assume contiguous space of size nArrays*pitch)
    double data[nArrays];    // local copy of the data when isReference==true

    ProxyPoint() { isReference = false; }
    ProxyPoint(uint64_t pos, double *soa, uint64_t pitch);

    ProxyPoint(const ProxyPoint &other) {
        // Essential Change for Swap-by-Value: Preserve Reference Nature!
        // If other is a Reference, we become a Reference to the same data (Handle Copy).
        // If other is a Value, we become a Value Copy.
        isReference = other.isReference;
        if (isReference) {
             pos = other.pos;
             soa = other.soa;
             pitch = other.pitch;
             // Do NOT copy data[], we point to SOA.
        } else {
             // Deep copy of data
             for(int i=0; i<nArrays; ++i) data[i] = other.data[i];
        }
    }
    ProxyPoint& operator=(const ProxyPoint &other);

    // access data
    double getValue(size_t valueIdx);   // valueIdx < nArrays
    void setValue(size_t valueIdx, double value);
    uint64_t getValueUInt64(size_t valueIdx);
    void setValueUInt64(size_t valueIdx, uint64_t value);
    Eigen::Matrix2f getTensor(size_t valueIdx);

    Eigen::Vector2d getPos();
    Eigen::Vector2d getPos(double cellsize);
    Eigen::Vector2d getVelocity();
    bool getCrushedStatus();
    bool getCrackedStatus();
    bool getDisabledStatus();

    bool getFractureTension();
    bool getFractureCompressionShear();
    bool getFractureCrush();

    long long getCellIndex(int GridY);  // index of the grid cell at the point's location
    unsigned getCellX();

    // other
    void ConvertToIntegerCellFormat(double h);

    friend void swap(ProxyPoint a, ProxyPoint b) {
        // We must perform a deep swap of the underlying data in the SOA.
        // 'a' and 'b' are local proxies acting as references to the SOA slots.
        
        // Use default constructor to create a value-holding proxy (isReference=false)
        ProxyPoint temp; 
        
        // Snapshot a's data into temp
        temp = a;
        
        // Write b's data into a's SOA slot
        a = b;
        
        // Write temp's data (original a) into b's SOA slot
        b = temp;
    }
};

#endif
