#include "proxypoint2d.h"
#include <cstring>

// ====================================================== ProxyPoint
ProxyPoint::ProxyPoint(uint64_t pos, double *soa, uint64_t pitch)
    : pos(pos), pitch(pitch), soa(soa), isReference(true)
{
}



ProxyPoint& ProxyPoint::operator=(const ProxyPoint &other)
{
    if(isReference)
    {
        // distribute into soa
        if(other.isReference)
        {
            for(size_t i=0; i<nArrays; i++) soa[pos + i*(size_t)pitch] = other.soa[other.pos + i*(size_t)other.pitch];
        }
        else
        {
            for(size_t i=0; i<nArrays; i++) soa[pos + i*(size_t)pitch] = other.data[i];
        }
    }
    else
    {
        // local copy
        if(other.isReference)
        {
            for(size_t i=0; i<nArrays; i++) data[i] = other.soa[other.pos + i*(size_t)other.pitch];
        }
        else
        {
            for(size_t i=0; i<nArrays; i++) data[i] = other.data[i];
        }
    }
    return *this;
}

Eigen::Vector2d ProxyPoint::getPos()
{
    Eigen::Vector2d result;
    for(int i=0;i<SimParams::dim;i++) result[i] = getValue(SimParams::PtArrIdx::posx+i);
    if(std::abs(result.x()) > 0.5 || std::abs(result.y()) > 0.5)
    {
        LOGR("point out of cell bounds; idx {}; local coord {} x {}", pos, result.x(), result.y());
    }
    return result;
}

Eigen::Vector2d ProxyPoint::getVelocity()
{
    Eigen::Vector2d result;
    for(int i=0;i<SimParams::dim;i++) result[i] = getValue(SimParams::PtArrIdx::velx+i);
    return result;
}

double ProxyPoint::getValue(size_t valueIdx)
{
    if(isReference)
        return soa[pos + (size_t)pitch*valueIdx];
    else
        return data[valueIdx];
}

Eigen::Matrix2f ProxyPoint::getTensor(size_t valueIdx)
{
    Eigen::Matrix2f result;
    // pull point data from SOA
    for(int i=0; i<SimParams::dim; i++)
    {
        for(int j=0; j<SimParams::dim; j++)
        {
            result(i,j) = (float)getValue(valueIdx + i*SimParams::dim + j);
        }
    }
    return result;
}


void ProxyPoint::setValue(size_t valueIdx, double value)
{
    if(isReference)
        soa[pos + (size_t)pitch*valueIdx] = value;
    else
        data[valueIdx] = value;
}

uint64_t ProxyPoint::getValueUInt64(size_t valueIdx)
{
    if(isReference)
        return *reinterpret_cast<uint64_t*>(&soa[pos + (size_t)pitch*valueIdx]);
    else
        return *reinterpret_cast<uint64_t*>(&data[valueIdx]);
}

void ProxyPoint::setValueUInt64(size_t valueIdx, uint64_t value)
{
    if(isReference)
        *reinterpret_cast<uint64_t*>(&soa[pos + (size_t)pitch*valueIdx]) = value;
    else
        *reinterpret_cast<uint64_t*>(&data[valueIdx]) = value;
}


bool ProxyPoint::getCrushedStatus()
{
    uint64_t val = getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::status_crushed);
}

bool ProxyPoint::getCrackedStatus()
{
    uint64_t val = getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::status_cracked);
}

bool ProxyPoint::getDisabledStatus()
{
    uint64_t val = getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::status_disabled);
}



bool ProxyPoint::getFractureTension()
{
    uint64_t val = getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::fracture_tension);
}

bool ProxyPoint::getFractureCompressionShear()
{
    uint64_t val = getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::fracture_compression_shear);
}

bool ProxyPoint::getFractureCrush()
{
    uint64_t val = getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::fracture_crush);
}

long long ProxyPoint::getCellIndex(int GridY)
{
    uint64_t cell = getValueUInt64(SimParams::PtArrIdx::integer_cell_idx);
    long long x_idx = (long long)(cell & 0xffffffff);
    long long y_idx = (long long)(cell >> 32);
    return x_idx * (long long)GridY + y_idx;
}

unsigned ProxyPoint::getCellX()
{
    uint64_t cell = getValueUInt64(SimParams::PtArrIdx::integer_cell_idx);
    uint64_t x_idx = cell & 0xffffffff;
    return (unsigned) x_idx;
}


void ProxyPoint::ConvertToIntegerCellFormat(double h)
{
    const double hinv = 1.0f/h;
    double x = getValue(SimParams::PtArrIdx::posx);
    double y = getValue(SimParams::PtArrIdx::posx+1);
    uint64_t x_idx = (uint64_t)(x*hinv + 0.5);
    uint64_t y_idx = (uint64_t)(y*hinv + 0.5);
    uint64_t cell = (y_idx << 32) | x_idx;
    setValueUInt64(SimParams::PtArrIdx::integer_cell_idx, cell);

    x = x*hinv - (double)x_idx;
    y = y*hinv - (double)y_idx;
    setValue(SimParams::PtArrIdx::posx, x);
    setValue(SimParams::PtArrIdx::posx+1, y);
}

Eigen::Vector2d ProxyPoint::getPos(double cellsize)
{
    uint64_t cell = getValueUInt64(SimParams::PtArrIdx::integer_cell_idx);
    uint64_t x_idx = cell & 0xffffffff;
    uint64_t y_idx = (cell >> 32);
    Eigen::Vector2d cell_pos(x_idx, y_idx);
    return (getPos() + cell_pos) * cellsize;
}
