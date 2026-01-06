#include "proxypoint2d.h"

// ====================================================== ProxyPoint
ProxyPoint::ProxyPoint(const ProxyPoint &other)
{
    isReference = false;
    *this = other;
}

ProxyPoint& ProxyPoint::operator=(const ProxyPoint &other)
{
    if(isReference)
    {
        // distribute into soa
        if(other.isReference)
        {
            for(int i=0;i<nArrays;i++) soa[pos + i*pitch] = other.soa[other.pos + i*other.pitch];
        }
        else
        {
            for(int i=0;i<nArrays;i++) soa[pos + i*pitch] = other.data[i];
        }
    }
    else
    {
        // local copy
        if(other.isReference)
        {
            for(int i=0;i<nArrays;i++) data[i] = other.soa[other.pos + i*other.pitch];
        }
        else
        {
            for(int i=0;i<nArrays;i++) data[i] = other.data[i];
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
        return soa[pos + pitch*valueIdx];
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
        soa[pos + pitch*valueIdx] = value;
    else
        data[valueIdx] = value;
}

uint32_t ProxyPoint::getValueInt(size_t valueIdx)
{
    if(isReference)
        return *reinterpret_cast<uint32_t*>(&soa[pos + pitch*valueIdx]);
    else
        return *reinterpret_cast<uint32_t*>(&data[valueIdx]);
}

void ProxyPoint::setValueInt(size_t valueIdx, uint32_t value)
{
    if(isReference)
        *reinterpret_cast<uint32_t*>(&soa[pos + pitch*valueIdx]) = value;
    else
        *reinterpret_cast<uint32_t*>(&data[valueIdx]) = value;
}

uint64_t ProxyPoint::getValueUInt64(size_t valueIdx)
{
    if(isReference)
        return *reinterpret_cast<uint64_t*>(&soa[pos + pitch*valueIdx]);
    else
        return *reinterpret_cast<uint64_t*>(&data[valueIdx]);
}

void ProxyPoint::setValueUInt64(size_t valueIdx, uint64_t value)
{
    if(isReference)
        *reinterpret_cast<uint64_t*>(&soa[pos + pitch*valueIdx]) = value;
    else
        *reinterpret_cast<uint64_t*>(&data[valueIdx]) = value;
}


bool ProxyPoint::getCrushedStatus()
{
    uint32_t val = getValueInt(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::status_crushed);
}

bool ProxyPoint::getCrackedStatus()
{
    uint32_t val = getValueInt(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::status_cracked);
}

bool ProxyPoint::getDisabledStatus()
{
    uint32_t val = getValueInt(SimParams::PtArrIdx::idx_utility_data);
    return (val & SimParams::status_disabled);
}


uint16_t ProxyPoint::getGrain()
{
    uint32_t val = getValueInt(SimParams::PtArrIdx::idx_utility_data);
    return (val & 0xffff);
}

int ProxyPoint::getCellIndex(int GridY)
{
    uint32_t cell = getValueInt(SimParams::PtArrIdx::integer_cell_idx);
    uint32_t x_idx = cell & 0xffff;
    uint32_t y_idx = (cell >> 16);
    return x_idx*GridY + y_idx;
}

unsigned ProxyPoint::getCellX()
{
    uint32_t cell = getValueInt(SimParams::PtArrIdx::integer_cell_idx);
    uint32_t x_idx = cell & 0xffff;
    return (unsigned) x_idx;
}


void ProxyPoint::ConvertToIntegerCellFormat(double h)
{
    const double hinv = 1.0f/h;
    double x = getValue(SimParams::PtArrIdx::posx);
    double y = getValue(SimParams::PtArrIdx::posx+1);
    uint32_t x_idx = (uint32_t)(x*hinv + 0.5);
    uint32_t y_idx = (uint32_t)(y*hinv + 0.5);
    uint32_t cell = (y_idx << 16) | x_idx;
    setValueInt(SimParams::PtArrIdx::integer_cell_idx, cell);

    x = x*hinv - (double)x_idx;
    y = y*hinv - (double)y_idx;
    setValue(SimParams::PtArrIdx::posx, x);
    setValue(SimParams::PtArrIdx::posx+1, y);
}

Eigen::Vector2d ProxyPoint::getPos(double cellsize)
{
    uint32_t cell = getValueInt(SimParams::PtArrIdx::integer_cell_idx);
    uint32_t x_idx = cell & 0xffff;
    uint32_t y_idx = (cell >> 16);
    Eigen::Vector2d cell_pos(x_idx, y_idx);
    return (getPos() + cell_pos) * cellsize;
}
