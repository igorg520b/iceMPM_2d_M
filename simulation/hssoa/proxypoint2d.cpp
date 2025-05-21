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

PointVector2r ProxyPoint::getPos()
{
    PointVector2r result;
    for(int i=0;i<SimParams::dim;i++) result[i] = getValue(SimParams::posx+i);
    if(std::abs(result.x()) > 0.5 || std::abs(result.y()) > 0.5)
    {
        LOGR("point out of cell bounds; idx {}; local coord {} x {}", pos, result.x(), result.y());
    }
    return result;
}

PointVector2r ProxyPoint::getVelocity()
{
    PointVector2r result;
    for(int i=0;i<SimParams::dim;i++) result[i] = getValue(SimParams::velx+i);
    return result;
}

t_PointReal ProxyPoint::getValue(size_t valueIdx)
{
    if(isReference)
        return soa[pos + pitch*valueIdx];
    else
        return data[valueIdx];
}


void ProxyPoint::setValue(size_t valueIdx, t_PointReal value)
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


bool ProxyPoint::getCrushedStatus()
{
    uint32_t val = getValueInt(SimParams::idx_utility_data);
    return (val & 0x10000);
}

bool ProxyPoint::getDisabledStatus()
{
    uint32_t val = getValueInt(SimParams::idx_utility_data);
    return (val & 0x20000);
}

bool ProxyPoint::getWeakenedStatus()
{
    uint32_t val = getValueInt(SimParams::idx_utility_data);
    return (val & 0x40000);
}



uint16_t ProxyPoint::getGrain()
{
    uint32_t val = getValueInt(SimParams::idx_utility_data);
    return (val & 0xffff);
}

int ProxyPoint::getCellIndex(int GridY)
{
    uint32_t cell = getValueInt(SimParams::integer_cell_idx);
    uint32_t x_idx = cell & 0xffff;
    uint32_t y_idx = (cell >> 16);
    return x_idx*GridY + y_idx;
}

void ProxyPoint::ConvertToIntegerCellFormat(t_PointReal h)
{
    const t_PointReal hinv = 1.0f/h;
    t_PointReal x = getValue(SimParams::posx);
    t_PointReal y = getValue(SimParams::posx+1);
    uint32_t x_idx = (uint32_t)(x*hinv + 0.5);
    uint32_t y_idx = (uint32_t)(y*hinv + 0.5);
    uint32_t cell = (y_idx << 16) | x_idx;
    setValueInt(SimParams::integer_cell_idx, cell);

    x = x*hinv - (t_PointReal)x_idx;
    y = y*hinv - (t_PointReal)y_idx;
    setValue(SimParams::posx, x);
    setValue(SimParams::posx+1, y);
}

PointVector2r ProxyPoint::getPos(t_PointReal cellsize)
{
    uint32_t cell = getValueInt(SimParams::integer_cell_idx);
    uint32_t x_idx = cell & 0xffff;
    uint32_t y_idx = (cell >> 16);
    PointVector2r cell_pos(x_idx, y_idx);
    return (getPos() + cell_pos) * cellsize;
}
