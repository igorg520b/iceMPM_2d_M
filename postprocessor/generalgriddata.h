// generalgriddata.h

#ifndef GENERALGRIDDATA_H
#define GENERALGRIDDATA_H

#include <vector>
#include <string>

#include "parameters_sim.h"


struct GeneralGridData
{
    GeneralGridData() { prms.Reset();}

    void ReadParameterFile(std::string parameterFileName);
    void ScanDirectory(std::string frameFileName);

    SimParams prms;
    int countFrames;
    std::string frameDirectory;

    std::vector<int> path_indices;
    std::vector<uint8_t> original_image_colors_rgb; // from PNG image
};

#endif // GENERALGRIDDATA_H
