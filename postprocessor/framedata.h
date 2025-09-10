// framedata.h

#ifndef FRAMEDATA_H
#define FRAMEDATA_H

#include <string>
#include <vector>
#include <spdlog/spdlog.h>

#include "parameters_sim.h"
#include "generalgriddata.h"

// Forward declaration is sufficient
namespace icy { class VisualRepresentation; }

struct FrameData
{
    GeneralGridData &ggd;
    std::vector<uint8_t> rgb;
    std::vector<t_GridReal> host_grid_buffer;
    int currentFrameNumber = -1;
    double simulationTime = 0.0;
    int simulationStep = 0;

    FrameData(GeneralGridData &ggd_);
    bool LoadFrame(int frameNumber);  // Loads a specific frame from disk.

    // Links the data stored in this object to a representation object.
    void LinkRepresentation(icy::VisualRepresentation &representation);

private:
    static void readGridDataset(const H5::H5File& file, const std::string& dataset_name,
                                std::vector<t_GridReal>& dest_buffer, size_t offset);
};

#endif // FRAMEDATA_H
