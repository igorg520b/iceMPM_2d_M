// flowfieldgenerator.h

#ifndef FLOWFIELDGENERATOR_H
#define FLOWFIELDGENERATOR_H

#include <string>
#include <vector>

// Forward declaration to avoid circular includes
struct ParameterParser;

class FlowFieldGenerator
{
public:
    FlowFieldGenerator() = default;
    ~FlowFieldGenerator() = default;

    void GenerateFlow(const ParameterParser& params,
                     int gx, int gy, double cellsize, int ox, int oy,
                     int imageWidth, int imageHeight,
                     const std::string& projectDirectory);

    // Generate FLUENT flow field from CAS/DAT files
    void GenerateFluentFlow(const ParameterParser& params);

    // Add wind flow data to existing (or new) flow field HDF5
    // Reads `windFilePath`, extracts region, and appends to `outputFlowPath`
    void AddWindToFlowFieldHDF5(const std::string& windFilePath, const std::string& outputFlowPath, bool compressFlow);

private:
    // Grid and project settings
    int gx = 0;
    int gy = 0;
    double cellsize = 0.0;
    int ox = 0;
    int oy = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    std::string projectDirectory;

    // Reusable buffers for frame generation
    std::vector<double> vx_frame;
    std::vector<double> vy_frame;

    // Helper method to create HDF5 file structure (called once at start)
    void CreateFlowFieldHDF5(const std::string& flowType, int num_frames, double time_interval,
                            int loop_mode, bool compressFlow);

    // Helper method to write a single frame to HDF5 (called for each frame)
    // Uses member buffers vx_frame, vy_frame, eta_frame
    void WriteFrameToHDF5(int frame_index);

    // Legacy wrapper for writing all frames at once (for constant/wave flow)
    void WriteFlowFieldToHDF5(const std::string& flowType, int num_frames, double time_interval,
                             int loop_mode, bool compressFlow,
                             const std::vector<std::vector<double>>& vx_frames,
                             const std::vector<std::vector<double>>& vy_frames);
};

#endif // FLOWFIELDGENERATOR_H
