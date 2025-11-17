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

    // Main dispatcher method: generate flow field based on parameters
    // Routes to appropriate method based on params.FlowType:
    // - "constant": GenerateConstantFlow()
    // - "wave": GenerateWaveFlow()
    // - "FLUENT-static": GenerateFluentFlow()
    // Parameters:
    //   gx, gy: modeled region grid dimensions
    //   cellsize: grid cell size
    //   ox, oy: offset of modeled region from image origin
    //   imageWidth, imageHeight: initialization image dimensions (for FLUENT rasterization)
    void GenerateFlow(const ParameterParser& params,
                     int gx, int gy, double cellsize, int ox, int oy,
                     int imageWidth, int imageHeight,
                     const std::string& projectDirectory);

    // Generate constant flow field
    void GenerateConstantFlow(int gx, int gy, double cellsize, int ox, int oy,
                             double flowBearing, double flowSpeed,
                             const std::string& projectDirectory, bool compressFlow);

    // Generate wave flow field
    void GenerateWaveFlow(int gx, int gy, double cellsize, int ox, int oy,
                         double flowBearing, double waveAmplitude, double waveLength, double phaseSpeed,
                         int nFrames, const std::string& projectDirectory, bool compressFlow);

    // Generate FLUENT flow field from CAS/DAT files
    void GenerateFluentFlow(const ParameterParser& params,
                           int gx, int gy, int ox, int oy,
                           int imageWidth, int imageHeight,
                           const std::string& projectDirectory);

private:
    // Helper method to write velocity data to HDF5
    void WriteFlowFieldToHDF5(int gx, int gy, int num_frames, double time_interval,
                             int loop_mode, const std::string& projectDirectory,
                             bool compressFlow,
                             const std::vector<std::vector<double>>& vx_frames,
                             const std::vector<std::vector<double>>& vy_frames);
};

#endif // FLOWFIELDGENERATOR_H
