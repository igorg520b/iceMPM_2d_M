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
    void GenerateConstantFlow(double flowBearing, double flowSpeed, bool compressFlow);

    // Generate wave flow field
    void GenerateWaveFlow(double flowBearing, double waveAmplitude, double waveLength, double phaseSpeed,
                         int nFrames, bool compressFlow);

    // Generate Kelvin wake flow field using Havelock source integration
    // Uses precompute-and-pan approach: compute large grid at t=0, then extract frames
    // Time step: dt = (cellsize * frameSkip) / shipSpeed
    void GenerateKelvinWakeFlow(double waveAmplitude, double shipSpeed, double xShip, double yShip,
                               int frameSkip, double timeScale, double flowBearing,
                               bool compressFlow);

    // Generate river flow field
    void GenerateRiverFlow(double flowSpeed, double waveAmplitude, bool compressFlow);

    // Generate FLUENT flow field from CAS/DAT files
    void GenerateFluentFlow(const ParameterParser& params);


    // Generate dynamic bending test flow field
    // Water level amplitude increases linearly from 0 to maximum over timeScale
    // At end of timeScale, last frame is held
    // Water level: eta(x,y) = amplitude(t) * sin(k*x), where k = pi/DimensionHorizontal
    // All velocities are zero
    void GenerateBendingTestDynamic(double waveAmplitude, double dimensionHorizontal, double timeScale, bool compressFlow);

    // Generate standing wave flow field
    // Periodic standing wave along x-axis with amplitude ramp over first 5 periods
    // Water level: eta(x,t) = A(t) * sin(k*x) * sin(omega*t)
    // where A(t) = A_max * min(1.0, t/(5*T)), k = 2*pi/wavelength, omega = 2*pi/period
    // Maximum elevation reaches A(t) when both sin(k*x)=1 and sin(omega*t)=1
    // Horizontal velocity: u(x,t) = (A(t)*omega) * sin(k*x) * cos(omega*t)
    // Periodic with period = WavePeriod after amplitude reaches maximum
    void GenerateStandingWave(double flowBearing, double waveAmplitude, double waveLength, double wavePeriod, int nFrames, bool compressFlow);

private:
    // Kelvin wake computation using Havelock source integration
    // Calculates water elevation and velocity at a point relative to ship
    static void calculateKelvinWake(double x, double y, double speed,
                                   double& out_eta, double& out_vx, double& out_vy);

    static void calculateKelvinWakeWithGradients(double x, double y, double speed,
                                                              double& out_eta,
                                                              double& out_vx, double& out_vy,
                                                              double& out_grad_x, double& out_grad_y);

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
    std::vector<double> eta_frame;
    std::vector<double> d_eta_dx_frame;
    std::vector<double> d_eta_dy_frame;

    // Helper method to create HDF5 file structure (called once at start)
    void CreateFlowFieldHDF5(const std::string& flowType, int num_frames, double time_interval,
                            int loop_mode, bool compressFlow);

    // Helper method to create static 2D HDF5 for Kelvin wake (panorama)
    void CreateKelvinStaticHDF5(const std::string& flowType, int gx_large, int gy_large, double ship_speed, double cellsize, int extra_cells, int y_offset_px, double flow_bearing, double x_ship, double y_ship, bool compressFlow);

    // Helper method to write a single frame to HDF5 (called for each frame)
    // Uses member buffers vx_frame, vy_frame, eta_frame
    void WriteFrameToHDF5(int frame_index);

    // Legacy wrapper for writing all frames at once (for constant/wave flow)
    void WriteFlowFieldToHDF5(const std::string& flowType, int num_frames, double time_interval,
                             int loop_mode, bool compressFlow,
                             const std::vector<std::vector<double>>& vx_frames,
                             const std::vector<std::vector<double>>& vy_frames,
                             const std::vector<std::vector<double>>& eta_frames);
};

#endif // FLOWFIELDGENERATOR_H
