#ifndef PARAMETERPARSER_H
#define PARAMETERPARSER_H

#include <string>

#include <rapidjson/reader.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <Eigen/Core>

struct ParameterParser
{
    std::string ProjectName, ProjectDirectory;
    std::string ImageColor, ImageCrushedMask, ImageIceMask, ImageLandMask, ImageThicknessMask;
    std::string ConfigFileDirectory;  // directory containing the JSON config file
    int PointsPerCell = 5;
    int height = 0;
    int width = 0;
    double DimensionHorizontal = 0.0;
    double ThicknessFrom = 1.0;  // default: no scaling
    double ThicknessTo = 1.0;    // default: no scaling

    // Flow field parameters
    std::string FlowType = "";              // "constant", "wave", "Kelvin_wake", "FLUENT-static" (empty = no flow)
    double FlowBearing = 0.0;               // bearing in degrees (0=north, 90=east, etc.)
    double FlowSpeed = 0.0;                 // speed for constant flow
    double WaveAmplitude = 0.0;             // amplitude for wave flow
    double WaveLength = 0.0;                // wavelength for wave flow
    double WavePeriod = 0.0;                // wave period for standing_wave flow type
    double PhaseSpeed = 0.0;                // phase speed for wave flow
    double AmplitudeDecayDistance = 0.0;    // distance over which amplitude decays by factor of 2 (Kelvin wake)
    double ShipSpeed = 0.0;                 // speed of ship generating Kelvin wake (only for Kelvin_wake flow type)
    double X_Ship = 0.0;                    // X-coordinate of ship position at t=0 (default: 0.0)
    double Y_Ship = 0.0;                    // Y-coordinate of ship centerline (default: 0.0)
    int FrameSkip = 5;                      // Time step parameter: dt = (cellsize * FrameSkip) / U_ship (default: 5)
    int NFrames = 1;                        // number of frames per wave period (default=1 for static, 30 for wave/Kelvin_wake)
    double TimeScale = 1000.0;              // time scale for animation slider (slider 0-1000 maps to 0-TimeScale seconds, default=1000)
    bool CompressFlow = false;              // whether to compress flow field HDF5

    // FLUENT-specific parameters (optional, only used when FlowType == "FLUENT-static")
    std::string InputFluentDAT = "";        // HDF5 file: velocity data
    std::string InputFluentCAS = "";        // HDF5 file: mesh definition
    std::string SVG = "";                   // SVG file: geometry + path definitions
    std::string RectanglePathID = "";       // SVG path ID: image bounds
    std::string FluentPathID = "";          // SVG path ID: FLUENT grid bounds
    double VelocityMultiplier = 1.0;        // Multiplier for FLUENT velocity field (default: 1.0)

    void LoadParamsFile(std::string fileName);
};

#endif // PARAMETERPARSER_H
