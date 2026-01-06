#include "parameterparser.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <iostream>


void ParameterParser::LoadParamsFile(std::string fileName)
{
    if(!std::filesystem::exists(fileName))
    {
        spdlog::info("params.json does not exist");
        return;
    }

    // Extract directory path from JSON filename
    std::filesystem::path configPath(fileName);
    ConfigFileDirectory = configPath.parent_path().string();
    if(ConfigFileDirectory.empty()) ConfigFileDirectory = ".";

    std::ifstream fileStream(fileName);
    std::string strConfigFile;
    strConfigFile.resize(std::filesystem::file_size(fileName));
    fileStream.read(strConfigFile.data(), strConfigFile.length());
    fileStream.close();


    rapidjson::Document doc;
    doc.Parse(strConfigFile.data());
    if(!doc.IsObject()) throw std::runtime_error("configuration file is not JSON");

    if(doc.HasMember("ProjectName")) ProjectName = doc["ProjectName"].GetString();
    else ProjectName  = "default_project";
    ProjectDirectory = std::string("input/") + ProjectName;
    std::filesystem::create_directories(ProjectDirectory);

    ImageIceMask = doc["ImageIceMask"].GetString();

    if(doc.HasMember("ImageColor")) ImageColor = doc["ImageColor"].GetString();
    if(doc.HasMember("ImageCrushedMask")) ImageCrushedMask = doc["ImageCrushedMask"].GetString();
    if(doc.HasMember("ImageLandMask")) ImageLandMask = doc["ImageLandMask"].GetString();
    if(doc.HasMember("ImageThicknessMask")) ImageThicknessMask = doc["ImageThicknessMask"].GetString();

    if(doc.HasMember("PointsPerCell")) PointsPerCell = doc["PointsPerCell"].GetInt();
    if(doc.HasMember("DimensionHorizontal")) DimensionHorizontal = doc["DimensionHorizontal"].GetDouble();

    // Ice thickness scaling parameters
    if(doc.HasMember("ThicknessFrom")) ThicknessFrom = doc["ThicknessFrom"].GetDouble();
    if(doc.HasMember("ThicknessTo")) ThicknessTo = doc["ThicknessTo"].GetDouble();

    // Flow field parameters
    if(doc.HasMember("FlowType")) FlowType = doc["FlowType"].GetString();
    if(doc.HasMember("FlowBearing")) FlowBearing = doc["FlowBearing"].GetDouble();
    if(doc.HasMember("FlowSpeed")) FlowSpeed = doc["FlowSpeed"].GetDouble();
    if(doc.HasMember("WaveAmplitude")) WaveAmplitude = doc["WaveAmplitude"].GetDouble();
    if(doc.HasMember("WaveLength")) WaveLength = doc["WaveLength"].GetDouble();
    if(doc.HasMember("WavePeriod")) WavePeriod = doc["WavePeriod"].GetDouble();
    if(doc.HasMember("PhaseSpeed")) PhaseSpeed = doc["PhaseSpeed"].GetDouble();
    if(doc.HasMember("AmplitudeDecayDistance")) AmplitudeDecayDistance = doc["AmplitudeDecayDistance"].GetDouble();
    if(doc.HasMember("ShipSpeed")) ShipSpeed = doc["ShipSpeed"].GetDouble();
    if(doc.HasMember("X_Ship")) X_Ship = doc["X_Ship"].GetDouble();
    if(doc.HasMember("Y_Ship")) Y_Ship = doc["Y_Ship"].GetDouble();
    if(doc.HasMember("FrameSkip")) FrameSkip = doc["FrameSkip"].GetInt();
    if(doc.HasMember("TimeScale")) TimeScale = doc["TimeScale"].GetDouble();
    if(doc.HasMember("CompressFlow")) CompressFlow = doc["CompressFlow"].GetBool();

    // Set NFrames default based on FlowType (before parsing explicit value)
    // Default: 30 frames per period for dynamic flows, 1 frame for static flows
    if (FlowType == "wave" || FlowType == "Kelvin_wake" || FlowType == "Kelvin_wake_alt" || FlowType == "standing_wave") {
        NFrames = 30;  // Dynamic flow: default to 30 frames per wave period
    } else {
        NFrames = 1;   // Static flow (constant, FLUENT-static, or empty): single frame
    }

    // Override with explicit NFrames from JSON if provided
    if(doc.HasMember("NFrames")) NFrames = doc["NFrames"].GetInt();

    // Log flow field parameters for debugging
    if (!FlowType.empty()) {
        spdlog::info("Flow field parameters: FlowType={}, NFrames={}, TimeScale={}", FlowType, NFrames, TimeScale);
        if (FlowType == "constant") {
            spdlog::info("  Constant flow: bearing={}, speed={}", FlowBearing, FlowSpeed);
        } else if (FlowType == "wave") {
            spdlog::info("  Wave parameters: amplitude={}, wavelength={}, phase_speed={}, bearing={}",
                        WaveAmplitude, WaveLength, PhaseSpeed, FlowBearing);
        }
    }

    // FLUENT-specific parameters
    if(doc.HasMember("InputFluentDAT")) InputFluentDAT = doc["InputFluentDAT"].GetString();
    if(doc.HasMember("InputFluentCAS")) InputFluentCAS = doc["InputFluentCAS"].GetString();
    if(doc.HasMember("SVG")) SVG = doc["SVG"].GetString();
    if(doc.HasMember("RectanglePathID")) RectanglePathID = doc["RectanglePathID"].GetString();
    if(doc.HasMember("FluentPathID")) FluentPathID = doc["FluentPathID"].GetString();
    if(doc.HasMember("VelocityMultiplier")) VelocityMultiplier = doc["VelocityMultiplier"].GetDouble();

    spdlog::info("parameter file loaded\n");
}
