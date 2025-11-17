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
    if(doc.HasMember("PhaseSpeed")) PhaseSpeed = doc["PhaseSpeed"].GetDouble();
    if(doc.HasMember("NFrames")) NFrames = doc["NFrames"].GetInt();
    if(doc.HasMember("CompressFlow")) CompressFlow = doc["CompressFlow"].GetBool();

    // FLUENT-specific parameters
    if(doc.HasMember("InputFluentDAT")) InputFluentDAT = doc["InputFluentDAT"].GetString();
    if(doc.HasMember("InputFluentCAS")) InputFluentCAS = doc["InputFluentCAS"].GetString();
    if(doc.HasMember("SVG")) SVG = doc["SVG"].GetString();
    if(doc.HasMember("RectanglePathID")) RectanglePathID = doc["RectanglePathID"].GetString();
    if(doc.HasMember("FluentPathID")) FluentPathID = doc["FluentPathID"].GetString();
    if(doc.HasMember("VelocityMultiplier")) VelocityMultiplier = doc["VelocityMultiplier"].GetDouble();

    spdlog::info("parameter file loaded\n");
}
