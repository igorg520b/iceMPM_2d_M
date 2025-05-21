#include "parameterparser.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <iostream>

ParameterParser::ParameterParser()
{
    std::string fileName = "params.json";
    if(!std::filesystem::exists(fileName))
    {
        spdlog::info("params.json does not exist");
        return;
    }

    std::ifstream fileStream(fileName);
    std::string strConfigFile;
    strConfigFile.resize(std::filesystem::file_size(fileName));
    fileStream.read(strConfigFile.data(), strConfigFile.length());
    fileStream.close();

    rapidjson::Document doc;
    doc.Parse(strConfigFile.data());
    if(!doc.IsObject()) throw std::runtime_error("configuration file is not JSON");

    if(doc.HasMember("ProjectName")) ProjectName = doc["ProjectName"].GetString();
    else ProjectName  = "p1";

    if(doc.HasMember("InputSVG")) fileNameSVG = doc["InputSVG"].GetString();
    if(doc.HasMember("InputPNG")) fileNamePNG = doc["InputPNG"].GetString();
    if(doc.HasMember("MainPathID")) MainPathID = doc["MainPathID"].GetString();

    if(doc.HasMember("InputFluentDAT")) fileNameFluentDAT = doc["InputFluentDAT"].GetString();
    if(doc.HasMember("InputFluentCAS")) fileNameFluentCAS = doc["InputFluentCAS"].GetString();

    height = doc["Height"].GetInt();
    width = doc["Width"].GetInt();

    if(doc.HasMember("FLUENT_Scale")) FLUENT_Scale = doc["FLUENT_Scale"].GetDouble();
    if(doc.HasMember("FLUENT_OffsetX")) FLUENT_OffsetX = doc["FLUENT_OffsetX"].GetDouble();
    if(doc.HasMember("FLUENT_OffsetY")) FLUENT_OffsetY = doc["FLUENT_OffsetY"].GetDouble();


    // color values
    colordata_OpenWater.clear();
    colordata_Solid.clear();
    // --- Load Open Water Colors ---
    if (!doc.HasMember("openWaterColors") || !doc["openWaterColors"].IsArray()) {
        std::cerr << "Error: JSON missing 'openWaterColors' array." << std::endl;
        throw std::runtime_error("Error: JSON missing 'openWaterColors' array.");
    }
    const rapidjson::Value& waterColors = doc["openWaterColors"];
    colordata_OpenWater.reserve(waterColors.Size());
    for (rapidjson::SizeType i = 0; i < waterColors.Size(); ++i) {
        if (!waterColors[i].IsArray() || waterColors[i].Size() != 3) {
            std::cerr << "Error: Invalid format in 'openWaterColors' at index " << i << ". Expected array of 3 floats." << std::endl;
            throw std::runtime_error("JSON error");
        }
        Eigen::Vector3f color;
        for (rapidjson::SizeType j = 0; j < 3; ++j) {
            if (!waterColors[i][j].IsNumber()) {
                std::cerr << "Error: Non-numeric value in 'openWaterColors' at index " << i << ", element " << j << std::endl;
                throw std::runtime_error("JSON error");
            }
            color[j] = waterColors[i][j].GetFloat();
        }
        // Optional: Clamp/validate color values are in [0, 1] range
        // color = color.cwiseMax(0.0f).cwiseMin(1.0f);
        colordata_OpenWater.push_back(color);
    }
    if (colordata_OpenWater.size() < 2) {
        std::cerr << "Warning: 'openWaterColors' curve needs at least 2 points." << std::endl;
        // Decide if this is a fatal error or just a warning
    }


    // --- Load Solid Colors ---
    if (!doc.HasMember("solidColors") || !doc["solidColors"].IsArray()) {
        std::cerr << "Error: JSON missing 'solidColors' array." << std::endl;
        throw std::runtime_error("JSON error");
    }
    const rapidjson::Value& solidColors = doc["solidColors"];
    colordata_Solid.reserve(solidColors.Size());
    for (rapidjson::SizeType i = 0; i < solidColors.Size(); ++i) {
        if (!solidColors[i].IsArray() || solidColors[i].Size() != 3) {
            std::cerr << "Error: Invalid format in 'solidColors' at index " << i << ". Expected array of 3 floats." << std::endl;
            throw std::runtime_error("JSON error");
        }
        Eigen::Vector3f color;
        for (rapidjson::SizeType j = 0; j < 3; ++j) {
            if (!solidColors[i][j].IsNumber()) {
                std::cerr << "Error: Non-numeric value in 'solidColors' at index " << i << ", element " << j << std::endl;
                throw std::runtime_error("JSON error");
            }
            color[j] = solidColors[i][j].GetFloat();
        }
        // Optional: Clamp/validate color values are in [0, 1] range
        // color = color.cwiseMax(0.0f).cwiseMin(1.0f);
        colordata_Solid.push_back(color);
    }
    if (colordata_Solid.size() < 2) {
        std::cerr << "Warning: 'solidColors' curve needs at least 2 points." << std::endl;
        // Decide if this is a fatal error or just a warning
    }


    // create output folder
    ProjectDirectory = std::string("output/") + ProjectName;
    std::filesystem::create_directories(ProjectDirectory);

    spdlog::info("parameter file loaded");
    spdlog::info("fileNameSVG: {}", fileNameSVG);
    spdlog::info("fileNamePNG: {}", fileNamePNG);
    spdlog::info("MainPathID: {}", MainPathID);
    spdlog::info("dimension {} x {}", width, height);
    spdlog::info("fluent DAT {}", fileNameFluentDAT);
    spdlog::info("fluent CAS {}", fileNameFluentCAS);
    std::cout << '\n';
}
