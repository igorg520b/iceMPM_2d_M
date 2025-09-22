#ifndef PARAMETERPARSER_H
#define PARAMETERPARSER_H

#include <string>

#include <rapidjson/reader.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <Eigen/Core>

class ParameterParser
{
public:
    ParameterParser() = default;
    void LoadParamsFile(std::string fileName);

    std::string ProjectName, ProjectDirectory;
    std::string fileNameSVG, fileNamePNG, fileNameWindData;
    std::string MainPathID;
    std::string RectanglePathID;
    std::string FluentPathID;
    std::string fileNameFluentDAT, fileNameFluentCAS;
    int height, width;

    double FLUENT_Scale = 0;
    double FLUENT_OffsetX = 0;
    double FLUENT_OffsetY = 0;

    double FlowX = 0;
    double FlowY = 0;
    bool ConstFlow = false;
    bool MakeAllIceSolid = false;

    std::vector<Eigen::Vector3f> colordata_OpenWater, colordata_Solid, colordata_Crushed;
    Eigen::Vector3i pierColor;
    std::vector<std::string> renderedPaths;
};

#endif // PARAMETERPARSER_H
