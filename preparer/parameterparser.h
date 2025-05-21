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
    ParameterParser();

    std::string ProjectName, ProjectDirectory;
    std::string fileNameSVG, fileNamePNG, fileNameWindData;
    std::string MainPathID;
    std::string fileNameFluentDAT, fileNameFluentCAS;
    int height, width;

    double FLUENT_Scale = 0;
    double FLUENT_OffsetX = 0;
    double FLUENT_OffsetY = 0;

    std::vector<Eigen::Vector3f> colordata_OpenWater;
    std::vector<Eigen::Vector3f> colordata_Solid;
};

#endif // PARAMETERPARSER_H
