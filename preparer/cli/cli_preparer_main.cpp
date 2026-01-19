#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>

#include <cxxopts.hpp>
#include "parameters_sim.h"
#include "host_side_data.h"
#include "data_preparer.h"
#include "parameterparser.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    std::string prepare_parameter_filename;
    // parse options
    std::string parameter_filename;
    std::string resume_filename; // Defaults to empty

    cxxopts::Options options("Ice MPM", "CLI version of MPM simulation");
    options.add_options()
        // Define options and link them to variables
        ("parameter", "Preapare parameter file (JSON, required)", cxxopts::value<std::string>(parameter_filename));

    // Mark 'parameter' as the positional argument
    options.parse_positional({"parameter"});
    auto result = options.parse(argc, argv); // Parse arguments

    // Check if required parameter file was provided
    if (!result.count("parameter")) {
        LOGR("Error: Parameter file argument is required."); // Minimal error output
        throw std::runtime_error("Parameter file is required."); // Still need to signal failure
    }
    parameter_filename = result["parameter"].as<std::string>();

    // Check if the provided path is a directory (if so, append "prepare.json")
    if (fs::is_directory(parameter_filename)) {
        prepare_parameter_filename = (fs::path(parameter_filename) / "prepare.json").string();
    } else {
        prepare_parameter_filename = parameter_filename;
    }

    LOGR("Loading parameters from: {}", prepare_parameter_filename);

    try {
        ParameterParser params;
        params.LoadParamsFile(prepare_parameter_filename);

        // Construct file paths
        std::string configDir = params.ConfigFileDirectory;
        std::string landmaskPath = params.ImageLandMask.empty() ? "" : (configDir + "/" + params.ImageLandMask);
        std::string colorPath = configDir + "/" + params.ImageColor;
        std::string icemaskPath = configDir + "/" + params.ImageIceMask;
        std::string crushedmaskPath = params.ImageCrushedMask.empty() ? "" : (configDir + "/" + params.ImageCrushedMask);
        std::string crackedmaskPath = params.ImageCrackedMask.empty() ? "" : (configDir + "/" + params.ImageCrackedMask);
        std::string thicknessmaskPath = params.ImageThicknessMask.empty() ? "" : (configDir + "/" + params.ImageThicknessMask);
        std::string projectDir = params.ProjectDirectory;

        // Create HostSideData and DataPreparer
        HostSideData hsd;
        DataPreparer preparer(hsd);

        LOGR("Starting CLI Data Preparation...");
        
        // Execute preparation
        // allocate_dense_grid = false (CLI optimization)
        preparer.PrepareGridAndPoints(landmaskPath, colorPath, icemaskPath, crushedmaskPath, crackedmaskPath,
                                      projectDir, params.DimensionHorizontal, params.PointsPerCell,
                                      params.ThicknessFrom, params.ThicknessTo,
                                      params.ProportionOfCrackedPoints, params.StdDevOfThickness,
                                      thicknessmaskPath,
                                      false);

        LOGR("Preparation Complete.");
        LOGR("  Grid Dimensions: {} x {}", hsd.prms.GridXTotal, hsd.prms.GridYTotal);
        LOGR("  Total Points: {}", hsd.prms.nPtsInitial);
        LOGR("  Data saved to: {}", hsd.data_directory);

    } catch (const std::exception& e) {
        LOGR("Error during preparation: {}", e.what());
        return 1;
    }

    return 0;
}
