#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <vector>

#include "simulation/data_manager/host_side_data.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // Setup simple logging
    spdlog::set_level(spdlog::level::info);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_snapshot_h5> [output_dir] [compress: 1/0]" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputDir = "";
    
    // Default to compressed since this is "snapshot_compressor"
    bool compress = true;

    if (argc >= 3) {
        outputDir = argv[2];
    }
    if (argc >= 4) {
        compress = (std::stoi(argv[3]) != 0);
    }

    // Determine output directory if not provided or empty
    if (outputDir.empty()) {
        outputDir = fs::path(inputPath).parent_path().string();
    }
    if (outputDir.empty()) {
        outputDir = ".";
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    HostSideData hsd;
    
    LOGR("Loading snapshot from {}", inputPath);
    
    // This will read points, attributes, and perform RemoveDisabledAndSort internaly
    hsd.prms.extra_space_pts = 0;   // for this compressor purposes, no extra space necessary
    hsd.ReadPointsFromSnapshot(inputPath);

    std::string prefix = compress ? "sc" : "s";
    
    // Extract frame index from filename to preserve numbering (e.g. s00123.h5 -> 123)
    // We assume the filename format contains the frame number.
    int frameIndex = -1;
    std::string filename = fs::path(inputPath).filename().string();
    
    try {
        // Find the sequence of digits in the filename
        size_t firstDigit = filename.find_first_of("0123456789");
        if (firstDigit != std::string::npos) {
            size_t lastDigit = filename.find_last_of("0123456789");
            // Extract the number
            std::string numStr = filename.substr(firstDigit, lastDigit - firstDigit + 1);
            frameIndex = std::stoi(numStr);
        }
    } catch (...) {
        LOGR("Warning: Could not extract frame index from filename '{}', defaulting to calculated step.", filename);
    }

    LOGR("Saving snapshot to '{}' (prefix: '{}', frame: {})", outputDir, prefix, frameIndex);

    // Save using the reused function
    // Note: UpdateEveryNthStep might be default (1) since we didn't load json. 
    // Passing frameIndex forces the frame number in the filename.
    hsd.SaveSnapshot(hsd.prms.SimulationStep, hsd.prms.SimulationTime, compress, outputDir, prefix, frameIndex);

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Print time
    int minutes = (int)(elapsed.count() / 60);
    double seconds = elapsed.count() - (minutes * 60);
    
    std::cout << "Execution time: " << minutes << "m " << seconds << "s" << std::endl;

    return 0;
}
