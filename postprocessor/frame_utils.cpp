// frame_utils.cpp

#include "frame_utils.h"
#include <filesystem>
#include <regex>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "parameters_sim.h"

namespace frame_utils {

int ScanFrameDirectory(const std::string& directoryName)
{
    LOGR("frame_utils::ScanFrameDirectory scanning: {}", directoryName);

    std::filesystem::path dirPath(directoryName);
    
    // Check for new split format: frames/color
    std::filesystem::path colorDir = dirPath / "color";
    if (std::filesystem::exists(colorDir) && std::filesystem::is_directory(colorDir)) {
        LOGR("Detected split frame format, scanning: {}", colorDir.string());
        dirPath = colorDir;
    }

    // Define the regex pattern for files like frame_00001.h5 or f00001.h5
    const std::regex filePattern(R"(^(frame_|f)\d+\.h5$)");

    int foundCount = 0;

    // Iterate through the directory entries
    if (std::filesystem::exists(dirPath)) {
        for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
            // Check if it's a regular file (not a directory, symlink, etc.)
            if (entry.is_regular_file()) {
                // Get the filename part of the path
                const std::string filename = entry.path().filename().string();

                // Check if the filename matches the pattern
                if (std::regex_match(filename, filePattern)) {
                    foundCount++;
                }
            }
        }
    }

    LOGR("frame_utils::ScanFrameDirectory found {} matching frame files in directory {}", foundCount, dirPath.string());
    return foundCount;
}


std::string GetFramePath(const std::string& frameDirectory, int frameNumber)
{
    std::filesystem::path dirPath(frameDirectory);
    std::filesystem::path colorDir = dirPath / "color";
    
    if (std::filesystem::exists(colorDir)) {
        std::string baseName = fmt::format(fmt::runtime("frame_{:05d}.h5"), frameNumber);
        return (colorDir / baseName).string();
    } else {
        // Old format
        std::string baseName = fmt::format(fmt::runtime("f{:05d}.h5"), frameNumber);
        return (dirPath / baseName).string();
    }
}

} // namespace frame_utils
