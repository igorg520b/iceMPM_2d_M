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

    const std::filesystem::path dirPath(directoryName);

    // Define the regex pattern for files like f00001.h5
    const std::regex filePattern(R"(^f\d+\.h5$)");

    int foundCount = 0;

    // Iterate through the directory entries
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

    LOGR("frame_utils::ScanFrameDirectory found {} matching frame files in directory {}", foundCount, directoryName);
    return foundCount;
}


std::string GetFramePath(const std::string& frameDirectory, int frameNumber)
{
    const std::string baseName = fmt::format(fmt::runtime("f{:05d}.h5"), frameNumber);
    return (std::filesystem::path(frameDirectory) / baseName).string();
}

} // namespace frame_utils
