// frame_utils.h
// Standalone utilities for discovering and accessing frame files in post-processor

#ifndef FRAME_UTILS_H
#define FRAME_UTILS_H

#include <string>
#include <vector>

namespace frame_utils {
    // Scan a directory for frame files matching the pattern f00000.h5, f00001.h5, etc.
    // Returns the count of matching frames found
    int ScanFrameDirectory(const std::string& frameDirectory);

    // Get the full filesystem path to a frame file by frame number
    // Example: GetFramePath("/path/to/frames", 42) -> "/path/to/frames/f00042.h5"
    std::string GetFramePath(const std::string& frameDirectory, int frameNumber);
}

#endif // FRAME_UTILS_H
