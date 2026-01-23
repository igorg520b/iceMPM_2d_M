#include "core.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <cstring>

int main(int argc, char* argv[]) {
    // Usage: compressor <project_directory> <frame_from> <frame_to> [--overwrite]

    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <project_directory> <frame_from> <frame_to> [--overwrite]" << std::endl;
        return 1;
    }

    std::string projectDir = argv[1];
    int frame_from = std::stoi(argv[2]);
    int frame_to   = std::stoi(argv[3]);

    bool overwrite = false;
    if (argc == 5) {
        if (std::string(argv[4]) == "--overwrite") {
            overwrite = true;
        } else {
            std::cerr << "Unknown argument: " << argv[4] << std::endl;
            return 1;
        }
    }

    if (frame_from > frame_to) {
        std::cerr << "Error: frame_from must be <= frame_to" << std::endl;
        return 1;
    }

    std::string framesDir = projectDir + "/frames";
    std::string outDir    = projectDir + "/frames_compressed";

    if (!overwrite) {
        // Ensure output directory exists only if we are NOT purely overwriting in place?
        // Actually, even in overwrite mode, we write to a temp (or similar) but here we follow the plan:
        // Compress to outDir then move. So outDir MUST exist.
        std::string mkdir_cmd = "mkdir -p \"" + outDir + "\"";
        system(mkdir_cmd.c_str());
    } else {
         // Also create it for the temporary compressed files
        std::string mkdir_cmd = "mkdir -p \"" + outDir + "\"";
        system(mkdir_cmd.c_str());
    }

    std::cout << "Processing frames from " << frame_from << " to " << frame_to 
              << (overwrite ? " (OVERWRITE)" : "") << std::endl;

    for (int frame = frame_from; frame <= frame_to; ++frame) {
        std::ostringstream fname_ss;
        fname_ss << "f" << std::setw(5) << std::setfill('0') << frame << ".h5";
        std::string filename = fname_ss.str();

        for (const auto& sub : SUBCATEGORIES) {
            std::string subInDir = framesDir + "/" + sub;
            std::string subOutDir = outDir + "/" + sub;

            // Ensure output subdirectory exists
            // (Only strictly needed if not overwriting IN PLACE, but we effectively copy then move)
            // If overwrite is true, we still use outDir as temp.
            // Using system("mkdir -p") inside loop is inefficient but safe. 
            // Better to create all at start, but simple logic first.
            // Only create if input exists?
            struct stat st;
            std::string inputFile = subInDir + "/" + filename;
            if (stat(inputFile.c_str(), &st) != 0) {
                 // Input file doesn't exist, skip
                 continue; 
            }

            // Create output subdir
            std::string mkdir_cmd = "mkdir -p \"" + subOutDir + "\"";
            system(mkdir_cmd.c_str());

            std::string outputFile = subOutDir + "/" + filename;

            bool result = process_frame_file(inputFile, outputFile, overwrite);
            if (result) {
                if (overwrite) {
                   // std::cout << "  [OW] " << sub << "/" << filename << std::endl;
                } else {
                   // std::cout << "  " << sub << "/" << filename << std::endl;
                }
            } else {
                std::cerr << "  Failed: " << inputFile << std::endl;
            }
        }
        if (frame % 10 == 0) std::cout << "Processed frame " << frame << std::endl;
    }

    if (overwrite) {
       // Optional: Clean up empty frames_compressed if it shouldn't persist?
       // Leaving it for now as per plan/instruction simplicity in case user wants to inspect.
    }

    return 0;
}
