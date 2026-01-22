#include "core.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <mpi.h>
#include <sys/stat.h>

int main(int argc, char* argv[]) {
    // Usage: compressor_mpi <project_directory> <frame_from> <frame_to> [--overwrite]

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 4 || argc > 5) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] 
                      << " <project_directory> <frame_from> <frame_to> [--overwrite]" << std::endl;
        }
        MPI_Finalize();
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
            if (world_rank == 0) {
                 std::cerr << "Unknown argument: " << argv[4] << std::endl;
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (frame_from > frame_to) {
        if (world_rank == 0) {
            std::cerr << "Error: frame_from must be <= frame_to" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string framesDir = projectDir + "/frames";
    std::string outDir    = projectDir + "/frames_compressed";

    // Only master process creates directory structure
    if (world_rank == 0) {
        std::string mkdir_cmd = "mkdir -p " + outDir;
        system(mkdir_cmd.c_str());
        
        // Create subdirectories
        for (const auto& sub : SUBCATEGORIES) {
            std::string subOutCmd = "mkdir -p " + outDir + "/" + sub;
            system(subOutCmd.c_str());
        }

        std::cout << "Processing frames from " << frame_from << " to " << frame_to 
                  << " using " << world_size << " processes" 
                  << (overwrite ? " (OVERWRITE)" : "") << "." << std::endl;
    }

    // Barrier to ensure directory structure exists
    MPI_Barrier(MPI_COMM_WORLD);

    for (int frame = frame_from; frame <= frame_to; ++frame) {
        // Simple modulo distribution
        if ((frame - frame_from) % world_size == world_rank) {
             std::ostringstream fname_ss;
             fname_ss << "f" << std::setw(5) << std::setfill('0') << frame << ".h5";
             std::string filename = fname_ss.str();

             for (const auto& sub : SUBCATEGORIES) {
                std::string inputFile  = framesDir + "/" + sub + "/" + filename;
                std::string outputFile = outDir + "/" + sub + "/" + filename;
                
                // Check if input exists
                struct stat st;
                if (stat(inputFile.c_str(), &st) != 0) continue;

                // Log attempt (verbose?)
                // std::cout << "[Process " << world_rank << "] Processing: " << inputFile << std::endl;

                bool result = process_frame_file(inputFile, outputFile, overwrite);
                if (!result) {
                     std::cerr << "[Process " << world_rank << "] Failed to process " << inputFile << std::endl;
                }
             }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "All compression tasks completed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
