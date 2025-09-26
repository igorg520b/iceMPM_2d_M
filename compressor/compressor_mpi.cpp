#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <sys/stat.h>
#include <hdf5.h>
#include <ctime>
#include <iomanip>
#include <mpi.h> // <-- Include the MPI header

// Grid datasets to compress
const std::vector<std::string> GRID_DATASET_NAMES = {
    "grid_idx_px", "grid_idx_py", "grid_idx_mass", "grid_idx_vis_pts_density",
    "grid_idx_vis_Jpinv", "grid_idx_vis_P", "grid_idx_vis_Q",
    "grid_idx_vis_strain_vonMises"
};

void process_frame_file(const std::string& inputFile, const std::string& outputFile);

int main(int argc, char* argv[]) {
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 4) {
        // Only let the master process (rank 0) print the usage error.
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <project_directory> <frame_from> <frame_to>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string projectDir = argv[1];
    int frame_from = std::stoi(argv[2]);
    int frame_to   = std::stoi(argv[3]);

    if (frame_from > frame_to) {
        if (world_rank == 0) {
            std::cerr << "Error: frame_from must be <= frame_to" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string framesDir = projectDir + "/frames";
    std::string outDir    = projectDir + "/frames_compressed";

    // --- Directory Creation ---
    // Let only the master process create the directory to avoid race conditions.
    if (world_rank == 0) {
        std::string mkdir_cmd = "mkdir -p " + outDir;
        system(mkdir_cmd.c_str());
        std::cout << "Processing frames from " << frame_from << " to "
                  << frame_to << " using " << world_size << " processes." << std::endl;
    }
    
    // --- MPI Work Distribution ---
    // Each process iterates through the total range of frames
    // and processes a file only if the frame number is assigned to its rank.
    for (int frame = frame_from; frame <= frame_to; ++frame) {
        if ((frame - frame_from) % world_size == world_rank) {
            std::ostringstream fname;
            fname << "f" << std::setw(5) << std::setfill('0') << frame << ".h5";

            std::string inputFile  = framesDir + "/" + fname.str();
            std::string outputFile = outDir + "/" + fname.str();
            
            // Add rank to the output for better logging
            std::cout << "[Process " << world_rank << "] Processing: " << inputFile << std::endl;
            process_frame_file(inputFile, outputFile);
        }
    }
    
    // Add a barrier to wait for all processes to finish before finalizing.
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "All compression tasks completed." << std::endl;
    }

    // --- MPI Finalization ---
    MPI_Finalize();

    return 0;
}

void process_frame_file(const std::string& inputFile, const std::string& outputFile) {
    hid_t fsrc = H5Fopen(inputFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fsrc < 0) {
        // It's useful to know which process failed to open a file
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "  [Process " << rank << "] Could not open " << inputFile << std::endl;
        return;
    }

    hid_t fdst = H5Fcreate(outputFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // --- RGB dataset ---
    hid_t dset = H5Dopen(fsrc, "rgb", H5P_DEFAULT);
    if (dset >= 0) {
        hid_t space = H5Dget_space(dset);
        hssize_t npoints = H5Sget_simple_extent_npoints(space);

        std::vector<uint8_t> buf(npoints);
        H5Dread(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

        hsize_t chunk_dims[3] = {128, 128, 3};
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 3, chunk_dims);
        H5Pset_deflate(plist, 8);

        hid_t dset_new = H5Dcreate(fdst, "rgb", H5T_NATIVE_UINT8, space,
                                   H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dwrite(dset_new, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

        // Copy attributes
        if (H5Aexists(dset, "SimulationStep") > 0) {
            int simStep;
            hid_t attr = H5Aopen(dset, "SimulationStep", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_INT, &simStep);
            hid_t attr_new = H5Acreate(dset_new, "SimulationStep", H5T_NATIVE_INT,
                                       H5Screate(H5S_SCALAR), H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attr_new, H5T_NATIVE_INT, &simStep);
            H5Aclose(attr);
            H5Aclose(attr_new);
        }

        if (H5Aexists(dset, "SimulationTime") > 0) {
            double simTime;
            hid_t attr = H5Aopen(dset, "SimulationTime", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &simTime);
            hid_t attr_new = H5Acreate(dset_new, "SimulationTime", H5T_NATIVE_DOUBLE,
                                       H5Screate(H5S_SCALAR), H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attr_new, H5T_NATIVE_DOUBLE, &simTime);
            H5Aclose(attr);
            H5Aclose(attr_new);
        }

        H5Pclose(plist);
        H5Dclose(dset_new);
        H5Sclose(space);
        H5Dclose(dset);
    }

    // --- Grid datasets ---
    for (auto& name : GRID_DATASET_NAMES) {
        if (H5Lexists(fsrc, name.c_str(), H5P_DEFAULT) <= 0) continue;

        hid_t dsetg = H5Dopen(fsrc, name.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dsetg);
        hssize_t npoints = H5Sget_simple_extent_npoints(space);

        std::vector<float> buf(npoints);
        H5Dread(dsetg, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

        hsize_t chunk_dims[2] = {256, 256};
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 2, chunk_dims);
        H5Pset_deflate(plist, 4);

        hid_t dset_new = H5Dcreate(fdst, name.c_str(), H5T_NATIVE_FLOAT, space,
                                   H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dwrite(dset_new, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

        H5Pclose(plist);
        H5Dclose(dset_new);
        H5Sclose(space);
        H5Dclose(dsetg);
    }

    H5Fclose(fdst);
    H5Fclose(fsrc);
    
    // Silence this message to avoid cluttering the log with thousands of lines.
    // The message in main() is sufficient to track progress.
    // std::cout << "  Successfully compressed: " << inputFile << std::endl;
}