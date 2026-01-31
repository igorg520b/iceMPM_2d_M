#include "core.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <sys/sysinfo.h>

static size_t get_available_memory_bytes()
{
    struct sysinfo info;
    if (sysinfo(&info) != 0) return 0;
    return static_cast<size_t>(info.freeram) * info.mem_unit;
}

template<typename T>
static std::vector<T> checked_alloc(size_t n,
                                    const std::string& dset_name,
                                    int world_rank)
{
    size_t bytes = n * sizeof(T);
    size_t avail = get_available_memory_bytes();

    if (avail > 0 && bytes > avail * 8 / 10) {
        std::cerr << "[Rank " << world_rank << "] ERROR: "
                  << "Insufficient memory for dataset '" << dset_name << "'\n"
                  << "  Requested: " << bytes / (1024.0 * 1024.0) << " MB\n"
                  << "  Available: " << avail / (1024.0 * 1024.0) << " MB\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    try {
        return std::vector<T>(n);
    } catch (const std::bad_alloc&) {
        std::cerr << "[Rank " << world_rank << "] FATAL: std::bad_alloc for dataset '"
                  << dset_name << "' (" << bytes / (1024.0 * 1024.0) << " MB)\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
}


// --- Datasets Definitions ---

// Floating point datasets (2D grids)
const std::vector<std::string> FLOAT_DATASETS = {
    // Physics
    "mass", "vx", "vy",
    // Strains
    "strain_eqv", "strain_vm",
    // Pressure / Flow
    "P", "Q", "Jpinv", "glen_flow",
    // Fracture Status
    "thickness"
};

// UInt8 datasets (2D grids)
const std::vector<std::string> UINT8_DATASETS = {
    // Fracture Status
    "crushed", "cracked",
    // Fracture Type
    "tension", "shear", "crush"
};

// Note: "rgba" is special (3D uint8) and handled separately.


// Start of subcategories definition
const std::vector<std::string> SUBCATEGORIES = {
    "color",
    "fracture_status",
    "fracture_type",
    "physics",
    "pressure",
    "strains"
};

bool process_frame_file(const std::string& inputFile, const std::string& outputFile, bool overwrite) {
    hid_t fsrc = H5Fopen(inputFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fsrc < 0) {
        // Function called blindly on files that might not have all datasets, 
        // OR called on files that might not exist?
        // Caller checks existence.
        std::cerr << "  Could not open " << inputFile << std::endl;
        return false;
    }

    hid_t fdst = H5Fcreate(outputFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fdst < 0) {
        std::cerr << "  Could not create output file " << outputFile << std::endl;
        H5Fclose(fsrc);
        return false;
    }

    bool success = true;

    // --- 1. Float Datasets ---
    for (const auto& name : FLOAT_DATASETS) {
        if (H5Lexists(fsrc, name.c_str(), H5P_DEFAULT) <= 0) continue;

        hid_t dsetg = H5Dopen(fsrc, name.c_str(), H5P_DEFAULT);
        if (dsetg < 0) continue;

        hid_t space = H5Dget_space(dsetg);
        hssize_t npoints = H5Sget_simple_extent_npoints(space);

//        std::vector<float> buf(npoints);
        std::vector<float> buf =
            checked_alloc<float>(npoints, name, 0);

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

    // --- 2. UInt8 Datasets ---
    for (const auto& name : UINT8_DATASETS) {
        if (H5Lexists(fsrc, name.c_str(), H5P_DEFAULT) <= 0) continue;

        hid_t dset = H5Dopen(fsrc, name.c_str(), H5P_DEFAULT);
        if (dset < 0) continue;

        hid_t space = H5Dget_space(dset);
        hssize_t npoints = H5Sget_simple_extent_npoints(space);

        // std::vector<uint8_t> buf(npoints);

        std::vector<uint8_t> buf =
            checked_alloc<uint8_t>(npoints, name, 0);

        H5Dread(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

        // Use similar compression for these 2D uint8 maps
        hsize_t chunk_dims[2] = {256, 256};
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 2, chunk_dims);
        H5Pset_deflate(plist, 4); 

        hid_t dset_new = H5Dcreate(fdst, name.c_str(), H5T_NATIVE_UINT8, space,
                                   H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dwrite(dset_new, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

        H5Pclose(plist);
        H5Dclose(dset_new);
        H5Sclose(space);
        H5Dclose(dset);
    }
    
    // "rgba" handling (Special because of 3D chunks and attributes)
    if (H5Lexists(fsrc, "rgba", H5P_DEFAULT) > 0) {
        hid_t dset = H5Dopen(fsrc, "rgba", H5P_DEFAULT);
        if (dset >= 0) {
            hid_t space = H5Dget_space(dset);
            hssize_t npoints = H5Sget_simple_extent_npoints(space);
            //std::vector<uint8_t> buf(npoints);
            std::vector<uint8_t> buf = checked_alloc<uint8_t>(npoints, "rgba", 0);

            H5Dread(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
            
            hsize_t chunk_dims[3] = {128, 128, 4}; // RGBA is depth 4
            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(plist, 3, chunk_dims);
            H5Pset_deflate(plist, 8); // High compression for images
            
            hid_t dset_new = H5Dcreate(fdst, "rgba", H5T_NATIVE_UINT8, space, H5P_DEFAULT, plist, H5P_DEFAULT);
            H5Dwrite(dset_new, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
            
            H5Pclose(plist); H5Dclose(dset_new); H5Sclose(space); H5Dclose(dset);
        }
    }

    // Copy File-level Attributes (SimulationStep, SimulationTime) if they exist
    // This handles attributes from 'color' directory files or any other files.
    // Copy File-level Attributes (SimulationStep, SimulationTime) - Assumed to exist per requirement
    if (H5Aexists(fsrc, "SimulationStep") > 0)
    {
        int simStep;
        hid_t attr = H5Aopen(fsrc, "SimulationStep", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_INT, &simStep);
        H5Aclose(attr);

        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr_new = H5Acreate(fdst, "SimulationStep", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr_new, H5T_NATIVE_INT, &simStep);
        H5Aclose(attr_new);
        H5Sclose(space);
    }

    if (H5Aexists(fsrc, "SimulationTime") > 0)
    {
        double simTime;
        hid_t attr = H5Aopen(fsrc, "SimulationTime", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_DOUBLE, &simTime);
        H5Aclose(attr);

        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr_new = H5Acreate(fdst, "SimulationTime", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr_new, H5T_NATIVE_DOUBLE, &simTime);
        H5Aclose(attr_new);
        H5Sclose(space);
    }



    H5Fclose(fdst);
    H5Fclose(fsrc);

    if (overwrite) {
        // Safe overwrite: delete original, move new to original
        try {
            if (std::filesystem::exists(inputFile)) {
                std::filesystem::remove(inputFile);
            }
            //std::filesystem::rename(outputFile, inputFile);
            // inputFile is now the compressed file
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "  Error overwriting file " << inputFile << ": " << e.what() << std::endl;
            success = false;
        }
    } else {
        // std::cout << "  Successfully compressed: " << inputFile << " -> " << outputFile << std::endl;
    }

    return success;
}
