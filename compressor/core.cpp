#include "core.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <filesystem>

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

    // --- 2. UInt8 Datasets ---
    for (const auto& name : UINT8_DATASETS) {
        if (H5Lexists(fsrc, name.c_str(), H5P_DEFAULT) <= 0) continue;

        hid_t dset = H5Dopen(fsrc, name.c_str(), H5P_DEFAULT);
        if (dset < 0) continue;

        hid_t space = H5Dget_space(dset);
        hssize_t npoints = H5Sget_simple_extent_npoints(space);

        std::vector<uint8_t> buf(npoints);
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
            std::vector<uint8_t> buf(npoints);
            H5Dread(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
            
            hsize_t chunk_dims[3] = {128, 128, 4}; // RGBA is depth 4
            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(plist, 3, chunk_dims);
            H5Pset_deflate(plist, 8); // High compression for images
            
            hid_t dset_new = H5Dcreate(fdst, "rgba", H5T_NATIVE_UINT8, space, H5P_DEFAULT, plist, H5P_DEFAULT);
            H5Dwrite(dset_new, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
            
            // Attributes
             if (H5Aexists(dset, "SimulationStep") > 0) {
                int simStep;
                hid_t attr = H5Aopen(dset, "SimulationStep", H5P_DEFAULT);
                H5Aread(attr, H5T_NATIVE_INT, &simStep);
                hid_t attr_new = H5Acreate(dset_new, "SimulationStep", H5T_NATIVE_INT, H5Screate(H5S_SCALAR), H5P_DEFAULT, H5P_DEFAULT);
                H5Awrite(attr_new, H5T_NATIVE_INT, &simStep);
                H5Aclose(attr); H5Aclose(attr_new);
            }
            if (H5Aexists(dset, "SimulationTime") > 0) {
                double simTime;
                hid_t attr = H5Aopen(dset, "SimulationTime", H5P_DEFAULT);
                H5Aread(attr, H5T_NATIVE_DOUBLE, &simTime);
                hid_t attr_new = H5Acreate(dset_new, "SimulationTime", H5T_NATIVE_DOUBLE, H5Screate(H5S_SCALAR), H5P_DEFAULT, H5P_DEFAULT);
                H5Awrite(attr_new, H5T_NATIVE_DOUBLE, &simTime);
                H5Aclose(attr); H5Aclose(attr_new);
            }

            H5Pclose(plist); H5Dclose(dset_new); H5Sclose(space); H5Dclose(dset);
        }
    }

    H5Fclose(fdst);
    H5Fclose(fsrc);

    if (overwrite) {
        // Safe overwrite: delete original, move new to original
        try {
            if (std::filesystem::exists(inputFile)) {
                std::filesystem::remove(inputFile);
            }
            std::filesystem::rename(outputFile, inputFile);
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
