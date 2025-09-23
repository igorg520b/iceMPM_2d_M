#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

#include <hdf5.h>

// The list of datasets to compare
const std::vector<std::string> GRID_DATASET_NAMES = {
    "grid_idx_px", "grid_idx_py", "grid_idx_mass", "grid_idx_vis_pts_density",
    "grid_idx_vis_Jpinv", "grid_idx_vis_P", "grid_idx_vis_Q",
    "grid_idx_vis_strain_vonMises"
};
const std::string RGB_DATASET_NAME = "rgb";


// Function prototypes
bool compare_float_datasets(hid_t file1, hid_t file2, const std::string& dset_name);
bool compare_uint8_datasets(hid_t file1, hid_t file2, const std::string& dset_name);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file1.h5> <file2.h5>" << std::endl;
        return 1;
    }

    std::string file1_name = argv[1];
    std::string file2_name = argv[2];

    hid_t file1 = H5Fopen(file1_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t file2 = H5Fopen(file2_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file1 < 0 || file2 < 0) {
        std::cerr << "Error opening one or both files." << std::endl;
        if (file1 >= 0) H5Fclose(file1);
        if (file2 >= 0) H5Fclose(file2);
        return 1;
    }

    bool all_match = true;

    // Compare the RGB dataset
    if (!compare_uint8_datasets(file1, file2, RGB_DATASET_NAME)) {
        all_match = false;
    }

    // Compare the grid datasets
    for (const auto& name : GRID_DATASET_NAMES) {
        if (!compare_float_datasets(file1, file2, name)) {
            all_match = false;
        }
    }

    H5Fclose(file1);
    H5Fclose(file2);

    if (all_match) {
        std::cout << "Files appear to be identical." << std::endl;
        return 0;
    } else {
        std::cout << "Files are different." << std::endl;
        return 1;
    }
}

/**
 * @brief Compare a floating-point dataset in two HDF5 files.
 */
bool compare_float_datasets(hid_t file1, hid_t file2, const std::string& dset_name) {
    htri_t exists1 = H5Lexists(file1, dset_name.c_str(), H5P_DEFAULT);
    htri_t exists2 = H5Lexists(file2, dset_name.c_str(), H5P_DEFAULT);

    if (exists1 > 0 && exists2 > 0) {
        hid_t dset1 = H5Dopen(file1, dset_name.c_str(), H5P_DEFAULT);
        hid_t dset2 = H5Dopen(file2, dset_name.c_str(), H5P_DEFAULT);

        hid_t space1 = H5Dget_space(dset1);
        hid_t space2 = H5Dget_space(dset2);

        int rank1 = H5Sget_simple_extent_ndims(space1);
        int rank2 = H5Sget_simple_extent_ndims(space2);

        if (rank1 != rank2) {
            std::cerr << "Dataset '" << dset_name << "' has different ranks." << std::endl;
            return false;
        }

        std::vector<hsize_t> dims1(rank1), dims2(rank1);
        H5Sget_simple_extent_dims(space1, dims1.data(), NULL);
        H5Sget_simple_extent_dims(space2, dims2.data(), NULL);

        if (dims1 != dims2) {
            std::cerr << "Dataset '" << dset_name << "' has different dimensions." << std::endl;
            return false;
        }

        hsize_t n_elements = 1;
        for(int i=0; i<rank1; ++i) n_elements *= dims1[i];

        std::vector<float> data1(n_elements), data2(n_elements);
        H5Dread(dset1, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data1.data());
        H5Dread(dset2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data());

        H5Sclose(space1);
        H5Sclose(space2);
        H5Dclose(dset1);
        H5Dclose(dset2);
        
        if (data1 != data2) {
            std::cerr << "Dataset '" << dset_name << "' has different data." << std::endl;
            return false;
        }

    } else if (exists1 != exists2) {
        std::cerr << "Dataset '" << dset_name << "' exists in one file but not the other." << std::endl;
        return false;
    }
    return true;
}


/**
 * @brief Compare a uint8_t dataset in two HDF5 files.
 */
bool compare_uint8_datasets(hid_t file1, hid_t file2, const std::string& dset_name) {
    htri_t exists1 = H5Lexists(file1, dset_name.c_str(), H5P_DEFAULT);
    htri_t exists2 = H5Lexists(file2, dset_name.c_str(), H5P_DEFAULT);

    if (exists1 > 0 && exists2 > 0) {
        hid_t dset1 = H5Dopen(file1, dset_name.c_str(), H5P_DEFAULT);
        hid_t dset2 = H5Dopen(file2, dset_name.c_str(), H5P_DEFAULT);

        hid_t space1 = H5Dget_space(dset1);
        hid_t space2 = H5Dget_space(dset2);

        int rank1 = H5Sget_simple_extent_ndims(space1);
        int rank2 = H5Sget_simple_extent_ndims(space2);

        if (rank1 != rank2) {
            std::cerr << "Dataset '" << dset_name << "' has different ranks." << std::endl;
            return false;
        }

        std::vector<hsize_t> dims1(rank1), dims2(rank1);
        H5Sget_simple_extent_dims(space1, dims1.data(), NULL);
        H5Sget_simple_extent_dims(space2, dims2.data(), NULL);

        if (dims1 != dims2) {
            std::cerr << "Dataset '" << dset_name << "' has different dimensions." << std::endl;
            return false;
        }

        hsize_t n_elements = 1;
        for(int i=0; i<rank1; ++i) n_elements *= dims1[i];

        std::vector<uint8_t> data1(n_elements), data2(n_elements);
        H5Dread(dset1, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data1.data());
        H5Dread(dset2, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data());

        H5Sclose(space1);
        H5Sclose(space2);
        H5Dclose(dset1);
        H5Dclose(dset2);

        if (data1 != data2) {
            std::cerr << "Dataset '" << dset_name << "' has different data." << std::endl;
            return false;
        }

    } else if (exists1 != exists2) {
        std::cerr << "Dataset '" << dset_name << "' exists in one file but not the other." << std::endl;
        return false;
    }
    return true;
}