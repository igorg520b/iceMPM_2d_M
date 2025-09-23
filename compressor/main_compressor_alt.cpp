#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <hdf5.h>
#include <sys/stat.h>  // mkdir()

// Grid datasets to compress
const char* GRID_DATASET_NAMES[] = {
    "grid_idx_px", "grid_idx_py", "grid_idx_mass", "grid_idx_vis_pts_density",
    "grid_idx_vis_Jpinv", "grid_idx_vis_P", "grid_idx_vis_Q",
    "grid_idx_vis_strain_vonMises"
};
const int NUM_GRID_DATASETS = 8;

void process_frame_file(const char* inputFile, const char* outputFile);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <frame_from> <frame_to>\n", argv[0]);
        return 1;
    }

    int frame_from = atoi(argv[1]);
    int frame_to   = atoi(argv[2]);
    if (frame_from > frame_to) {
        fprintf(stderr, "Error: frame_from must be <= frame_to\n");
        return 1;
    }

    printf("Processing frames from %d to %d\n", frame_from, frame_to);

    // Create output directory if it doesn’t exist
    mkdir("frames_compressed", 0777);

    clock_t start = clock();

    for (int frame = frame_from; frame <= frame_to; ++frame) {
        char fname[256];
        snprintf(fname, sizeof(fname), "f%d.h5", frame);

        char outname[256];
        snprintf(outname, sizeof(outname), "frames_compressed/f%d.h5", frame);

        process_frame_file(fname, outname);
    }

    clock_t end = clock();
    double secs = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\n--------------------------------------------------\n");
    printf("Compression complete.\n");
    printf("Processed %d files in %.1f seconds.\n",
           (frame_to - frame_from + 1), secs);
    printf("--------------------------------------------------\n");

    return 0;
}

void process_frame_file(const char* inputFile, const char* outputFile) {
    hid_t fsrc = H5Fopen(inputFile, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fsrc < 0) {
        fprintf(stderr, "  Could not open %s\n", inputFile);
        return;
    }

    hid_t fdst = H5Fcreate(outputFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // --- Process "rgb" dataset ---
    hid_t dset = H5Dopen(fsrc, "rgb", H5P_DEFAULT);
    if (dset >= 0) {
        hid_t space = H5Dget_space(dset);
        hssize_t npoints = H5Sget_simple_extent_npoints(space);

        uint8_t* buf = (uint8_t*)malloc(npoints * sizeof(uint8_t));
        H5Dread(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

        // Create compression properties
        hsize_t chunk_dims[3] = {128, 128, 3};
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 3, chunk_dims);
        H5Pset_deflate(plist, 8);

        hid_t dset_new = H5Dcreate(fdst, "rgb", H5T_NATIVE_UINT8, space,
                                   H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dwrite(dset_new, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

        // Copy attributes (SimulationStep, SimulationTime)
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

        free(buf);
        H5Pclose(plist);
        H5Dclose(dset_new);
        H5Sclose(space);
        H5Dclose(dset);
    }

    // --- Grid datasets ---
    for (int i = 0; i < NUM_GRID_DATASETS; ++i) {
        const char* name = GRID_DATASET_NAMES[i];
        if (H5Lexists(fsrc, name, H5P_DEFAULT) <= 0) continue;

        hid_t dsetg = H5Dopen(fsrc, name, H5P_DEFAULT);
        hid_t space = H5Dget_space(dsetg);
        hssize_t npoints = H5Sget_simple_extent_npoints(space);

        float* buf = (float*)malloc(npoints * sizeof(float));
        H5Dread(dsetg, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

        hsize_t chunk_dims[2] = {256, 256};
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 2, chunk_dims);
        H5Pset_deflate(plist, 4);

        hid_t dset_new = H5Dcreate(fdst, name, H5T_NATIVE_FLOAT, space,
                                   H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dwrite(dset_new, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

        free(buf);
        H5Pclose(plist);
        H5Dclose(dset_new);
        H5Sclose(space);
        H5Dclose(dsetg);
    }

    H5Fclose(fdst);
    H5Fclose(fsrc);

    printf("  Successfully compressed: %s\n", inputFile);
}
