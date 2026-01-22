#!/bin/bash

# Load HDF5 module if needed (e.g., module load hdf5)
# Ensure HDF5 libraries/headers are in search path or set variables:
# HDF5_INCL="-I/path/to/hdf5/include"
# HDF5_LIB="-L/path/to/hdf5/lib"

echo "Note: std::filesystem requires C++17 support (-Nclang -std=c++17 or equivalent)."

# --- 1. Serial Version ---
echo "Compiling Serial Version (compressor)..."
FCCpx -Nclang -std=c++17 -O3 -I$HDF5_DIR/include \
    main_serial.cpp core.cpp \
    -L$HDF5_DIR/lib -lhdf5 -lz \
    -lstdc++fs \
    -o compressor

# --- 2. MPI Version ---
echo "Compiling MPI Version (compressor_mpi)..."
mpiFCCpx -Nclang -std=c++17 -O3 -I$HDF5_DIR/include \
    main_mpi.cpp core.cpp \
    -L$HDF5_DIR/lib -lhdf5 -lz \
    -lstdc++fs \
    -o compressor_mpi
