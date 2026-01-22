cmake_minimum_required(VERSION 3.20)

project(compressor_project LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
# Explicitly request only C component for HDF5 as C++ bindings are unavailable
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(MPI)

# --- 1. Serial Executable (Always built) ---
add_executable(compressor main_serial.cpp core.cpp)
target_include_directories(compressor PRIVATE ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compressor PRIVATE ${HDF5_LIBRARIES})
# Link 'stdc++fs' for C++17 filesystem support on older compilers
target_link_libraries(compressor PRIVATE stdc++fs)


# --- 2. MPI Executable (Built only if MPI found) ---
if(MPI_FOUND)
    add_executable(compressor_mpi main_mpi.cpp core.cpp)
    target_include_directories(compressor_mpi PRIVATE ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
    
    # Link HDF5 (C libraries)
    target_link_libraries(compressor_mpi PRIVATE ${HDF5_LIBRARIES})
    
    # Link MPI
    target_link_libraries(compressor_mpi PRIVATE MPI::MPI_CXX)
    
    # Link filesystem
    target_link_libraries(compressor_mpi PRIVATE stdc++fs)
    
    message(STATUS "MPI found. Building 'compressor_mpi' executable.")
else()
    message(WARNING "MPI not found. Skipping 'compressor_mpi' executable.")
endif()
