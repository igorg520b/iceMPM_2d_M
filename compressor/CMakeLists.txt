cmake_minimum_required(VERSION 3.20)

project(compressor_project LANGUAGES CXX C)

# Fujitsu FCCpx is C++17-ish, so C++20 may not work on Odyssey
# Use 17 for maximum compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)

add_executable(compressor main_compressor_alt.cpp)

# Includes
target_include_directories(compressor PRIVATE ${HDF5_INCLUDE_DIRS})

# Link against HDF5 (CMake provides targets in newer releases)
if(TARGET HDF5::HDF5_CXX)
    target_link_libraries(compressor PRIVATE HDF5::HDF5_CXX)
elseif(HDF5_LIBRARIES)
    target_link_libraries(compressor PRIVATE ${HDF5_LIBRARIES})
else()
    target_link_libraries(compressor PRIVATE hdf5_cpp-shared)
endif()
