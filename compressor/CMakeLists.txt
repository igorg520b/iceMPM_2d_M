cmake_minimum_required(VERSION 3.20.2)

project(compressor_project LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(HDF5 REQUIRED COMPONENTS CXX)

add_executable(compressor main_compressor.cpp)

target_include_directories(compressor PRIVATE ${HDF5_INCLUDE_DIRS})

if(HDF5_INCLUDE_DIRS)
    target_link_libraries(compressor PRIVATE ${HDF5_LIBRARIES})
else()
    target_link_libraries(compressor PRIVATE hdf5_cpp-shared)
endif()



