#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <filesystem>
#include <H5Cpp.h>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <cxxopts.hpp>
#include "poisson_disk_sampling.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    cxxopts::Options options("raw_point_generator", "Generate Poisson disk point clouds for MPM simulation");

    options.add_options()
        ("width", "Grid width", cxxopts::value<int>()->default_value("22012"))
        ("height", "Grid height", cxxopts::value<int>()->default_value("26164"))
        ("ppc", "Points per cell", cxxopts::value<float>()->default_value("5"))
        ("help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    int width = result["width"].as<int>();
    int height = result["height"].as<int>();
    float ppc = result["ppc"].as<float>();

    spdlog::info("Parameters: width={}, height={}, ppc={}", width, height, ppc);

    const float dy = (float)height / width;
    const std::array<float, 2> x_min{0, 0};
    const std::array<float, 2> x_max{1, dy};
    
    // Logic from DataPreparer::generate_and_save_poisson
    constexpr float magic_constant = 0.6f;
    const float radius = std::sqrt(magic_constant / (ppc * (float)width * width));

    spdlog::info("Generating Poisson points... radius={:.6e}, dy={:.4f}", radius, dy);
    
    std::vector<std::array<float, 2>> buffer = thinks::PoissonDiskSampling(radius, x_min, x_max);
    
    size_t raw_count = buffer.size();
    const float raw_ppc = (float)raw_count / ((float)width * height);
    spdlog::info("Generated {} raw Poisson points. Raw achieved ppc: {:.4f} (target was {:.4f})", raw_count, raw_ppc, ppc);

    // Scale points to achieve target ppc
    const float scale = std::sqrt(raw_ppc / (ppc * 1.0005f));
    spdlog::info("Scaling points by factor: {:.4f}", scale);

    for (auto &pt : buffer) {
        pt[0] *= scale;
        pt[1] *= scale;
    }

    // SKIP out-of-bounds point removal as requested
    /*
    auto result_it = std::remove_if(buffer.begin(), buffer.end(),
                                    [&](const std::array<float, 2> &pt) {
                                        return (pt[0] > 1.0f || pt[1] > dy || pt[0] < 0.0f || pt[1] < 0.0f);
                                    });
    buffer.erase(result_it, buffer.end());
    */

    size_t final_count = buffer.size();
    const float final_ppc = (float)final_count / ((float)width * height);
    double ram_used_mb = (double)(final_count * sizeof(std::array<float, 2>)) / (1024.0 * 1024.0);

    spdlog::info("Final point count: {}", final_count);
    spdlog::info("Final achieved PPC: {:.4f}", final_count / ((float)width * (float)height));
    spdlog::info("Estimated RAM usage for buffer: {:.2f} MB", ram_used_mb);

    // Save to HDF5
    std::string cache_dir = "_data/poisson_cache";
    std::string filename = fmt::format("{}/points_{}x{}_{:d}.h5", cache_dir, width, height, (int)ppc);
    
    fs::create_directories(cache_dir);
    
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);

        hsize_t dims[2] = {static_cast<hsize_t>(final_count), 2};
        H5::DataSpace space(2, dims);

        H5::DataSet dataset = file.createDataSet("coords", H5::PredType::NATIVE_FLOAT, space);
        dataset.write(buffer.data(), H5::PredType::NATIVE_FLOAT);

        file.close();
        spdlog::info("Saved Poisson points to cache: {}", filename);
    } catch (const H5::Exception &e) {
        spdlog::error("Failed to save HDF5 file: {}", e.getDetailMsg());
        return 1;
    }

    return 0;
}
