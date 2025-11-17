#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <random>
#include <mutex>
#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <future>

#include "parameters_sim.h"
#include "gpu_implementation5.h"
#include "windandcurrentinterpolator.h"
#include "host_side_data.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/logger.h>


class Model
{
public:
    Model();
    ~Model();

    // initialize the simulation from a parameter file
    void LoadParameterFile(std::string fileName, std::string resumeSnapshotFileName);

    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI

    HostSideData sim_data;
    GPU_Implementation5 gpu;
    SimParams &prms;

    bool SyncTopologyRequired;  // especially when some points get removed
    std::mutex lock_data_for_GUI; // locked until the current cycle results' are copied to host and processed

    int intentionalSlowdown = 0; // add delay after each computation step to unload GPU
    std::function<void()> transfer_completion_callback;     // invoked after frames are prepared for render

private:
    std::future<void> m_save_future;

    void PrintTimingTable();
};

#endif
