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

#include "parameters_sim.h"
#include "gpu_implementation5.h"
#include "windandcurrentinterpolator.h"
#include "snapshotmanager.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/logger.h>


namespace icy { class Model; }

class icy::Model
{
public:
    Model();
    ~Model();

    // initialize the simulation from a parameter file
    void LoadParameterFile(std::string fileName, std::string resumeSnapshotFileName, bool onlyGeneratePoints = false);

    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void UnlockCycleMutex();
    void SaveFrameRequest(int SimulationStep, double SimulationTime);

    SimParams prms;
    WindAndCurrentInterpolator wac_interpolator;
    icy::SnapshotManager snapshot;

    GPU_Implementation5 gpu;
    bool SyncTopologyRequired;  // especially when some points get removed

    std::mutex processing_current_cycle_data; // locked until the current cycle results' are copied to host and processed
    std::mutex accessing_point_data;

    int intentionalSlowdown = 0; // add delay after each computation step to unload GPU

private:
    void SaveThread();
    std::thread saver_thread;
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    bool frame_ready;
    std::atomic<bool> done;
    int saving_SimulationStep = -1;
    double saving_SimulationTime;
};

#endif
