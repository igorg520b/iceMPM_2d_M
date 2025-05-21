#include "model.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <thread>


bool icy::Model::Step()
{
    std::cout << '\n';
    LOGR("step {} ({}) started; sim_time {:>6.3}; host pts {}; cap {}",
                 prms.SimulationStep, prms.AnimationFrameNumber(), prms.SimulationTime, gpu.hssoa.size, gpu.hssoa.capacity);

    gpu.reset_timings();
    gpu.clear_force_accumulator();
    double simulation_time;
    int count_unupdated_steps = 0;

    do
    {
        const int step = prms.SimulationStep + count_unupdated_steps;
        simulation_time = prms.InitialTimeStep * step;

        gpu.reset_grid();
        gpu.p2g();

//        if(prms.UseWindData && wind_interpolator.setTime(simulation_time)) gpu.update_wind_velocity_grid();

        gpu.update_nodes(simulation_time, 0, 0);
        const bool isCycleStart = step % prms.UpdateEveryNthStep == 0;
        gpu.g2p(isCycleStart, false, false);
        gpu.record_timings(false);

        count_unupdated_steps++;
        if(intentionalSlowdown)
        {
            gpu.synchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(intentionalSlowdown));
        }
    } while((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep != 0);

    processing_current_cycle_data.lock();   // if locked, previous results are not yet processed by the host
    accessing_point_data.lock();

    gpu.transfer_from_device(count_unupdated_steps);
    snapshot.PrepareFrameArrays();
    prms.SimulationTime = simulation_time;
    prms.SimulationStep += count_unupdated_steps;
    LOGR("finished {:>8.1f} of {:>8.1f} ({}); host pts {}; cap {}; err {:#x}", prms.SimulationTime, prms.SimulationEndTime,
         prms.AnimationFrameNumber(), gpu.hssoa.size, gpu.hssoa.capacity, gpu.error_code);

    // print out timings
    LOGR("{:^3s} {:^8s} {:^8s} {:^7s} | {:^5s} {:^5s} {:^5s} | {:^5s} {:^5s} {:^5s} {:^5s} {:^5s} | {:^6s}",
                 "P-D",  "pts", "free", "dis",  "p2g",  "s2",  "S12",     "u",  "g2p", "psnt", "prcv","S36", "tot");
    GPU_Partition &p = gpu.partitions.front();
    p.normalize_timings(count_unupdated_steps);
    LOGR("{:>1}-{:>1} {:>8} {:>8} {:>7} | {:>5.1f} {:>5.1f} {:>5.1f} | {:>5.1f} {:>5.1f} {:>5.1f} {:5.1f} {:5.1f} | {:>6.1f}",
                 p.PartitionID, p.Device, p.nPts_partition, (p.nPtsPitch-p.nPts_partition), p.disabled_points_count,
                 p.timing_10_P2GAndHalo, p.timing_20_acceptHalo, (p.timing_10_P2GAndHalo + p.timing_20_acceptHalo),
                 p.timing_30_updateGrid, p.timing_40_G2P, p.timing_60_ptsSent, p.timing_70_ptsAccepted,
                 (p.timing_30_updateGrid + p.timing_40_G2P + p.timing_60_ptsSent + p.timing_70_ptsAccepted),
                 p.timing_stepTotal);

    const float disabled_proportion = (float)p.disabled_points_count/p.nPts_partition;
    if(disabled_proportion > SimParams::disabled_pts_proportion_threshold)
    {
        LOGV("Model::Step() squeezing and sorting HSSOA");
        gpu.hssoa.RemoveDisabledAndSort(prms.GridYTotal);
        gpu.split_hssoa_into_partitions();
        gpu.transfer_to_device();
        SyncTopologyRequired = true;
        LOGV("Model::Step() rebalancing done");
    }

    accessing_point_data.unlock();
    if(prms.SaveSnapshots) SaveFrameRequest(prms.SimulationStep, prms.SimulationTime);

    return (prms.SimulationTime < prms.SimulationEndTime && !gpu.error_code);
}




icy::Model::Model() : frame_ready(false), done(false), wac_interpolator(prms)
{
    snapshot.model = this;
    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    SyncTopologyRequired = true;

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto lg = std::make_shared<spdlog::logger>("multi_sink", spdlog::sinks_init_list({console_sink, file_sink}));
    spdlog::set_default_logger(lg);
    spdlog::set_pattern("%v");

    prms.Reset();
    gpu.model = this;
    GPU_Partition::prms = &this->prms;
    saver_thread = std::thread(&icy::Model::SaveThread, this);
    LOGV("Model constructor");
}


void icy::Model::SaveFrameRequest(int SimulationStep, double SimulationTime)
{
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        LOGR("icy::Model::SaveFrameRequest; step {}",SimulationStep);
        saving_SimulationStep = SimulationStep;
        saving_SimulationTime = SimulationTime;
        frame_ready = true; // Indicate that a new frame is ready
    }
    frame_cv.notify_one(); // Notify the saver thread
}


icy::Model::~Model()
{
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        done = true; // Signal that we're done
        saving_SimulationStep = -1;
    }
    frame_cv.notify_one(); // Notify the saver thread
    saver_thread.join();   // Wait for the thread to finish
}


// Frame-saving thread function
void icy::Model::SaveThread() {
    while (true) {
        // Wait for a frame to save or for the simulation to finish
        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            frame_cv.wait(lock, [this] { return frame_ready || done.load(); });

            if (frame_ready) {
                frame_ready = false; // Mark the frame as consumed
            } else if (done.load()) {
                break; // Exit if simulation is finished
            }

            // Save the frame
            if (saving_SimulationStep != -1) {
                bool saveSnapshot = (saving_SimulationStep/prms.UpdateEveryNthStep)%prms.SnapshotPeriod == 0;
                if (prms.SimulationTime >= prms.SimulationEndTime) saveSnapshot = true;
                accessing_point_data.lock();
                if(saveSnapshot) snapshot.SaveSnapshot(saving_SimulationStep, saving_SimulationTime);
                snapshot.SaveFrameCompressed(saving_SimulationStep, saving_SimulationTime);
                //snapshot.SaveFrame(saving_SimulationStep, saving_SimulationTime);

                accessing_point_data.unlock();

                saving_SimulationStep = -1;
            }
        }
    }
}



void icy::Model::UnlockCycleMutex()
{
    // current data was handled by host - allow next cycle to proceed
    processing_current_cycle_data.unlock();
}


void icy::Model::Prepare()
{
    LOGV("icy::Model::Prepare()");
    //abortRequested = false;
    gpu.update_constants();

    wac_interpolator.SetTime(prms.SimulationTime);

    gpu.transfer_wind_and_current_data_to_device();
}



void icy::Model::LoadParameterFile(std::string fileName, std::string resumeSnapshotFileName, bool onlyGeneratePoints)
{
    LOGR("icy::Model::LoadParameterFile {}", fileName);

    std::map<std::string,std::string> additionalFiles = prms.ParseFile(fileName);

    snapshot.SimulationTitle = additionalFiles["SimulationTitle"];

    snapshot.PrepareGrid(additionalFiles["InputPNG"], additionalFiles["InputMap"]);

    if(resumeSnapshotFileName.empty())
    {
        snapshot.PopulatePoints(additionalFiles["InputMap"], onlyGeneratePoints);
        if(onlyGeneratePoints) return;
    }
    else
    {
        std::filesystem::path inputPath(resumeSnapshotFileName);
        if (!inputPath.has_parent_path())
        {
            std::filesystem::path outputDir = "output";
            std::filesystem::path snapshotsDir = "snapshots";
            std::filesystem::path targetPath = outputDir / snapshot.SimulationTitle / snapshotsDir / resumeSnapshotFileName;
            resumeSnapshotFileName = targetPath.string();
        }

        // try to load snapshot file
        snapshot.ReadPointsFromSnapshot(resumeSnapshotFileName);
    }
    snapshot.SplitIntoPartitionsAndTransferToDevice();

    if(additionalFiles.count("InputFlowVelocity"))
    {
        prms.UseCurrentData = true;
        wac_interpolator.OpenCustomHDF5(additionalFiles["InputFlowVelocity"]);
    }
    snapshot.PrepareFrameArrays();

    //    if(additionalFiles.count("InputWindData")) model.snapshot.LoadWindData(additionalFiles["InputWindData"]);
}

