#include "model.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <thread>
#include "parameters_sim.h"

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
        const bool isCycleEnd = (step + 1) % prms.UpdateEveryNthStep == 0;
        gpu.g2p(isCycleEnd);

        bool attempt_point_transfer = (step) % prms.PointTransferPeriod == 0;
        if(attempt_point_transfer) gpu.point_transfer();
        gpu.record_timings();

        count_unupdated_steps++;
        if(intentionalSlowdown)
        {
            gpu.synchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(intentionalSlowdown));
        }
    } while((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep != 0);

    prms.SimulationTime = simulation_time;
    prms.SimulationStep += count_unupdated_steps;

    if(m_save_future.valid())
    {
        auto start_time = std::chrono::steady_clock::now();
        m_save_future.get(); // wait until frame is saved
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        LOGR("waited to finish saving frame: {} ms", (int)duration_ms);
    }

    bool saveSnapshot = ((prms.SimulationStep / prms.UpdateEveryNthStep) % prms.SnapshotPeriod == 0) ||
                        (prms.SimulationTime >= prms.SimulationEndTime);
    // wait until snapshot is saved, only if we need to save it this turn
    if(saveSnapshot && m_save_full_snapshot_future.valid())
    {
        LOGR("waiting to finish saving snapshot; step (after update) {}; time (after update) {}", prms.SimulationStep, prms.SimulationTime);
        m_save_full_snapshot_future.get();
    }

    gpu.render_visualized_data();

    lock_data_for_GUI.lock(); // prevent GUI from accessing data
    gpu.transfer_from_device();

    LOGR("finished {:>8.1f} of {:>8.1f} ({}); host pts {}; cap {}; err {:#x}", prms.SimulationTime, prms.SimulationEndTime,
         prms.AnimationFrameNumber(), gpu.hssoa.size, gpu.hssoa.capacity, gpu.error_code);
    // print out timings
    LOGR("{0:^3s} {1:^9s} {2:^7s} {3:^7s} | {4:^8s} {5:^5s} {6:^8s} | {7:^5s} {8:^8s} {9:^7s} {10:^5s} {11:^8s} | {12:^8s}",
           "P-D",  "pts", "free",  "dis",    "p2g",  "s2",  "S12",      "u",  "g2p",   "psnt", "prcv",   "S36",    "tot");

    bool squeeze_required = false;
    for(GPU_Partition &p : gpu.partitions)
    {
        p.normalize_timings(count_unupdated_steps);
        const unsigned pts_free_slots = p.pparams.pitch_pts-p.pparams.count_pts;

        LOGR("{0:>1}-{1:>1} {2:>9} {3:>7} {4:>7} | {5:>8.1f} {6:>5.1f} {7:>8.1f} | {8:>5.1f} {9:>8.1f} {10:>7.1f} {11:5.1f} {12:8.1f} | {13:>8.1f}",
             p.pparams.PartitionID, // 0  P-D
             p.Device,              // 1
             p.pparams.count_pts,   // 2 pts
             pts_free_slots, // 3 free space
             p.get_disabled_pts(),   // 4 disabled
             p.timing_10_P2GAndHalo,    // 5
             p.timing_20_acceptHalo,    // 6
             (p.timing_10_P2GAndHalo + p.timing_20_acceptHalo),     // 7
             p.timing_30_updateGrid,    // 8
             p.timing_40_G2P,           // 9
             p.timing_60_ptsSent,       // 10
             p.timing_70_ptsAccepted,   // 11
             (p.timing_30_updateGrid + p.timing_40_G2P + p.timing_60_ptsSent + p.timing_70_ptsAccepted),    // 12
             p.timing_stepTotal);       // 13

        const float disabled_proportion = (float)p.get_disabled_pts()/p.pparams.count_pts;
        if(disabled_proportion > SimParams::disabled_pts_proportion_threshold) squeeze_required = true;
        const float free_space_proportion = (float)pts_free_slots/p.pparams.pitch_pts;
        if(gpu.partitions.size() > 1 && free_space_proportion < SimParams::free_space_threshold) squeeze_required = true;
    }

    if(squeeze_required)
    {
        LOGV("Model::Step() squeezing and sorting HSSOA");
        gpu.hssoa.RemoveDisabledAndSort(prms.GridYTotal);
        gpu.split_hssoa_into_partitions();
        gpu.transfer_to_device();
        wac_interpolator.SetTime(prms.SimulationTime);
        gpu.transfer_wind_and_current_data_to_device();
        SyncTopologyRequired = true;
        LOGV("Model::Step() rebalancing done");
    }

    lock_data_for_GUI.unlock();     // allow GUI to access the data
    if(transfer_completion_callback) transfer_completion_callback();    // signal GUI to udpate

    // request async snapshot
    m_save_future = std::async(std::launch::async, &icy::Model::AsyncSaveFrameTask, this, prms.SimulationStep, prms.SimulationTime);

    if(saveSnapshot)
    {
        gpu.hssoa.transferToSecondBuffer();
        m_save_full_snapshot_future = std::async(std::launch::async, &icy::Model::AsyncSaveFullSnapshotTask, this,
                                                 prms.SimulationStep, prms.SimulationTime);
    }
    return (prms.SimulationTime < prms.SimulationEndTime && !gpu.error_code);
}


icy::Model::Model() : wac_interpolator(prms)
{
    snapshot.model = this;
    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    SyncTopologyRequired = true;

    prms.Reset();
    gpu.model = this;
    GPU_Partition::prms = &this->prms;
    LOGV("Model constructor");
}

icy::Model::~Model()
{
    if (m_save_future.valid()) m_save_future.get();
    if (m_save_full_snapshot_future.valid()) m_save_full_snapshot_future.get();
}


void icy::Model::Prepare()
{
    LOGV("icy::Model::Prepare()");
    gpu.update_constants();
    wac_interpolator.SetTime(prms.SimulationTime);
    gpu.transfer_wind_and_current_data_to_device();
}



void icy::Model::LoadParameterFile(std::string fileName, std::string resumeSnapshotFileName, bool onlyGeneratePoints)
{
    LOGR("icy::Model::LoadParameterFile {}", fileName);

    std::map<std::string,std::string> additionalFiles = prms.ParseFile(fileName);

    snapshot.SimulationTitle = additionalFiles["SimulationTitle"];

    std::filesystem::path outputDir = "output";
    std::filesystem::path logDir = "logs";
    std::filesystem::path targetLogPath = outputDir / snapshot.SimulationTitle / logDir;
    std::filesystem::create_directories(targetLogPath);
    std::filesystem::path fullLogPath = targetLogPath / "multisink.txt";

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fullLogPath.string(), true);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto lg = std::make_shared<spdlog::logger>("multi_sink", spdlog::sinks_init_list({console_sink, file_sink}));
    spdlog::set_default_logger(lg);
    spdlog::set_pattern("%v");

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

    Prepare();
    gpu.render_visualized_data();
    gpu.transfer_from_device();

    // saved snapshot at step 0 (if needed, the snapshot can be uploaded and resumed on a remote machine)
    if(resumeSnapshotFileName.empty())
    {
        m_save_future = std::async(std::launch::async, &icy::Model::AsyncSaveFrameTask, this,
                                   prms.SimulationStep, prms.SimulationTime);
        gpu.hssoa.transferToSecondBuffer();
        m_save_full_snapshot_future = std::async(std::launch::async, &icy::Model::AsyncSaveFullSnapshotTask, this,
                                                 prms.SimulationStep, prms.SimulationTime);
    }
    LOGR("LoadParameterFile done {}", fileName);
}


void icy::Model::AsyncSaveFrameTask(int simulationStep, double simulationTime)
{
    // Save the frame
    if (prms.SaveSnapshots && simulationStep != -1)
    {
        snapshot.SaveFrame(simulationStep, simulationTime);
    }
}

void icy::Model::AsyncSaveFullSnapshotTask(int simulationStep, double simulationTime)
{
    if (prms.SaveSnapshots && simulationStep != -1)
    {
        bool saveSnapshot = ((simulationStep / prms.UpdateEveryNthStep) % prms.SnapshotPeriod == 0) || (simulationTime >= prms.SimulationEndTime);
        if(saveSnapshot)
        {
            snapshot.SaveSnapshot(simulationStep, simulationTime);
        }
    }
}
