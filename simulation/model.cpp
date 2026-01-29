#include "model.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <thread>
#include "parameters_sim.h"



bool Model::Step()
{
    std::cout << '\n';
    LOGR("step {} ({}) started; sim_time {:>6.3}; host pts {}; cap {}",
                 sim_data.prms.SimulationStep, sim_data.prms.AnimationFrameNumber(),
         sim_data.prms.SimulationTime, sim_data.hssoa.size, sim_data.hssoa.capacity);
    spdlog::default_logger()->flush();

    gpu.reset_timings();
    // gpu.clear_force_accumulator(); // currently don't record forces
    double simulation_time;
    int count_unupdated_steps = 0;

    do
    {
        const int step = sim_data.prms.SimulationStep + count_unupdated_steps;
        simulation_time = sim_data.prms.InitialTimeStep * step;

        gpu.reset_grid();
        gpu.p2g();

        bool frames_changed = false;
        {
            std::lock_guard<std::mutex> lg(lock_data_for_GUI);
            auto [ocean_changed, wind_changed] = sim_data.waci.SetTime(simulation_time);
            if(ocean_changed) gpu.update_ocean_current_field(sim_data.waci);
            if(wind_changed) gpu.update_wind_field(sim_data.waci);
            frames_changed = ocean_changed || wind_changed;
        }

        gpu.update_nodes(simulation_time);
        const bool isCycleEnd = (step + 1) % sim_data.prms.UpdateEveryNthStep == 0;
        gpu.g2p(isCycleEnd, step);

        bool attempt_point_transfer = (step) % sim_data.prms.PointTransferPeriod == 0;
        if(attempt_point_transfer) gpu.point_transfer();
        gpu.record_timings();

        count_unupdated_steps++;
        if(intentionalSlowdown) // for GUI to unfreeze
        {
            gpu.synchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(intentionalSlowdown));
        }
    } while((sim_data.prms.SimulationStep+count_unupdated_steps) % sim_data.prms.UpdateEveryNthStep != 0);

    sim_data.prms.SimulationTime = simulation_time;
    sim_data.prms.SimulationStep += count_unupdated_steps;

    if(m_save_future.valid()) m_save_future.get();
    gpu.render_visualized_data();

    {
        std::lock_guard<std::mutex> lg(lock_data_for_GUI);
        gpu.transfer_from_device();

        // Normalize timings and check for squeeze conditions
        bool squeeze_required = false;
        for(GPU_Partition &p : gpu.partitions)
        {
            p.normalize_timings(count_unupdated_steps);
            const unsigned pts_free_slots = p.pparams.pitch_pts-p.pparams.count_pts;

            const float disabled_proportion = (float)p.get_disabled_pts()/p.pparams.count_pts;
            if(disabled_proportion > SimParams::disabled_pts_proportion_threshold) squeeze_required = true;
            const float free_space_proportion = (float)pts_free_slots/p.pparams.pitch_pts;
            if(gpu.partitions.size() > 1 && free_space_proportion < SimParams::free_space_threshold) squeeze_required = true;
        }
        PrintTimingTable();

        if(squeeze_required)
        {
            sim_data.hssoa.RemoveDisabledAndSort(sim_data.prms.GridYTotal);
            gpu.split_hssoa_into_partitions();
            gpu.transfer_to_device();
            sim_data.waci.SetTime(prms.SimulationTime);
            gpu.update_ocean_current_field(sim_data.waci);
            gpu.update_wind_field(sim_data.waci);
            SyncTopologyRequired = true;
            LOGR("Model::Step() squeezing and sorting HSSOA done\n");
        }
    }

    if(transfer_completion_callback) transfer_completion_callback();    // signal GUI to udpate

    // snapshot is synchronous, frame save is async
    bool saveSnapshot = ((sim_data.prms.SimulationStep / sim_data.prms.UpdateEveryNthStep) % sim_data.prms.SnapshotPeriod == 0) ||
                        (sim_data.prms.SimulationTime >= sim_data.prms.SimulationEndTime);
    if(saveSnapshot) sim_data.SaveSnapshot(sim_data.prms.SimulationStep, sim_data.prms.SimulationTime, false, sim_data.snapshot_directory);  // synchronous

    m_save_future = std::async(std::launch::async, &HostSideData::SaveFrame, &sim_data,
                              sim_data.prms.SimulationStep, sim_data.prms.SimulationTime);

    return (sim_data.prms.SimulationTime < sim_data.prms.SimulationEndTime && !gpu.error_code && sim_data.hssoa.size);
}

Model::Model() : gpu(sim_data), prms(sim_data.prms)
{
    SyncTopologyRequired = true;
    LOGR("Model constructor done");
}

Model::~Model()
{
    if (m_save_future.valid()) m_save_future.get();
    LOGR("Model destructor done");
}


void Model::Prepare()
{
    LOGR("Model::Prepare()");
    gpu.update_constants();
    sim_data.waci.SetTime(prms.SimulationTime);
    gpu.update_ocean_current_field(sim_data.waci);
    gpu.update_wind_field(sim_data.waci);
}


void Model::PrintTimingTable()
{
    LOGR("finished {:>8.1f} of {:>8.1f} ({}); host pts {}; cap {}; err {:#x}",
         sim_data.prms.SimulationTime, sim_data.prms.SimulationEndTime,
         sim_data.prms.AnimationFrameNumber(), sim_data.hssoa.size, sim_data.hssoa.capacity, gpu.error_code);

    // print out timings
    LOGR("{0:^3s} {1:^9s} {2:^7s} {3:^7s} | {4:^8s} {5:^5s} {6:^8s} | {7:^5s} {8:^8s} {9:^7s} {10:^5s} {11:^8s} | {12:^8s}",
         "P-D",  "pts", "free",  "dis",    "p2g",  "s2",  "S12",      "u",  "g2p",   "psnt", "prcv",   "S36",    "tot");

    for(GPU_Partition &p : gpu.partitions)
    {
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
    }
    LOGR("\n");
}


void Model::LoadParameterFile(std::string fileName)
{
    LOGR("Model::LoadParameterFile {}", fileName);

    // Get JSON directory for resolving relative paths
    std::filesystem::path jsonFileDir = std::filesystem::path(fileName).parent_path();
    if (jsonFileDir.empty()) {
        jsonFileDir = ".";
    }

    // Parse configuration from JSON
    std::map<std::string, std::string> parseResult = sim_data.prms.ParseFile(fileName);
    sim_data.SimulationTitle = parseResult["SimulationTitle"];

    // Setup output directory (using jsonFileDir / "output")
    std::filesystem::path outputDir = jsonFileDir / "output";
    std::filesystem::path logDir = outputDir / "logs";
    std::filesystem::create_directories(logDir);
    std::filesystem::path fullLogPath = logDir / "multisink.txt";

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fullLogPath.string(), true);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto lg = std::make_shared<spdlog::logger>("multi_sink", spdlog::sinks_init_list({console_sink, file_sink}));
    spdlog::set_default_logger(lg);
    spdlog::set_pattern("%v");

    // Load grid data (pre-created by plate_preparer)
    std::filesystem::path gridPath = jsonFileDir / parseResult["GridData"];
    sim_data.LoadGridDataFromFile(gridPath.string());

    // Cleaned up directory management
    sim_data.data_directory = jsonFileDir.string();
    sim_data.output_directory = outputDir.string();
    sim_data.snapshot_directory = (jsonFileDir / "snapshots").string();

    // Create directories
    std::filesystem::create_directories(sim_data.output_directory);
    std::filesystem::create_directories(sim_data.snapshot_directory);

    std::filesystem::path framesDir = outputDir / "frames";
    std::filesystem::create_directories(framesDir);

    // Load points from snapshot - STRICT CHECKING
    std::filesystem::path snapshotPath;
    if (parseResult.count("Snapshot")) {
        snapshotPath = parseResult["Snapshot"];
        if (snapshotPath.is_relative()) {
            snapshotPath = jsonFileDir / "snapshots" / snapshotPath;
        }
    } else {
        throw std::runtime_error("Starting simulation requires 'Snapshot' parameter in .json file");
    }

    if (!std::filesystem::exists(snapshotPath)) {
        throw std::runtime_error(fmt::format("Snapshot file not found: {}", snapshotPath.string()));
    }
    LOGR("Loading snapshot from: {}", snapshotPath.string());
    sim_data.ReadPointsFromSnapshot(snapshotPath.string());

    // Allocate point arrays and transfer to GPU partitions
    sim_data.AllocatePointArrays();
    gpu.SplitIntoPartitionsAndTransferToDevice();


    // Load flow field data (mandatory)
    if(parseResult.count("CurrentVelocityData"))
    {
        std::filesystem::path flowPath = jsonFileDir / parseResult["CurrentVelocityData"];
        sim_data.waci.SetHDF5Path(flowPath.string());
    }
    
    if (parseResult.count("ERA5Data")) {
        std::string rawPath = parseResult["ERA5Data"];
        std::filesystem::path era5Path(rawPath);
        if (era5Path.is_relative()) {
            era5Path = jsonFileDir / era5Path;
        }
        sim_data.waci.SetEra5Path(era5Path.string());
    }

    // Load GLO12 if present
    if (parseResult.count("GLO12Data")) {
        std::filesystem::path glo12Path = jsonFileDir / parseResult["GLO12Data"];
        sim_data.waci.SetGLO12Path(glo12Path.string());
    }

    if (parseResult.count("GLO12Tides")) {
        std::filesystem::path glo12TidesPath = jsonFileDir / parseResult["GLO12Tides"];
        sim_data.waci.SetGLO12TidesPath(glo12TidesPath.string());
    }

    LOGR("Model::LoadParameterFile() about to invoke waci.SetTime");
    spdlog::default_logger()->flush();

    sim_data.waci.SetTime(sim_data.prms.SimulationTime);
    gpu.update_ocean_current_field(sim_data.waci);
    gpu.update_wind_field(sim_data.waci);

    // Print memory allocation summary
    LOGR("");
    LOGR("Memory allocation summary:");
    LOGR("  Grid data:    {:.3f} GB", sim_data.allocated_bytes[0] / 1e9);
    LOGR("  Particle data: {:.3f} GB", sim_data.allocated_bytes[1] / 1e9);
    LOGR("  Total:        {:.3f} GB", (sim_data.allocated_bytes[0] + sim_data.allocated_bytes[1]) / 1e9);
    LOGR("");
    spdlog::default_logger()->flush();

    // Final GPU preparation and rendering
    Prepare();
    LOGR("LoadParameterFile - invoking gpu.render_visualized_data();");
    spdlog::default_logger()->flush();
    gpu.render_visualized_data();
    gpu.transfer_from_device();

    LOGR("LoadParameterFile completed successfully");
    spdlog::default_logger()->flush();
}


