#ifndef P_SIM_H
#define P_SIM_H

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#define LOGR(fmtstr, ...) spdlog::info(fmt::format(fmt::runtime(fmtstr), ##__VA_ARGS__))

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <map>

#include <Eigen/Core>
#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

// variables related to the formulation of the model

struct SimParams
{
public:
    constexpr static float disabled_pts_proportion_threshold = 0.05; // when exceeded, re-balance occurs
    constexpr static float free_space_threshold = 0.01; // when crossed, re-balance occurs
    constexpr static float MPM_points_per_cell = 5.0;    // approximate average value
    constexpr static double g = 9.8;
    constexpr static double pi = 3.14159265358979323846;
    constexpr static double gravity = 9.81;
    constexpr static double rho_water = 1030.0;         // water density

    constexpr static int dim = 2;
    constexpr static int MAX_REGIONS = 255;
    constexpr static int ModelledAreaIndicator = 255;

    // Status flags (bit masks for utility_data)
    constexpr static uint32_t status_crushed = 0x10000;
    constexpr static uint32_t status_cracked = 0x20000;
    constexpr static uint32_t status_disabled = 0x40000;

    // status flags for types of fracture (recorded for visualization)
    constexpr static uint32_t fracture_tension = 0x80000; // bit 19
    constexpr static uint32_t fracture_compression_shear = 0x100000; // bit 20
    constexpr static uint32_t fracture_crush = 0x200000; // bit 21 (22 and 23 are free)

    // GPU allocation
    constexpr static double extra_space_pts = 0.15;               // reserved additional space on devices for points
    constexpr static double points_transfer_buffer_fraction = 0.07;  // % of points that could "fly over" during a given cycle

    // layout of the grid arrays
    constexpr static int grid_arrays_to_clear = 3;  // at reset_grid, which should be cleared
    enum GPUGridArrayIndex : size_t {
        // --- Persistent Arrays (Group 0) ---
        // These arrays persist across the entire time step and are not overwritten by visualization logic
        gpu_grid_idx_mass = 0,
        gpu_grid_idx_px = 1,
        gpu_grid_idx_py = 2,
        nGridArraysGPU = 3, // total count for allocation on GPU

        // group 1
        gpu_grid_idx_vis_r = 0,
        gpu_grid_idx_vis_g = 1,
        gpu_grid_idx_vis_b = 2,

        // group 2
        gpu_grid_idx_vis_Jpinv = 0,
        gpu_grid_idx_vis_P = 1,
        gpu_grid_idx_vis_Q = 2,          

        // group 3
        gpu_grid_idx_vis_pts_density = 0,
        gpu_grid_idx_vis_strain_EqvGreenLagrange = 1,
        gpu_grid_idx_vis_strain_vonMises = 2,

        // group 4
        gpu_grid_idx_vis_crushed = 0,
        gpu_grid_idx_vis_cracked = 1,
        gpu_grid_idx_vis_thickness = 2,

        // group 5
        gpu_grid_idx_fracture_tension = 0,
        gpu_grid_idx_fracture_shear = 1,
        gpu_grid_idx_fracture_crush = 2
    };


    enum HostGridArrayIndex : size_t {
        host_grid_idx_mass = 0,
        grid_idx_px = 1,
        grid_idx_py = 2,

        grid_idx_vis_r = 3,
        grid_idx_vis_g = 4,
        grid_idx_vis_b = 5,
        grid_idx_vis_Jpinv = 6,
        grid_idx_vis_P = 7,
        grid_idx_vis_Q = 8,
        grid_idx_vis_strain_EqvGreenLagrange = 9,
        grid_idx_vis_strain_vonMises = 10,
        grid_idx_vis_pts_density = 11,

        grid_idx_vis_crushed = 12,
        grid_idx_vis_cracked = 13,
        grid_idx_vis_thickness = 14,
        
        grid_idx_fracture_tension = 15,
        grid_idx_fracture_shear = 16,
        grid_idx_fracture_crush = 17,

        nGridArraysHost = 18
    };

    static bool IsPersistentGridArray(int idx);


    // indices in the grid_forcing_buffer to access
    enum GridForcingFramesIndex : size_t {
         grid_idx_current_vx_frame0 = 0,
         grid_idx_current_vy_frame0 = 1,
         grid_idx_current_vx_frame1 = 2,
         grid_idx_current_vy_frame1 = 3,
         
         grid_idx_wind_vx_frame0 = 4,
         grid_idx_wind_vy_frame0 = 5,
         grid_idx_wind_vx_frame1 = 6,
         grid_idx_wind_vy_frame1 = 7,

         nGridForcingArrays = 8
    };

    // index of the corresponding array in SoA
    enum PtArrIdx : size_t {
        // --- Standard Model (Indices 0-19) ---
        idx_utility_data = 0,
        integer_cell_idx = 1,
        
        idx_P = 2,
        idx_Q = 3,
        idx_Jp_inv = 4,

        posx = 5,
        posy = 6,
        
        velx = 7,
        vely = 8,

        Fe00 = 9,       // size 4: deformation gradient (9,10,11,12)
        Bp00 = 13,      // size 4: grad of v with respect to x,y (13,14,15,16)
        
        idx_thickness = 17,
        idx_glen_flow = 18,
        nPtsArrays = 19,
    };

    // GPU and multi-GPU-related params
    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation
    unsigned nPartitions;           // number of partitions split between GPU devices
    unsigned GridHaloSize;
    unsigned HaloDiffusionThreshold;    // must be <GridHaloSize-1
    unsigned PointTransferPeriod;       // how often do we try to transfer points (~GridHaloSize)

    int nPtsInitial;
    double InitialTimeStep, SimulationEndTime;
    double AnimationFramePeriod;
    int SimulationStep;
    double SimulationTime;
    bool SaveSnapshots;
    int SnapshotPeriod;

    // grid
    int GridXTotal, GridYTotal;     // actually used in simulation
    int ModeledRegionOffsetX, ModeledRegionOffsetY;
    int InitializationImageSizeX, InitializationImageSizeY;
    double DimensionHorizontal; // with respect to initialization image

    // wind and/or current data
    double waterDragEffectiveQuadratic;
    double windDragEffectiveQuadratic;

    // Wind and ERA5 Data
    bool UseWindData;

    // GLO12 Ocean Data
    // GLO12 Ocean Data
    bool UseGLO12Data;
    bool UseGLO12Tides;

    // Orthographic Projection Parameters
    double PROJ_LAT_0;
    double PROJ_LON_0;
    static constexpr double PROJ_R = 6371000.0;
    double PROJ_TRANSFORM_COEFFS[6];
    double PROJ_RESIZE_FACTOR;

    // material properties
    double IceDensity, PoissonsRatio, YoungsModulus;
    double IceCompressiveStrength, IceTensileStrength, IceShearStrength, IceTensileStrength2;
    double IceShearStrengthFractured;
    double IceCompressiveThreshold;     // exceding this causes the material to crush

    double RidgeFormationCoeff;
    double GlenA; // Glen's flow law parameter (A) for ice rheology


    double DP_phi, DP_threshold_p;
    double cellsize;
    double ParticleArea, ParticleViewSize;

    // computed parameters/properties
    double dt_vol_Dpinv, vmax;
    double lambda, mu, kappa; // Lame
    double ParticleMass;
    double cellsize_inv, Dp_inv;
    int UpdateEveryNthStep;

    void Reset();
    std::map<std::string,std::string> ParseFile(std::string fileName);  // return additional filenames to load

    void ComputeLame();
    void ComputeHelperVariables();
    int AnimationFrameNumber() { return SimulationStep / UpdateEveryNthStep;}

    void Printout();    // for testing
    size_t getHaloElementCount() { return GridYTotal*GridHaloSize*2; }
};

#endif
