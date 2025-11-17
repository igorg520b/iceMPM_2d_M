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

#include <H5Cpp.h>
// variables related to the formulation of the model

struct SimParams
{
public:
    constexpr static float disabled_pts_proportion_threshold = 0.05; // when exceeded, re-balance occurs
    constexpr static float free_space_threshold = 0.01; // when crossed, re-balance occurs
    constexpr static float MPM_points_per_cell = 5.0;    // approximate average value
    constexpr static double g = 9.8;
    constexpr static double pi = 3.14159265358979323846;

    constexpr static int dim = 2;
    constexpr static int MAX_REGIONS = 255;
    constexpr static int ModelledAreaIndicator = 255;

    // layout of the grid arrays
    enum GridArrayIndex : size_t {
        grid_idx_mass = 0,
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
        grid_idx_fx = 12,
        grid_idx_fy = 13,
        grid_idx_current_vx_frame0 = 14,
        grid_idx_current_vx_frame1 = 15,
        grid_idx_current_vy_frame0 = 16,
        grid_idx_current_vy_frame1 = 17,
        nGridArrays = 18
    };

    // index of the corresponding array in SoA
    enum PtArrIdx : size_t {
        idx_utility_data = 0,
        integer_cell_idx = 1,
        integer_point_idx = 2,
        idx_P = 3,
        idx_Q = 4,
        idx_Jp_inv = 5,
        posx = 6,
        velx = 8,
        Fe00 = 10,
        Bp00 = 14,
        idx_thickness = 18,
        idx_pt_color_RGB = 19,
        nPtsArrays = 22
    };

    // GPU and multi-GPU-related params
    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation
    unsigned nPartitions;           // number of partitions split between GPU devices
    unsigned GridHaloSize;
    unsigned HaloDiffusionThreshold;    // must be <GridHaloSize-1
    unsigned PointTransferPeriod;       // how often do we try to transfer points (~GridHaloSize)
    float extra_space_pts;               // reserved additional space on devices for points
    float points_transfer_buffer_fraction;  // % of points that could "fly over" during a given cycle

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
    double windDragCoeff_airDensity;
    bool UseWindData, UseCurrentData;
    double sea_water_density;
    double waterDragEffectiveLinear, waterDragEffectiveQuadratic;

    // material properties
    double IceDensity, PoissonsRatio, YoungsModulus;
    double IceCompressiveStrength, IceTensileStrength, IceShearStrength, IceTensileStrength2;
    double DP_phi, DP_threshold_p;
    double RidgeFormationCoeff;
    double cellsize;
    double ParticleVolume, ParticleViewSize;

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
