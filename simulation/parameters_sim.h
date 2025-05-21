#ifndef P_SIM_H
#define P_SIM_H

#include <spdlog/spdlog.h>

#define LOGR(fmtstr, ...) spdlog::info(fmt::format(fmt::runtime(fmtstr), __VA_ARGS__))
#define LOGV(fmtstr) spdlog::info(fmtstr)


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

typedef double real;
//typedef float real;
typedef real t_GridReal;      // data type for grid data
typedef real t_PointReal;     // data type to store point data

typedef Eigen::Matrix<t_GridReal, 2, 1> GridVector2r;
typedef Eigen::Matrix<t_GridReal, 2, 2> GridMatrix2r;
typedef Eigen::Matrix<t_PointReal, 2, 1> PointVector2r;
typedef Eigen::Matrix<t_PointReal, 2, 2> PointMatrix2r;
typedef Eigen::Array<t_PointReal, 2, 1> PointArray2r;

struct SimParams
{
public:
    constexpr static float disabled_pts_proportion_threshold = 0.05; // when exceeded, disabled points are removed
    constexpr static t_PointReal pi = 3.14159265358979323846;
    constexpr static double Earth_Radius = 6371000.0;
    constexpr static float MPM_points_per_cell = 5.0;    // approximate average value
    constexpr static double g = 9.8;

    constexpr static int dim = 2;
    constexpr static int MAX_REGIONS = 100;

    // layout of the grid arrays
    constexpr static size_t grid_idx_mass           = 0;
    constexpr static size_t grid_idx_px             = 1;
    constexpr static size_t grid_idx_py             = 2;
    constexpr static size_t grid_idx_bc_normal_nx   = 3;
    constexpr static size_t grid_idx_bc_normal_ny   = 4;
    constexpr static size_t grid_idx_current_vx     = 5;
    constexpr static size_t grid_idx_current_vy     = 6;
    constexpr static size_t grid_idx_fx             = 7;
    constexpr static size_t grid_idx_fy             = 8;
    constexpr static size_t nGridArrays             = 9;
//    constexpr static size_t grid_idx_wind1_vx       = 7;
//    constexpr static size_t grid_idx_wind1_vy       = 8;
//    constexpr static size_t grid_idx_wind2_vx       = 9;
//    constexpr static size_t grid_idx_wind2_vy       = 10;

    // index of the corresponding array in SoA
    constexpr static size_t idx_utility_data = 0;
    constexpr static size_t integer_cell_idx = idx_utility_data + 1;
    constexpr static size_t integer_point_idx = integer_cell_idx + 1;
    constexpr static size_t idx_P = integer_point_idx + 1;
    constexpr static size_t idx_Q = idx_P + 1;
    constexpr static size_t idx_Jp_inv = idx_Q + 1;
    constexpr static size_t posx = idx_Jp_inv + 1;
    constexpr static size_t velx = posx + 2;
    constexpr static size_t Fe00 = velx + 2;
    constexpr static size_t Bp00 = Fe00 + 4;
    constexpr static size_t idx_thickness = Bp00 + 4;
    constexpr static size_t nPtsArrays = idx_thickness + 1;

    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation

    int nPtsInitial;
    int64_t SimulationStartUnixTime;
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
};

#endif
