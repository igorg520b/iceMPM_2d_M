#include "parameters_sim.h"
#include <spdlog/spdlog.h>

#include <string>
#include <filesystem>

void SimParams::Reset()
{
    SimulationStep = 0;
    SimulationTime = 0;
    GridXTotal = GridYTotal = 0;
    ModeledRegionOffsetX = ModeledRegionOffsetY = 0;
    InitializationImageSizeX = InitializationImageSizeY = 0;

    nPartitions = 1;
    GridHaloSize = 10;
    HaloDiffusionThreshold = 5;
    PointTransferPeriod = 20;

    DimensionHorizontal = 0;
    SaveSnapshots = false;
    SnapshotPeriod = 25;

    nPtsInitial = 0;

    waterDragEffectiveQuadratic = 0.1;
    windDragEffectiveQuadratic = 0.001;

    InitialTimeStep = 3.e-5;
    YoungsModulus = 5.e8;
    ParticleViewSize = 2.5f;

    SimulationEndTime = 20000;
    AnimationFramePeriod = 200;

    PoissonsRatio = 0.3;
    IceDensity = 916;

    IceCompressiveStrength = 100e6;
    IceTensileStrength = 10e6;

    IceCompressiveThreshold = 3e6;

    IceShearStrength = 1e6;
    IceShearStrengthFractured = 1e6;
    IceTensileStrength2 = 10e6;
    RidgeFormationCoeff = 30;

    // Glen's Flow Law Parameter A (Pa^-3 s^-1)
    // Default for temperate ice (0C): ~2.4e-24
    // Cold ice (-10C): ~3.5e-25
    GlenA = 2.4e-24;

    DP_phi = 62;
    DP_threshold_p = 1e4;

    tpb_P2G = 256;
    tpb_Upd = 512;
    tpb_G2P = 128;

    InitializationImageSizeX = InitializationImageSizeY = 0;

    UseWindData = false;
    UseGLO12Data = false;
    UseGLO12Tides = false;

    extra_space_pts = 0.15;     // extra allocation for storage of points (to allow transfer between GPU partitions)

    ComputeLame();
    ComputeHelperVariables();
    spdlog::info("SimParams reset");
}



std::map<std::string,std::string> SimParams::ParseFile(std::string fileName)
{
    LOGR("SimParams ParseFile {}",fileName);
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("configuration file is not found");
    std::ifstream fileStream(fileName);
    std::string strConfigFile;
    strConfigFile.resize(std::filesystem::file_size(fileName));
    fileStream.read(strConfigFile.data(), strConfigFile.length());
    fileStream.close();


    rapidjson::Document doc;
    doc.Parse(strConfigFile.data());
    if(!doc.IsObject()) throw std::runtime_error("configuration file is not JSON");

    // parse strings and save into "result"
    std::map<std::string,std::string> result;
    if(doc.HasMember("GridData")) result["GridData"] = doc["GridData"].GetString();
    if(doc.HasMember("CurrentVelocityData")) result["CurrentVelocityData"] = doc["CurrentVelocityData"].GetString();
    if(doc.HasMember("Snapshot")) result["Snapshot"] = doc["Snapshot"].GetString();

    if(doc.HasMember("UseWindData")) UseWindData = doc["UseWindData"].GetBool();
    if(doc.HasMember("ERA5Data")) {
        result["ERA5Data"] = doc["ERA5Data"].GetString();
        UseWindData = true; // Auto-enable if path provided
    }
    
    if(doc.HasMember("GLO12Data")) {
        result["GLO12Data"] = doc["GLO12Data"].GetString();
        UseGLO12Data = true;
    }
    
    if(doc.HasMember("GLO12Tides")) {
        result["GLO12Tides"] = doc["GLO12Tides"].GetString();
        UseGLO12Tides = true;
    }

    // Projection params
    if(doc.HasMember("PROJ_LAT_0")) PROJ_LAT_0 = doc["PROJ_LAT_0"].GetDouble();
    if(doc.HasMember("PROJ_LON_0")) PROJ_LON_0 = doc["PROJ_LON_0"].GetDouble();
    // PROJ_R is static constexpr
    if(doc.HasMember("PROJ_RESIZE_FACTOR")) PROJ_RESIZE_FACTOR = doc["PROJ_RESIZE_FACTOR"].GetDouble();
    if(doc.HasMember("PROJ_TRANSFORM_COEFFS")) {
        const rapidjson::Value& a = doc["PROJ_TRANSFORM_COEFFS"];
        if(a.IsArray() && a.Size() == 6) {
            for(int i = 0; i < 6; i++) PROJ_TRANSFORM_COEFFS[i] = a[i].GetDouble();
        }
    }

    if(doc.HasMember("SimulationTitle")) result["SimulationTitle"] = doc["SimulationTitle"].GetString();
    else result["SimulationTitle"] = "default_simulation";

    if(doc.HasMember("SaveSnapshots")) SaveSnapshots = doc["SaveSnapshots"].GetBool();



    if(doc.HasMember("SnapshotPeriod")) SnapshotPeriod = doc["SnapshotPeriod"].GetInt();
    if(doc.HasMember("SimulationEndTime")) SimulationEndTime = doc["SimulationEndTime"].GetDouble();

    if(doc.HasMember("InitialTimeStep")) InitialTimeStep = doc["InitialTimeStep"].GetDouble();
    if(doc.HasMember("ParticleViewSize")) ParticleViewSize = doc["ParticleViewSize"].GetDouble();
    if(doc.HasMember("AnimationFramePeriod")) AnimationFramePeriod = doc["AnimationFramePeriod"].GetDouble();

    if(doc.HasMember("YoungsModulus")) YoungsModulus = doc["YoungsModulus"].GetDouble();
    if(doc.HasMember("PoissonsRatio")) PoissonsRatio = doc["PoissonsRatio"].GetDouble();
    if(doc.HasMember("IceDensity")) IceDensity = doc["IceDensity"].GetDouble();

    if(doc.HasMember("IceCompressiveStrength")) IceCompressiveStrength = doc["IceCompressiveStrength"].GetDouble();
    if(doc.HasMember("IceTensileStrength")) IceTensileStrength = doc["IceTensileStrength"].GetDouble();
    if(doc.HasMember("IceTensileStrength2")) IceTensileStrength2 = doc["IceTensileStrength2"].GetDouble();
    if(doc.HasMember("IceShearStrength")) IceShearStrength = doc["IceShearStrength"].GetDouble();
    if(doc.HasMember("IceShearStrengthFractured")) IceShearStrengthFractured = doc["IceShearStrengthFractured"].GetDouble();


    if(doc.HasMember("IceCompressiveThreshold")) IceCompressiveThreshold = doc["IceCompressiveThreshold"].GetDouble();
    if(doc.HasMember("DP_phi")) DP_phi = doc["DP_phi"].GetDouble();
    if(doc.HasMember("DP_threshold_p")) DP_threshold_p = doc["DP_threshold_p"].GetDouble();
    if(doc.HasMember("RidgeFormationCoeff")) RidgeFormationCoeff = doc["RidgeFormationCoeff"].GetDouble();
    if(doc.HasMember("GlenA")) GlenA = doc["GlenA"].GetDouble();

    if(doc.HasMember("waterDragEffectiveQuadratic")) waterDragEffectiveQuadratic = doc["waterDragEffectiveQuadratic"].GetDouble();
    if(doc.HasMember("windDragEffectiveQuadratic")) windDragEffectiveQuadratic = doc["windDragEffectiveQuadratic"].GetDouble();

    if(doc.HasMember("tpb_P2G")) tpb_P2G = doc["tpb_P2G"].GetInt();
    if(doc.HasMember("tpb_Upd")) tpb_Upd = doc["tpb_Upd"].GetInt();
    if(doc.HasMember("tpb_G2P")) tpb_G2P = doc["tpb_G2P"].GetInt();
    if(doc.HasMember("nPartitions")) nPartitions = doc["nPartitions"].GetUint();
    if(doc.HasMember("GridHaloSize")) GridHaloSize = doc["GridHaloSize"].GetUint();
    if(doc.HasMember("HaloDiffusionThreshold")) HaloDiffusionThreshold = doc["HaloDiffusionThreshold"].GetUint();

    spdlog::info("SimParams::ParseFile done");
    return result;
}

void SimParams::ComputeLame()
{
    lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
    mu = YoungsModulus/(2*(1+PoissonsRatio));
    kappa = mu*2./3. + lambda;
}

void SimParams::ComputeHelperVariables()
{
    ParticleMass = ParticleArea * IceDensity;

    UpdateEveryNthStep = (int)(AnimationFramePeriod / InitialTimeStep);
    cellsize_inv = 1./cellsize; // cellsize itself is set when loading .h5 file
    Dp_inv = 4./(cellsize*cellsize);
    dt_vol_Dpinv = InitialTimeStep*ParticleArea*Dp_inv;
    vmax = 0.25*cellsize/InitialTimeStep;

    if(nPartitions == 1) extra_space_pts = 0;

    // compute suggested time step
//    double suggested_dt = 0.8 * cellsize * sqrt(IceDensity*ThicknessFrom/YoungsModulus);
//    LOGR("SimParams::ComputeHelperVariables(): suggested_dt: {}", suggested_dt);

    ComputeLame();
}



void SimParams::Printout()
{
    spdlog::info(fmt::format(fmt::runtime("Simulation Parameters:")));
    spdlog::info(fmt::format(fmt::runtime("nPartitions: {}"), nPartitions));
    spdlog::info(fmt::format(fmt::runtime("InitialTimeStep: {}, SimulationEndTime: {}"), InitialTimeStep, SimulationEndTime));
    spdlog::info(fmt::format(fmt::runtime("AnimationFramePeriod: {}"), AnimationFramePeriod));
    spdlog::info(fmt::format(fmt::runtime("GridXTotal: {}, GridYTotal: {}"), GridXTotal, GridYTotal));
    spdlog::info(fmt::format(fmt::runtime("ModeledRegionOffsetX: {}, ModeledRegionOffsetY: {}"), ModeledRegionOffsetX, ModeledRegionOffsetY));
    spdlog::info(fmt::format(fmt::runtime("InitializationImageSizeX: {}, InitializationImageSizeY: {}"), InitializationImageSizeX, InitializationImageSizeY));
    spdlog::info(fmt::format(fmt::runtime("DimensionHorizontal: {}"), DimensionHorizontal));
    spdlog::info(fmt::format(fmt::runtime("SimulationStep: {}"), SimulationStep));
    spdlog::info(fmt::format(fmt::runtime("SimulationTime: {}"), SimulationTime));
    spdlog::info(fmt::format(fmt::runtime("UpdateEveryNthStep: {}"), UpdateEveryNthStep));


    // parameters
    spdlog::info("");
    spdlog::info(fmt::format(fmt::runtime("Parameters:")));
    spdlog::info(fmt::format(fmt::runtime("dt_vol_Dpinv: {}, vmax: {}"), dt_vol_Dpinv, vmax));

    spdlog::info(fmt::format(fmt::runtime("lambda: {}, mu: {}, kappa: {}"), lambda, mu, kappa));
    spdlog::info(fmt::format(fmt::runtime("ParticleArea: {}, ParticleViewSize: {}"), ParticleArea, ParticleViewSize));
    spdlog::info(fmt::format(fmt::runtime("ParticleMass: {}"), ParticleMass));
    spdlog::info(fmt::format(fmt::runtime("DP_phi: {}, DP_threshold_p: {}"), DP_phi, DP_threshold_p));
    spdlog::info(fmt::format(fmt::runtime("PoissonsRatio: {}, YoungsModulus: {}"),
                             PoissonsRatio, YoungsModulus));
    spdlog::info(fmt::format(fmt::runtime("IceCompressiveStrength: {}, IceTensileStrength: {}, IceShearStrength: {}, IceTensileStrength2: {}, IceShearStrengthFractured: {}"),
                             IceCompressiveStrength, IceTensileStrength, IceShearStrength, IceTensileStrength2, IceShearStrengthFractured));
    spdlog::info(fmt::format(fmt::runtime("RidgeFormationCoeff: {}"),
                             RidgeFormationCoeff));
    // points
    spdlog::info("");
    spdlog::info(fmt::format(fmt::runtime("Points:")));
    spdlog::info(fmt::format(fmt::runtime("nPtsInitial: {}"), nPtsInitial));

    // grid
    spdlog::info("");
    spdlog::info(fmt::format(fmt::runtime("Grid:")));
    spdlog::info(fmt::format(fmt::runtime("cellsize: {}; cellsize*InitializationImageSizeX: {}"),
                             cellsize, cellsize * InitializationImageSizeX));
    spdlog::info(fmt::format(fmt::runtime("Sim grid: {} x {}"), GridXTotal, GridYTotal));
    spdlog::info(fmt::format(fmt::runtime("Original image: {} x {}"), InitializationImageSizeX, InitializationImageSizeY));
    spdlog::info(fmt::format(fmt::runtime("Offset of the modelled region: [{}, {}]"),
                             ModeledRegionOffsetX, ModeledRegionOffsetY));
}

bool SimParams::IsPersistentGridArray(int idx)
{
    // All GPU arrays are cleared before visualization rendering
    // Forces (fx, fy) are summarized in a separate phase before render_visualized_data()
    // So no arrays need to be marked as persistent
    return false;
}
