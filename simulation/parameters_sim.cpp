#include "parameters_sim.h"
#include <spdlog/spdlog.h>

#include <string>
#include <filesystem>

void SimParams::Reset()
{
    UseWindData = false;
    UseCurrentData = false;
    SimulationStartUnixTime = 0;
    DimensionHorizontal = 0;
    SaveSnapshots = false;
    SnapshotPeriod = 25;

    nPtsInitial = 0;
    windDragCoeff_airDensity = 0.0025 * 1.2;

    waterDragEffectiveLinear = 0.01;
    waterDragEffectiveQuadratic = 0.01;

    InitialTimeStep = 3.e-5;
    YoungsModulus = 5.e8;
    ParticleViewSize = 2.5f;
    sea_water_density = 1030;

    SimulationEndTime = 20000;
    AnimationFramePeriod = 200;

    PoissonsRatio = 0.3;
    IceDensity = 916;

    SimulationStep = 0;
    SimulationTime = 0;

    IceCompressiveStrength = 100e6;
    IceTensileStrength = 10e6;
    IceShearStrength = 1e6;
    IceTensileStrength2 = 10e6;
    RidgeFormationCoeff = 30;

    DP_phi = 62;
    DP_threshold_p = 1e4;

    tpb_P2G = 256;
    tpb_Upd = 512;
    tpb_G2P = 128;

    InitializationImageSizeX = InitializationImageSizeY = 0;

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
    result["InputPNG"] = doc["InputPNG"].GetString();
    result["InputMap"] = doc["InputMap"].GetString();
    if(doc.HasMember("InputFlowVelocity")) result["InputFlowVelocity"] = doc["InputFlowVelocity"].GetString();
    if(doc.HasMember("SimulationTitle")) result["SimulationTitle"] = doc["SimulationTitle"].GetString();
    else result["SimulationTitle"] = "sim1";

    if(doc.HasMember("DimensionHorizontal")) DimensionHorizontal = doc["DimensionHorizontal"].GetDouble();
    if(doc.HasMember("SaveSnapshots")) SaveSnapshots = doc["SaveSnapshots"].GetBool();
    if(doc.HasMember("UseCurrentData")) UseCurrentData = doc["UseCurrentData"].GetBool();

    if(doc.HasMember("SnapshotPeriod")) SnapshotPeriod = doc["SnapshotPeriod"].GetInt();

    if(doc.HasMember("SimulationStartUnixTime")) SimulationStartUnixTime = doc["SimulationStartUnixTime"].GetInt64();
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

    if(doc.HasMember("DP_phi")) DP_phi = doc["DP_phi"].GetDouble();
    if(doc.HasMember("DP_threshold_p")) DP_threshold_p = doc["DP_threshold_p"].GetDouble();
    if(doc.HasMember("RidgeFormationCoeff")) RidgeFormationCoeff = doc["RidgeFormationCoeff"].GetDouble();

    if(doc.HasMember("waterDragEffectiveLinear")) waterDragEffectiveLinear = doc["waterDragEffectiveLinear"].GetDouble();
    if(doc.HasMember("waterDragEffectiveQuadratic")) waterDragEffectiveQuadratic = doc["waterDragEffectiveQuadratic"].GetDouble();

    if(doc.HasMember("tpb_P2G")) tpb_P2G = doc["tpb_P2G"].GetInt();
    if(doc.HasMember("tpb_Upd")) tpb_Upd = doc["tpb_Upd"].GetInt();
    if(doc.HasMember("tpb_G2P")) tpb_G2P = doc["tpb_G2P"].GetInt();

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
    ParticleMass = ParticleVolume * IceDensity;

    UpdateEveryNthStep = (int)(AnimationFramePeriod / InitialTimeStep);
    cellsize_inv = 1./cellsize; // cellsize itself is set when loading .h5 file
    Dp_inv = 4./(cellsize*cellsize);
    dt_vol_Dpinv = InitialTimeStep*ParticleVolume*Dp_inv;
    vmax = 0.25*cellsize/InitialTimeStep;

    ComputeLame();
}



void SimParams::Printout()
{
    spdlog::info(fmt::format(fmt::runtime("Simulation Parameters:")));
    spdlog::info(fmt::format(fmt::runtime("SimulationStartUnixTime: {}"), SimulationStartUnixTime));
    spdlog::info(fmt::format(fmt::runtime("InitialTimeStep: {}, SimulationEndTime: {}"), InitialTimeStep, SimulationEndTime));
    spdlog::info(fmt::format(fmt::runtime("AnimationFramePeriod: {}"), AnimationFramePeriod));
    spdlog::info(fmt::format(fmt::runtime("SimulationStep: {}"), SimulationStep));
    spdlog::info(fmt::format(fmt::runtime("SimulationTime: {}"), SimulationTime));
    spdlog::info(fmt::format(fmt::runtime("UpdateEveryNthStep: {}"), UpdateEveryNthStep));
    spdlog::info(fmt::format(fmt::runtime("UseWindData: {}"), UseWindData));
    spdlog::info(fmt::format(fmt::runtime("UseCurrentData: {}"), UseCurrentData));

    // parameters
    spdlog::info("");
    spdlog::info(fmt::format(fmt::runtime("Parameters:")));
    spdlog::info(fmt::format(fmt::runtime("dt_vol_Dpinv: {}, vmax: {}"), dt_vol_Dpinv, vmax));
    spdlog::info(fmt::format(fmt::runtime("windDragCoeff_airDensity: {}"), windDragCoeff_airDensity));
    spdlog::info(fmt::format(fmt::runtime("lambda: {}, mu: {}, kappa: {}"), lambda, mu, kappa));
    spdlog::info(fmt::format(fmt::runtime("ParticleVolume: {}, ParticleViewSize: {}"), ParticleVolume, ParticleViewSize));
    spdlog::info(fmt::format(fmt::runtime("ParticleMass: {}"), ParticleMass));
    spdlog::info(fmt::format(fmt::runtime("DP_phi: {}, DP_threshold_p: {}"), DP_phi, DP_threshold_p));
    spdlog::info(fmt::format(fmt::runtime("PoissonsRatio: {}, YoungsModulus: {}"),
                             PoissonsRatio, YoungsModulus));
    spdlog::info(fmt::format(fmt::runtime("IceCompressiveStrength: {}, IceTensileStrength: {}, IceShearStrength: {}, IceTensileStrength2: {}"),
                             IceCompressiveStrength, IceTensileStrength, IceShearStrength, IceTensileStrength2));
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
