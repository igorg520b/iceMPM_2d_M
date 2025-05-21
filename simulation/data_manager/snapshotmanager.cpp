// snapshotmanager.cpp

#include "snapshotmanager.h"
#include "model.h"
#include "poisson_disk_sampling.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>

#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <utility>
#include <type_traits>

#include <cmath>        // For std::round, std::abs, std::isnormal, M_PI
#include <limits>       // For std::numeric_limits
#include <stdexcept>    // For std::runtime_error
#include <iostream>     // For LOGR placeholder
#include <cstring>      // For memcpy
#include <map>          // For HDF5 attribute maps


#include <fmt/format.h>
#include <fmt/std.h>

#include <openjpeg.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;


icy::SnapshotManager::SnapshotManager()
{
    data_ready_flag_.store(false);
}

icy::SnapshotManager::~SnapshotManager()
{
    if (decoding_thread_.joinable()) decoding_thread_.join();
}



void icy::SnapshotManager::PrepareGrid(std::string fileNamePNG, std::string fileNameModelledArea)
{
    LOGR("icy::SnapshotManager::PrepareGrid {}", fileNamePNG);
    LOGR("modeled area file: {}", fileNameModelledArea);

    // (1) Load from HDF5: attributes, normals and path_indices
    H5::H5File file(fileNameModelledArea, H5F_ACC_RDONLY);
    int nPaths;

    H5::DataSet ds_path_indices = file.openDataSet("path_indices");
    ds_path_indices.openAttribute("nPaths").read(H5::PredType::NATIVE_INT, &nPaths);
    ds_path_indices.openAttribute("width").read(H5::PredType::NATIVE_INT, &model->prms.InitializationImageSizeX);
    ds_path_indices.openAttribute("height").read(H5::PredType::NATIVE_INT, &model->prms.InitializationImageSizeY);

    ds_path_indices.openAttribute("ModeledRegionOffsetX").read(H5::PredType::NATIVE_INT, &model->prms.ModeledRegionOffsetX);
    ds_path_indices.openAttribute("ModeledRegionOffsetY").read(H5::PredType::NATIVE_INT, &model->prms.ModeledRegionOffsetY);
    ds_path_indices.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &model->prms.GridXTotal);
    ds_path_indices.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &model->prms.GridYTotal);

    const int &width = model->prms.InitializationImageSizeX;
    const int &height = model->prms.InitializationImageSizeY;
    const int &ox = model->prms.ModeledRegionOffsetX;
    const int &oy = model->prms.ModeledRegionOffsetY;
    const int &gx = model->prms.GridXTotal;
    const int &gy = model->prms.GridYTotal;
    model->prms.cellsize = model->prms.DimensionHorizontal / (model->prms.InitializationImageSizeX-1);
    model->prms.cellsize_inv = 1.0/model->prms.cellsize;


    LOGR("initialization image: {} x {}", model->prms.InitializationImageSizeX, model->prms.InitializationImageSizeY);
    LOGR("grid size: {} x {}", model->prms.GridXTotal, model->prms.GridYTotal);
    LOGR("modeled area offset: {}, {}", model->prms.ModeledRegionOffsetX, model->prms.ModeledRegionOffsetY);

    const size_t sz = width*height;

    // temporary buffers for HDF5 data
    std::vector<Eigen::Vector2f> normals(sz);
    std::vector<int> path_indices(sz);

    ds_path_indices.read(path_indices.data(), H5::PredType::NATIVE_INT);
    file.openDataSet("normals").read(normals.data(), H5::PredType::NATIVE_FLOAT);
    file.close();



    // (2) allocate grid arrays host-side
    model->gpu.allocate_host_arrays_grid();



    // (3) Load PNG image (only for rendering)
    int channels, imgx, imgy;
    unsigned char *png_data = stbi_load(fileNamePNG.c_str(), &imgx, &imgy, &channels, 3); // expect 3 channels - RGB
    if(!png_data || channels != 3 || imgx != width || imgy != height)
    {
        LOGR("filename {} not loaded; channels {}", fileNamePNG, channels);
        LOGR("png expected size {} x {}; actual size {} x {}", width, height, imgx, imgy);
        throw std::runtime_error("png1 not loaded");
    }

    // function for obtaining index in png_data from the pixel's 2D index (i,j)
    auto idxInPng = [&](int i, int j) -> int { return 3*((height - j - 1)*width + i); };

    // save original colors for the whole image
    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
            for(int k=0;k<3;k++)
                model->gpu.original_image_colors_rgb[(i+j*width)*3+k] = png_data[idxInPng(i, j)+k];

    stbi_image_free(png_data);



    // (4) Set grid normals and "status" byte
    auto transformPathIdx = [](const int &idx) -> uint8_t {
        if(idx < 1000) return (uint8_t)(idx + 1);
        else return (uint8_t)(idx-900);
    };


    // transfer / set grid land bit
    for(int i=0;i<gx;i++)
        for(int j=0;j<gy;j++)
        {
            size_t idx1 = (i+ox) + (j+oy)*width;
            size_t idx_host = j + i*gy; // store as row-major
            model->gpu.grid_status_buffer[idx_host] = transformPathIdx(path_indices[idx1]);
            model->gpu.grid_boundary_normals[idx_host] = (t_GridReal)normals[idx1].x();
            model->gpu.grid_boundary_normals[idx_host + gx*gy] = (t_GridReal)normals[idx1].y();
        }
    LOGV("PrepareGrid done \n");
}



void icy::SnapshotManager::PopulatePoints(std::string fileNameModelledAreaHDF5, bool onlyGenerateCache)
{
    LOGR("\nicy::SnapshotManager::PopulatePoints: {}", fileNameModelledAreaHDF5);

    // (0) params
    const int &width = model->prms.InitializationImageSizeX;
    const int &height = model->prms.InitializationImageSizeY;
    const int &ox = model->prms.ModeledRegionOffsetX;
    const int &oy = model->prms.ModeledRegionOffsetY;
    const int &gx = model->prms.GridXTotal;
    const int &gy = model->prms.GridYTotal;
    const size_t sz = width*height;
    const t_PointReal &h = model->prms.cellsize;

    std::vector<uint8_t> iceStatus(sz);     // 0 for open water, 1 for crushed, 2 for "intact"
    std::vector<float> iceThickness(sz);    // quantitive measure of "thickness"

    // (1) load iceStatus from HDF5
    H5::H5File file(fileNameModelledAreaHDF5, H5F_ACC_RDONLY);
    file.openDataSet("iceStatus").read(iceStatus.data(), H5::PredType::NATIVE_UINT8);
    file.openDataSet("iceThickness").read(iceThickness.data(), H5::PredType::NATIVE_FLOAT);
    file.close();


    // (2) generate points (or load from cache)
    std::vector<std::array<float, 2>> pt_buffer;   // initial buffer for reading/generating points
    generate_points(gx, gy, SimParams::MPM_points_per_cell, pt_buffer);
    if(onlyGenerateCache) return;   // the reason is to prepare point cache on the "login" node server-side

    const size_t initial_pt_count = pt_buffer.size();
    model->prms.ParticleVolume = h*h*gx*gy/ initial_pt_count;


    // convert unscaled point coordinates to (i,j) pair on the grid
    auto idxPt = [&](const std::array<float,2> &pt) -> std::pair<int,int> {
        const double scale = gx-1;
        return {(int)(pt[0]*scale + 0.5), (int)(pt[1]*scale + 0.5)};
    };


    // return true if the underlying cell is land or open water
    auto shouldRemove = [&](const std::array<float,2> &pt) -> bool {
        auto [i,j] = idxPt(pt);
        if(i<=1 || j<=1 || i>= (gx-2) || j>= (gy-2)) return true; // exclude points right at the boundary

        uint8_t status = iceStatus[(i+ox) + (j+oy)*width];
        return (status != 1 && status != 2);    // 1 - crushed, 2 - intact ice
    };


    std::erase_if(pt_buffer, shouldRemove);  // C++ 20
    model->prms.nPtsInitial = pt_buffer.size();


    // (3) allocate host-side array for points
    // allocate space for points on hssoa
    model->gpu.allocate_host_arrays_points();

    model->gpu.hssoa.size = model->prms.nPtsInitial; // set the actual number of points stored in HSSOA
    LOGR("icy::SnapshotManager::PopulatePoints: allocated {} points on HSSOA", model->prms.nPtsInitial);


    // (6) transfer points
    const double pointScale = (gx-1) * h;

    HostSideSOA &hssoa = model->gpu.hssoa;
    for(size_t k = 0; k<pt_buffer.size(); k++)
    {
        std::array<float,2> &pt = pt_buffer[k];

        auto [i,j] = idxPt(pt);

        // write into SOA
        SOAIterator it = hssoa.begin()+k;
        ProxyPoint &p = *it;
        p.setValue(SimParams::posx, pt[0]*pointScale);      // set point's x-position in SOA
        p.setValue(SimParams::posx+1, pt[1]*pointScale);    // set point's y-position in SOA

        const size_t idx_in_image = ((i+ox) + (j+oy)*width)*3;
        uint32_t r = model->gpu.original_image_colors_rgb[idx_in_image+0];
        uint32_t g = model->gpu.original_image_colors_rgb[idx_in_image+1];
        uint32_t b = model->gpu.original_image_colors_rgb[idx_in_image+2];
        uint32_t rgba = (r << 16) | (g << 8) | b;
        model->gpu.point_colors_rgb[k] = rgba;
        p.setValueInt(SimParams::integer_point_idx, k);

        // set ice type
        int idx = (i+ox) + (j+oy)*width;
        float thickness = iceThickness[idx];
        p.setValue(SimParams::idx_thickness, thickness);

        uint8_t status = iceStatus[idx]; // 1 - crushed; 2 - solid
        uint32_t val = 0;
        if(status == 1) val |= 0x10000;     // crushed
        p.setValueInt(SimParams::idx_utility_data, val);

        p.setValue(SimParams::idx_P, 0.f);
        p.setValue(SimParams::idx_Q, 0.f);
        p.setValue(SimParams::idx_Jp_inv, 1.f);
        for(int idx=0; idx<SimParams::dim; idx++)
        {
            p.setValue(SimParams::Fe00+idx*2+idx, 1.f);
            p.setValue(SimParams::velx+0, 0.f);
            p.setValue(SimParams::velx+1, 0.f);
        }
    }
    LOGV("Points transferred to HSSOA");

    // convert points to cell-based local coordinates
    hssoa.convertToIntegerCellFormat(h);

    // we no longer need the original color in the sattelite photo
    FillModelledAreaWithBlueColor();

    if(model->prms.SaveSnapshots)
    {
        SavePointColors();
        SaveSnapshot(0, 0);
    }
    LOGV("PopulatePoints done \n");
}



void icy::SnapshotManager::FillModelledAreaWithBlueColor()
{
    const int &width = model->prms.InitializationImageSizeX;
    const int &height = model->prms.InitializationImageSizeY;
    const int &ox = model->prms.ModeledRegionOffsetX;
    const int &oy = model->prms.ModeledRegionOffsetY;
    const int &gx = model->prms.GridXTotal;
    const int &gy = model->prms.GridYTotal;

    for(int i=0;i<gx;i++)
        for(int j=0;j<gy;j++)
        {
            uint8_t status = model->gpu.grid_status_buffer[j + i*gy];
            if(status == 100)
            {
                for(int k=0;k<3;k++)
                    model->gpu.original_image_colors_rgb[((i+ox)+(j+oy)*width)*3+k] = waterColor[k];
            }
        }
}

void icy::SnapshotManager::SplitIntoPartitionsAndTransferToDevice()
{
    // particle volume and mass
    model->prms.ComputeHelperVariables();
    model->prms.Printout();

    // allocate GPU partitions
    model->gpu.initialize();
    model->gpu.split_hssoa_into_partitions();
    model->gpu.allocate_device_arrays();
    model->gpu.transfer_to_device();
}


void icy::SnapshotManager::ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5)
{
    LOGR("ReadPointsFromSnapshot {}", fileNameSnapshotHDF5);
    // instead of generating the points, read them from the provided HDF5 (snapshot) file
    H5::H5File file(fileNameSnapshotHDF5, H5F_ACC_RDONLY);
    H5::DataSet ds = file.openDataSet("pts_data");

    ds.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &model->prms.SimulationStep);
    ds.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &model->prms.SimulationTime);
    ds.openAttribute("HSSOA_size").read(H5::PredType::NATIVE_UINT, &model->gpu.hssoa.size);

    int nPtsArrays;
    ds.openAttribute("nPtsArrays").read(H5::PredType::NATIVE_INT, &nPtsArrays);

    H5::DataSpace dsp = ds.getSpace();
    hsize_t dims[2];
    dsp.getSimpleExtentDims(dims, nullptr);

    if(dims[0] != SimParams::nPtsArrays || dims[1] != model->gpu.hssoa.size || nPtsArrays != SimParams::nPtsArrays)
        throw std::runtime_error("icy::SnapshotManager::ReadSnapshot array size mismatch");


    model->prms.nPtsInitial = model->gpu.hssoa.size;
    model->gpu.allocate_host_arrays_points();
    model->gpu.hssoa.size = model->prms.nPtsInitial;

    // Define hyperslab
    hsize_t dims_mem[2] = {SimParams::nPtsArrays, model->gpu.hssoa.capacity};
    H5::DataSpace memspace(2, dims_mem);
    hsize_t offset[2] = {0, 0};      // Start reading at the origin
    hsize_t count[2] = {dims[0], dims[1]}; // Size of the data in the file
    memspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    H5::DataType dtype;
    if constexpr(std::is_same_v<t_PointReal, float>) dtype = H5::PredType::NATIVE_FLOAT;
    else dtype = H5::PredType::NATIVE_DOUBLE;

    ds.read(model->gpu.hssoa.host_buffer, dtype, memspace, dsp);


//    model->gpu.hssoa.RemoveDisabledAndSort(model->prms.GridYTotal);

    FillModelledAreaWithBlueColor();
    ReadPointColors();
    PrepareFrameArrays();
    LOGR("ReadPointsFromSnapshot; hssoa capacity {}; size {}", model->gpu.hssoa.capacity, model->gpu.hssoa.size);
}


void icy::SnapshotManager::SaveSnapshot(int SimulationStep, double SimulationTime)
{
    // ensure that the output directory exists
    fs::path outputDir = "output";
    fs::path snapshotsDir = "snapshots";
    fs::path targetPath = outputDir / SimulationTitle / snapshotsDir;
    fs::create_directories(targetPath);

    // save current state
    const int frame = SimulationStep / model->prms.UpdateEveryNthStep;
//    std::string baseName = std::format("s{:05d}.h5", frame);
    std::string baseName = fmt::format(fmt::runtime("s{:05d}.h5"), frame);

    fs::path fullPath = targetPath / baseName;


    H5::H5File file(fullPath.string(), H5F_ACC_TRUNC);

    // points
    hsize_t dims_points[2] = {SimParams::nPtsArrays, model->gpu.hssoa.size};
    H5::DataSpace dataspace_points(2, dims_points);

    // We are zipping the data, so using chunks
    H5::DSetCreatPropList proplist;
    hsize_t chunk_size = (hsize_t)std::min((unsigned)256*1024, model->gpu.hssoa.size);
    hsize_t chunk_dims[2] = {SimParams::nPtsArrays, chunk_size};
    proplist.setChunk(2, chunk_dims);
    proplist.setDeflate(5);

    H5::DataType dtype;
    if constexpr(std::is_same_v<t_PointReal, float>) dtype = H5::PredType::NATIVE_FLOAT;
    else dtype = H5::PredType::NATIVE_DOUBLE;

    // Define the hyperslab in the memory space
    hsize_t mem_dims[2] = {SimParams::nPtsArrays, model->gpu.hssoa.capacity};
    H5::DataSpace memspace(2, mem_dims);
    hsize_t mem_offset[2] = {0, 0};  // Start at the beginning of the memory array
    hsize_t mem_count[2] = {SimParams::nPtsArrays, model->gpu.hssoa.size};  // Number of elements to select
    memspace.selectHyperslab(H5S_SELECT_SET, mem_count, mem_offset);

    // Write the data to the dataset
    H5::DataSet dataset_pts = file.createDataSet("pts_data", dtype, dataspace_points, proplist);
    dataset_pts.write(model->gpu.hssoa.host_buffer, dtype, memspace, dataspace_points);

    H5::DataSpace att_dspace(H5S_SCALAR);
    dataset_pts.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
    dataset_pts.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
    dataset_pts.createAttribute("HSSOA_size", H5::PredType::NATIVE_UINT, att_dspace).write(H5::PredType::NATIVE_UINT, &model->gpu.hssoa.size);
    int nPtsArrays = SimParams::nPtsArrays;
    dataset_pts.createAttribute("nPtsArrays", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &nPtsArrays);
}


void icy::SnapshotManager::SavePointColors()
{
    // ensure that the output directory exists
    fs::path outputDir = "output";
    fs::path snapshotsDir = "snapshots";
    fs::path targetPath = outputDir / SimulationTitle / snapshotsDir;
    fs::create_directories(targetPath);

    // save current state
    fs::path fullPath = targetPath / "point_colors.h5";

    H5::H5File file(fullPath.string(), H5F_ACC_TRUNC);

    hsize_t dims_points[1] = {model->gpu.point_colors_rgb.size()};
    H5::DataSpace dataspace_points(1, dims_points);

    // We are zipping the data, so using chunks
    H5::DSetCreatPropList proplist;
    hsize_t chunk_size = (hsize_t)std::min((hsize_t)256*1024, dims_points[0]);
    hsize_t chunk_dims[1] = {chunk_size};
    proplist.setChunk(1, chunk_dims);
    proplist.setDeflate(5);

    H5::DataSet dataset_pts = file.createDataSet("pts_colors", H5::PredType::NATIVE_UINT32, dataspace_points, proplist);
    dataset_pts.write(model->gpu.point_colors_rgb.data(), H5::PredType::NATIVE_UINT32);
}


void icy::SnapshotManager::ReadPointColors()
{
    // ensure that the output directory exists
    fs::path outputDir = "output";
    fs::path snapshotsDir = "snapshots";
    fs::path targetPath = outputDir / SimulationTitle / snapshotsDir;
    fs::create_directories(targetPath);

    // save current state
    fs::path fullPath = targetPath / "point_colors.h5";

    H5::H5File file(fullPath.string(), H5F_ACC_RDONLY);
    file.openDataSet("pts_colors").read(model->gpu.point_colors_rgb.data(), H5::PredType::NATIVE_UINT32);
}




std::string icy::SnapshotManager::prepare_file_name(int gx, int gy)
{
//    return std::format("{}/point_cache_{:05d}_{:05d}.h5", pts_cache_path, gx, gy);
    return fmt::format("{}/point_cache_{:05d}_{:05d}.h5", pts_cache_path, gx, gy);
}



// =========================  POINTS GENERATION

void icy::SnapshotManager::generate_points(int gx, int gy, float points_per_cell,
                                                  std::vector<std::array<float, 2>> &buffer)
{
    bool cache_result = attempt_to_fill_from_cache(gx, gy, buffer);
    if(!cache_result) generate_and_save(gx, gy, SimParams::MPM_points_per_cell, buffer);
}


void icy::SnapshotManager::generate_and_save(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer)
{
    const float dy = 1.0f*gy/gx;

    const std::array<float, 2>kXMin {0, 0};
    const std::array<float, 2>kXMax {1, dy};
    constexpr float magic_constant = 0.6;
    const float kRadius = sqrt(magic_constant/(points_per_cell*gx*gx));

    LOGR("generate_and_save: attempting to generate {} points in {}x{} grid", (points_per_cell*gx*gy), gx, gy);
    buffer = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);

    const float result_ppc = (float)buffer.size()/(gx*gy);
    LOGR("grid: {}x{}; generated pts {:>8}; density {:>7.4}",gx, gy, buffer.size(), result_ppc);

    const float scale = sqrt(result_ppc/(points_per_cell*1.0005));
    if(scale<1.0)
    {
        LOGR("requested ppc {}; generated ppc {}", points_per_cell, result_ppc);
        throw std::runtime_error("point generation error");
    }

    LOGR("requested ppc {}; generated ppc {}; scale {}%", points_per_cell, result_ppc, 100*(scale-1.f));

    for(auto &pt : buffer)
    {
        pt[0] *= scale;
        pt[1] *= scale;
    }

    auto result_it = std::remove_if(buffer.begin(),buffer.end(), [&](std::array<float,2> &pt)
                                    {return (pt[0]>1.f || pt[1]>dy ||pt[0] < 0 || pt[1] < 0);
                                    });
    buffer.erase(result_it, buffer.end());
    size_t final_count = buffer.size();
    float final_ppc = (float)final_count/(gx*gy);
    LOGR("grid: {}x{}; updated pts {:>8}; updated_density {:>7.4}",gx, gy, buffer.size(), final_ppc);

    // sort
    double hinv = (float)(gx-1);
    auto computeIdx = [&](std::array<float,2> &pt) -> int {
        float &x = pt[0];
        float &y = pt[1];

        int x_idx = (uint32_t)(x*hinv + 0.5);
        int y_idx = (uint32_t)(y*hinv + 0.5);
        int result_idx = x_idx*gy + y_idx;
        return result_idx;
    };

    LOGV("sorting started");
    std::sort(buffer.begin(),buffer.end(),
              [&](std::array<float,2> &pt1, std::array<float,2> &pt2)
              {return computeIdx(pt1) < computeIdx(pt2);}
              );

    LOGR("sorting finished; first point: {}, {}", buffer.front()[0], buffer.front()[1]);

    std::vector<int> grid_count(gx*gy,0);
    for(auto &pt : buffer) grid_count[computeIdx(pt)]++;

    auto [it_min, it_max] = std::minmax_element(grid_count.begin(),grid_count.end());

    LOGR("grid fill min/max {} to {} ", *it_min, *it_max);

    std::vector<int> histogram(std::max((int)points_per_cell*3, *it_max), 0);
    for(int &count : grid_count) histogram[count]++;
    for(auto val : histogram) std::cout << val << ' ';
    std::cout << std::endl;

    // write file
    std::filesystem::path directory_path(pts_cache_path);
    if (!std::filesystem::exists(directory_path)) std::filesystem::create_directories(directory_path);
    std::string fileName = prepare_file_name(gx, gy);
    LOGR("saving file {}", fileName);
    H5::H5File file(fileName, H5F_ACC_TRUNC);

    const size_t nPts = buffer.size();
    hsize_t dims_pts[2] = {nPts,2};
    H5::DataSpace dataspace_pts(2, dims_pts);

    H5::DataSet dataset = file.createDataSet("coords", H5::PredType::NATIVE_FLOAT, dataspace_pts);
    dataset.write(buffer.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSpace att_dspace(H5S_SCALAR);
    dataset.createAttribute("gx", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &gx);
    dataset.createAttribute("gy", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &gy);
    file.close();
}


bool icy::SnapshotManager::attempt_to_fill_from_cache(int gx, int gy, std::vector<std::array<float, 2>> &buffer)
{
    LOGV("attempting to load from cache");
    std::string fileName = prepare_file_name(gx, gy);
    std::filesystem::path file_path(fileName);
    if (!std::filesystem::exists(file_path))
    {
        LOGV("cached file does not exist");
        return false;
    }

    H5::H5File file(fileName, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("coords");

    hsize_t dims[2];
    dataset.getSpace().getSimpleExtentDims(dims,NULL);
    if(dims[1] != 2) {
        LOGR("something is wrong with cache file - dims[1] is {}",dims[1]);
        return false;
    }

    int saved_gx, saved_gy;
    dataset.openAttribute("gx").read(H5::PredType::NATIVE_INT, &saved_gx);
    dataset.openAttribute("gy").read(H5::PredType::NATIVE_INT, &saved_gy);
    if(saved_gx != gx || saved_gy != gy)
    {
        LOGR("cache error - expected grid {}x{}; cached grid {}x{}",gx, gy, saved_gx, saved_gy);
        return false;
    }

    buffer.resize(dims[0]);
    LOGR("reading {} points from file", dims[0]);
    dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
    LOGR("attempt_to_fill_from_cache: finished reading from file; {} points", buffer.size());
    LOGR("attempt_to_fill_from_cache: first point: {}, {}", buffer.front()[0], buffer.front()[1]);

    return true;
}




// =============================  READ AND WRITE SNAPSHOTS


void icy::SnapshotManager::CalculateWeightCoeffs(const PointVector2r &pos, PointArray2r ww[3])
{
    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    PointArray2r arr_v0 = 0.5 - pos.array();
    PointArray2r arr_v1 = pos.array();
    PointArray2r arr_v2 = pos.array() + 0.5;
    ww[0] = 0.5*arr_v0*arr_v0;
    ww[1] = 0.75-arr_v1*arr_v1;
    ww[2] = 0.5*arr_v2*arr_v2;
}


void icy::SnapshotManager::PrepareFrameArrays()
{
    LOGR("PrepareFrameArrays() {} x {}", model->prms.GridXTotal, model->prms.GridYTotal);
    const int gridSize = model->prms.GridXTotal*model->prms.GridYTotal;

    count.assign(gridSize,0);

    vis_point_density.assign(gridSize, 0);
    vis_mass.assign(gridSize, 0);
    vis_vx.assign(gridSize,0);
    vis_vy.assign(gridSize,0);
    vis_r.assign(gridSize,0);
    vis_g.assign(gridSize,0);
    vis_b.assign(gridSize,0);
    vis_Jpinv.assign(gridSize,0);
    vis_P.assign(gridSize,0);
    vis_Q.assign(gridSize,0);
    rgb.assign(gridSize*3,255);
    mass_mask.assign(gridSize,0);

    const int &width = model->prms.InitializationImageSizeX;
    const int &height = model->prms.InitializationImageSizeY;
    const int &ox = model->prms.ModeledRegionOffsetX;
    const int &oy = model->prms.ModeledRegionOffsetY;
    const int &gx = model->prms.GridXTotal;
    const int &gy = model->prms.GridYTotal;

    const int nPts = model->gpu.hssoa.size;

    for(int i=0;i<nPts;i++)
    {
        SOAIterator s = model->gpu.hssoa.begin()+i;
        bool disabled = s->getDisabledStatus();
        if(disabled) continue;
        //PointVector2r pos = s->getPos(model->prms.cellsize);
        const int GridY = model->prms.GridYTotal;
        int cellIdx = s->getCellIndex(GridY);
        Eigen::Vector2i cell(cellIdx/GridY, cellIdx % GridY);
        PointVector2r pos;
        pos.x() = s->getValue(SimParams::posx+0);
        pos.y() = s->getValue(SimParams::posx+1);
        const t_PointReal vx = s->getValue(SimParams::velx+0);
        const t_PointReal vy = s->getValue(SimParams::velx+1);

        const t_PointReal Jpinv = s->getValue(SimParams::idx_Jp_inv);
        const t_PointReal P = s->getValue(SimParams::idx_P);
        const t_PointReal Q = s->getValue(SimParams::idx_Q);

        const t_PointReal thickness = s->getValue(SimParams::idx_thickness);
        const double particle_mass = model->prms.ParticleMass * thickness;

        int pt_idx = s->getValueInt(SimParams::integer_point_idx);
        uint32_t rgb = model->gpu.point_colors_rgb[pt_idx];
        uint8_t r = (rgb >> 16) & 0xff;
        uint8_t g = (rgb >> 8) & 0xff;
        uint8_t b = rgb & 0xff;

        PointArray2r ww[3];
        CalculateWeightCoeffs(pos, ww);

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const size_t idx_gridnode = (j+cell[1]) + (i+cell[0])*GridY;
                const t_PointReal Wip = ww[i+1][0]*ww[j+1][1];

                const t_PointReal incM = Wip*particle_mass;
                vis_point_density[idx_gridnode] += Wip;
                vis_mass[idx_gridnode] += incM;
                vis_vx[idx_gridnode] += incM*vx;
                vis_vy[idx_gridnode] += incM*vy;

                vis_Jpinv[idx_gridnode] += Wip*Jpinv;
                vis_P[idx_gridnode] += Wip * P;
                vis_Q[idx_gridnode] += Wip * Q;

                vis_r[idx_gridnode] += Wip * (float)r;
                vis_g[idx_gridnode] += Wip * (float)g;
                vis_b[idx_gridnode] += Wip * (float)b;
            }
        count[cellIdx]++;
    }


#pragma omp parallel for
    for(int i=0;i<gridSize;i++)
    {
        float density = vis_point_density[i];
        float mass = vis_mass[i];
        if(density>0)
        {
            const float coeff = 1./density;

            vis_vx[i] /= mass;
            vis_vy[i] /= mass;

            vis_r[i] *= coeff;
            vis_g[i] *= coeff;
            vis_b[i] *= coeff;

            vis_Jpinv[i] *= coeff;
            vis_Jpinv[i] -= 1.0;
            vis_P[i] *= coeff;
            vis_Q[i] *= coeff;

            // Scale the float values to the range [0, 255]
          //  rgb[i*3+0] = static_cast<uint8_t>(std::clamp(vis_r[i],0.f,255.f));
          //  rgb[i*3+1] = static_cast<uint8_t>(std::clamp(vis_g[i],0.f,255.f));
          //  rgb[i*3+2] = static_cast<uint8_t>(std::clamp(vis_b[i],0.f,255.f));
        }

        const float epsilon = 1e-1;
        mass_mask[i] = density > epsilon ? 1 : 0;

        if(!mass_mask[i] || model->gpu.grid_status_buffer[i] != 100)
        {
            vis_point_density[i] = 0;
            vis_mass[i] = 0;
            vis_vx[i] = vis_vy[i] = vis_Jpinv[i] = vis_P[i] = vis_Q[i] = 0;
            vis_Jpinv[i] = 0;
        }

        // special treatment for rgb
        float alpha = std::clamp(density, 0.f, 1.f);
        rgb[i*3+0] = alpha*(static_cast<uint8_t>(std::clamp(vis_r[i],0.f,255.f))) + (1-alpha)*ColorMap::rgb_water[0];
        rgb[i*3+1] = alpha*(static_cast<uint8_t>(std::clamp(vis_g[i],0.f,255.f))) + (1-alpha)*ColorMap::rgb_water[1];
        rgb[i*3+2] = alpha*(static_cast<uint8_t>(std::clamp(vis_b[i],0.f,255.f))) + (1-alpha)*ColorMap::rgb_water[2];
    }


    for(int i=0;i<SimParams::MAX_REGIONS;i++)
        forces_per_region[i].setZero();

    // forces per regions
    for(int i=0;i<gridSize;i++)
    {
        uint8_t region = model->gpu.grid_status_buffer[i];
        int idx = region >=100 ? (region-99) : region;
        forces_per_region[idx].x() += model->gpu.grid_boundary_forces[i];
        forces_per_region[idx].y() += model->gpu.grid_boundary_forces[i+gx*gy];
    }

}



void icy::SnapshotManager::SaveFrame(int SimulationStep, double SimulationTime)
{
    LOGR("SnapshotManager::SaveFrame: step {}, time {}", SimulationStep, SimulationTime);
    const int frame = SimulationStep / model->prms.UpdateEveryNthStep;
    // SaveImagesJGP(frame);

    // save as HDF5
    fs::path outputDir = "output";
    fs::path framesDir = "frames";
    fs::path targetPath = outputDir / SimulationTitle / framesDir;
    fs::create_directories(targetPath);

    // save current state
//    std::string baseName = std::format("f{:05d}.h5", frame);
    std::string baseName = fmt::format(fmt::runtime("f{:05d}.h5"), frame);

    fs::path fullPath = targetPath / baseName;
    H5::H5File file(fullPath.string(), H5F_ACC_TRUNC);

    hsize_t dims_grid[2] = {(hsize_t)model->prms.GridXTotal, (hsize_t)model->prms.GridYTotal};
    H5::DataSpace dataspace_grid(2, dims_grid);

    H5::DSetCreatPropList proplist;
    hsize_t chunk_dims[2] = {128, 128};
    proplist.setChunk(2, chunk_dims);
    proplist.setDeflate(6);

    H5::DataSet ds_vis_vx = file.createDataSet("vis_vx", H5::PredType::NATIVE_FLOAT, dataspace_grid, proplist);
    ds_vis_vx.write(vis_vx.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet ds_vis_vy = file.createDataSet("vis_vy", H5::PredType::NATIVE_FLOAT, dataspace_grid, proplist);
    ds_vis_vy.write(vis_vy.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet ds_vis_JpInv = file.createDataSet("vis_Jpinv", H5::PredType::NATIVE_FLOAT, dataspace_grid, proplist);
    ds_vis_JpInv.write(vis_Jpinv.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet ds_vis_P = file.createDataSet("vis_P", H5::PredType::NATIVE_FLOAT, dataspace_grid, proplist);
    ds_vis_P.write(vis_P.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet ds_vis_Q = file.createDataSet("vis_Q", H5::PredType::NATIVE_FLOAT, dataspace_grid, proplist);
    ds_vis_Q.write(vis_Q.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet ds_count = file.createDataSet("count", H5::PredType::NATIVE_UINT8, dataspace_grid, proplist);
    ds_count.write(count.data(), H5::PredType::NATIVE_UINT8);

    hsize_t dims_rgb[3] = {(hsize_t)model->prms.GridXTotal, (hsize_t)model->prms.GridYTotal, 3};
    H5::DataSpace dataspace_rgb(3, dims_rgb);

    H5::DSetCreatPropList proplist2;
    hsize_t chunk_dims2[3] = {128, 128, 3};
    proplist2.setChunk(3, chunk_dims2);
    proplist2.setDeflate(6);

    H5::DataSet ds_rgb = file.createDataSet("rgb", H5::PredType::NATIVE_UINT8, dataspace_rgb, proplist2);
    ds_rgb.write(rgb.data(), H5::PredType::NATIVE_UINT8);

    H5::DataSpace att_dspace(H5S_SCALAR);
    ds_count.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace).write(H5::PredType::NATIVE_INT, &SimulationStep);
    ds_count.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace).write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);

   // SaveFrameCompressed(SimulationStep, SimulationTime);
}















// ==================================== saving frame with OpenJPEG


// Write callback: writes data to the buffer and resizes it if necessary
OPJ_SIZE_T icy::SnapshotManager::mem_stream_write(void* p_buffer, OPJ_SIZE_T size, void* p_user_data) {
    MemoryStream* mem = reinterpret_cast<MemoryStream*>(p_user_data);
    size_t required_size = mem->position + size;
    if (required_size > mem->buffer->size())
        mem->buffer->resize(required_size);

    std::memcpy(mem->buffer->data() + mem->position, p_buffer, size);
    mem->position += size;
    return size;
}

// Skip callback (used by OpenJPEG)
OPJ_OFF_T icy::SnapshotManager::mem_stream_skip(OPJ_OFF_T n, void* p_user_data) {
    MemoryStream* mem = reinterpret_cast<MemoryStream*>(p_user_data);
    mem->position += n;
    return n;
}

// Seek callback
OPJ_BOOL icy::SnapshotManager::mem_stream_seek(OPJ_OFF_T pos, void* p_user_data) {
    MemoryStream* mem = reinterpret_cast<MemoryStream*>(p_user_data);
    if (pos < 0) return OPJ_FALSE;
    mem->position = static_cast<size_t>(pos);
    return OPJ_TRUE;
}

// Main compression function
bool icy::SnapshotManager::compress_grayscale_jp2(const uint16_t* data_ptr,
                                                  const int width, const int height,
                            std::vector<uint8_t>& out_compressed_data) const
{
    opj_image_cmptparm_t cmptparm{};
    cmptparm.dx = 1;
    cmptparm.dy = 1;
    cmptparm.w = width;
    cmptparm.h = height;
    cmptparm.sgnd = 0;
    cmptparm.prec = DEFAULT_DISCRETIZATION_BITS;

    opj_image_t* image = opj_image_create(1, &cmptparm, OPJ_CLRSPC_GRAY);
    if (!image)  throw std::runtime_error("icy::SnapshotManager::compress_grayscale_jp2: Failed to create OpenJPEG image.");
    image->x1 = width;
    image->y1 = height;

    // Copy pixel data
    // OpenJPEG's image->comps[c].data is OPJ_INT32*
    for(int i = 0; i < width * height; ++i) {
        image->comps[0].data[i] = static_cast<OPJ_INT32>(data_ptr[i]);
    }

    opj_cparameters_t prms;
    opj_set_default_encoder_parameters(&prms);
    prms.tcp_rates[0] = DEFAULT_OPENJPEG_COMPRESSION_RATE;
    prms.tcp_numlayers = 1;
    prms.cp_disto_alloc = 1;
    prms.irreversible = 1;
    prms.cod_format = 1; // JP2 format

    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_JP2);
    if (!codec) throw std::runtime_error("Failed to create OpenJPEG codec.");

    if (!opj_setup_encoder(codec, &prms, image)) throw std::runtime_error("Failed to setup OpenJPEG encoder.");


    opj_stream_t* stream = opj_stream_create(OPJ_J2K_STREAM_CHUNK_SIZE, OPJ_FALSE);
    if(!stream) throw std::runtime_error("Failed to create OpenJPEG stream.");

    out_compressed_data.clear(); // Ensure output vector is clean
    out_compressed_data.reserve(static_cast<size_t>(width) * height * (DEFAULT_DISCRETIZATION_BITS + 7) / 8);

    MemoryStream mem_stream_user_data;
    mem_stream_user_data.buffer = &out_compressed_data;
    mem_stream_user_data.position = 0;


    opj_stream_set_user_data(stream, &mem_stream_user_data, nullptr);
    opj_stream_set_write_function(stream, mem_stream_write);
    opj_stream_set_skip_function(stream, mem_stream_skip);
    opj_stream_set_seek_function(stream, mem_stream_seek);
    opj_stream_set_user_data_length(stream, static_cast<OPJ_UINT64>(width * height * 2)); // conservative

    bool success = opj_start_compress(codec, image, stream)
                   && opj_encode(codec, stream)
                   && opj_end_compress(codec, stream);

    if (!success) throw std::runtime_error("Compression failed.");

    out_compressed_data.resize(mem_stream_user_data.position);

    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);

    return success;
}


bool icy::SnapshotManager::compress_rgb_jp2(const uint8_t* data_ptr,
                                            const int width, const int height,
                                            std::vector<uint8_t>& out_compressed_data) const
{
    // Optional parameter validation
    if (!data_ptr || width <= 0 || height <= 0) {
        throw std::invalid_argument("icy::SnapshotManager::compress_rgb_jp2: Invalid parameters.");
    }

    const int num_components = 3;

    // 1. Setup component parameters for R, G, B
    opj_image_cmptparm_t cmptparms[num_components]; // Array for 3 components
    for (int i = 0; i < num_components; ++i) {
        std::memset(&cmptparms[i], 0, sizeof(opj_image_cmptparm_t)); // Use std::memset
        cmptparms[i].dx = 1; // Sample spacing horizontal
        cmptparms[i].dy = 1; // Sample spacing vertical
        cmptparms[i].w = width;
        cmptparms[i].h = height;
        cmptparms[i].sgnd = 0; // Unsigned
        cmptparms[i].prec = 8; // 8 bits per channel
    }

    // 2. Create the OpenJPEG image structure for sRGB
    opj_image_t* image = opj_image_create(num_components, cmptparms, OPJ_CLRSPC_SRGB);
    if (!image) {
        throw std::runtime_error("icy::SnapshotManager::compress_rgb_jp2: Failed to create OpenJPEG image.");
    }
    // If program terminates on throw, image is leaked until termination.

    // Set image canvas dimensions (optional but good practice)
    image->x0 = 0;
    image->y0 = 0;
    image->x1 = width;
    image->y1 = height;

    // 3. De-interleave pixel data from input (RGBRGB...) to planar (RRR..., GGG..., BBB...)
    // OpenJPEG's image->comps[c].data is OPJ_INT32*
    for (int i = 0; i < width * height; ++i) {
        image->comps[0].data[i] = static_cast<OPJ_INT32>(data_ptr[num_components * i + 0]); // R
        image->comps[1].data[i] = static_cast<OPJ_INT32>(data_ptr[num_components * i + 1]); // G
        image->comps[2].data[i] = static_cast<OPJ_INT32>(data_ptr[num_components * i + 2]); // B
    }

    // 4. Setup compression parameters
    opj_cparameters_t prms;
    opj_set_default_encoder_parameters(&prms);
    prms.tcp_rates[0] = DEFAULT_OPENJPEG_COMPRESSION_RATE_RGB; // Target compression ratio
    prms.tcp_numlayers = 1;
    prms.cp_disto_alloc = 1; // Rate distortion allocation
    prms.irreversible = 1;   // Use lossy (irreversible) transforms
    prms.cod_format = 1;     // JP2 file format

    // 5. Create the compressor codec
    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_JP2);
    if (!codec) {
        opj_image_destroy(image); // Manual cleanup before throw
        throw std::runtime_error("icy::SnapshotManager::compress_rgb_jp2: Failed to create OpenJPEG codec.");
    }
    // If program terminates on throw, codec (and image) leaked until termination.

    // 6. Setup the encoder
    if (!opj_setup_encoder(codec, &prms, image)) {
        opj_destroy_codec(codec); // Manual cleanup
        opj_image_destroy(image); // Manual cleanup
        throw std::runtime_error("icy::SnapshotManager::compress_rgb_jp2: Failed to setup OpenJPEG encoder.");
    }
    // If program terminates on throw, codec and image leaked until termination.

    // 7. Create the output stream
    opj_stream_t* stream = opj_stream_create(OPJ_J2K_STREAM_CHUNK_SIZE, OPJ_FALSE); // OPJ_FALSE for writing
    if(!stream) {
        opj_destroy_codec(codec); // Manual cleanup
        opj_image_destroy(image); // Manual cleanup
        throw std::runtime_error("icy::SnapshotManager::compress_rgb_jp2: Failed to create OpenJPEG stream.");
    }
    // If program terminates on throw, stream, codec, image leaked until termination.

    // 8. Prepare output buffer and MemoryStream user data
    out_compressed_data.clear();
    // Reserve based on uncompressed size (W * H * 3 components * 1 byte/component)
    out_compressed_data.reserve(static_cast<size_t>(width) * height * num_components);

    MemoryStream mem_stream_user_data; // Assumes icy::SnapshotManager::MemoryStream
    mem_stream_user_data.buffer = &out_compressed_data;
    mem_stream_user_data.position = 0;

    // 9. Configure the stream with callbacks and user data
    opj_stream_set_user_data(stream, &mem_stream_user_data, nullptr /* no custom free function */);
    opj_stream_set_write_function(stream, icy::SnapshotManager::mem_stream_write); // Static member
    opj_stream_set_skip_function(stream, icy::SnapshotManager::mem_stream_skip);   // Static member
    opj_stream_set_seek_function(stream, icy::SnapshotManager::mem_stream_seek);   // Static member
    // opj_stream_set_user_data_length(stream, estimated_length); // Optional

    // 10. Perform the compression
    bool success = opj_start_compress(codec, image, stream)
                   && opj_encode(codec, stream)
                   && opj_end_compress(codec, stream);

    // 11. Handle compression result
    if (!success) {
        opj_stream_destroy(stream); // Manual cleanup before throw
        opj_destroy_codec(codec);   // Manual cleanup
        opj_image_destroy(image); // Manual cleanup
        throw std::runtime_error("icy::SnapshotManager::compress_rgb_jp2: Compression encoding failed.");
    }

    // 12. Resize output vector to actual compressed size
    out_compressed_data.resize(mem_stream_user_data.position);

    // 13. Cleanup resources on successful path
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);

    return success;
}


// --- Float Data Processing ---
void icy::SnapshotManager::normalize_and_discretize(const std::vector<float>& input,
                                                    std::vector<uint16_t>& output,
                                                    float& minVal, float& maxVal,
                                                    int bits) const
{
    auto mm = std::minmax_element(input.begin(), input.end());
    minVal = *mm.first;
    maxVal = *mm.second;

    output.resize(input.size());
    float range = maxVal - minVal;
    const uint16_t max_discrete_val = static_cast<uint16_t>((1 << bits) - 1);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<uint16_t>(
            std::round((input[i] - minVal) / range * max_discrete_val)
            );
    }
}




void icy::SnapshotManager::save_compressed_float_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                                            const std::vector<float>& data_vec,
                                                            int width, int height) const
{
    // 1. Discretize the float data
    std::vector<uint16_t> discretized_data;
    float min_val, max_val;
    normalize_and_discretize(data_vec, discretized_data, min_val, max_val, DEFAULT_DISCRETIZATION_BITS);

    // 2. Compress the discretized data using OpenJPEG
    std::vector<uint8_t> compressed_blob;
    bool success = compress_grayscale_jp2(discretized_data.data(),
                                          width, height,
                                          compressed_blob);
    if (!success) {
        throw std::runtime_error("icy::SnapshotManager::save_compressed_float_array_hdf5: Failed to compress float array: " + dataset_name);
    }

    // 3. Write the compressed blob as an HDF5 dataset
    hsize_t dims_blob[1] = {compressed_blob.size()};
    H5::DataSpace blob_space(1, dims_blob);

    // Create dataset - no chunking/filtering needed for the pre-compressed blob
    H5::DataSet dataset = file.createDataSet(dataset_name, H5::PredType::NATIVE_UINT8, blob_space);

    // Write the data (handle potentially empty blob if input was empty)
    if (!compressed_blob.empty()) {
        dataset.write(compressed_blob.data(), H5::PredType::NATIVE_UINT8);
    }

    // 4. Write attributes directly to the dataset
    H5::DataSpace att_space(H5S_SCALAR); // Dataspace for scalar attributes

    // Attribute: original_width
    H5::Attribute attr_width = dataset.createAttribute("original_width", H5::PredType::NATIVE_INT, att_space);
    attr_width.write(H5::PredType::NATIVE_INT, &width);

    // Attribute: original_height
    H5::Attribute attr_height = dataset.createAttribute("original_height", H5::PredType::NATIVE_INT, att_space);
    attr_height.write(H5::PredType::NATIVE_INT, &height);

    // Attribute: discretization_bits
    int disc_bits = DEFAULT_DISCRETIZATION_BITS; // Use the constant value
    H5::Attribute attr_bits = dataset.createAttribute("discretization_bits", H5::PredType::NATIVE_INT, att_space);
    attr_bits.write(H5::PredType::NATIVE_INT, &disc_bits);

    // Attribute: min_value
    H5::Attribute attr_min = dataset.createAttribute("min_value", H5::PredType::NATIVE_FLOAT, att_space);
    attr_min.write(H5::PredType::NATIVE_FLOAT, &min_val);

    // Attribute: max_value
    H5::Attribute attr_max = dataset.createAttribute("max_value", H5::PredType::NATIVE_FLOAT, att_space);
    attr_max.write(H5::PredType::NATIVE_FLOAT, &max_val);

    // Attributes are now written. No need to close them individually unless reusing the variable name.
    //LOGR("dataset {}: {}", dataset_name, compressed_blob.size());
}


void icy::SnapshotManager::save_compressed_rgb_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                                          const std::vector<uint8_t>& rgb_data, // interleaved RGB
                                                          int width, int height) const
{
    // 1. Compress the RGB data using OpenJPEG
    std::vector<uint8_t> compressed_blob;
    bool success = compress_rgb_jp2(rgb_data.data(), width, height, compressed_blob);
    if (!success) {
        throw std::runtime_error("icy::SnapshotManager::save_compressed_rgb_array_hdf5: Failed to compress RGB array: " + dataset_name);
    }

    // 2. Write the compressed blob as an HDF5 dataset
    hsize_t dims_blob[1] = {compressed_blob.size()};
    H5::DataSpace blob_space(1, dims_blob);

    // Create dataset - no chunking/filtering needed for the pre-compressed blob
    H5::DataSet dataset = file.createDataSet(dataset_name, H5::PredType::NATIVE_UINT8, blob_space);

    // Write the data (handle potentially empty blob if input was empty)
    if (!compressed_blob.empty()) {
        dataset.write(compressed_blob.data(), H5::PredType::NATIVE_UINT8);
    }

    // 3. Write attributes directly to the dataset
    H5::DataSpace att_space(H5S_SCALAR); // Dataspace for scalar attributes

    // Attribute: original_width
    H5::Attribute attr_width = dataset.createAttribute("original_width", H5::PredType::NATIVE_INT, att_space);
    attr_width.write(H5::PredType::NATIVE_INT, &width);

    // Attribute: original_height
    H5::Attribute attr_height = dataset.createAttribute("original_height", H5::PredType::NATIVE_INT, att_space);
    attr_height.write(H5::PredType::NATIVE_INT, &height);
    //LOGR("dataset {}: {}", dataset_name, compressed_blob.size());
}




void icy::SnapshotManager::SaveFrameCompressed(int SimulationStep, double SimulationTime)
{

    LOGR("SnapshotManager::SaveFrameCompressed: step {}, time {}", SimulationStep, SimulationTime);
    const int frame = SimulationStep / model->prms.UpdateEveryNthStep;

    fs::path outputDir = "output_compressed";
    fs::path framesDir = "frames";
    fs::path targetPath = outputDir / SimulationTitle / framesDir;
    fs::create_directories(targetPath);

    std::string baseName = fmt::format(fmt::runtime("f{:05d}.h5"), frame);
    fs::path fullPath = targetPath / baseName;

    H5::H5File file(fullPath.string(), H5F_ACC_TRUNC);
    const int grid_w = model->prms.GridXTotal;
    const int grid_h = model->prms.GridYTotal;

//    save_compressed_float_array_hdf5(file, "vis_point_density", vis_point_density, grid_w, grid_h);
//    save_compressed_float_array_hdf5(file, "vis_mass", vis_mass, grid_w, grid_h);
    save_compressed_float_array_hdf5(file, "vis_Jpinv", vis_Jpinv, grid_w, grid_h);
    save_compressed_float_array_hdf5(file, "vis_P", vis_P, grid_w, grid_h);
    save_compressed_float_array_hdf5(file, "vis_Q", vis_Q, grid_w, grid_h);
    save_compressed_float_array_hdf5(file, "vis_vx", vis_vx, grid_w, grid_h);
    save_compressed_float_array_hdf5(file, "vis_vy", vis_vy, grid_w, grid_h);

    if (!rgb.empty()) save_compressed_rgb_array_hdf5(file, "rgb", rgb, grid_w, grid_h);

    H5::DataSet vis_mass_dataset = file.openDataSet("vis_Jpinv");
    H5::DataSpace scalar_space(H5S_SCALAR);

    H5::Attribute attr_step = vis_mass_dataset.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, scalar_space);
    attr_step.write(H5::PredType::NATIVE_INT, &SimulationStep);

    H5::Attribute attr_time = vis_mass_dataset.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, scalar_space);
    attr_time.write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);


    // save mass mask
    hsize_t dims_mask_array[2] = {(hsize_t)grid_w, (hsize_t)grid_h};
    H5::DataSpace dataspace_mass_mask(2, dims_mask_array); // Rank 2

    H5::DSetCreatPropList proplist_mass_mask;
    // Define chunk dimensions, ensuring they are not larger than dataset dimensions
    hsize_t chunk_dims_mm[2] = {
        std::min((hsize_t)128, dims_mask_array[0]),
        std::min((hsize_t)128, dims_mask_array[1])
    };

    proplist_mass_mask.setChunk(2, chunk_dims_mm); // Rank 2
    proplist_mass_mask.setDeflate(9); // Compression level (e.g., 6)

    H5::DataSet ds_mass_mask = file.createDataSet("mass_mask",
                                                  H5::PredType::NATIVE_UINT8,
                                                  dataspace_mass_mask,
                                                  proplist_mass_mask);
    ds_mass_mask.write(mass_mask.data(), H5::PredType::NATIVE_UINT8);



    // save forces
    fs::path fullPathForces = targetPath / "forces.h5";
    bool file_exists = std::filesystem::exists(fullPathForces);
    H5::H5File file_forces(fullPathForces.string(), file_exists ? H5F_ACC_RDWR : H5F_ACC_TRUNC);
    H5::DataSet ds_forces;
    if(!file_exists)
    {
        hsize_t initial_dims[3] = {0, (hsize_t)SimParams::MAX_REGIONS, 2}; // Initially 0 frames
        hsize_t max_dims[3] = {H5S_UNLIMITED, (hsize_t)SimParams::MAX_REGIONS, 2}; // Frames are unlimited
        H5::DataSpace file_dataspace_for_creation(3, initial_dims, max_dims);

        // Define dataset creation property list for chunking and compression.
        H5::DSetCreatPropList dcpl;
        // Chunk dimensions: Store one full frame slice per chunk for efficient access.
        hsize_t chunk_dims[3] = {1, (hsize_t)SimParams::MAX_REGIONS, 2};
        dcpl.setChunk(3, chunk_dims);
        ds_forces = file_forces.createDataSet("ds_forces", H5::PredType::NATIVE_FLOAT, file_dataspace_for_creation, dcpl);

        H5::DataSpace scalar_space(H5S_SCALAR);
        H5::Attribute attr = ds_forces.createAttribute("cellsize", H5::PredType::NATIVE_DOUBLE, scalar_space);
        attr.write(H5::PredType::NATIVE_DOUBLE, &model->prms.cellsize);
    }
    else
    {
        ds_forces = file_forces.openDataSet("ds_forces");
    }

    // 1. Get current dataspace and dimensions from the dataset
    H5::DataSpace file_space = ds_forces.getSpace();
    hsize_t current_dims_on_file[3];
    file_space.getSimpleExtentDims(current_dims_on_file);

    // 2. Determine if extension is needed for the current 'frame' (0-based index)
    hsize_t required_frame_capacity = static_cast<hsize_t>(frame) + 1;
    if (required_frame_capacity > current_dims_on_file[0]) {
        hsize_t new_dims[3] = {required_frame_capacity,
                               static_cast<hsize_t>(SimParams::MAX_REGIONS),
                               2};
        ds_forces.extend(new_dims);
        file_space = ds_forces.getSpace(); // Refresh dataspace after extension
    }

    // 3. Define the hyperslab in the file for writing this frame's data
    hsize_t offset[3] = {static_cast<hsize_t>(frame), 0, 0};
    hsize_t slab_dims[3] = {1, static_cast<hsize_t>(SimParams::MAX_REGIONS), 2};
    file_space.selectHyperslab(H5S_SELECT_SET, slab_dims, offset);

    // 4. Define memory dataspace (matches the slab we're writing)
    H5::DataSpace memory_space(3, slab_dims);

    // 5. Write the data
    // forces_per_region[0].data() gives a float* to the contiguous data of the Eigen::Vector2f array
    ds_forces.write(forces_per_region[0].data(), H5::PredType::NATIVE_FLOAT,
                    memory_space, file_space);
    //LOGR("Successfully saved compressed frame: {}", fullPath.string());
}









// =========================== FRAME load

OPJ_SIZE_T icy::SnapshotManager::mem_stream_read(void* p_buffer, OPJ_SIZE_T size, void* p_user_data) {
    MemoryStream* mem = reinterpret_cast<MemoryStream*>(p_user_data);
    if (!mem || !mem->buffer) return (OPJ_SIZE_T)-1;

    OPJ_SIZE_T remaining_bytes = mem->buffer->size() - mem->position;
    OPJ_SIZE_T bytes_to_read = std::min(size, remaining_bytes);

    if (bytes_to_read > 0) {
        std::memcpy(p_buffer, mem->buffer->data() + mem->position, bytes_to_read);
        mem->position += bytes_to_read;
    }
    // If bytes_to_read < size it means EOF was reached.
    // OpenJPEG expects the number of bytes actually read, or -1 on error/true EOF condition.
    return bytes_to_read == 0 && size > 0 ? (OPJ_SIZE_T)-1 : bytes_to_read;
}

OPJ_OFF_T icy::SnapshotManager::mem_stream_read_skip(OPJ_OFF_T n, void* p_user_data) {
    MemoryStream* mem = reinterpret_cast<MemoryStream*>(p_user_data);
    if (!mem || !mem->buffer) return -1;

    OPJ_OFF_T current_pos_offt = static_cast<OPJ_OFF_T>(mem->position);
    OPJ_OFF_T new_pos_offt = current_pos_offt + n;

    if (new_pos_offt < 0) { // Trying to skip before beginning
        mem->position = 0;
        return -current_pos_offt; // Actual amount skipped back
    }
    // Check if skipping beyond buffer end
    if (static_cast<size_t>(new_pos_offt) > mem->buffer->size()) {
        OPJ_OFF_T actual_skip = static_cast<OPJ_OFF_T>(mem->buffer->size()) - current_pos_offt;
        mem->position = mem->buffer->size();
        return actual_skip; // Return actual amount skipped forward
    }

    mem->position = static_cast<size_t>(new_pos_offt);
    return n; // Return requested skip amount
}

OPJ_BOOL icy::SnapshotManager::mem_stream_read_seek(OPJ_OFF_T pos, void* p_user_data) {
    MemoryStream* mem = reinterpret_cast<MemoryStream*>(p_user_data);
    if (!mem || !mem->buffer) return OPJ_FALSE;
    if (pos < 0 || static_cast<size_t>(pos) > mem->buffer->size()) {
        // Cannot seek outside buffer bounds for reading
        return OPJ_FALSE;
    }
    mem->position = static_cast<size_t>(pos);
    return OPJ_TRUE;
}


// --- Decompression Functions ---

bool icy::SnapshotManager::decompress_grayscale_jp2(std::vector<uint8_t>& compressed_data,
                                                    std::vector<uint16_t>& out_data,
                                                    int& width, int& height, int& bits_per_sample) const
{
    if (compressed_data.empty()) {
        throw std::invalid_argument("icy::SnapshotManager::decompress_grayscale_jp2: No compressed data provided.");
    }

    opj_stream_t* stream = opj_stream_create(OPJ_J2K_STREAM_CHUNK_SIZE, OPJ_TRUE); // OPJ_TRUE for reading
    if (!stream) throw std::runtime_error("icy::SnapshotManager::decompress_grayscale_jp2: Failed to create OpenJPEG stream.");

    MemoryStream mem_stream_user_data;
    mem_stream_user_data.buffer = &compressed_data;
    mem_stream_user_data.position = 0;

    opj_stream_set_user_data(stream, &mem_stream_user_data, nullptr);
    opj_stream_set_read_function(stream, mem_stream_read);
    opj_stream_set_skip_function(stream, mem_stream_read_skip);
    opj_stream_set_seek_function(stream, mem_stream_read_seek);
    opj_stream_set_user_data_length(stream, compressed_data.size());

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_JP2); // Assuming JP2 format was used for encoding
    if (!codec) {
        opj_stream_destroy(stream);
        throw std::runtime_error("icy::SnapshotManager::decompress_grayscale_jp2: Failed to create OpenJPEG codec.");
    }

    opj_dparameters_t prms;
    opj_set_default_decoder_parameters(&prms);
    // Optional: prms.decod_format = 1; // Set if needed, usually auto-detected for JP2

    if (!opj_setup_decoder(codec, &prms)) {
        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
        throw std::runtime_error("icy::SnapshotManager::decompress_grayscale_jp2: Failed to setup OpenJPEG decoder.");
    }

    opj_image_t* image = nullptr;
    if (!opj_read_header(stream, codec, &image) || !image) {
        if (image) opj_image_destroy(image); // Destroy image if partially created
        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
        throw std::runtime_error("icy::SnapshotManager::decompress_grayscale_jp2: Failed to read OpenJPEG header.");
    }

    // Optional: Set decoding area if needed via opj_set_decode_area

    if (!opj_decode(codec, stream, image) || !opj_end_decompress(codec, stream)) {
        opj_image_destroy(image);
        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
        throw std::runtime_error("icy::SnapshotManager::decompress_grayscale_jp2: OpenJPEG decompression failed.");
    }

    if (image->numcomps != 1) {
        opj_image_destroy(image); opj_destroy_codec(codec); opj_stream_destroy(stream);
        throw std::runtime_error("icy::SnapshotManager::decompress_grayscale_jp2: Expected 1 component for grayscale.");
    }

    opj_image_comp_t* comp = &image->comps[0];
    width = comp->w;
    height = comp->h;
    bits_per_sample = comp->prec; // Precision from the codestream

    out_data.resize(static_cast<size_t>(width) * height);
    for (int i = 0; i < width * height; ++i) {
        // Data in comp->data is OPJ_INT32. Cast to uint16_t.
        // Add checks/handling for comp->sgnd if signed data is possible.
        out_data[i] = static_cast<uint16_t>(comp->data[i]);
    }

    opj_image_destroy(image);
    opj_destroy_codec(codec);
    opj_stream_destroy(stream);

    return true;
}


bool icy::SnapshotManager::decompress_rgb_jp2(std::vector<uint8_t>& compressed_data,
                                              std::vector<uint8_t>& out_data, // Interleaved RGB
                                              int& width, int& height) const
{
    if (compressed_data.empty()) {
        throw std::invalid_argument("icy::SnapshotManager::decompress_rgb_jp2: No compressed data provided.");
    }

    opj_stream_t* stream = opj_stream_create(OPJ_J2K_STREAM_CHUNK_SIZE, OPJ_TRUE); // Read mode
    if (!stream) throw std::runtime_error("icy::SnapshotManager::decompress_rgb_jp2: Failed to create OpenJPEG stream.");

    MemoryStream mem_stream_user_data;
    mem_stream_user_data.buffer = &compressed_data;
    mem_stream_user_data.position = 0;

    opj_stream_set_user_data(stream, &mem_stream_user_data, nullptr);
    opj_stream_set_read_function(stream, mem_stream_read);
    opj_stream_set_skip_function(stream, mem_stream_read_skip);
    opj_stream_set_seek_function(stream, mem_stream_read_seek);
    opj_stream_set_user_data_length(stream, compressed_data.size());

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_JP2);
    if (!codec) { opj_stream_destroy(stream); throw std::runtime_error("Failed to create codec."); }

    opj_dparameters_t prms;
    opj_set_default_decoder_parameters(&prms);

    if (!opj_setup_decoder(codec, &prms)) { opj_destroy_codec(codec); opj_stream_destroy(stream); throw std::runtime_error("Failed setup decoder."); }

    opj_image_t* image = nullptr;
    if (!opj_read_header(stream, codec, &image) || !image) {
        if(image) opj_image_destroy(image); opj_destroy_codec(codec); opj_stream_destroy(stream);
        throw std::runtime_error("Failed read header.");
    }

    if (!opj_decode(codec, stream, image) || !opj_end_decompress(codec, stream)) {
        opj_image_destroy(image); opj_destroy_codec(codec); opj_stream_destroy(stream);
        throw std::runtime_error("Decompression failed.");
    }

    if (image->numcomps < 3) { // Allow 3 (RGB) or 4 (RGBA, ignore A)
        opj_image_destroy(image); opj_destroy_codec(codec); opj_stream_destroy(stream);
        throw std::runtime_error("Expected at least 3 components for RGB.");
    }
    // Check precision (should be 8 for standard RGB)
    if (image->comps[0].prec != 8 || image->comps[1].prec != 8 || image->comps[2].prec != 8) {
        // Log warning or throw? For now, proceed but data might be misinterpreted.
    }

    width = image->comps[0].w;
    height = image->comps[0].h;
    const size_t num_pixels = static_cast<size_t>(width) * height;
    const int num_components_out = 3; // Outputting RGB
    out_data.resize(num_pixels * num_components_out);

    // Interleave planar components back into RGB
    for (size_t i = 0; i < num_pixels; ++i) {
        out_data[num_components_out * i + 0] = static_cast<uint8_t>(image->comps[0].data[i]); // R
        out_data[num_components_out * i + 1] = static_cast<uint8_t>(image->comps[1].data[i]); // G
        out_data[num_components_out * i + 2] = static_cast<uint8_t>(image->comps[2].data[i]); // B
    }

    opj_image_destroy(image);
    opj_destroy_codec(codec);
    opj_stream_destroy(stream);
    return true;
}


// --- Float Restoration ---

void icy::SnapshotManager::restore_from_discretized(std::vector<uint16_t>& input,
                                                    std::vector<float>& output,
                                                    float minVal, float maxVal,
                                                    int bits) const
{
    output.resize(input.size());
    float range = maxVal - minVal;
    const float max_discrete_val_float = static_cast<float>((1 << bits) - 1);

    // Handle potential division by zero if range is extremely small
    // Check if max_discrete_val_float is zero (only if bits <= 0, should not happen)
    if (range < 1e-9f || max_discrete_val_float == 0.0f) {
        // If range is near zero, all values should be minVal (or maxVal)
        std::fill(output.begin(), output.end(), minVal);
    } else {
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = minVal + (static_cast<float>(input[i]) / max_discrete_val_float) * range;
        }
    }
}


// --- HDF5 Loaders ---

void icy::SnapshotManager::load_compressed_float_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                                            std::vector<float>& data_vec) const
{
    // 1. Open dataset and read compressed blob
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace filespace = dataset.getSpace();
    hsize_t dims_out[1];
    filespace.getSimpleExtentDims(dims_out, nullptr);
    std::vector<uint8_t> compressed_blob(dims_out[0]);
    if (!compressed_blob.empty()) {
        dataset.read(compressed_blob.data(), H5::PredType::NATIVE_UINT8);
    }

    // 2. Read attributes
    int width, height, bits;
    float min_val, max_val;
    H5::Attribute attr_width = dataset.openAttribute("original_width");
    attr_width.read(H5::PredType::NATIVE_INT, &width);
    H5::Attribute attr_height = dataset.openAttribute("original_height");
    attr_height.read(H5::PredType::NATIVE_INT, &height);
    H5::Attribute attr_bits = dataset.openAttribute("discretization_bits");
    attr_bits.read(H5::PredType::NATIVE_INT, &bits);
    H5::Attribute attr_min = dataset.openAttribute("min_value");
    attr_min.read(H5::PredType::NATIVE_FLOAT, &min_val);
    H5::Attribute attr_max = dataset.openAttribute("max_value");
    attr_max.read(H5::PredType::NATIVE_FLOAT, &max_val);

    // 3. Decompress
    std::vector<uint16_t> discretized_data;
    int decomp_w, decomp_h, decomp_bits;
    bool success = decompress_grayscale_jp2(compressed_blob, discretized_data, decomp_w, decomp_h, decomp_bits);

    if (!success) {
        throw std::runtime_error("load_compressed_float_array_hdf5: Decompression failed for " + dataset_name);
    }
    // Optional: Check if decomp_w/h match attributes width/height
    if (decomp_w != width || decomp_h != height) {
        throw std::runtime_error("load_compressed_float_array_hdf5: Dimension mismatch for " + dataset_name);
    }
    // Optional: Check decomp_bits vs bits (though 'bits' from attribute is more reliable for restoration)

    // 4. Restore float values
    restore_from_discretized(discretized_data, data_vec, min_val, max_val, bits);

    // Ensure final size matches dimensions read from attributes
    // (Should already be correct if decompression worked, but good check)
    data_vec.resize(static_cast<size_t>(width) * height);
}


void icy::SnapshotManager::load_compressed_rgb_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                                          std::vector<uint8_t>& rgb_data) const
{
    // 1. Open dataset and read compressed blob
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace filespace = dataset.getSpace();
    hsize_t dims_out[1];
    filespace.getSimpleExtentDims(dims_out, nullptr);
    std::vector<uint8_t> compressed_blob(dims_out[0]);
    if (!compressed_blob.empty()) {
        dataset.read(compressed_blob.data(), H5::PredType::NATIVE_UINT8);
    }

    // 2. Read attributes (optional if relying solely on JP2 header)
    int width, height;
    H5::Attribute attr_width = dataset.openAttribute("original_width");
    attr_width.read(H5::PredType::NATIVE_INT, &width);
    H5::Attribute attr_height = dataset.openAttribute("original_height");
    attr_height.read(H5::PredType::NATIVE_INT, &height);

    // 3. Decompress
    int decomp_w, decomp_h;
    bool success = decompress_rgb_jp2(compressed_blob, rgb_data, decomp_w, decomp_h);

    if (!success) {
        throw std::runtime_error("load_compressed_rgb_array_hdf5: Decompression failed for " + dataset_name);
    }
    // Optional: Check if decomp_w/h match attributes width/height
    if (decomp_w != width || decomp_h != height) {
        throw std::runtime_error("load_compressed_rgb_array_hdf5: Dimension mismatch for " + dataset_name);
    }

    // Ensure final size matches dimensions read from attributes
    rgb_data.resize(static_cast<size_t>(width) * height * 3);
}



void icy::SnapshotManager::StartLoadFrameCompressedAsync(std::filesystem::path outputDir, int frame)
{
    this->FrameNumber = frame;
    std::string fileName = fmt::format(fmt::runtime("f{:05d}.h5"), frame);

    fs::path fullPath = outputDir / fileName;
    std::string fileNameSnapshotHDF5 = fullPath.string();

    // 1. Ensure any previous decoding thread is finished before starting a new one.
    if (decoding_thread_.joinable()) decoding_thread_.join();

    // 2. Reset the ready flag. The operation is not yet complete.
    //    memory_order_relaxed is fine for setting, as the acquire barrier will be on read.
    data_ready_flag_.store(false, std::memory_order_relaxed);

    // 3. Launch the new decoding thread.
    decoding_thread_ = std::thread([this, fileNameSnapshotHDF5]() {
        bool success = this->LoadFrameCompressed(fileNameSnapshotHDF5);
        this->data_ready_flag_.store(true, std::memory_order_release);
        this->data_ready_flag_.notify_one(); // Add for good measure
    });
}


bool icy::SnapshotManager::LoadFrameCompressed(const std::string& fileNameSnapshotHDF5)
{
    LOGR("SnapshotManager::LoadFrameCompressed from: {}", fileNameSnapshotHDF5);

    // H5::H5File uses RAII
    H5::H5File file(fileNameSnapshotHDF5, H5F_ACC_RDONLY);

    // Load float arrays - relies on exceptions for missing datasets
    load_compressed_float_array_hdf5(file, "vis_Jpinv", vis_Jpinv);
    load_compressed_float_array_hdf5(file, "vis_P", vis_P);
    load_compressed_float_array_hdf5(file, "vis_Q", vis_Q);
    load_compressed_float_array_hdf5(file, "vis_vx", vis_vx);
    load_compressed_float_array_hdf5(file, "vis_vy", vis_vy);

    // Load RGB image data - relies on exception if "rgb" dataset missing
    load_compressed_rgb_array_hdf5(file, "rgb", rgb);

    // Load simulation step and time from attributes on "vis_mass" dataset
    // Relies on exception if dataset or attributes are missing
    H5::DataSet vis_mass_dataset = file.openDataSet("vis_Jpinv");

    H5::Attribute attr_step = vis_mass_dataset.openAttribute("SimulationStep");
    attr_step.read(H5::PredType::NATIVE_INT, &SimulationStep);

    H5::Attribute attr_time = vis_mass_dataset.openAttribute("SimulationTime");
    attr_time.read(H5::PredType::NATIVE_DOUBLE, &SimulationTime);

    // load mask
    H5::DataSet ds_mass_mask = file.openDataSet("mass_mask");
    H5::DataSpace filespace_mask = ds_mass_mask.getSpace();

    hsize_t dims_mask_read[2]; // Assuming rank 2
    filespace_mask.getSimpleExtentDims(dims_mask_read, nullptr);

    // Resize the member vector mass_mask
    // dims_mask_read[0] corresponds to grid_h, dims_mask_read[1] to grid_w
    mass_mask.resize(dims_mask_read[0] * dims_mask_read[1]);

    ds_mass_mask.read(mass_mask.data(), H5::PredType::NATIVE_UINT8);

    LOGR("Successfully loaded compressed frame. Step: {}, Time: {}, frame {}", SimulationStep, SimulationTime, FrameNumber);
    return true; // Return true if no exceptions were thrown
}
