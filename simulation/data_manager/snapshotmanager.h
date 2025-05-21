// snapshotmanager.h

#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <filesystem>
#include <thread>

#include <H5Cpp.h>
#include <Eigen/Core>

#include "gui/colormap.h"
#include "parameters_sim.h"

#include <openjpeg.h>

namespace icy {class SnapshotManager; class Model;}


class icy::SnapshotManager
{
public:
    SnapshotManager();
    SnapshotManager(const icy::SnapshotManager &other) {};
    ~SnapshotManager();

    icy::Model *model;
    std::string SimulationTitle;
    int SimulationStep;    // only used when reading frame from file
    double SimulationTime; // only used when reading frame from file
    int FrameNumber = -1;       // used when loading frame
    std::atomic<bool> data_ready_flag_; // for postprocessor

    void PrepareGrid(std::string fileNamePNG, std::string fileNameModelledAreaHDF5);
    void PopulatePoints(std::string fileNameModelledAreaHDF5, bool onlyGenerateCache);
    void ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5);
    void SplitIntoPartitionsAndTransferToDevice();
    void SaveSnapshot(int SimulationStep, double SimulationTime);
    void PrepareFrameArrays(); // invoked from SaveFrame

    void SaveFrame(int SimulationStep, double SimulationTime);
    void SaveFrameCompressed(int SimulationStep, double SimulationTime);
    bool LoadFrameCompressed(const std::string&fileNameSnapshotHDF5);
    void StartLoadFrameCompressedAsync(std::filesystem::path outputDir, int frameIndex);

//    void LoadWindData(std::string fileName);    // netCDF4 data
//    void ReadSnapshot(std::string fileName);    // custom HDF5 file

    std::vector<float> vis_point_density, vis_mass;
    std::vector<float> vis_r, vis_g, vis_b, vis_Jpinv, vis_P, vis_Q, vis_vx, vis_vy;
    static constexpr uint8_t waterColor[3] = {0x15, 0x1f, 0x2f};
    std::vector<uint8_t> rgb, mass_mask;
    Eigen::Vector2f forces_per_region[SimParams::MAX_REGIONS];


private:
    std::thread decoding_thread_;   // for StartLoadFrameCompressedAsync

    ColorMap colormap;
    std::vector<uint8_t> count;   // used for counting points per cell and image generation

    void CalculateWeightCoeffs(const PointVector2r &pos, PointArray2r ww[3]);
    constexpr static std::string_view pts_cache_path = "_data/point_cache";

    static std::string prepare_file_name(int gx, int gy);
    static bool attempt_to_fill_from_cache(int gx, int gy, std::vector<std::array<float, 2>> &buffer);
    static void generate_and_save(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer);
    static void generate_points(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer);

    void FillModelledAreaWithBlueColor();
    void SavePointColors();
    void ReadPointColors();



    // Open JPEG - for saving raster data in compressed way
    const int DEFAULT_DISCRETIZATION_BITS = 12;
    const float DEFAULT_OPENJPEG_COMPRESSION_RATE = 15.f;
    const float DEFAULT_OPENJPEG_COMPRESSION_RATE_RGB = 15.f;

    struct MemoryStream {
        std::vector<uint8_t>* buffer;
        size_t position = 0;
    };

    static OPJ_SIZE_T mem_stream_read(void* p_buffer, OPJ_SIZE_T size, void* p_user_data);
    static OPJ_OFF_T mem_stream_read_skip(OPJ_OFF_T n, void* p_user_data);
    static OPJ_BOOL mem_stream_read_seek(OPJ_OFF_T pos, void* p_user_data);

    static OPJ_SIZE_T mem_stream_write(void* p_buffer, OPJ_SIZE_T size, void* p_user_data);
    static OPJ_OFF_T mem_stream_skip(OPJ_OFF_T n, void* p_user_data);
    static OPJ_BOOL mem_stream_seek(OPJ_OFF_T pos, void* p_user_data);


    bool compress_grayscale_jp2(const uint16_t* data_ptr,
                                const int width, const int height,
                                std::vector<uint8_t>& out_compressed_data) const;

    void normalize_and_discretize(const std::vector<float>& input,
                                  std::vector<uint16_t>& output,
                                  float& minVal, float& maxVal,
                                  int bits) const;


    bool compress_rgb_jp2(const uint8_t* data_ptr,
                                                const int width, const int height,
                          std::vector<uint8_t>& out_compressed_data) const;

    void save_compressed_float_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                          const std::vector<float>& data_vec,
                                          int width, int height) const;

    void save_compressed_rgb_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                        const std::vector<uint8_t>& rgb_data, // interleaved
                                        int width, int height) const;


    // decompress
    // Counterpart to compress_grayscale_jp2
    bool decompress_grayscale_jp2(std::vector<uint8_t>& compressed_data,
                                  std::vector<uint16_t>& out_data, // Always uint16_t for simplicity
                                  int& width, int& height, int& bits_per_sample) const;

    // Counterpart to compress_rgb_jp2
    bool decompress_rgb_jp2(std::vector<uint8_t>& compressed_data,
                            std::vector<uint8_t>& out_data, // Interleaved RGB
                            int& width, int& height) const;

    // Counterpart to normalize_and_discretize
    void restore_from_discretized(std::vector<uint16_t>& input,
                                  std::vector<float>& output,
                                  float minVal, float maxVal,
                                  int bits) const;

    // Loaders from HDF5
    void load_compressed_float_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                          std::vector<float>& data_vec) const; // Note: removed expected width/height

    void load_compressed_rgb_array_hdf5(H5::H5File &file, const std::string& dataset_name,
                                        std::vector<uint8_t>& rgb_data) const; // Note: removed expected width/height
};

#endif // SNAPSHOTWRITER_H
