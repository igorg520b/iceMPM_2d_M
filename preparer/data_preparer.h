#ifndef DATA_PREPARER_H
#define DATA_PREPARER_H

#include <vector>
#include <string>
#include <array>
#include <filesystem>

#include "host_side_data.h"
#include "parameters_sim.h"

class DataPreparer
{
public:
    DataPreparer(HostSideData& hsd);

    // Bitmask flags for m_flags
    static constexpr uint8_t FLAG_WATER   = 1 << 0;
    static constexpr uint8_t FLAG_ICE     = 1 << 1;
    static constexpr uint8_t FLAG_CRUSHED = 1 << 2;
    static constexpr uint8_t FLAG_CRACKED = 1 << 3;

    void PrepareGridAndPoints(std::string fileNameLandMask, std::string fileNameColor,
                              std::string fileNameIceMask, std::string fileNameCrushedMask,
                              std::string fileNameCrackedMask,
                              std::string projectDirectory, double dimensionHorizontal, int pointsPerCell,
                              double thicknessFrom, double thicknessTo,
                              double probCracked, double stdDevThickness,
                              std::string fileNameThicknessMask,
                              bool allocate_dense_grid);

private:
    HostSideData& hsd;

    int m_width = 0;
    int m_height = 0;
    std::vector<uint8_t> m_flags;      // Bitpacked flags (water, ice, crushed, cracked)
    std::vector<uint8_t> m_thickness;  // Thickness values (0-255)
    std::vector<uint8_t> m_color;      // RGB color data (3 bytes per pixel)

    // Helper method to set a flag in m_flags based on an image file
    // Returns true if the image was loaded and processed, false otherwise
    bool ProcessMaskLayer(const std::string& filename, uint8_t flag, bool invert = false, int threshold = 128);

    void PrepareGrid(std::string projectDirectory, double dimensionHorizontal, bool allocate_dense_grid);

    void PopulatePoints(int pointsPerCell, double thicknessFrom, double thicknessTo,
                        double probCracked, double stdDevThickness);

    // Poisson point generation helpers
    std::string prepare_cache_filename(int gx, int gy, int ppc);
    bool attempt_to_fill_from_cache(int gx, int gy, int ppc, std::vector<std::array<float, 2>> &buffer);
    void generate_and_save_poisson(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer);
};

#endif // DATA_PREPARER_H
