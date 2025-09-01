#ifndef COLORMAP_H
#define COLORMAP_H

#include <iostream>
#include <array>
#include <vector>
#include <Eigen/Dense>
#include <cstdint>
#include <algorithm>

class ColorMap {
public:
    // Fast enum-based colormap selection
    enum class Palette { SpecialJ, P2, Pressure, ANSYS, Pastel, NCD, Ridges, COUNT};

private:
    // Store colormaps using std::vector for variable sizes
    static const std::array<std::vector<Eigen::Vector3f>, static_cast<size_t>(Palette::COUNT)> colormaps;

public:

    // Get interpolated color as Eigen::Vector3f (values in range [0,1])
    static Eigen::Vector3f interpolateColor(Palette palette, float value);

    // Get interpolated color as uint8_t[3] (values in range [0,255])
    static std::array<uint8_t, 3> getColor(Palette palette, float value);

    // Get full color table for a given palette
    static const std::vector<Eigen::Vector3f>& getColorTable(Palette palette);

    static std::array<uint8_t, 3> mergeColors(uint32_t rgb, const std::array<uint8_t, 3>& colorArray, float alpha);
    static std::array<uint8_t, 3> mergeColors(const std::array<uint8_t, 3>& colorArray1,
                                              const std::array<uint8_t, 3>& colorArray2, float alpha);


    inline static constexpr std::array<uint8_t, 3> rgb_water = {0x15, 0x1f, 0x2f};
};


#endif // COLORMAP_H
