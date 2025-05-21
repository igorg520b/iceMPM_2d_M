#include "colormap.h"



Eigen::Vector3f ColorMap::interpolateColor(Palette palette, float value)
{
    const auto& colors = colormaps[static_cast<size_t>(palette)];
    if (colors.empty()) return {0, 0, 0}; // Fallback in case of empty palette

    value = std::clamp(value, 0.0f, 1.0f);
    float scaled = value * (colors.size() - 1);
    int idx = static_cast<int>(scaled);
    float t = scaled - idx;

    if (idx >= colors.size() - 1) return colors.back();
    return (1.0f - t) * colors[idx] + t * colors[idx + 1];
}

// Get interpolated color as uint8_t[3] (values in range [0,255])
std::array<uint8_t, 3> ColorMap::getColor(Palette palette, float value)
{
    Eigen::Vector3f color = interpolateColor(palette, value) * 255.0f;
    return { static_cast<uint8_t>(color.x()), static_cast<uint8_t>(color.y()), static_cast<uint8_t>(color.z()) };
}

// Get full color table for a given palette
const std::vector<Eigen::Vector3f>& ColorMap::getColorTable(Palette palette)
{
    return colormaps[static_cast<size_t>(palette)];
}

std::array<uint8_t, 3> ColorMap::mergeColors(uint32_t rgb, const std::array<uint8_t, 3>& colorArray, float alpha)
{
    // Ensure alpha is clamped between 0 and 1
    alpha = std::max(0.0f, std::min(1.0f, alpha));

    // Extract RGB components from the uint32_t color
    uint8_t r1 = (rgb >> 16) & 0xFF;
    uint8_t g1 = (rgb >> 8) & 0xFF;
    uint8_t b1 = rgb & 0xFF;

    // Get the second color components
    uint8_t r2 = colorArray[0];
    uint8_t g2 = colorArray[1];
    uint8_t b2 = colorArray[2];

    // Perform linear interpolation for each channel
    uint8_t r = static_cast<uint8_t>(r1 * (1 - alpha) + r2 * alpha);
    uint8_t g = static_cast<uint8_t>(g1 * (1 - alpha) + g2 * alpha);
    uint8_t b = static_cast<uint8_t>(b1 * (1 - alpha) + b2 * alpha);

    // Return the result as an std::array<uint8_t, 3>
    return {r, g, b};
}

std::array<uint8_t, 3> ColorMap::mergeColors(const std::array<uint8_t, 3>& colorArray1,
                                          const std::array<uint8_t, 3>& colorArray2, float alpha)
{
        // Ensure alpha is clamped between 0 and 1
        alpha = std::max(0.0f, std::min(1.0f, alpha));

        // Extract RGB components from the uint32_t color
        uint8_t r1 = colorArray1[0];
        uint8_t g1 = colorArray1[1];
        uint8_t b1 = colorArray1[2];

        // Get the second color components
        uint8_t r2 = colorArray2[0];
        uint8_t g2 = colorArray2[1];
        uint8_t b2 = colorArray2[2];

        // Perform linear interpolation for each channel
        uint8_t r = static_cast<uint8_t>(r1 * (1 - alpha) + r2 * alpha);
        uint8_t g = static_cast<uint8_t>(g1 * (1 - alpha) + g2 * alpha);
        uint8_t b = static_cast<uint8_t>(b1 * (1 - alpha) + b2 * alpha);

        // Return the result as an std::array<uint8_t, 3>
        return {r, g, b};
}


const std::array<std::vector<Eigen::Vector3f>, static_cast<size_t>(ColorMap::Palette::COUNT)>
    ColorMap::colormaps = {{
        // SpecialJ
        {
            {0.342992, 0.650614, 0.772702},
            {0.688376, 0.931066, 0.963615},
            {0.836049, 0.882901, 0.85903},
            {0xee/255.,0x6e/255.,0xba/255.},
            {0x94/255.,0x00/255.,0x58/255.}
        },
        //P2
        {{0xed/255.,0xdf/255.,0xd6/255.},
         {0xc6/255.,0xb8/255.,0xaf/255.},
         {0x9c/255.,0x8b/255.,0x7b/255.},
         {0x7a/255.,0x63/255.,0x55/255.},
         {0x52/255.,0x40/255.,0x32/255.},
         {0x2d/255.,0x1e/255.,0x17/255.}},

        //Pressure
        {    {0x03/255.,0x36/255.,0xb3/255.},
         {0x27/255.,0x5d/255.,0x96/255.},    //3
         {0x80/255.,0xac/255.,0xd9/255.},    //1
         {0x77/255.,0x9a/255.,0xae/255.},  // crushed 9
         {0xdf/255.,0xa7/255.,0xac/255.},             // -1
         {0xc2/255.,0x66/255.,0x6e/255.},             // -2
         {0xb7/255.,0x24/255.,0x30/255.}},

        // ANSYS

        {{0x00/255.,0x00/255.,0xff/255.},
         {0x00/255.,0x33/255.,0xff/255.},
         {0x00/255.,0x66/255.,0xff/255.},
         {0x00/255.,0x99/255.,0xff/255.},
         {0x00/255.,0xcc/255.,0xff/255.},
         {0x00/255.,0xff/255.,0xff/255.},
         {0x00/255.,0xff/255.,0xbf/255.},
         {0x00/255.,0xff/255.,0x7f/255.},
         {0x00/255.,0xff/255.,0x3f/255.},
         {0x00/255.,0xff/255.,0x00/255.},
         {0x3f/255.,0xff/255.,0x00/255.},
         {0x7f/255.,0xff/255.,0x00/255.},
         {0xbf/255.,0xff/255.,0x00/255.},
         {0xff/255.,0xff/255.,0x00/255.},
         {0xff/255.,0xd5/255.,0x00/255.},
         {0xff/255.,0xaa/255.,0x00/255.},
         {0xff/255.,0x7f/255.,0x00/255.},
         {0xff/255.,0x55/255.,0x00/255.},
         {0xff/255.,0x2a/255.,0x00/255.},
         {0xff/255.,0x00/255.,0x00/255.}},

        //Pastel
        {{1.0f, 0.8f, 0.8f},
         {1.0f, 0.85f, 0.7f},
         {1.0f, 0.95f, 0.8f},
         {0.85f, 1.0f, 0.8f},
         {0.7f, 1.0f, 0.7f},
         {0.7f, 1.0f, 0.9f},
         {0.75f, 0.9f, 1.0f},
         {0.7f, 0.7f, 1.0f},
         {0.85f, 0.75f, 1.0f},
         {0.95f, 0.8f, 1.0f},
         {1.0f, 0.8f, 0.9f},
         {0.9f, 0.9f, 0.9f},
         {1.0f, 1.0f, 0.8f}},

        //NCD
        {{0.f, 0.7f, 0.f},
         {0.7f, 0.f, 0.f},
         {0.1f, 0.1f, 0.1f}},


        // Ridges
                            {
                                {0.000, 0.500, 0.000},    // 1. Dark Green
                                {0.126, 0.500, 0.000},    // 2. Dark Green-Yellow
                                {0.251, 0.500, 0.000},    // 3. Dark Lime/Olive
                                {0.375, 0.500, 0.000},    // 4. Dark Yellow-Green
                                {0.500, 0.500, 0.000},    // 5. Dark Yellow / Olive
                                {0.500, 0.375, 0.000},    // 6. Dark Yellow-Orange / Brownish
                                {0.500, 0.251, 0.000},    // 7. Dark Orange / Brown
                                {0.500, 0.126, 0.000},    // 8. Dark Orange-Red / Rust
                                {0.500, 0.000, 0.000},    // 9. Dark Red / Maroon
                                {0.438, 0.000, 0.063},    // 10. Dark Red-Magenta
                                {0.251, 0.000, 0.251}     // 13. Dark Purple
                            }

    }};


