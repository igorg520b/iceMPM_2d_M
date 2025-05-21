// vtk_visualization.cpp

#include "vtk_visualization.h"
#include "parameters_sim.h"
//#include <omp.h>
#include <algorithm>
#include <iostream>
#include <spdlog/spdlog.h>
#include <Eigen/Core>

#include "framedata.h"

VTKVisualization::VTKVisualization(FrameData &fd_) : frameData(fd_)
{
    populateLut(ColorMap::Palette::Pressure, lut_Pressure);
    populateLut(ColorMap::Palette::P2, lut_P2);
    populateLut(ColorMap::Palette::ANSYS, lut_ANSYS);
    populateLut(ColorMap::Palette::Ridges, lut_ridges);

    // text
    constexpr int fontSize = 20;
    vtkTextProperty* txtprop = actor_text->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOff();
    txtprop->SetFontSize(fontSize);
    txtprop->ShadowOff();
    txtprop->SetColor(0,0,0);

    // scalar bar
    scalarBar->SetLookupTable(lut_Pressure);
    scalarBar->SetMaximumWidthInPixels(180);
    scalarBar->SetBarRatio(0.1);
    scalarBar->SetMaximumHeightInPixels(250);
    scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    scalarBar->GetPositionCoordinate()->SetValue(0.01,0.015, 0.0);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->GetLabelTextProperty()->BoldOff();
    scalarBar->GetLabelTextProperty()->ItalicOff();
    scalarBar->GetLabelTextProperty()->ShadowOff();
    scalarBar->GetLabelTextProperty()->SetColor(0.1,0.1,0.1);
    scalarBar->GetLabelTextProperty()->SetFontFamilyToTimes();
    scalarBar->SetUnconstrainedFontSize(true);
    scalarBar->GetLabelTextProperty()->SetFontSize(40);

    actor_text->SetDisplayPosition(600, 10);

    txtprop = actor_text_title->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOff();
    txtprop->ShadowOff();
    txtprop->SetColor(0,0,0);
    txtprop->SetFontSize(fontSize);

    actor_text_title->SetDisplayPosition(10, 500);
}


void VTKVisualization::ChangeVisualizationOption(int option)
{
    LOGR("VTKVisualization::ChangeVisualizationOption {}", option);
    VisualizingVariable = (VisOpt)option;
    if(option == VisOpt::P)
    {
        actor_text_title->SetInput("Pressure");
    }
    else if(option == VisOpt::Q)
    {
        actor_text_title->SetInput("Dev. stress");
    }
    else if(option == VisOpt::Jp_inv)
    {
        actor_text_title->SetInput("Rel.surf.dens.");
    }
    else if(option == VisOpt::ridges)
    {
        actor_text_title->SetInput("Ridges");
    }
    else if(option == VisOpt::grid_vnorm)
    {
        actor_text_title->SetInput("Vel.mag.");
    }
    else
    {
        actor_text_title->SetInput("-");
    }

    SynchronizeValues();
}


void VTKVisualization::SynchronizeValues()
{
    // Get a reference to the currently active snapshot via the parent FrameData
    // This is the core change:
    const icy::SnapshotManager& active_snapshot = frameData.frontSnapShot();

    // Ensure the data in the active_snapshot is actually ready before proceeding.
    // This is a crucial safety check. The caller of UpdateQueue (PPMainWindow)
    // should ensure this, but a check here is good for robustness if SynchronizeValues
    // could somehow be called independently.
    // For brevity as requested, I'll assume data_ready_flag_ is true.
    // if (!active_snapshot.data_ready_flag_.load(std::memory_order_acquire)) {
    //     spdlog::warn("VTKVisualization::SynchronizeValues called, but frontSnapShot data is not ready!");
    //     return; // Or handle error appropriately
    // }

    const SimParams &prms = frameData.ggd.prms;

    const int &width = prms.InitializationImageSizeX;
    const int &height = prms.InitializationImageSizeY;
    const int &ox = prms.ModeledRegionOffsetX;
    const int &oy = prms.ModeledRegionOffsetY;
    const int &gx = prms.GridXTotal;
    const int &gy = prms.GridYTotal;
    const double &h = prms.cellsize;

    if (!width) return; // Guard against uninitialized prms

    // Copy the base image
    renderedImage.assign(frameData.ggd.original_image_colors_rgb.begin(), frameData.ggd.original_image_colors_rgb.end());

    const double range_setting = std::pow(10, ranges[(int)VisualizingVariable]);
    LOGR("VTKVisualization::SynchronizeValues(), grid [{} x {}], h {}; VV {}; range {:>6.2}", width, height, h,
         VisualizingVariable, range_setting);

#pragma omp parallel for // This pragma should be fine as it's read-only access to active_snapshot data
    for (size_t i = 0; i < gx; i++) {
        for (size_t j = 0; j < gy; j++) {
            // path_indices is from GeneralGridData, not the snapshot
            int status = frameData.ggd.path_indices[(i + ox) + (j + oy) * width];
            if (status == 1000) { // Assuming 1000 means it's a cell with data to visualize
                size_t idx2 = j + i * gy; // Index for the flattened grid data arrays in SnapshotManager

                // All accesses to snapshot data now use 'active_snapshot'
                std::array<uint8_t, 3> _rgb_from_snapshot;
                _rgb_from_snapshot[0] = active_snapshot.rgb[idx2 * 3 + 0];
                _rgb_from_snapshot[1] = active_snapshot.rgb[idx2 * 3 + 1];
                _rgb_from_snapshot[2] = active_snapshot.rgb[idx2 * 3 + 2];

                float alpha = active_snapshot.mass_mask[idx2] ? 1.0f : 0.0f;
                std::array<uint8_t, 3> base_color_mixed_with_water = ColorMap::mergeColors(ColorMap::rgb_water, _rgb_from_snapshot, alpha);

                if (VisualizingVariable == Jp_inv) {
                    float val = active_snapshot.vis_Jpinv[idx2];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range_setting + 0.5);
                    const float mix_original_color = std::abs(val / range_setting * alpha); // Use alpha here
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(base_color_mixed_with_water, c1, mix_original_color);
                    for (int k = 0; k < 3; k++) renderedImage[((i + ox) + (j + oy) * width) * 3 + k] = c2[k];
                }
                else if (VisualizingVariable == ridges) {
                    float val = active_snapshot.vis_Jpinv[idx2]; // Ridges also use vis_Jpinv
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Ridges, val / range_setting);
                    const float mix_original_color = (val > 0) ? (alpha * val / range_setting) : 0.0f;
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(base_color_mixed_with_water, c1, mix_original_color);
                    for (int k = 0; k < 3; k++) renderedImage[((i + ox) + (j + oy) * width) * 3 + k] = c2[k];
                }
                else if (VisualizingVariable == P) {
                    float val = active_snapshot.vis_P[idx2];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range_setting + 0.5);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha); // Mix with water using alpha
                    for (int k = 0; k < 3; k++) renderedImage[((i + ox) + (j + oy) * width) * 3 + k] = c2[k];
                }
                else if (VisualizingVariable == Q) {
                    float val = active_snapshot.vis_Q[idx2];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range_setting);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[((i + ox) + (j + oy) * width) * 3 + k] = c2[k];
                }
                else if (VisualizingVariable == grid_vnorm) {
                    float val_x = active_snapshot.vis_vx[idx2];
                    float val_y = active_snapshot.vis_vy[idx2];
                    float vnorm = std::sqrt(val_x * val_x + val_y * val_y);
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, vnorm / range_setting);

                    // Mix scalar color with original point color, then with water
                    float scalar_mix_factor = std::min(1.0, vnorm / range_setting); // Ensure mix factor is [0,1]
                    std::array<uint8_t, 3> c2_scalar_and_point = ColorMap::mergeColors(_rgb_from_snapshot, c1, scalar_mix_factor);
                    std::array<uint8_t, 3> c3_final = ColorMap::mergeColors(ColorMap::rgb_water, c2_scalar_and_point, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[((i + ox) + (j + oy) * width) * 3 + k] = c3_final[k];
                }
                else if (VisualizingVariable == colors) {
                    // This case just uses base_color_mixed_with_water calculated earlier
                    for (int k = 0; k < 3; k++) renderedImage[((i + ox) + (j + oy) * width) * 3 + k] = base_color_mixed_with_water[k];
                }
                // Note: if VisualizingVariable == none, or any other unhandled case,
                // the original background image pixel (from frameData.ggd.original_image_colors_rgb)
                // remains in renderedImage for this cell, which is often the desired behavior.
            }
        }
    }

    raster_scalars->SetNumberOfComponents(3);
    raster_scalars->SetArray(renderedImage.data(), renderedImage.size(), 1); // 1 means VTK won't delete it
    raster_scalars->Modified();

    raster_imageData->SetDimensions(width, height, 1);
    raster_imageData->SetSpacing(1.0, 1.0, 1.0);
    raster_imageData->SetOrigin(0.0, 0.0, 0.0);
    raster_imageData->GetPointData()->SetScalars(raster_scalars);
    raster_imageData->Modified();

    raster_plane->SetOrigin(-h / 2.0, -h / 2.0, -1.0);
    raster_plane->SetPoint1((width - 0.5) * h, -h / 2.0, -1.0);
    raster_plane->SetPoint2(-h / 2.0, (height - 0.5) * h, -1.0);
    raster_plane->SetNormal(0.0, 0.0, 1.0);

    raster_mapper->SetInputConnection(raster_plane->GetOutputPort());
    raster_mapper->ScalarVisibilityOff();

    raster_texture->SetInputData(raster_imageData);
    raster_texture->InterpolateOff();

    raster_actor->SetMapper(raster_mapper);
    raster_actor->SetTexture(raster_texture);

    // It's good practice to call Update on mapper and texture if inputs changed,
    // though VTK's pipeline might handle it. Explicit is safer.
    // raster_mapper->Update(); // May not be necessary if actor modification triggers update
    // raster_texture->Update(); // May not be necessary

    // Update text actor with simulation time from the active snapshot
    // frameData.getSimulationTime() also works here as it calls frontSnapShot().SimulationTime
    int64_t display_date = static_cast<int64_t>(active_snapshot.SimulationTime) + prms.SimulationStartUnixTime;
    std::time_t unix_time = static_cast<std::time_t>(display_date);
    std::tm* tm_time = std::gmtime(&unix_time); // Use gmtime for UTC

    char buffer[100];
    if (tm_time) { // gmtime can return nullptr
        std::strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S UTC", tm_time);
        actor_text->SetInput(buffer);
    } else {
        actor_text->SetInput("Invalid Time");
    }


    // Scalar bar updates
    scalarBar->VisibilityOn();
    if (VisualizingVariable == ridges) {
        lut_ridges->SetTableRange(0, range_setting);
        scalarBar->SetLookupTable(lut_ridges);
        scalarBar->SetLabelFormat("%.2f");
    } else if (VisualizingVariable == Jp_inv || VisualizingVariable == P) { // Combined P and Jp_inv
        lut_Pressure->SetTableRange(-range_setting, range_setting);
        scalarBar->SetLookupTable(lut_Pressure);
        scalarBar->SetLabelFormat("%.1e");
    } else if (VisualizingVariable == Q) {
        lut_ANSYS->SetTableRange(0, range_setting);
        scalarBar->SetLookupTable(lut_ANSYS);
        scalarBar->SetLabelFormat("%.1e");
    } else if (VisualizingVariable == grid_vnorm) {
        lut_ANSYS->SetTableRange(0, range_setting);
        scalarBar->SetLookupTable(lut_ANSYS);
        scalarBar->SetLabelFormat("%.2f");
    } else { // Includes VisOpt::colors and VisOpt::none
        scalarBar->VisibilityOff();
    }
}


void VTKVisualization::populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table)
{
    const std::vector<Eigen::Vector3f>& colorTable = ColorMap::getColorTable(palette);
    int size = static_cast<int>(colorTable.size());

    if (size < 2) {
        std::cerr << "Error: Colormap must have at least two colors." << std::endl;
        return;
    }

    const int m = 256;  // Number of colors in the lookup table
    table->SetNumberOfTableValues(m);
    table->Build();

    for (int i = 0; i < m; ++i) {
        float t = static_cast<float>(i) / (m - 1); // Normalize index to [0, 1]

        // Scale t to the range [0, size-1] for interpolation
        float scaledT = t * (size - 1);
        int lowerIdx = static_cast<int>(std::floor(scaledT));
        int upperIdx = static_cast<int>(std::ceil(scaledT));
        float localT = scaledT - lowerIdx; // Fractional part for interpolation

        // Interpolate RGB components
        const Eigen::Vector3f& lowerColor = colorTable[lowerIdx];
        const Eigen::Vector3f& upperColor = colorTable[upperIdx];

        Eigen::Vector3f interpolatedColor = (1.0f - localT) * lowerColor + localT * upperColor;

        // Set interpolated color in the lookup table
        table->SetTableValue(i, interpolatedColor[0], interpolatedColor[1], interpolatedColor[2], 1.0);
    }
}

