// visual_representation.cpp

#include "visual_representation.h"
#include "host_side_data.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

#include <vtkProperty.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkProperty2D.h>
#include <vtkTextProperty.h>
#include <vtkPointData.h>
#include <vtkContourFilter.h>

VisualRepresentation::VisualRepresentation(HostSideData& data) : hsd(data)
{
    LOGR("VisualRepresentation constructor");

    populateLut(ColorMap::Palette::Pressure, lut_Pressure);
    populateLut(ColorMap::Palette::P2, lut_P2);
    populateLut(ColorMap::Palette::ANSYS, lut_ANSYS);
    populateLut(ColorMap::Palette::Ridges, lut_Ridges);

    pts_colors->SetNumberOfComponents(3);
    pts_colors->SetName("pts_colors");
    points_polydata->SetPoints(points);
    points_polydata->GetPointData()->AddArray(pts_colors);
    points_filter->SetInputData(points_polydata);
    points_filter->Update();
    points_mapper->SetInputData(points_filter->GetOutput());
    actor_points->SetMapper(points_mapper);
    actor_points->GetProperty()->SetPointSize(2);
    actor_points->GetProperty()->SetVertexColor(1, 0, 0);
    actor_points->GetProperty()->SetColor(151./255, 188./255, 215./255);
    actor_points->GetProperty()->LightingOff();
    actor_points->GetProperty()->ShadingOff();
    actor_points->GetProperty()->SetInterpolationToFlat();
    actor_points->PickableOff();

    scalarBar->SetMaximumWidthInPixels(150);
    scalarBar->SetBarRatio(0.1);
    scalarBar->SetMaximumHeightInPixels(200);
    scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    scalarBar->GetPositionCoordinate()->SetValue(0.01, 0.015, 0.0);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->GetLabelTextProperty()->BoldOff();
    scalarBar->GetLabelTextProperty()->ItalicOff();
    scalarBar->GetLabelTextProperty()->ShadowOff();
    scalarBar->GetLabelTextProperty()->SetColor(0.1, 0.1, 0.1);

    vtkTextProperty* scalarBarTitleProp = scalarBar->GetTitleTextProperty();
    scalarBarTitleProp->ShadowOff();
    scalarBarTitleProp->SetColor(0.1, 0.1, 0.1);
    scalarBarTitleProp->BoldOff();
    scalarBarTitleProp->ItalicOff();

    vtkTextProperty* txtprop = actorText->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOn();
    txtprop->SetFontSize(30);
    txtprop->ShadowOff();
    txtprop->SetColor(0.1, 0.1, 0.1);
    actorText->SetDisplayPosition(1600, 10);

    vtkTextProperty* titleTextProp = actorTextTitle->GetTextProperty();
    titleTextProp->SetFontFamilyToArial();
    titleTextProp->BoldOn();
    titleTextProp->SetFontSize(30);
    titleTextProp->ShadowOff();
    titleTextProp->SetColor(0.1, 0.1, 0.1);
    titleTextProp->SetJustificationToCentered();
    actorTextTitle->SetDisplayPosition(1000, 10);

    // Text background
    vtkNew<vtkPoints> textBgPoints;
    textBgPoints->InsertNextPoint(580, 5, 0);
    textBgPoints->InsertNextPoint(1910, 5, 0);
    textBgPoints->InsertNextPoint(1910, 60, 0);
    textBgPoints->InsertNextPoint(580, 60, 0);

    vtkNew<vtkCellArray> textBgPoly;
    vtkIdType textIds[4] = {0, 1, 2, 3};
    textBgPoly->InsertNextCell(4, textIds);

    vtkNew<vtkPolyData> textBgPolyData;
    textBgPolyData->SetPoints(textBgPoints);
    textBgPolyData->SetPolys(textBgPoly);

    vtkNew<vtkPolyDataMapper2D> textBgMapper;
    textBgMapper->SetInputData(textBgPolyData);

    textBgActor->SetMapper(textBgMapper);
    textBgActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    textBgActor->GetProperty()->SetOpacity(0.8);
    textBgActor->SetLayerNumber(2);

    // Scalar bar background
    vtkNew<vtkPoints> sbBgPoints;
    sbBgPoints->InsertNextPoint(10, 10, 0);
    sbBgPoints->InsertNextPoint(200, 10, 0);
    sbBgPoints->InsertNextPoint(200, 280, 0);
    sbBgPoints->InsertNextPoint(10, 280, 0);

    vtkNew<vtkCellArray> sbBgPoly;
    vtkIdType sbIds[4] = {0, 1, 2, 3};
    sbBgPoly->InsertNextCell(4, sbIds);

    vtkNew<vtkPolyData> sbBgPolyData;
    sbBgPolyData->SetPoints(sbBgPoints);
    sbBgPolyData->SetPolys(sbBgPoly);

    vtkNew<vtkPolyDataMapper2D> sbBgMapper;
    sbBgMapper->SetInputData(sbBgPolyData);

    scalarBarBgActor->SetMapper(sbBgMapper);
    scalarBarBgActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    scalarBarBgActor->GetProperty()->SetOpacity(0.8);
    scalarBarBgActor->SetLayerNumber(2);

    // Contours actor setup - REMOVED


    actorText->SetLayerNumber(1);
    actorTextTitle->SetLayerNumber(1);
    scalarBar->SetLayerNumber(1);

    LoadVisualizationState();
    LOGR("VisualRepresentation constructor done");
}

void VisualRepresentation::ChangeVisualizationOption(int option)
{
    LOGR("VisualRepresentation::ChangeVisualizationOption {}", option);
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}

void VisualRepresentation::populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table)
{
    const std::vector<Eigen::Vector3f>& colorTable = ColorMap::getColorTable(palette);
    int size = static_cast<int>(colorTable.size());
    if (size < 2) { return; }

    const int m = 256;
    table->SetNumberOfTableValues(m);
    table->Build();

    for (int i = 0; i < m; ++i) {
        float t = static_cast<float>(i) / (m - 1);
        float scaledT = t * (size - 1);
        int lowerIdx = static_cast<int>(std::floor(scaledT));
        int upperIdx = static_cast<int>(std::ceil(scaledT));
        float localT = scaledT - lowerIdx;
        const Eigen::Vector3f& lowerColor = colorTable[lowerIdx];
        const Eigen::Vector3f& upperColor = colorTable[upperIdx];
        Eigen::Vector3f interpolatedColor = (1.0f - localT) * lowerColor + localT * upperColor;
        table->SetTableValue(i, interpolatedColor[0], interpolatedColor[1], interpolatedColor[2], 1.0);
    }
}

void VisualRepresentation::SynchronizeTopology()
{
    LOGR("VisualRepresentation::SynchronizeTopology(): {}", (int)VisualizingVariable);

    const SimParams& prms = hsd.prms;
    const std::vector<uint8_t>& grid_status = hsd.landmask_buffer;
    const std::vector<uint8_t>& original_colors = hsd.original_image_colors_rgb;
    const std::vector<double>& grid_buffer = hsd.host_grid_buffer;

    const int width = prms.InitializationImageSizeX;
    const int height = prms.InitializationImageSizeY;
    const int ox = prms.ModeledRegionOffsetX;
    const int oy = prms.ModeledRegionOffsetY;
    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;
    const size_t gridSize = (size_t)gx * gy;
    const double h = prms.cellsize;
    const double range = std::pow(10, ranges[VisualizingVariable]);
    const double transparency = transparency_coeffs[VisualizingVariable];

    // Update raster image
    renderedImage.assign(original_colors.begin(), original_colors.end());

    // For regions mode, override exterior background with neutral color
    if (VisualizingVariable == VisOpt::regions) {
        // Fill entire image with light gray for exterior
        for (size_t idx = 0; idx < renderedImage.size(); idx += 3) {
            renderedImage[idx] = 200;      // R
            renderedImage[idx + 1] = 200;  // G
            renderedImage[idx + 2] = 200;  // B
        }
    }

    for (int i = 0; i < gx; i++) {
        for (int j = 0; j < gy; j++) {
            const size_t grid_idx = (size_t)j + (size_t)i * gy;
            const size_t render_idx = ((i + ox) + (j + oy) * width) * 3;

            if (grid_status[grid_idx] == SimParams::MAX_REGIONS)
            {
                // Modeled area
                double val_pt_density = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_pts_density];
                float alpha = std::min(val_pt_density * (2.0 / 5.0), 1.0);
                float val_crushed = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_crushed];

                // Original ice colors
                std::array<uint8_t, 3> _rgb;
                for (int k = 0; k < 3; k++) {
                    float v = grid_buffer[grid_idx + gridSize * (SimParams::grid_idx_vis_r + k)];
                    _rgb[k] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255);
                }

                std::array<uint8_t, 3> c = ColorMap::mergeColors(ColorMap::rgb_water, _rgb, alpha);

                if (VisualizingVariable == VisOpt::regions) {
                    // In regions mode, show water as a distinct color
                    std::array<uint8_t, 3> water_color = {100, 150, 200};  // Light blue for water
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = water_color[k];
                }

                if (VisualizingVariable == VisOpt::grid_mass) {
                    double val = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::host_grid_idx_mass];
                    const float mix = alpha * (1. - transparency);
                    std::array<uint8_t, 3> cm = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, cm, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_pt_count) {
                    double val = val_pt_density;
                    std::array<uint8_t, 3> cm = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = cm[k];
                }
                else if (VisualizingVariable == VisOpt::grid_colors) {
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                }
                else if (VisualizingVariable == VisOpt::grid_Jpinv) {
                    float val = grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_Jpinv] - 1.0f;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range + 0.5);
                    const float mix = std::abs(val / range * alpha) + (1. - transparency) * alpha;
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }

                else if (VisualizingVariable == VisOpt::grid_P) {
                    float val = grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_P];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range + 0.5);
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_Q) {
                    float val = grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_Q];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val/range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_vnorm) {
                    float vx = grid_buffer[grid_idx + gridSize * SimParams::grid_idx_px];
                    float vy = grid_buffer[grid_idx + gridSize * SimParams::grid_idx_py];
                    float val = std::sqrt(vx * vx + vy * vy);
                    const float mix = (1. - transparency) * alpha;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_cracked) {
                    float val_cracked = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_cracked];
                    float val_crushed = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_crushed];

                    // Cracked -> Green
                    // Crushed -> Red
                    // Combine them if both exist
                    std::array<uint8_t, 3> combined_color = c;
                    
                    if (val_cracked > 0.0) {
                         // Blend with green
                         float mix = std::min(1.0f, val_cracked); 
                         combined_color = ColorMap::mergeColors(combined_color, ColorMap::rgb_green, mix);
                    }
                    if (val_crushed > 0.0) {
                        // Blend with red
                        float mix = std::min(1.0f, val_crushed);
                        combined_color = ColorMap::mergeColors(combined_color, ColorMap::rgb_red, mix);
                    }
                    
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, combined_color, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::str_EqvGreenLagrange) {
                    float val = grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_strain_EqvGreenLagrange];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::str_vonMises) {
                    float val = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_ridges) {
                    float val = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv] - 1.0f;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Ridges, 0.5 * val / range + 0.5);
                    const float mix = (val > 0.01) ? ((alpha * val / range) + (1. - transparency) * alpha) : 0.0f;
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_thickness) {
                    float val = grid_buffer[grid_idx + gridSize * SimParams::HostGridArrayIndex::grid_idx_vis_thickness];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::v_ocean_norm) {
                    auto [vx, vy] = hsd.waci.GetOceanValue(i, j);
                    double val = std::sqrt(vx * vx + vy * vy);
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c1[k];
                }
                else if (VisualizingVariable == VisOpt::v_wind_norm) {
                    auto [vx, vy] = hsd.waci.GetWindValue(i, j);
                    double val = std::sqrt(vx * vx + vy * vy);
                    // use same colormap as v_norm for consistency
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c1[k];
                }
                else if (VisualizingVariable == VisOpt::vis_lat) {
                    auto [lat, lon] = hsd.waci.GetLatLon(i, j);
                    double val = lat - prms.PROJ_LAT_0;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c1[k];
                }
                else if (VisualizingVariable == VisOpt::vis_lon) {
                    auto [lat, lon] = hsd.waci.GetLatLon(i, j);
                    double val = lon - prms.PROJ_LON_0;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c1[k];
                }
            } else {
                // Non-modeled area
                if (VisualizingVariable == VisOpt::regions) {
                    // In regions mode, color non-modeled areas by region ID
                    uint8_t region_id = grid_status[grid_idx];
                    float val = (region_id % 13) / 12.0f;
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pastel, val);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                }
            }
        }
    }

    // Draw rectangle boundary of modeled area (if applicable)
    if (VisualizingVariable == VisOpt::regions || VisualizingVariable == VisOpt::none) {
        SetupRegionBoundary(gx, gy, ox, oy, h);
    } else {
        actor_region_boundary->VisibilityOff();
    }



    // Update VTK raster image
    raster_scalars->SetNumberOfComponents(3);
    raster_scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);
    raster_scalars->Modified();
    raster_imageData->SetDimensions(width, height, 1);
    raster_imageData->GetPointData()->SetScalars(raster_scalars);

    raster_plane->SetOrigin(-h / 2, -h / 2, -1.0);
    raster_plane->SetPoint1((width - 0.5) * h, -h / 2, -1.0);
    raster_plane->SetPoint2(-h / 2, (height - 0.5) * h, -1.0);

    raster_mapper->SetInputConnection(raster_plane->GetOutputPort());
    raster_texture->SetInputData(raster_imageData);
    raster_actor->SetMapper(raster_mapper);
    raster_actor->SetTexture(raster_texture);
    raster_mapper->Update();
    raster_texture->Update();

    // Contours logic removed


    SynchronizeValues();
    ConfigureScalarBar();
    UpdateTimeText();
}

// UpdateEtaContours removed



void VisualRepresentation::SynchronizeValues()
{
    HostSideSOA& hssoa = hsd.hssoa;
    const SimParams& prms = hsd.prms;
    const int nPts = hssoa.size;

    if (nPts == 0) {
        actor_points->VisibilityOff();
        return;
    }

    points->SetNumberOfPoints(nPts);
    pts_colors->SetNumberOfValues(nPts * 3);

    const int ox = prms.ModeledRegionOffsetX;
    const int oy = prms.ModeledRegionOffsetY;
    const double h = prms.cellsize;

    for (int i = 0; i < nPts; i++) {
        SOAIterator s = hssoa.begin() + i;
        Eigen::Vector2d pos = s->getPos(h);
        points->SetPoint((vtkIdType)i, pos[0] + ox * h, pos[1] + oy * h, 1.0);
    }

    actor_points->VisibilityOn();
    points_mapper->ScalarVisibilityOn();
    points_mapper->SetColorModeToDirectScalars();

    const double range = std::pow(10, ranges[VisualizingVariable]);
    const double transparency = transparency_coeffs[(int)VisualizingVariable];

    if (VisualizingVariable == VisOpt::pt_color) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;
            pts_colors->SetTuple3((vtkIdType)i, r, g, b);
        }
    } else if (VisualizingVariable == VisOpt::pt_status) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            std::array<uint8_t, 3> c = ColorMap::rgb_white; // Default Intact (White)
            if (utility & SimParams::status_crushed) {
                c = ColorMap::rgb_red; // Crushed (Red)
            } else if (utility & SimParams::status_cracked) {
                c = ColorMap::rgb_green; // Cracked (Green)
            }
            pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
        }
    } else if (VisualizingVariable == VisOpt::none) {
        for (int i = 0; i < nPts; i++) {
            pts_colors->SetTuple3((vtkIdType)i, 240, 122, 122);
        }
    } else if (VisualizingVariable == VisOpt::pt_Jp_inv) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            double val = s->getValue(SimParams::PtArrIdx::idx_Jp_inv) - 1.0;
            double value = val / range + 0.5;
            // Compute alpha based on transparency coefficient (see pt_P for details)
            const double base_alpha = std::min(1.0, std::abs(val) / range);
            double alpha = (1.0 - transparency) * 1.0 + transparency * base_alpha;
            
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;
            
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pressure, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::pt_ridges) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            double val = s->getValue(SimParams::PtArrIdx::idx_Jp_inv) - 1.0;
            double alpha = val > 0 ? 1.0 : 0.0;
            
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;
            
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Ridges, val / range);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::pt_P) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            const double val = s->getValue(SimParams::PtArrIdx::idx_P);
            double value = 0.5 * val / range + 0.5;
            const double base_alpha = std::min(1.0, std::abs(val) / range);
            double alpha = (1.0 - transparency) * 1.0 + transparency * base_alpha;
            
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;
            
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pressure, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::pt_ridges) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            double val = s->getValue(SimParams::PtArrIdx::idx_Jp_inv) - 1.0;
            double alpha = val > 0 ? 1.0 : 0.0;
            
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;
            
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Ridges, val / range);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::pt_Q) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            double val = s->getValue(SimParams::PtArrIdx::idx_Q);
            double value = val / range;
            // Compute alpha based on transparency coefficient (see pt_P for details)
            const double base_alpha = std::min(1.0, std::abs(val) / range);
            double alpha = (1.0 - transparency) * 1.0 + transparency * base_alpha;
            
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;
            
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::pt_glen_flow) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            // Use idx_glen_flow (18)
            double val = s->getValue(SimParams::PtArrIdx::idx_glen_flow);
            double value = val / range;
            
            const double base_alpha = std::min(1.0, std::abs(val) / range);
            double alpha = (1.0 - transparency) * 1.0 + transparency * base_alpha;
            
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;
            
            std::array<uint8_t, 3> original_color = {r, g, b};
            // Use ANSYS palette (same as Q)
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::pt_thickness) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            double val = s->getValue(SimParams::PtArrIdx::idx_thickness);
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
            pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
        }
        lut_ANSYS->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_ANSYS);
        scalarBar->SetLabelFormat("%.2f");
        scalarBar->VisibilityOn();
    } else if (VisualizingVariable == VisOpt::pt_partitions) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            uint32_t util = s->getValueInt(SimParams::PtArrIdx::idx_utility_data);
            uint16_t partition_idx = util & 0xFFFF;

            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::NCD, partition_idx / 8.0);
            pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
        }


    } else {
        // Default case: disable point rendering (for v_u, v_v, v_norm, and other unhandled modes)
        actor_points->VisibilityOff();
    }

    // Notify VTK that data has changed
    pts_colors->Modified();
    points_polydata->GetPointData()->SetActiveScalars("pts_colors");
    points_polydata->Modified();
    points->Modified();
    points_filter->Update();
    actor_points->GetProperty()->SetPointSize(prms.ParticleViewSize);
}

void VisualRepresentation::ConfigureScalarBar()
{
    const double range = std::pow(10, ranges[VisualizingVariable]);

    // Default to visible
    scalarBar->VisibilityOn();
    scalarBarBgActor->VisibilityOn();
    //scalarBar->SetTitle(visOptDescriptions[(int)VisualizingVariable].data());
    actorTextTitle->SetInput(visOptDescriptions[(int)VisualizingVariable].data());


    switch (VisualizingVariable)
    {
    case VisOpt::grid_ridges:
    case VisOpt::pt_ridges:
        lut_Ridges->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_Ridges);
        scalarBar->SetLabelFormat("%.2f");
        break;

    case VisOpt::pt_Jp_inv:
    case VisOpt::pt_P:
    case VisOpt::grid_Jpinv:
    case VisOpt::grid_P:
        lut_Pressure->SetTableRange(-range, range);
        scalarBar->SetLookupTable(lut_Pressure);
        scalarBar->SetLabelFormat("%.1e");
        break;

    case VisOpt::pt_Q:
    case VisOpt::pt_glen_flow:
        lut_ANSYS->SetTableRange(-range, range);
        scalarBar->SetLookupTable(lut_ANSYS);
        scalarBar->SetLabelFormat("%.1e");
        break;



    case VisOpt::pt_thickness:
    case VisOpt::grid_Q:
    case VisOpt::grid_vnorm:
    case VisOpt::grid_thickness:
    case VisOpt::str_EqvGreenLagrange:
    case VisOpt::str_vonMises:
    case VisOpt::grid_pt_count: // Enable scalar bar
    case VisOpt::grid_mass:
    case VisOpt::v_ocean_norm:
    case VisOpt::v_wind_norm:
    case VisOpt::vis_lat:
    case VisOpt::vis_lon:
        lut_ANSYS->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_ANSYS);
        scalarBar->SetLabelFormat("%.1e");
        scalarBar->VisibilityOn();
        break;

    default:
        scalarBar->VisibilityOff();
        scalarBarBgActor->VisibilityOff();
        break;
    }
}

void VisualRepresentation::UpdateTimeText()
{
    LOGR("VisualRepresentation::UpdateTimeText()");
    // Convert total seconds to days:hours:minutes
    long long total_seconds = static_cast<long long>(simulationTime);

    const long long seconds_per_day = 24 * 3600;
    const long long seconds_per_hour = 3600;

    int days = total_seconds / seconds_per_day;
    long long remaining_seconds = total_seconds % seconds_per_day;
    int hours = remaining_seconds / seconds_per_hour;
    remaining_seconds %= seconds_per_hour;
    int minutes = remaining_seconds / 60;
    int seconds = total_seconds % 60;

    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d:%02d", days, hours, minutes, seconds);
    actorText->SetInput(buffer);
}

void VisualRepresentation::SetupRegionBoundary(int gx, int gy, int ox, int oy, double h)
{
    // Rectangle corners at cell boundaries, with cell centers at grid nodes
    // (ox, oy) is the grid offset in image coordinates
    // Grid extends from (ox, oy) to (ox + gx - 1, oy + gy - 1) in cell centers
    // Rectangle edges are at half-cell distance from corners

    double x_min = (ox - 0.5) * h;
    double x_max = (ox + gx - 0.5) * h;
    double y_min = (oy - 0.5) * h;
    double y_max = (oy + gy - 0.5) * h;

    vtkNew<vtkPoints> boundary_points;
    boundary_points->InsertNextPoint(x_min, y_min, 0.0);  // 0: bottom-left
    boundary_points->InsertNextPoint(x_max, y_min, 0.0);  // 1: bottom-right
    boundary_points->InsertNextPoint(x_max, y_max, 0.0);  // 2: top-right
    boundary_points->InsertNextPoint(x_min, y_max, 0.0);  // 3: top-left

    vtkNew<vtkCellArray> boundary_lines;
    vtkIdType line_ids[2];
    // Bottom edge
    line_ids[0] = 0; line_ids[1] = 1;
    boundary_lines->InsertNextCell(2, line_ids);
    // Right edge
    line_ids[0] = 1; line_ids[1] = 2;
    boundary_lines->InsertNextCell(2, line_ids);
    // Top edge
    line_ids[0] = 2; line_ids[1] = 3;
    boundary_lines->InsertNextCell(2, line_ids);
    // Left edge
    line_ids[0] = 3; line_ids[1] = 0;
    boundary_lines->InsertNextCell(2, line_ids);

    vtkNew<vtkPolyData> boundary_polydata;
    boundary_polydata->SetPoints(boundary_points);
    boundary_polydata->SetLines(boundary_lines);

    vtkNew<vtkPolyDataMapper> boundary_mapper;
    boundary_mapper->SetInputData(boundary_polydata);
    actor_region_boundary->SetMapper(boundary_mapper);
    actor_region_boundary->GetProperty()->SetColor(0.0, 0.0, 0.0);  // Black lines
    actor_region_boundary->GetProperty()->SetLineWidth(2.0);
    actor_region_boundary->VisibilityOn();
}




VisualRepresentation::~VisualRepresentation()
{
    SaveVisualizationState();
}

void VisualRepresentation::SaveVisualizationState()
{
    std::string state_file = ".plateMPM_vis_state";
    try {
        std::ofstream out(state_file, std::ios::binary);
        if(!out) {
            spdlog::debug("Could not open visualization state file for writing: {}", state_file);
            return;
        }

        // Save ranges and transparency coefficients as raw binary
        out.write(reinterpret_cast<const char*>(ranges), sizeof(ranges));
        out.write(reinterpret_cast<const char*>(transparency_coeffs), sizeof(transparency_coeffs));
        out.close();

        spdlog::debug("Visualization state saved to {}", state_file);
    } catch (const std::exception& e) {
        spdlog::debug("Error saving visualization state: {}", e.what());
    }
}

void VisualRepresentation::LoadVisualizationState()
{
    std::string state_file = ".plateMPM_vis_state";
    try {
        std::ifstream in(state_file, std::ios::binary);
        if(!in) {
            spdlog::debug("Visualization state file not found: {}", state_file);
            return;
        }

        // Load ranges and transparency coefficients from binary file
        in.read(reinterpret_cast<char*>(ranges), sizeof(ranges));
        in.read(reinterpret_cast<char*>(transparency_coeffs), sizeof(transparency_coeffs));
        in.close();

        spdlog::debug("Visualization state loaded from {}", state_file);
    } catch (const std::exception& e) {
        spdlog::debug("Error loading visualization state: {}", e.what());
    }
}
