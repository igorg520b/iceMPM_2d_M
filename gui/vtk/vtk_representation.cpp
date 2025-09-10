#include "vtk_representation.h" // Assuming your header is now vtk_representation.hh or similar
#include <algorithm>
#include <iostream>
#include <spdlog/spdlog.h>

#include "parameters_sim.h"
#include "host_side_soa.h"
#include "windandcurrentinterpolator.h"



// Constructor remains the same
icy::VisualRepresentation::VisualRepresentation()
{
    LOGV("icy::VisualRepresentation::VisualRepresentation() constructor");
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
    actor_points->GetProperty()->SetVertexColor(1,0,0);
    actor_points->GetProperty()->SetColor(151./255,188./255,215./255);
    actor_points->GetProperty()->LightingOff();
    actor_points->GetProperty()->ShadingOff();
    actor_points->GetProperty()->SetInterpolationToFlat();
    actor_points->PickableOff();
    scalarBar->SetMaximumWidthInPixels(150);
    scalarBar->SetBarRatio(0.1);
    scalarBar->SetMaximumHeightInPixels(200);
    scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    scalarBar->GetPositionCoordinate()->SetValue(0.01,0.015, 0.0);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->GetLabelTextProperty()->BoldOff();
    scalarBar->GetLabelTextProperty()->ItalicOff();
    scalarBar->GetLabelTextProperty()->ShadowOff();
    scalarBar->GetLabelTextProperty()->SetColor(0.1,0.1,0.1);

    vtkTextProperty* txtprop = actorText->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOn();
    txtprop->SetFontSize(30);
    txtprop->ShadowOff();
    txtprop->SetColor(0.1,0.1,0.1);
    actorText->SetDisplayPosition(1600, 10);


    // backgrounds for scalarBar and actorText
    // 1. Setup for the text background
    vtkNew<vtkPoints> textBgPoints;
    textBgPoints->InsertNextPoint(1580, 5, 0);  // Bottom-left
    textBgPoints->InsertNextPoint(1910, 5, 0);  // Bottom-right
    textBgPoints->InsertNextPoint(1910, 60, 0); // Top-right
    textBgPoints->InsertNextPoint(1580, 60, 0); // Top-left

    vtkNew<vtkCellArray> textBgPoly;
    vtkIdType textIds[4] = {0, 1, 2, 3};
    textBgPoly->InsertNextCell(4, textIds);

    vtkNew<vtkPolyData> textBgPolyData;
    textBgPolyData->SetPoints(textBgPoints);
    textBgPolyData->SetPolys(textBgPoly);

    vtkNew<vtkPolyDataMapper2D> textBgMapper;
    textBgMapper->SetInputData(textBgPolyData);

    textBgActor->SetMapper(textBgMapper);
    textBgActor->GetProperty()->SetColor(1.0, 1.0, 1.0); // White
    textBgActor->GetProperty()->SetOpacity(0.8);
    textBgActor->SetLayerNumber(2); // Draw first (bottom layer)

    // 2. Setup for the scalar bar background
    // These coordinates should match the scalar bar's position, plus padding.
    // The scalar bar is at (0.01, 0.015) in normalized display coordinates.
    // In a 1920x1080 window, this is approx (19, 16).
    vtkNew<vtkPoints> sbBgPoints;
    sbBgPoints->InsertNextPoint(10, 10, 0);   // Bottom-left
    sbBgPoints->InsertNextPoint(200, 10, 0);  // Bottom-right
    sbBgPoints->InsertNextPoint(200, 280, 0); // Top-right
    sbBgPoints->InsertNextPoint(10, 280, 0);  // Top-left

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
    scalarBarBgActor->SetLayerNumber(2); // Also in the bottom layer

    // 3. Set the foreground actors to a higher layer to ensure they draw on top.
    actorText->SetLayerNumber(1);
    scalarBar->SetLayerNumber(1);

    LOGV("icy::VisualRepresentation::VisualRepresentation() constructor done");
}

void icy::VisualRepresentation::ChangeVisualizationOption(int option)
{
    LOGR("void icy::VisualRepresentation::ChangeVisualizationOption {}", option);
    // This function remains unchanged as it only modifies internal state
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}


// The populateLut helper function has no dependencies and remains unchanged.
void icy::VisualRepresentation::populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table)
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





void icy::VisualRepresentation::SynchronizeTopology()
{
    LOGR("icy::VisualRepresentation::SynchronizeTopology(): {}", (int) VisualizingVariable);
    // --- Step 1: Safety checks ---
    if (!prms || !host_grid_buffer || !grid_status_buffer || !original_image_colors_rgb) {
        LOGV("VisualRepresentation::SynchronizeTopology - Aborting: Essential data pointers are not set.");
        if(!prms) LOGV("prms is null");
        if(!host_grid_buffer) LOGV("host_grid_buffer is null");
        if(!grid_status_buffer) LOGV("grid_status_buffer is null");
        if(!original_image_colors_rgb) LOGV("original_image_colors_rgb is null");
        return;
    }

    // --- Step 2: Cache frequently used parameters ---
    const int width = prms->InitializationImageSizeX;
    const int height = prms->InitializationImageSizeY;
    const int ox = prms->ModeledRegionOffsetX;
    const int oy = prms->ModeledRegionOffsetY;
    const int gx = prms->GridXTotal;
    const int gy = prms->GridYTotal;
    const size_t gridSize = (size_t)gx * gy;
    const double h = prms->cellsize;
    const double range = std::pow(10, ranges[VisualizingVariable]);

    // --- Step 3: Update Raster (Background) Image ---
    renderedImage.assign(original_image_colors_rgb->begin(), original_image_colors_rgb->end());

    for (int i = 0; i < gx; i++) {
        for (int j = 0; j < gy; j++) {
            const size_t grid_idx = (size_t)j + (size_t)i * gy;
            const size_t render_idx = ((i + ox) + (j + oy) * width) * 3;

            if (grid_status_buffer[grid_idx] == 100)
            { // Modeled area
                // Common alpha calculation for blended visualizations
                t_GridReal val_pt_density = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_pts_density];
                float alpha = std::min((double)val_pt_density * (2.0 / 5.0), 1.0);

                if (VisualizingVariable == VisOpt::grid_mass) {
                    t_GridReal val = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_mass];
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                }
                else if (VisualizingVariable == VisOpt::grid_colors) {
                    std::array<uint8_t, 3> _rgb;
                    for (int k = 0; k < 3; k++) {
                        float v = host_grid_buffer[grid_idx + gridSize * (SimParams::grid_idx_vis_r + k)];
                        _rgb[k] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255);
                    }
                    std::array<uint8_t, 3> c = ColorMap::mergeColors(ColorMap::rgb_water, _rgb, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                }
                else if (VisualizingVariable == VisOpt::grid_Jpinv) {
                    std::array<uint8_t, 3> _rgb;
                    for (int k = 0; k < 3; k++) {
                        float v = host_grid_buffer[grid_idx + gridSize * (SimParams::grid_idx_vis_r + k)];
                        _rgb[k] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255);
                    }
                    std::array<uint8_t, 3> c = ColorMap::mergeColors(ColorMap::rgb_water, _rgb, alpha);
                    float val = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_Jpinv] - 1.0f;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range + 0.5);
                    const float mix = std::abs(val / range * alpha);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }

                else if (VisualizingVariable == VisOpt::grid_ridges) {
                    std::array<uint8_t, 3> _rgb;
                    for (int k = 0; k < 3; k++) {
                        float v = host_grid_buffer[grid_idx + gridSize * (SimParams::grid_idx_vis_r + k)];
                        _rgb[k] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255);
                    }
                    std::array<uint8_t, 3> c = ColorMap::mergeColors(ColorMap::rgb_water, _rgb, alpha);
                    float val = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_Jpinv] - 1.0f;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Ridges, 0.5 * val / range + 0.5);

                    const float mix = (val > 0) ? (alpha * val / range) : 0.0f;
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }

                else if (VisualizingVariable == VisOpt::grid_P) {
                    float val = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_P];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range + 0.5);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_Q) {
                    float val = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_Q];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_vnorm) {
                    float vx = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_px];
                    float vy = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_py];
                    float val = std::sqrt(vx * vx + vy * vy);
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::str_EqvGreenLagrange) {
                    float val = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_strain_EqvGreenLagrange];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::str_vonMises) {
                    float val = host_grid_buffer[grid_idx + gridSize * SimParams::grid_idx_vis_strain_vonMises];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_pointdensity) {
                    float val = val_pt_density; // Use the value before alpha scaling
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(ColorMap::rgb_water, c1, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                // Wind/current data visualization
                else if (wac_interpolator) {
                    if(VisualizingVariable == VisOpt::v_norm) {
                        t_GridReal vx = wac_interpolator->current_flow_data[grid_idx];
                        t_GridReal vy = wac_interpolator->current_flow_data[grid_idx + gridSize];
                        float norm = sqrt(vx * vx + vy * vy);
                        std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, norm / range);
                        for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                    } else if(VisualizingVariable == VisOpt::v_u) {
                        t_GridReal vx = wac_interpolator->current_flow_data[grid_idx];
                        std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, 0.5 + vx / range);
                        for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                    } else if(VisualizingVariable == VisOpt::v_v) {
                        t_GridReal vy = wac_interpolator->current_flow_data[grid_idx + gridSize];
                        std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, 0.5 + vy / range);
                        for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                    }
                }
            } else { // Non-modeled area
                if(VisualizingVariable == VisOpt::regions) {
                    uint8_t region_id = grid_status_buffer[grid_idx];
                    float val = (region_id % 13) / 12.0f;
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pastel, val);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                }
            }
        }
    }

    // --- Step 4: Update VTK objects for the raster image ---
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

    // --- Step 5: Update Points ---
    SynchronizeValues();
    ConfigureScalarBar();
    UpdateTimeText();
}

void icy::VisualRepresentation::SynchronizeValues()
{
    // --- Step 1: Safety checks and parameter caching ---
    if (!prms || !hssoa) {
        actor_points->VisibilityOff();
        return;
    }
    const int nPts = hssoa->size;
    if (nPts == 0) {
        actor_points->VisibilityOff();
        return;
    }

    // --- Step 2: Update point geometry and color arrays ---
    points->SetNumberOfPoints(nPts);
    pts_colors->SetNumberOfValues(nPts * 3);

    const int ox = prms->ModeledRegionOffsetX;
    const int oy = prms->ModeledRegionOffsetY;
    const double h = prms->cellsize;

    for (int i = 0; i < nPts; i++) {
        SOAIterator s = hssoa->begin() + i;
        PointVector2r pos = s->getPos(prms->cellsize);
        points->SetPoint((vtkIdType)i, pos[0] + ox * h, pos[1] + oy * h, 1.0);
    }

    // --- Step 3: Update point visibility and colors ---
    actor_points->VisibilityOn();
    scalarBar->VisibilityOff();
    points_mapper->ScalarVisibilityOn();
    points_mapper->SetColorModeToDirectScalars();

    const double range = std::pow(10, ranges[VisualizingVariable]);

    if (VisualizingVariable == VisOpt::color) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            uint8_t r = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 0) * 255);
            uint8_t g = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 1) * 255);
            uint8_t b = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 2) * 255);
            pts_colors->SetTuple3((vtkIdType)i, r, g, b);
        }
    } else if (VisualizingVariable == VisOpt::status) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            double val = s->getCrushedStatus() ? 1.0 : (s->getDisabledStatus() ? 2.0 : 0.0);
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::NCD, val / 2.0);
            pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
        }
    } else if (VisualizingVariable == VisOpt::none || VisualizingVariable == VisOpt::regions) {
        for (int i = 0; i < nPts; i++) {
            pts_colors->SetTuple3((vtkIdType)i, 240, 122, 122);
        }
    } else if (VisualizingVariable == VisOpt::Jp_inv) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            double val = s->getValue(SimParams::idx_Jp_inv) - 1.0;
            double value = (val) / range + 0.5;
            double alpha = std::min(1.0, std::abs(val) / range);
            uint8_t r = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 0) * 255);
            uint8_t g = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 1) * 255);
            uint8_t b = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 2) * 255);
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pressure, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::ridges) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            double val = s->getValue(SimParams::idx_Jp_inv) - 1.0;
            double alpha = val > 0 ? 1.0 : 0.0;
            uint8_t r = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 0) * 255);
            uint8_t g = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 1) * 255);
            uint8_t b = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 2) * 255);
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Ridges, val / range);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
        lut_Ridges->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_Ridges);
        scalarBar->SetLabelFormat("%.2f");
        scalarBar->VisibilityOn();
    } else if (VisualizingVariable == VisOpt::P) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            const double val = s->getValue(SimParams::idx_P);
            double value = val / range + 0.5;
            double alpha = std::min(1.0, std::abs(val) / range);
            uint8_t r = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 0) * 255);
            uint8_t g = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 1) * 255);
            uint8_t b = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 2) * 255);
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pressure, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::Q) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            double val = s->getValue(SimParams::idx_Q);
            double value = val / range + 0.5;
            double alpha = std::min(1.0, std::abs(val) / range);
            uint8_t r = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 0) * 255);
            uint8_t g = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 1) * 255);
            uint8_t b = (uint8_t)(s->getValue(SimParams::idx_pt_color_RGB + 2) * 255);
            std::array<uint8_t, 3> original_color = {r, g, b};
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::P2, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(original_color, c, alpha);
            pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
        }
    } else if (VisualizingVariable == VisOpt::thickness) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            double val = s->getValue(SimParams::idx_thickness);
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
            pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
        }
        lut_ANSYS->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_ANSYS);
        scalarBar->SetLabelFormat("%.2f");
        scalarBar->VisibilityOn();
    } else if (VisualizingVariable == VisOpt::partitions && point_partitions) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa->begin() + i;
            int pt_idx = s->getValueInt(SimParams::integer_point_idx);
            if (pt_idx < point_partitions->size()) {
                uint8_t partition_idx = (*point_partitions)[pt_idx];
                std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::NCD, partition_idx / 8.0);
                pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
            }
        }
    } else {
        actor_points->VisibilityOff();
    }

    // --- Step 4: Notify VTK that data has changed ---
    points_filter->Update();

    points_polydata->GetPointData()->SetActiveScalars("pts_colors");
    pts_colors->Modified();
    points_polydata->Modified();
    points->Modified();

    actor_points->GetProperty()->SetPointSize(prms->ParticleViewSize);
}


void icy::VisualRepresentation::ConfigureScalarBar()
{
    const double range = std::pow(10, ranges[VisualizingVariable]);

    // Default to visible; we will hide it only in the default case.
    scalarBar->VisibilityOn();
    scalarBarBgActor->VisibilityOn();

    // A switch statement is perfect here for clarity and performance.
    switch (VisualizingVariable)
    {
    case VisOpt::grid_ridges:
    case VisOpt::ridges:
    case VisOpt::thickness: // Assuming thickness also uses ridges/ANSYS LUT
        lut_Ridges->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_Ridges);
        scalarBar->SetLabelFormat("%.2f");
        break;

    case VisOpt::Jp_inv: // Note: Live sim uses grid_Jpinv
    case VisOpt::P:      // Note: Live sim uses grid_P
    case VisOpt::grid_Jpinv:
    case VisOpt::grid_P:
        lut_Pressure->SetTableRange(-range, range);
        scalarBar->SetLookupTable(lut_Pressure);
        scalarBar->SetLabelFormat("%.1e");
        break;

    case VisOpt::Q: // Note: Live sim uses grid_Q
    case VisOpt::grid_Q:
    case VisOpt::grid_vnorm:
    case VisOpt::str_EqvGreenLagrange:
    case VisOpt::str_vonMises:
    case VisOpt::grid_pointdensity:
    case VisOpt::grid_mass:
    case VisOpt::v_norm: // For wind/current
        lut_ANSYS->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_ANSYS);
        scalarBar->SetLabelFormat("%.1e");
        break;

        // For any other case, hide the scalar bar.
    default:
        scalarBar->VisibilityOff();
        scalarBarBgActor->VisibilityOff();
        break;
    }
}


void icy::VisualRepresentation::UpdateTimeText()
{
    if (prms) { // Ensure prms is valid before accessing time
        // Convert total seconds (double) to a whole number of seconds for calculation.
        long long total_seconds = static_cast<long long>(simulationTime);

        const long long seconds_per_day = 24 * 3600;
        const long long seconds_per_hour = 3600;

        // Calculate days, hours, and minutes.
        int days = total_seconds / seconds_per_day;
        long long remaining_seconds = total_seconds % seconds_per_day;
        int hours = remaining_seconds / seconds_per_hour;
        remaining_seconds %= seconds_per_hour;
        int minutes = remaining_seconds / 60;

        // Format the string with zero-padding.
        char buffer[100];
        snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", days, hours, minutes);
        actorText->SetInput(buffer);
    }
}


