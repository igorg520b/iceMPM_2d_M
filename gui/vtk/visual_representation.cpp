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

    // Flow Visualization Setup
    flow_vectors->SetNumberOfComponents(3);
    flow_vectors->SetName("flow_vectors");
    flow_grid->GetPointData()->SetVectors(flow_vectors);

    // HedgeHog
    hedgehog->SetInputData(flow_grid);
    hedgehog->SetVectorModeToUseVector();
    hedgehog->SetScaleFactor(1.0); // Dynamic update later?
    hedgehog_mapper->SetInputConnection(hedgehog->GetOutputPort());
    actor_vectors->SetMapper(hedgehog_mapper);
    actor_vectors->GetProperty()->SetColor(1.0, 1.0, 1.0); // White vectors
    actor_vectors->GetProperty()->SetLineWidth(1.0);
    actor_vectors->VisibilityOff();

    // Streamlines
    streamer->SetInputData(flow_grid);
    streamer->SetIntegrationDirectionToBoth();
    streamer->SetComputeVorticity(false);
    
    // Direct PolyData mapping (no tubes)
    stream_mapper->SetInputConnection(streamer->GetOutputPort());
    stream_mapper->ScalarVisibilityOff(); // Solid color
    
    actor_streamlines->SetMapper(stream_mapper);
    actor_streamlines->GetProperty()->SetColor(0.2, 0.2, 0.2); // Dark Grey/Black
    actor_streamlines->GetProperty()->SetLineWidth(2.0); // Default thickness
    actor_streamlines->VisibilityOff();

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
    if (VisualizingVariable == VisOpt::regions)
        std::fill(renderedImage.begin(), renderedImage.end(), 200);

    // Get pointers to all grid fields (nullptr if not allocated or index invalid)
    // Common
    const float* ptr_density = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_pts_density);
    const float* ptr_crushed = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_crushed);
    const float* ptr_r = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_r);
    const float* ptr_g = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_g);
    const float* ptr_b = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_b);

    // Specific variables
    const float* ptr_mass = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::host_grid_idx_mass);
    const float* ptr_Jpinv = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv);
    const float* ptr_P = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_P);
    const float* ptr_Q = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_Q);
    const float* ptr_px = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_px);
    const float* ptr_py = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_py);
    const float* ptr_cracked = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_cracked);
    
    const float* ptr_frac_tension = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_fracture_tension);
    const float* ptr_frac_shear = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_fracture_shear);
    const float* ptr_frac_crush = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_fracture_crush);

    const float* ptr_strain_eqv = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange);
    const float* ptr_strain_vm = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises);
    const float* ptr_glen_flow = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_glen_flow);
    const float* ptr_thickness = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_thickness);

    // Flow Vis Logic
    bool show_vectors = (VisualizingVariable == VisOpt::v_ocean_norm || VisualizingVariable == VisOpt::v_wind_norm);
    bool show_streamlines = (VisualizingVariable == VisOpt::ocean_streamlines || VisualizingVariable == VisOpt::wind_streamlines);
    bool is_ocean = (VisualizingVariable == VisOpt::v_ocean_norm || VisualizingVariable == VisOpt::ocean_streamlines);
    bool is_wind = (VisualizingVariable == VisOpt::v_wind_norm || VisualizingVariable == VisOpt::wind_streamlines);

    if (show_vectors || show_streamlines) {
        flow_grid->SetDimensions(gx, gy, 1);
        flow_vectors->SetNumberOfTuples(gx * gy);
        
        // Initialize points if needed (assuming static grid for simpler implementation)
        // If grid changes size, this logic might need check.
        if (flow_grid->GetNumberOfPoints() != gx * gy) {
            vtkNew<vtkPoints> gp;
            gp->SetNumberOfPoints(gx * gy);
            for (int j = 0; j < gy; j++) {
                for (int i = 0; i < gx; i++) {
                    // VTK structured grid ordering: i moves fastest
                    vtkIdType vtk_idx = i + j * gx;
                    gp->SetPoint(vtk_idx, (ox + i) * h, (oy + j) * h, 0.0);
                }
            }
            flow_grid->SetPoints(gp);
        }
    }

    actor_vectors->VisibilityOff();
    actor_streamlines->VisibilityOff();

#pragma omp parallel for
    for (int i = 0; i < gx; i++) {
        for (int j = 0; j < gy; j++) {
            const size_t grid_idx = (size_t)j + (size_t)i * gy;
            const size_t render_idx = ((i + ox) + (j + oy) * width) * 3;

            // ... (Region/Background Logic) ...
            
            // Flow Data Collection
            if (show_vectors || show_streamlines) {
                float vx = 0, vy = 0;
                if (is_ocean) {
                    auto [uv, vv] = hsd.waci.GetOceanValue(i, j);
                    vx = uv; vy = vv;
                } else if (is_wind) {
                    auto [uv, vv] = hsd.waci.GetWindValue(i, j);
                    vx = uv; vy = vv;
                }
                
                vtkIdType vtk_idx = i + j * gx;
                flow_vectors->SetTuple3(vtk_idx, vx, vy, 0.0);
            }
            // End Flow Data Collection

            if (grid_status[grid_idx] == SimParams::MAX_REGIONS)
            {
                // Modeled area
                double val_pt_density = 0, val_crushed = 0;
                std::array<uint8_t, 3> _rgb = {0, 0, 0};
                float alpha = 0.0f;

                if(ptr_density) val_pt_density = ptr_density[grid_idx];
                if(ptr_crushed) val_crushed = ptr_crushed[grid_idx];

                // Determine base color and alpha
                if (hsd.frame_rgba.size() == gridSize * 4) {
                    _rgb[0] = hsd.frame_rgba[grid_idx * 4 + 0];
                    _rgb[1] = hsd.frame_rgba[grid_idx * 4 + 1];
                    _rgb[2] = hsd.frame_rgba[grid_idx * 4 + 2];
                    alpha = hsd.frame_rgba[grid_idx * 4 + 3] / 255.0f;
                } 
                else if (ptr_r && ptr_g && ptr_b) {
                    // Original ice colors from grid variables
                    float vr = ptr_r[grid_idx];
                    float vg = ptr_g[grid_idx];
                    float vb = ptr_b[grid_idx];
                    _rgb[0] = (uint8_t)(std::clamp(vr, 0.f, 1.f) * 255);
                    _rgb[1] = (uint8_t)(std::clamp(vg, 0.f, 1.f) * 255);
                    _rgb[2] = (uint8_t)(std::clamp(vb, 0.f, 1.f) * 255);

                    // Calculate alpha from density
                    alpha = std::min(val_pt_density * (2.0 / 5.0), 1.0);
                }

                std::array<uint8_t, 3> c = ColorMap::mergeColors(ColorMap::rgb_water, _rgb, alpha);

                if (VisualizingVariable == VisOpt::regions) {
                    // In regions mode, show water as a distinct color
                    std::array<uint8_t, 3> water_color = {100, 150, 200};  // Light blue for water
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = water_color[k];
                }

                else if (VisualizingVariable == VisOpt::grid_colors) {
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c[k];
                }
                else if (VisualizingVariable == VisOpt::grid_mass) {
                    if(!ptr_mass) continue;
                    double val = ptr_mass[grid_idx];
                    const float mix = alpha * (1. - transparency);
                    std::array<uint8_t, 3> cm = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, cm, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_pt_count) {
                    if(!ptr_density) continue; // Use ptr_density as proxy for ptr_vis_var
                    double val = ptr_density[grid_idx];
                    std::array<uint8_t, 3> cm = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = cm[k];
                }
                else if (VisualizingVariable == VisOpt::grid_Jpinv) {
                    if(!ptr_Jpinv) continue;
                    float val = ptr_Jpinv[grid_idx] - 1.0f;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range + 0.5);
                    const float mix = std::abs(val / range * alpha) + (1. - transparency) * alpha;
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }

                else if (VisualizingVariable == VisOpt::grid_P) {
                    if(!ptr_P) continue;
                    float val = ptr_P[grid_idx];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5 * val / range + 0.5);
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_Q) {
                    if(!ptr_Q) continue;
                    float val = ptr_Q[grid_idx];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val/range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_vnorm) {
                    if(!ptr_px || !ptr_py) continue;
                    float vx = ptr_px[grid_idx];
                    float vy = ptr_py[grid_idx];
                    float val = std::sqrt(vx * vx + vy * vy);
                    const float mix = (1. - transparency) * alpha;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::grid_cracked) {
                    if(!ptr_cracked) continue; 
                    float val_cracked = ptr_cracked[grid_idx];
                    float val_crushed = ptr_crushed ? ptr_crushed[grid_idx] : 0.0f;

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
                else if (VisualizingVariable == VisOpt::grid_fracture_type) {
                    if(!ptr_frac_tension || !ptr_frac_shear || !ptr_frac_crush) continue;
                    float val_tension = ptr_frac_tension[grid_idx];
                    float val_shear = ptr_frac_shear[grid_idx];
                    float val_crush = ptr_frac_crush[grid_idx];

                    val_tension = std::clamp(val_tension, 0.0f, 1.0f);
                    val_shear = std::clamp(val_shear, 0.0f, 1.0f);
                    val_crush = std::clamp(val_crush, 0.0f, 1.0f);

                    // Reconstruct color based on flag logic
                    // Start as black (fractured base)
                    std::array<float, 3> frac_rgb = {0.0f, 0.0f, 0.0f};

                    // Tension -> Blue, Shear -> Green
                    frac_rgb[2] = val_tension;
                    frac_rgb[1] = val_shear;

                    // Crush -> Red (Dominates/Overwrites)
                    // Interpolate Current -> Red based on crush val
                    for(int k=0; k<3; k++) {
                        frac_rgb[k] = frac_rgb[k] * (1.0f - val_crush) + (k==0 ? 1.0f : 0.0f) * val_crush;
                    }

                    std::array<uint8_t, 3> c_frac;
                    for(int k=0; k<3; k++) c_frac[k] = (uint8_t)(std::clamp(frac_rgb[k], 0.0f, 1.0f) * 255.0f);

                    // Blend Intact Color (c) -> Fracture Color
                    float fracture_intensity = std::max({val_tension, val_shear, val_crush});
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c_frac, fracture_intensity);

                    // Blend with Water
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c3[k];
                }
                else if (VisualizingVariable == VisOpt::str_EqvGreenLagrange) {
                    if(!ptr_strain_eqv) continue;
                    float val = ptr_strain_eqv[grid_idx];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::str_vonMises) {
                    if(!ptr_strain_vm) continue;
                    float val = ptr_strain_vm[grid_idx];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_glen_flow) {
                    if(!ptr_glen_flow) continue;
                    float val = ptr_glen_flow[grid_idx];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_ridges) {
                    if(!ptr_Jpinv) continue;
                    float val = ptr_Jpinv[grid_idx] - 1.0f;
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Ridges, 0.5 * val / range + 0.5);
                    const float mix = (val > 0.01) ? ((alpha * val / range) + (1. - transparency) * alpha) : 0.0f;
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::grid_thickness) {
                    if(!ptr_thickness) continue;
                    float val = ptr_thickness[grid_idx];
                    const float mix = alpha * (std::abs(val / range) + (1. - transparency));
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c2[k];
                }
                else if (VisualizingVariable == VisOpt::v_ocean_norm || VisualizingVariable == VisOpt::ocean_streamlines) {
                    auto [vx, vy] = hsd.waci.GetOceanValue(i, j);
                    double val = std::sqrt(vx * vx + vy * vy);
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::ANSYS, val / range);
                    for (int k = 0; k < 3; k++) renderedImage[render_idx + k] = c1[k];
                }
                else if (VisualizingVariable == VisOpt::v_wind_norm || VisualizingVariable == VisOpt::wind_streamlines) {
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


    // Update Actors
    if (show_vectors) {
        // Subsampling logic: Target ~50 vectors horizontally
        int stride = std::max(1, gx / 50);
        
        // Apply stride mask (zero out vectors not on stride)
        for(int j=0; j<gy; j++) {
            for(int i=0; i<gx; i++) {
                if(i % stride != 0 || j % stride != 0) {
                     // Set vector to 0,0,0
                     flow_vectors->SetTuple3(i + j*gx, 0.0, 0.0, 0.0);
                }
            }
        }

        flow_vectors->Modified();
        
        // Scale by transparency control (user requested)
        // Using transparency_coeffs which is controlled by transparency spinner
        double scale_control = transparency_coeffs[(int)VisualizingVariable]; 
        // Base scale factor logic
        hedgehog->SetScaleFactor(h * 5.0 * scale_control); 
        
        hedgehog->Update();
        actor_vectors->VisibilityOn();
        actor_streamlines->VisibilityOff();
    } else if (show_streamlines) {
        flow_vectors->Modified();
        streamer->SetMaximumPropagation(width * h * 1.5); // Increased propagation
        streamer->SetInitialIntegrationStep(h * 0.2);
        
        // Seed Plane Setup
        // Should cover the visual area with reduced density seeds (3x less than 25x25 -> ~15x15)
        seed_plane->SetOrigin(-h / 2, -h / 2, 0.0);
        seed_plane->SetPoint1((width - 0.5) * h, -h / 2, 0.0);
        seed_plane->SetPoint2(-h / 2, (height - 0.5) * h, 0.0);
        seed_plane->SetResolution(15, 15); 
        
        streamer->SetSourceConnection(seed_plane->GetOutputPort());
        
        streamer->Update();

        // Control line width (thickness) using transparency_coeffs
        double thickness_control = transparency_coeffs[(int)VisualizingVariable];
        // Direct map as requested
        actor_streamlines->GetProperty()->SetLineWidth(thickness_control);
        
        actor_streamlines->VisibilityOn();
        actor_vectors->VisibilityOff();
    } else {
        actor_vectors->VisibilityOff();
        actor_streamlines->VisibilityOff();
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

    } else if (VisualizingVariable == VisOpt::pt_fracture_type) {
        for (int i = 0; i < nPts; i++) {
            SOAIterator s = hssoa.begin() + i;
            uint64_t utility = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
            
            // Start with original color
            uint8_t r = (utility >> 24) & 0xFF;
            uint8_t g = (utility >> 32) & 0xFF;
            uint8_t b = (utility >> 40) & 0xFF;

            if (utility & (SimParams::fracture_tension | 
                SimParams::fracture_compression_shear |
                utility & SimParams::fracture_crush))
            {
                r = g = b = 0;
            }
            // Overwrite components based on fracture flags
            if (utility & SimParams::fracture_tension) {
                b = 255; // Blue
            }
            if (utility & SimParams::fracture_compression_shear) {
                g = 255; // Green
            }
            if (utility & SimParams::fracture_crush) {
                r = 255; // Red
                g = 0;
                b = 0;
            }

            pts_colors->SetTuple3((vtkIdType)i, r, g, b);
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
            uint64_t util = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
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
    case VisOpt::ocean_streamlines:
    case VisOpt::wind_streamlines:
    case VisOpt::vis_lat:
    case VisOpt::vis_lon:
    case VisOpt::pt_Q:
    case VisOpt::pt_glen_flow:
    case VisOpt::grid_glen_flow:
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
    std::string state_file(state_file_name);
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
    std::string state_file(state_file_name);
    LOGR("VisualRepresentation::LoadVisualizationState()");

    try {
        std::ifstream in(state_file, std::ios::binary);
        if(!in) {
            LOGR("Visualization state file not found: {}", state_file);
            ranges[(int)pt_Jp_inv] = ranges[(int)grid_Jpinv] = -1.0;
            ranges[(int)grid_P] = ranges[(int)grid_Q] = ranges[(int)pt_P] = ranges[(int)pt_Q] = 5.75;
            ranges[(int)grid_glen_flow] = -2.0;
            ranges[(int)str_vonMises] = ranges[(int)str_EqvGreenLagrange] = -3.0;
            ranges[(int)pt_thickness] = ranges[(int)grid_thickness] = 0.35;

            ranges[(int)v_wind_norm] = 1.25;
            transparency_coeffs[(int)v_wind_norm] = 0.6;

            ranges[(int)v_ocean_norm] = -0.25;
            transparency_coeffs[(int)v_ocean_norm] = 100.;

            transparency_coeffs[(int)grid_Jpinv] = transparency_coeffs[(int)pt_Jp_inv] = 1.0;
            transparency_coeffs[(int)grid_glen_flow] = 0.9;

            return;
        }

        // Load ranges and transparency coefficients from binary file
        in.read(reinterpret_cast<char*>(ranges), sizeof(ranges));
        in.read(reinterpret_cast<char*>(transparency_coeffs), sizeof(transparency_coeffs));
        in.close();

        LOGR("Visualization state loaded from {}", state_file);
    } catch (const std::exception& e) {
        LOGR("Error loading visualization state: {}", e.what());

    }
}

std::vector<int> VisualRepresentation::GetRequiredGridArrays(VisOpt visualizationOptionIndex)
{
    std::vector<int> required;
    using HI = SimParams::HostGridArrayIndex;

    switch (visualizationOptionIndex)
    {
    case grid_mass:
        required.push_back(HI::host_grid_idx_mass);
        break;
    case grid_pt_count:
        required.push_back(HI::grid_idx_vis_pts_density);
        break;
    case grid_Jpinv:
        required.push_back(HI::grid_idx_vis_Jpinv);
        break;
    case grid_ridges:
        required.push_back(HI::grid_idx_vis_Jpinv);
        // Ridges might require EqvGreenLagrange or other derived fields, 
        // but for now we assume they are pre-calculated or not strictly grid-dependent in this map
        // If checking SynchronizeTopology reveals otherwise, add here.
        break;
    case grid_P:
        required.push_back(HI::grid_idx_vis_P);
        break;
    case grid_Q:
        required.push_back(HI::grid_idx_vis_Q);
        break;
    case grid_colors:
        // No specific grid arrays required; uses hsd.frame_rgba which is always loaded.
        break;
    case grid_vnorm:
        required.push_back(HI::grid_idx_px);
        required.push_back(HI::grid_idx_py);
        required.push_back(HI::host_grid_idx_mass); 
        break;
    case grid_cracked:
        required.push_back(HI::grid_idx_vis_cracked);
        required.push_back(HI::grid_idx_vis_crushed); 
        break;
    case grid_thickness:
        required.push_back(HI::grid_idx_vis_thickness);
        break;
    case grid_fracture_type:
        required.push_back(HI::grid_idx_fracture_tension);
        required.push_back(HI::grid_idx_fracture_shear);
        required.push_back(HI::grid_idx_fracture_crush);
        break;
    case grid_glen_flow:
        required.push_back(HI::grid_idx_glen_flow);
        break;
    case str_EqvGreenLagrange:
        required.push_back(HI::grid_idx_vis_strain_EqvGreenLagrange);
        break;
    case str_vonMises:
        required.push_back(HI::grid_idx_vis_strain_vonMises);
        break;
    default:
        break;
    }
    
    return required;
}
