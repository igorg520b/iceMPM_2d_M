#include "vtk_representation.h"
#include "model.h"
#include "parameters_sim.h"
#include "gpu_partition.h"
//#include <omp.h>
#include <algorithm>
#include <iostream>
#include <spdlog/spdlog.h>


icy::VisualRepresentation::VisualRepresentation()
{
    populateLut(ColorMap::Palette::Pressure, lut_Pressure);
    populateLut(ColorMap::Palette::P2, lut_P2);
    populateLut(ColorMap::Palette::ANSYS, lut_ANSYS);
    populateLut(ColorMap::Palette::Ridges, lut_Ridges);


    // points
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

    // scalar bar
    //scalarBar->SetLookupTable(lutMPM);
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

    // text
    vtkTextProperty* txtprop = actorText->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOff();
    txtprop->SetFontSize(14);
    txtprop->ShadowOff();
    txtprop->SetColor(0,0,0);
    actorText->SetDisplayPosition(500, 30);
}



void icy::VisualRepresentation::SynchronizeTopology()
{
    LOGV("SynchronizeTopology()");
    std::lock_guard<std::mutex> lg(model->accessing_point_data);

    const int &width = model->prms.InitializationImageSizeX;
    const int &height = model->prms.InitializationImageSizeY;
    const int &ox = model->prms.ModeledRegionOffsetX;
    const int &oy = model->prms.ModeledRegionOffsetY;
    const int &gx = model->prms.GridXTotal;
    const int &gy = model->prms.GridYTotal;
    const double &h = model->prms.cellsize;
    const double range = std::pow(10,ranges[VisualizingVariable]);


    if(!model->gpu.original_image_colors_rgb.size()) return;

    // (1) background image (but cover the modelled area in blue)
    renderedImage.assign(model->gpu.original_image_colors_rgb.begin(), model->gpu.original_image_colors_rgb.end());
    const std::array<uint8_t, 3> _rgb_water = {0x15, 0x1f, 0x2f};

    for(size_t i=0;i<gx;i++)
        for(size_t j=0;j<gy;j++)
        {
            uint8_t status = model->gpu.grid_status_buffer[j + i*gy];



            if(status == 100)
            {
                if(VisualizingVariable == VisOpt::v_norm)
                {
                    // visualize current flow
                    size_t idx2 = j + i*gy;
                    t_GridReal vx = model->wac_interpolator.current_flow_data[idx2];
                    t_GridReal vy = model->wac_interpolator.current_flow_data[idx2 + gx*gy];
                    float norm = sqrt(vx*vx + vy*vy);

                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, norm/range);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c[k];
                }
                else if(VisualizingVariable == VisOpt::v_u)
                {
                    // visualize current flow
                    size_t idx2 = j + i*gy;
                    t_GridReal vx = model->wac_interpolator.current_flow_data[idx2];
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, 0.5+vx/range);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c[k];
                }
                else if(VisualizingVariable == VisOpt::v_v)
                {
                    // visualize current flow
                    size_t idx2 = j + i*gy;
                    t_GridReal vy = model->wac_interpolator.current_flow_data[idx2 + gx*gy];
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, 0.5+vy/range);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c[k];
                }


                else if(VisualizingVariable == VisOpt::grid_mass)
                {
                    if(!model->snapshot.vis_mass.size())continue;
                    // visualize Jpinv from the prepared grid / visual array
                    size_t idx2 = j + i*gy;
                    float val = model->snapshot.vis_mass[idx2];
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::P2, val/range);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c[k];
                }
                else if(VisualizingVariable == VisOpt::grid_colors)
                {
                    if(!model->snapshot.vis_mass.size())continue;
                    // visualize Jpinv from the prepared grid / visual array
                    size_t idx2 = j + i*gy;

                    std::array<uint8_t, 3> _rgb;
                    _rgb[0] = model->snapshot.rgb[idx2*3+0];
                    _rgb[1] = model->snapshot.rgb[idx2*3+1];
                    _rgb[2] = model->snapshot.rgb[idx2*3+2];

                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = _rgb[k];
                }
                else if(VisualizingVariable == VisOpt::grid_Jpinv)
                {
                    if(!model->snapshot.vis_mass.size())continue;
                    // visualize Jpinv from the prepared grid / visual array
                    size_t idx2 = j + i*gy;

                    std::array<uint8_t, 3> _rgb;
                    _rgb[0] = model->snapshot.rgb[idx2*3+0];
                    _rgb[1] = model->snapshot.rgb[idx2*3+1];
                    _rgb[2] = model->snapshot.rgb[idx2*3+2];

                    float alpha = model->snapshot.mass_mask[idx2] ? 1 : 0;
                    std::array<uint8_t, 3> c = ColorMap::mergeColors(_rgb_water, _rgb, alpha);

                    float val = model->snapshot.vis_Jpinv[idx2];
                    std::array<uint8_t, 3> c1 = colormap.getColor(ColorMap::Palette::Pressure, 0.5*val/range + 0.5);

                    const float mix_original_color = std::abs(val/range*alpha);

                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(c, c1, mix_original_color);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c2[k];
                }
                else if(VisualizingVariable == VisOpt::grid_pointdensity)
                {
                    if(!model->snapshot.vis_Jpinv.size()) continue;
                    size_t idx2 = j + i*gy;
                    float val = model->snapshot.vis_point_density[idx2]/SimParams::MPM_points_per_cell/range;
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, val);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c[k];
                }
                else if(VisualizingVariable == VisOpt::grid_P)
                {
                    if(!model->snapshot.vis_Jpinv.size()) continue;
                    size_t idx2 = j + i*gy;
                    float alpha = model->snapshot.mass_mask[idx2] ? 1 : 0;
                    float val = model->snapshot.vis_P[idx2];
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pressure, 0.5*val/range+0.5);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(_rgb_water, c, alpha);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c2[k];
                }
                else if(VisualizingVariable == VisOpt::grid_Q)
                {
                    if(!model->snapshot.vis_Jpinv.size()) continue;
                    size_t idx2 = j + i*gy;
                    float alpha = model->snapshot.mass_mask[idx2] ? 1 : 0;
                    float val = model->snapshot.vis_Q[idx2];
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, val/range);
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(_rgb_water, c, alpha);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c2[k];
                }

                else if(VisualizingVariable == VisOpt::grid_vnorm)
                {
                    if(!model->snapshot.vis_Jpinv.size()) continue;
                    // visualize Jpinv from the prepared grid / visual array
                    size_t idx2 = j + i*gy;
                    float val_x = model->snapshot.vis_vx[idx2];
                    float val_y = model->snapshot.vis_vy[idx2];
                    float vnorm = std::sqrt(val_x*val_x + val_y*val_y);
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, vnorm/range);

                    std::array<uint8_t, 3> _rgb;
                    _rgb[0] = model->snapshot.rgb[idx2*3+0];
                    _rgb[1] = model->snapshot.rgb[idx2*3+1];
                    _rgb[2] = model->snapshot.rgb[idx2*3+2];
                    std::array<uint8_t, 3> c2 = ColorMap::mergeColors(_rgb, c, vnorm/range);


                    float alpha = model->snapshot.mass_mask[idx2] ? 1 : 0;
                    std::array<uint8_t, 3> c3 = ColorMap::mergeColors(_rgb_water, c2, alpha);

                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c3[k];
                }
            }
            else
            {
                // not modelled area
                if(VisualizingVariable == VisOpt::regions)
                {
                    // visualize current flow
                    size_t idx2 = j + i*gy;
                    uint8_t region_id = model->gpu.grid_status_buffer[idx2];
                    float val = (region_id % 13) / 12.;
                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pastel, val);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c[k];
                }

                if(VisualizingVariable == VisOpt::grid_force)
                {
                    size_t idx2 = j + i*gy;
                    t_GridReal fx = model->gpu.grid_boundary_forces[idx2];
                    t_GridReal fy = model->gpu.grid_boundary_forces[idx2 + gx*gy];
                    float norm = sqrt(fx*fx + fy*fy);

                    std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, norm/range);
                    for(int k=0;k<3;k++) renderedImage[((i+ox) + (j+oy)*width)*3+k] = c[k];
                }
            }
        }


    raster_scalars->SetNumberOfComponents(3);          // RGB has 3 components
    raster_scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);
    raster_scalars->Modified();

    raster_imageData->SetDimensions(width, height, 1); // 2D image, depth = 1
    raster_imageData->SetSpacing(1.0, 1.0, 1.0);      // Pixel spacing
    raster_imageData->SetOrigin(0.0, 0.0, 0.0);       // Origin at (0,0,0)
    raster_imageData->GetPointData()->SetScalars(raster_scalars);

    raster_plane->SetOrigin(-h/2, -h/2, -1.0);           // Bottom-left corner
    raster_plane->SetPoint1((width-0.5)*h, -h/2, -1.0);         // Bottom-right (x-axis)
    raster_plane->SetPoint2(-h/2, (height-0.5)*h, -1.0);        // Top-left (y-axis)
    raster_plane->SetNormal(0.0, 0.0, 1.0);           // Normal along z-axis (facing forward)

    raster_mapper->SetInputConnection(raster_plane->GetOutputPort());

    raster_texture->SetInputData(raster_imageData);
    raster_texture->InterpolateOff(); // Smooth texture rendering

    raster_actor->SetMapper(raster_mapper);
    raster_actor->SetTexture(raster_texture);

    raster_mapper->Update();
    raster_texture->Update();



    // (2) points
    const int nPts = model->gpu.hssoa.size;
    if(!nPts) return;

    points->SetNumberOfPoints(nPts);
    pts_colors->SetNumberOfValues(nPts*3);

    SynchronizeValues();
}


void icy::VisualRepresentation::SynchronizeValues()
{
    const int &width = model->prms.InitializationImageSizeX;
    const int &height = model->prms.InitializationImageSizeY;
    const int &ox = model->prms.ModeledRegionOffsetX;
    const int &oy = model->prms.ModeledRegionOffsetY;
    const int &gx = model->prms.GridXTotal;
    const int &gy = model->prms.GridYTotal;
    const double &h = model->prms.cellsize;

    // points' coordinates
    const int nPts = model->gpu.hssoa.size;
    for(int i=0;i<nPts;i++)
    {
        SOAIterator s = model->gpu.hssoa.begin()+i;
        PointVector2r pos = s->getPos(model->prms.cellsize);
        points->SetPoint((vtkIdType)i, pos[0]+ox*h, pos[1]+oy*h, 1.0);
    }


    // point visibility and color
    actor_points->VisibilityOn();
    scalarBar->VisibilityOff();
    points_mapper->ScalarVisibilityOn();
    points_mapper->SetColorModeToDirectScalars();
    points_mapper->Modified();

    const double range = std::pow(10,ranges[VisualizingVariable]);

    if(VisualizingVariable == VisOpt::color)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            int pt_idx = s->getValueInt(SimParams::integer_point_idx);
            uint32_t rgb = model->gpu.point_colors_rgb[pt_idx];
            uint8_t r = (rgb >> 16) & 0xff;
            uint8_t g = (rgb >> 8) & 0xff;
            uint8_t b = rgb & 0xff;
            pts_colors->SetValue((vtkIdType)(i*3+0), r);
            pts_colors->SetValue((vtkIdType)(i*3+1), g);
            pts_colors->SetValue((vtkIdType)(i*3+2), b);
        }
    }
    else if(VisualizingVariable == VisOpt::status)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            bool crushed = s->getCrushedStatus();
            bool disabled = s->getDisabledStatus();

            double val = 0;
            if(crushed) val = 1;
            if(disabled) val = 2;
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::NCD, val/2.);
            for(int k=0;k<3;k++) pts_colors->SetValue((vtkIdType)(i*3+k), c[k]);
        }
    }
    else if(VisualizingVariable == VisOpt::none || VisualizingVariable == VisOpt::regions)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            pts_colors->SetValue((vtkIdType)(i*3+0), 240);
            pts_colors->SetValue((vtkIdType)(i*3+1), 122);
            pts_colors->SetValue((vtkIdType)(i*3+2), 122);
        }
    }
    else if(VisualizingVariable == VisOpt::Jp_inv)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;

            double val = s->getValue(SimParams::idx_Jp_inv)-1.;
            double value = (val)/range + 0.5;
            double alpha = abs(val)/range;
            int pt_idx = s->getValueInt(SimParams::integer_point_idx);
            uint32_t rgb = model->gpu.point_colors_rgb[pt_idx];
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pressure, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(rgb, c, alpha);
            for(int k=0;k<3;k++) pts_colors->SetValue((vtkIdType)(i*3+k), c2[k]);
        }
    }
    else if(VisualizingVariable == VisOpt::ridges)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;

            double val = s->getValue(SimParams::idx_Jp_inv)-1.;
            double value = (val)/range;
            double alpha = val > 0 ? 1 : 0;
            int pt_idx = s->getValueInt(SimParams::integer_point_idx);
            uint32_t rgb = model->gpu.point_colors_rgb[pt_idx];
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Ridges, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(rgb, c, alpha);
            for(int k=0;k<3;k++) pts_colors->SetValue((vtkIdType)(i*3+k), c2[k]);
        }
        lut_Ridges->SetTableRange(0, range);
        scalarBar->SetLookupTable(lut_Ridges);
        scalarBar->SetLabelFormat("%.2f");
        scalarBar->VisibilityOn();
    }
    else if(VisualizingVariable == VisOpt::P)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;

            const double val = s->getValue(SimParams::idx_P);
            double value = (val)/range+0.5;
            double alpha = abs(val)/range;

            int pt_idx = s->getValueInt(SimParams::integer_point_idx);
            uint32_t rgb = model->gpu.point_colors_rgb[pt_idx];
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::Pressure, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(rgb, c, alpha);
            for(int k=0;k<3;k++) pts_colors->SetValue((vtkIdType)(i*3+k), c2[k]);
        }
    }
    else if(VisualizingVariable == VisOpt::Q)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            double val = s->getValue(SimParams::idx_Q);
            double value = (val)/range+0.5;
            double alpha = abs(val)/range;

            int pt_idx = s->getValueInt(SimParams::integer_point_idx);
            uint32_t rgb = model->gpu.point_colors_rgb[pt_idx];
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::P2, value);
            std::array<uint8_t, 3> c2 = colormap.mergeColors(rgb, c, alpha);

            for(int k=0;k<3;k++) pts_colors->SetValue((vtkIdType)(i*3+k), c2[k]);
        }
    }
    else if(VisualizingVariable == VisOpt::thickness)
    {
        for(int i=0;i<nPts;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            double val = s->getValue(SimParams::idx_thickness);
            double value = (val-0.8)/0.2;
            std::array<uint8_t, 3> c = colormap.getColor(ColorMap::Palette::ANSYS, value);
            for(int k=0;k<3;k++) pts_colors->SetValue((vtkIdType)(i*3+k), c[k]);
        }
    }
    else
    {
        actor_points->VisibilityOff();
    }

    points_filter->Update();

    points_polydata->GetPointData()->SetActiveScalars("pts_colors");
    pts_colors->Modified();
    points_polydata->Modified();
    points->Modified();

    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);
}


void icy::VisualRepresentation::ChangeVisualizationOption(int option)
{
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}




void icy::VisualRepresentation::populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table)
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


