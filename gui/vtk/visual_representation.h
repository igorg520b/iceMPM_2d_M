// visual_representation.h
#ifndef VISUAL_REPRESENTATION_H
#define VISUAL_REPRESENTATION_H

#include <QObject>
#include <vector>

#include <vtkNew.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkLookupTable.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkScalarBarActor.h>
#include <vtkTextActor.h>
#include <vtkPlaneSource.h>
#include <vtkTexture.h>
#include <vtkActor2D.h>
#include <vtkUnsignedCharArray.h>
#include <vtkImageData.h>

#include "colormap.h"

class HostSideData;

class VisualRepresentation : public QObject
{
    Q_OBJECT

public:
    explicit VisualRepresentation(HostSideData& data);
    ~VisualRepresentation();

    vtkNew<vtkActor> actor_points;
    vtkNew<vtkActor> raster_actor;
    vtkNew<vtkActor> actor_region_boundary;  // boundary lines for regions visualization
    vtkNew<vtkActor> actor_debug_grid;       // debug grid points at 10-cell intervals
    vtkNew<vtkTextActor> actorText;
    vtkNew<vtkScalarBarActor> scalarBar;
    vtkNew<vtkActor2D> textBgActor;
    vtkNew<vtkActor2D> scalarBarBgActor;

    HostSideData& hsd;

    double wind_visualization_time = 0;
    double simulationTime = 0;

    enum VisOpt {
        none,
        regions,    // only basic grid data is needed
        // points must be available
        pt_status, pt_color,
        pt_Jp_inv, pt_ridges,
        pt_P, pt_Q, pt_thickness,
        pt_partitions,
        // grid-based visualizations
        grid_ridges, grid_Jpinv, grid_mass, grid_pointdensity, grid_P, grid_Q,
        grid_colors, grid_vnorm, grid_force,
        str_EqvGreenLagrange, str_vonMises,
        // visualization of external currents/forces (to be implemented later)
        v_u, v_v, v_norm
    };
    Q_ENUM(VisOpt)

    VisOpt VisualizingVariable = VisOpt::none;
    constexpr static int max_vis_opts = 50;
    double ranges[max_vis_opts] = {};
    double transparency_coeffs[max_vis_opts] = {};

    bool enableDebugGrid = false;  // Debug grid only shown in plate_preparer

    void SynchronizeTopology();
    void ChangeVisualizationOption(int option);
    void ConfigureScalarBar();
    void UpdateTimeText();

private:
    ColorMap colormap;
    void SynchronizeValues();
    void SetupDebugGrid(int width, int height, double h);
    void SetupRegionBoundary(int gx, int gy, int ox, int oy, double h);  // draw rectangle for modeled area

    void SaveVisualizationState();
    void LoadVisualizationState();

    vtkNew<vtkLookupTable> lut_Pressure, lut_P2, lut_ANSYS, lut_Ridges;
    void populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table);

    // points
    vtkNew<vtkPoints> points;
    vtkNew<vtkPolyData> points_polydata;
    vtkNew<vtkPolyDataMapper> points_mapper;
    vtkNew<vtkVertexGlyphFilter> points_filter;
    vtkNew<vtkUnsignedCharArray> pts_colors;

    // background image
    std::vector<uint8_t> renderedImage;
    vtkNew<vtkImageData> raster_imageData;
    vtkNew<vtkUnsignedCharArray> raster_scalars;
    vtkNew<vtkPlaneSource> raster_plane;
    vtkNew<vtkTexture> raster_texture;
    vtkNew<vtkPolyDataMapper> raster_mapper;
};

#endif
