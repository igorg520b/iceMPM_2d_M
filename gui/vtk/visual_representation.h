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
#include <vtkStructuredGrid.h>
#include <vtkHedgeHog.h>
#include <vtkStreamTracer.h>
#include <vtkTubeFilter.h>
#include <vtkFloatArray.h>

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
    
    // New actors for flow visualization
    vtkNew<vtkActor> actor_vectors;      // For HedgeHog (Vector Field)
    vtkNew<vtkActor> actor_streamlines;  // For Streamlines

    vtkNew<vtkTextActor> actorText;
    vtkNew<vtkTextActor> actorTextTitle;
    vtkNew<vtkScalarBarActor> scalarBar;
    vtkNew<vtkActor2D> textBgActor;
    vtkNew<vtkActor2D> scalarBarBgActor;


    HostSideData& hsd;


    double simulationTime = 0;

    enum VisOpt {
        none,
        regions,
        // points must be available
        pt_status,
        pt_color,
        pt_Jp_inv,
        pt_ridges,
        pt_P,
        pt_Q,
        pt_thickness,
        pt_partitions,
        pt_glen_flow,
        pt_fracture_type,
        pt_damage,
        pt_strain_energy,
        // grid-based visualizations
        grid_mass,
        grid_pt_count,
        grid_Jpinv,
        grid_ridges,
        grid_P,
        grid_Q,
        grid_colors,
        grid_vnorm,
        grid_cracked,
        grid_thickness,
        grid_fracture_type,
        grid_glen_flow,
        str_EqvGreenLagrange,
        str_vonMises,
        // visualization of external currents/forces
        v_ocean_norm,
        v_wind_norm,
        ocean_streamlines,
        wind_streamlines
    };
    Q_ENUM(VisOpt)

    inline static const std::map<VisOpt, std::string> visOptDescriptions = {
        {none, ""},
        {regions, "Regions"},
        {pt_status, "Status"},
        {pt_color, "Color"},
        {pt_Jp_inv, "Change in Surf. Density"},
        {pt_ridges, "Ridges"},
        {pt_P, "In-plane Pressure"},
        {pt_Q, "Deviatoric Stress"},
        {pt_thickness, "Thickness"},
        {pt_partitions, "GPU Partitions"},
        {pt_glen_flow, "Glen Flow"},
        {pt_fracture_type, "Fracture Type"},
        {pt_damage, "Point Damage"},
        {pt_strain_energy, "Strain Energy Density"},
        {grid_mass, "Mass"},
        {grid_pt_count, "Point count"},
        {grid_Jpinv, "Jp_inv"},
        {grid_ridges, "Ridges"},
        {grid_P, "In-plane Pressure P"},
        {grid_Q, "Deviatoric Stress Q"},
        {grid_colors, "Colors"},
        {grid_vnorm, "Ice velocity norm"},
        {grid_cracked, "Cracked/Crushed Material"},
        {grid_thickness, "Ice Thickness"},
        {grid_fracture_type, "Fracture Type"},
        {grid_glen_flow, "Glen Flow"},
        {str_EqvGreenLagrange, "Green-Lagrange Strain"},
        {str_vonMises, "von Mises Strain"},
        {v_ocean_norm, "Ocean Current Velocity Norm"},
        {v_wind_norm, "Wind Velocity Norm"},
        {ocean_streamlines, "Ocean Streamlines"},
        {wind_streamlines, "Wind Streamlines"}
    };



    VisOpt VisualizingVariable = VisOpt::none;
    constexpr static int max_vis_opts = 50;
    double ranges[max_vis_opts] = {};
    double transparency_coeffs[max_vis_opts] = {};

    // actor_contours moved to top

    void SynchronizeTopology();
    void ChangeVisualizationOption(int option);
    void ConfigureScalarBar();
    void UpdateTimeText();

    std::vector<int> GetRequiredGridArrays(VisOpt visualizationOptionIndex);

private:
    constexpr static std::string_view state_file_name = "plateMPM_vis_state";

    ColorMap colormap;
    void SynchronizeValues();
    void UpdateFlowData();

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
    
    // Flow Visualization Infrastructure
    vtkNew<vtkStructuredGrid> flow_grid;
    vtkNew<vtkFloatArray> flow_vectors;

    // HedgeHog
    vtkNew<vtkHedgeHog> hedgehog;
    vtkNew<vtkPolyDataMapper> hedgehog_mapper;

    // Streamlines
    vtkNew<vtkPlaneSource> seed_plane;
    vtkNew<vtkStreamTracer> streamer;
    vtkNew<vtkPolyDataMapper> stream_mapper;
};

#endif
