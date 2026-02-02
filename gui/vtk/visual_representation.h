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
        none,   // 0
        regions,    //1
        // points must be available
        pt_status, //2
        pt_color, // 3
        pt_Jp_inv, //4
        pt_ridges, //5
        pt_P, //6
        pt_Q, //7
        pt_thickness, //8
        pt_partitions, //9
        pt_glen_flow, // 10
        pt_fracture_type, //11
        // grid-based visualizations
        grid_mass, //12
        grid_pt_count, //13
        grid_Jpinv, //14
        grid_ridges, //15
        grid_P, //16
        grid_Q, //17
        grid_colors, //18
        grid_vnorm, //19
        grid_cracked, //20
        grid_thickness, //21
        grid_fracture_type, //22
        grid_glen_flow,     // 23
        str_EqvGreenLagrange, //24
        str_vonMises, //25
        // visualization of external currents/forces
        v_ocean_norm, //26
        v_wind_norm,  // 27
        ocean_streamlines, //28
        wind_streamlines,  // 29
        vis_lat,
        vis_lon
    };
    Q_ENUM(VisOpt)

    inline static constexpr std::array<std::string_view, 32> visOptDescriptions = {
        "", "Regions", "Status", "Color",
        "Change in Surf. Density", "Ridges",
        "In-plane Pressure", "Deviatoric Stress", "Thickness", "GPU Partitions", "Glen Flow", 
        "Fracture Type",

        "Mass", "Point count", "Jp_inv", "Ridges",

        "In-plane Pressure P", "Deviatoric Stress Q",
        "Colors", "Ice velocity norm", "Cracked/Crushed Material",
        "Ice Thickness", "Fracture Type", "Glen Flow",
        "Green-Lagrange Strain", "von Mises Strain",
        "Ocean Current Velocity Norm", "Wind Velocity Norm", 
        "Ocean Streamlines", "Wind Streamlines",
        "Latitude", "Longitude"
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
