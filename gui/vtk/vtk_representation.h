#ifndef MESHREPRESENTATION_H
#define MESHREPRESENTATION_H

#include <QObject>

#include <vtkNew.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellType.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkNamedColors.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkLookupTable.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyLine.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkStructuredGrid.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkTextActor.h>
#include <vtkPlaneSource.h>
#include <vtkTexture.h>
#include <vtkActor2D.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkProperty2D.h>

#include <vtkRegularPolygonSource.h>
#include <vtkCylinderSource.h>

#include <vtkUnsignedCharArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkImageData.h>
#include <vtkUniformGrid.h>

#include "colormap.h"
#include "parameters_sim.h"
#include "host_side_soa.h"
#include "windandcurrentinterpolator.h"

namespace icy { class VisualRepresentation; class Model;}

class icy::VisualRepresentation : public QObject
{
    Q_OBJECT

public:
    VisualRepresentation();

    vtkNew<vtkActor> actor_points;
    vtkNew<vtkActor> raster_actor;
    vtkNew<vtkTextActor> actorText;
    vtkNew<vtkScalarBarActor> scalarBar;
    vtkNew<vtkActor2D> textBgActor;
    vtkNew<vtkActor2D> scalarBarBgActor;

    const SimParams* prms = nullptr;
    HostSideSOA* hssoa = nullptr;
    const WindAndCurrentInterpolator* wac_interpolator = nullptr;
    const uint8_t *grid_status_buffer = nullptr;
    const t_GridReal *host_grid_buffer = nullptr;
    const std::vector<uint8_t>* original_image_colors_rgb = nullptr;
    const std::vector<uint8_t>* point_partitions = nullptr;

    double wind_visualization_time;
    double simulationTime = 0;

    enum VisOpt { none, status, Jp_inv, P, Q, color, v_u, v_v, v_norm, thickness,
                  regions, ridges, grid_Jpinv, grid_mass, grid_pointdensity, grid_P, grid_Q, grid_colors, grid_vnorm, grid_force,
                  partitions, str_EqvGreenLagrange, str_vonMises, grid_ridges};
    Q_ENUM(VisOpt)
    VisOpt VisualizingVariable = VisOpt::none;
    double ranges[30] = {};

    void SynchronizeTopology();
    void ChangeVisualizationOption(int option);  // invoked from GUI/main thread
    void ConfigureScalarBar();
    void UpdateTimeText();




private:
    ColorMap colormap;
    void SynchronizeValues();

    vtkNew<vtkLookupTable> lut_Pressure, lut_P2, lut_ANSYS, lut_Ridges;
    void populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table);

    // points
    vtkNew<vtkPoints> points;
    vtkNew<vtkPolyData> points_polydata;
    vtkNew<vtkPolyDataMapper> points_mapper;
    vtkNew<vtkVertexGlyphFilter> points_filter;
    vtkNew<vtkUnsignedCharArray> pts_colors;    // for color visualization per point

    // background image
    std::vector<uint8_t> renderedImage;
    vtkNew<vtkImageData> raster_imageData;
    vtkNew<vtkUnsignedCharArray> raster_scalars;
    vtkNew<vtkPlaneSource> raster_plane;
    vtkNew<vtkTexture> raster_texture;
    vtkNew<vtkPolyDataMapper> raster_mapper;

    void UpdateOverlayBackgrounds(vtkRenderer* renderer);

};
#endif
