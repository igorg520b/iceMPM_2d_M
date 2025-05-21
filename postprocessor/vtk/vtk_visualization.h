// vtk_visualization.h

#ifndef VTKVISUALIZATION_H
#define VTKVISUALIZATION_H

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

#include <vtkRegularPolygonSource.h>
#include <vtkCylinderSource.h>

#include <vtkUnsignedCharArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkArrowSource.h>
#include <vtkGlyph2D.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <Eigen/Core>

#include "colormap.h"

struct FrameData;

class VTKVisualization : public QObject
{
    Q_OBJECT

public:
    FrameData &frameData;

    VTKVisualization(FrameData &fd_);

    double wind_visualization_time;

    enum VisOpt { none, density, colors, Jp_inv, P, Q, grid_vnorm, ridges};
    Q_ENUM(VisOpt)
    VisOpt VisualizingVariable = VisOpt::none;
    double ranges[30] = {};

    void SynchronizeValues();
    void ChangeVisualizationOption(int option);  // invoked from GUI/main thread

    vtkNew<vtkActor> raster_actor;

    vtkNew<vtkScalarBarActor> scalarBar;
    vtkNew<vtkTextActor> actor_text;
    vtkNew<vtkTextActor> actor_text_title;

private:
    ColorMap colormap;

    vtkNew<vtkLookupTable> lut_Pressure, lut_P2, lut_ANSYS, lut_ridges;

    // background image
    std::vector<uint8_t> renderedImage;
    vtkNew<vtkImageData> raster_imageData;
    vtkNew<vtkUnsignedCharArray> raster_scalars;
    vtkNew<vtkPlaneSource> raster_plane;
    vtkNew<vtkTexture> raster_texture;
    vtkNew<vtkPolyDataMapper> raster_mapper;


    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    vtkNew<vtkPNGWriter> writerPNG;
    vtkNew<vtkRenderer> offscreenRenderer;

    void populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table);
};
#endif
