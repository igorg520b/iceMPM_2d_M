// framedata.h

#ifndef FRAMEDATA_H
#define FRAMEDATA_H

#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include <iostream>

#include <H5Cpp.h>
#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include "parameters_sim.h"
#include "vtk_visualization.h"
#include "generalgriddata.h"

#include <vtkRenderWindow.h>
#include <vtkWindowToImageFilter.h>
#include <vtkJPEGWriter.h>
#include <vtkRenderer.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkCamera.h>

#include "snapshotmanager.h"



struct FrameData
{
    FrameData(GeneralGridData &ggd_, int prefetch_buffer_size_);

    GeneralGridData &ggd;
    VTKVisualization representation;

    void SetUpOffscreenRender(const FrameData &guiFD, vtkCamera* sourceGuiCamera);
    void RenderFrame(VTKVisualization::VisOpt visopt);
    void UpdateQueue(int frameNumber, int frameTo);

    double getSimulationTime() {return frontSnapShot().SimulationTime;};
    icy::SnapshotManager& frontSnapShot() {return snapshot_pool[circular_buffer_top];}; // for rendering

    vtkNew<vtkRenderer> renderer;

private:
    const int PREFETCH_BUFFER_SIZE;
    int circular_buffer_top;    // which snapshot is the front of the queue
    std::vector<icy::SnapshotManager> snapshot_pool;

    // offscreenrender
    vtkNew<vtkRenderWindow> offscreenRenderWindow;
    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    vtkNew<vtkJPEGWriter> writer;
};

#endif // FRAMEDATA_H
