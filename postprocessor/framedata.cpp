// framedata.cpp

#include "framedata.h"
#include <fmt/format.h>
#include <fmt/std.h>

#include <vtkVersionMacros.h>
#include <QDebug>

namespace fs = std::filesystem;


FrameData::FrameData(GeneralGridData &ggd_, int prefetch_buffer_size_) : ggd(ggd_),
    representation(*this), PREFETCH_BUFFER_SIZE(prefetch_buffer_size_)
{
    std::cout << "VTK Version (from macros):" << std::endl;
    std::cout << "  Full Version String (VTK_VERSION): " << VTK_VERSION << std::endl;
    std::cout << "  Major Version (VTK_MAJOR_VERSION): " << VTK_MAJOR_VERSION << std::endl;
    std::cout << "  Minor Version (VTK_MINOR_VERSION): " << VTK_MINOR_VERSION << std::endl;
    std::cout << "  Build Version (VTK_BUILD_VERSION): " << VTK_BUILD_VERSION << std::endl;
    snapshot_pool.resize(PREFETCH_BUFFER_SIZE);
}

void FrameData::UpdateQueue(int frameNumber, int frameTo)
{
    circular_buffer_top++;
    circular_buffer_top %= snapshot_pool.size();

    // frameNumber must end up on "top" of the queue
    if(this->frontSnapShot().FrameNumber != frameNumber)
    {
        circular_buffer_top = 0;
        for(int i=0;i<snapshot_pool.size();i++)
            if(i+frameNumber <= frameTo)
            {
                LOGR("invoking StartLoadFrameCompressedAsync; i {}; frame {}", i, i+frameNumber);
                snapshot_pool[i].StartLoadFrameCompressedAsync(ggd.frameDirectory, i+frameNumber);
            }
    }
    else
    {
        const int frame_to_load = frameNumber + PREFETCH_BUFFER_SIZE - 1;
        if(frame_to_load <= frameTo)
        {
            int last_in_queue = (circular_buffer_top + PREFETCH_BUFFER_SIZE - 1) % snapshot_pool.size();
            snapshot_pool[last_in_queue].StartLoadFrameCompressedAsync(ggd.frameDirectory, frame_to_load);
        }
    }
    frontSnapShot().data_ready_flag_.wait(false); // wait until flag is set to true
    LOGV("UpdateQueue done");
}


void FrameData::SetUpOffscreenRender(const FrameData &guiFD, vtkCamera* sourceGuiCamera)
{
    std::copy(std::begin(guiFD.representation.ranges), std::end(guiFD.representation.ranges), std::begin(this->representation.ranges));

    offscreenRenderWindow->Initialize();
    offscreenRenderWindow->SetOffScreenRendering(true);
    offscreenRenderWindow->SetSize(1920, 1080);
    offscreenRenderWindow->DoubleBufferOff();

    // renderer -> offscreen
    offscreenRenderWindow->AddRenderer(renderer);
    renderer->SetBackground(1.0, 1.0, 1.0);

    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.actor_text);
    renderer->AddActor(representation.scalarBar);
    renderer->AddActor(representation.actor_text_title);

    windowToImageFilter->SetInput(offscreenRenderWindow);
    windowToImageFilter->SetScale(1);
    windowToImageFilter->SetInputBufferTypeToRGB();
//    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    windowToImageFilter->ReadFrontBufferOn();
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());


    // set up camera
    renderer->ResetCamera();
    vtkCamera* cam = renderer->GetActiveCamera();
    cam->DeepCopy(sourceGuiCamera);
    cam->ParallelProjectionOn();
    cam->Modified();
}


void FrameData::RenderFrame(VTKVisualization::VisOpt visopt)
{
    representation.ChangeVisualizationOption(visopt);

    offscreenRenderWindow->Render();
    windowToImageFilter->Modified(); // this is extra important
    std::string renderFileName = fmt::format("{:05d}.jpg", frontSnapShot().FrameNumber);

    fs::path outputDir = "output/raster";
    fs::path imageDir;
    if(visopt == VTKVisualization::VisOpt::colors) imageDir = "colors";
    else if(visopt == VTKVisualization::VisOpt::Jp_inv) imageDir = "Jpinv";
    else if(visopt == VTKVisualization::VisOpt::P) imageDir = "P";
    else if(visopt == VTKVisualization::VisOpt::Q) imageDir = "Q";
    else if(visopt == VTKVisualization::VisOpt::ridges) imageDir = "Ridges";
    else if(visopt == VTKVisualization::VisOpt::grid_vnorm) imageDir = "velocity";


    fs::path targetPath = outputDir / imageDir;
    fs::create_directories(targetPath);

    fs::path fullPath = targetPath / renderFileName;

    LOGR("FrameData::RenderFrame; writing {}", renderFileName);

    writer->SetFileName(fullPath.string().c_str());
    writer->Write();
}


