// pp_mainwindow.cpp

#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QCloseEvent>
#include <QStringList>
#include <QCoreApplication>
#include <QMessageBox>
#include <QProgressDialog>

#include <algorithm>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "pp_mainwindow.h"
#include "./ui_pp_mainwindow.h"



PPMainWindow::~PPMainWindow() {delete ui;}

PPMainWindow::PPMainWindow(QWidget *parent)
    : QMainWindow(parent), frameData(ggd,1)
    , ui(new Ui::PPMainWindow)
{
    ui->setupUi(this);

    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    setCentralWidget(qt_vtk_widget);

    renderWindow = qt_vtk_widget->renderWindow();

//    qt_vtk_widget->setRenderWindow(renderWindow);


    renderer->SetBackground(1.0,0.95,0.95);
    renderWindow->AddRenderer(renderer);

    renderWindow->GetInteractor()->SetInteractorStyle(interactor);

    renderer->AddActor(frameData.representation.raster_actor);
    renderer->AddActor(frameData.representation.actor_text);
    renderer->AddActor(frameData.representation.scalarBar);
    renderer->AddActor(frameData.representation.actor_text_title);


    // toolbar - combobox
    comboBox_visualizations = new QComboBox();
    ui->toolBar->addWidget(comboBox_visualizations);

    // double spin box
    qdsbValRange = new QDoubleSpinBox();
    qdsbValRange->setRange(-10, 10);
    qdsbValRange->setValue(-2);
    qdsbValRange->setDecimals(2);
    qdsbValRange->setSingleStep(0.25);
    ui->toolBar->addWidget(qdsbValRange);

    qsbFrameFrom = new QSpinBox();
    qsbFrameTo = new QSpinBox();
    ui->toolBar->addWidget(qsbFrameFrom);
    ui->toolBar->addWidget(qsbFrameTo);


    slider2 = new QSlider(Qt::Horizontal);
    ui->toolBar->addWidget(slider2);
    slider2->setTracking(true);
    slider2->setMinimum(1);
    slider2->setMaximum(10000);
    connect(slider2, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));

    // populate combobox
    QMetaEnum qme = QMetaEnum::fromType<VTKVisualization::VisOpt>();
    for(int i=0;i<qme.keyCount();i++) comboBox_visualizations->addItem(qme.key(i));


    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_visualizations(index); });

    // read/restore saved settings
    settingsFileName = QDir::currentPath() + "/ppcm.ini";
    QFileInfo fi(settingsFileName);

    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();

    if(fi.exists())
    {
        QSettings settings(settingsFileName,QSettings::IniFormat);
        QVariant var;


        var = settings.value("camData");
        if(!var.isNull())
        {
            double *vec = (double*)var.toByteArray().constData();
            camera->SetClippingRange(1e-1,1e4);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetPosition(vec[0],vec[1],vec[2]);
            camera->SetFocalPoint(vec[3],vec[4],vec[5]);
            camera->SetParallelScale(vec[6]);
            camera->ParallelProjectionOn();
            camera->Modified();
        }

        var = settings.value("visualization_ranges");
        if(!var.isNull())
        {
            QByteArray ba = var.toByteArray();
            memcpy(frameData.representation.ranges, ba.constData(), ba.size());
        }

        var = settings.value("vis_option");
        if(!var.isNull())
        {
            comboBox_visualizations->setCurrentIndex(var.toInt());
            qdsbValRange->setValue(frameData.representation.ranges[var.toInt()]);
        }
    }
    else
    {
        cameraReset_triggered();
    }



    connect(ui->action_camera_reset, &QAction::triggered, this, &PPMainWindow::cameraReset_triggered);
    connect(ui->actionRender_Frame, &QAction::triggered, this, &PPMainWindow::render_frame_triggered);
    connect(ui->actionRender_All, &QAction::triggered, this, &PPMainWindow::render_all_triggered);

//    connect(ui->actionOpen_Frame, &QAction::triggered, this, &PPMainWindow::open_frame_triggered);
//    connect(ui->actionGenerate_Script, &QAction::triggered, this, &PPMainWindow::generate_script_triggered);
//    connect(ui->actionShow_Wind, &QAction::triggered, this, &PPMainWindow::toggle_wind_visualization);
//    connect(ui->actionRender_All_2, &QAction::triggered, this, &PPMainWindow::render_all2_triggered);
//    connect(ui->actionLoad_Selected_Frame, &QAction::triggered, this, &PPMainWindow::load_selected_frame_triggered);

    connect(qdsbValRange,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &PPMainWindow::limits_changed);

    // off-screen rendering
    // Copy the renderers from the current render window

    ui->actionShow_Wind->setChecked(false);
    // Set the target resolution for the off-screen render window

    qDebug() << "PPMainWindow constructor done";

}




void PPMainWindow::closeEvent(QCloseEvent* event)
{
    qDebug() << "close event";

    QSettings settings(settingsFileName,QSettings::IniFormat);
    qDebug() << "PPMainWindow: closing main window; " << settings.fileName();

    double data[10];
    renderer->GetActiveCamera()->GetPosition(&data[0]);
    renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
    data[6] = renderer->GetActiveCamera()->GetParallelScale();

    qDebug() << "cam pos " << data[0] << "," << data[1] << "," << data[2];
    qDebug() << "cam focal pt " << data[3] << "," << data[4] << "," << data[5];
    qDebug() << "cam par scale " << data[6];

    QByteArray arr((char*)data, sizeof(data));
    settings.setValue("camData", arr);

    QByteArray ranges((char*)frameData.representation.ranges, sizeof(frameData.representation.ranges));
    settings.setValue("visualization_ranges", ranges);

    settings.setValue("vis_option", comboBox_visualizations->currentIndex());

    event->accept();
}


void PPMainWindow::limits_changed(double val_)
{
    qDebug() << "limits_changed";
    int idx = (int)frameData.representation.VisualizingVariable;
    frameData.representation.ranges[idx] = val_;
    frameData.representation.SynchronizeValues();
    renderWindow->Render();
}



void PPMainWindow::cameraReset_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);

    const double dx = ggd.prms.cellsize*ggd.prms.InitializationImageSizeX/2;
    const double dy = ggd.prms.cellsize*ggd.prms.InitializationImageSizeY/2;

    qDebug() << "dx " << dx << "\ndy " << dy;

    camera->SetPosition(dx, dy, 50.);
    camera->SetFocalPoint(dx, dy, 0.);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(std::min(dx,dy)*1.1);

    camera->Modified();
    renderWindow->Render();
}




// ==================================================

void PPMainWindow::LoadParametersFile(QString fileName)
{
    ggd.ReadParameterFile(fileName.toStdString());
}

void PPMainWindow::LoadFramesDirectory(QString framesDirectory)
{
    qDebug() << "PPMainWindow::LoadFramesDirectory: " << framesDirectory;
    ggd.ScanDirectory(framesDirectory.toStdString());
    slider2->setMaximum(ggd.countFrames);

    qsbFrameTo->setMaximum(ggd.countFrames);
    qsbFrameTo->setValue(ggd.countFrames);

    qsbFrameFrom->setMaximum(ggd.countFrames-1);
    qsbFrameFrom->setValue(1);

    frameData.UpdateQueue(1, ggd.countFrames-1);
    frameData.representation.SynchronizeValues();
    renderWindow->Render();

    generate_ffmpeg_script();
}

void PPMainWindow::sliderValueChanged(int val)
{
    qDebug() << "slider set to " << val;
    frameData.UpdateQueue(val, ggd.countFrames-1);
    frameData.representation.SynchronizeValues();
    renderWindow->Render();
}


void PPMainWindow::comboboxIndexChanged_visualizations(int index)
{
    frameData.representation.ChangeVisualizationOption(index);
    qdsbValRange->setValue(frameData.representation.ranges[index]);
    renderWindow->Render();
}



void PPMainWindow::render_frame_triggered()
{
    const int selectedFrame = slider2->value();
    qDebug() << "render_frame_triggered() " << selectedFrame;


    FrameData fd(ggd,1);

    vtkCamera *cam = renderer->GetActiveCamera();

    fd.SetUpOffscreenRender(this->frameData, cam);
    fd.UpdateQueue(selectedFrame, ggd.countFrames-1);
    fd.RenderFrame(frameData.representation.VisualizingVariable);

}


void PPMainWindow::render_all_triggered()
{
    const int frameFrom = qsbFrameFrom->value();
    const int frameTo = qsbFrameTo->value();
    const int totalFramesInRange = (frameTo - frameFrom) + 1;

    if (totalFramesInRange <= 0) {
        qDebug() << "PPMainWindow::render_all_triggered() - Invalid frame range.";
        QMessageBox::warning(this, "Invalid Range", "Frame 'To' must be greater than or equal to Frame 'From'.");
        return;
    }

    qDebug() << "PPMainWindow::render_all_triggered() from frame" << frameFrom << "to" << frameTo;

    std::vector<VTKVisualization::VisOpt> visOptsToRender = {
        VTKVisualization::VisOpt::Jp_inv,
        VTKVisualization::VisOpt::colors,
        VTKVisualization::VisOpt::P,
        VTKVisualization::VisOpt::Q,
        VTKVisualization::VisOpt::ridges,
        VTKVisualization::VisOpt::grid_vnorm
    };

    if (visOptsToRender.empty()) {
        qDebug() << "No visualization options selected for rendering.";
        QMessageBox::information(this, "Nothing to Render", "No visualization types were specified for batch rendering.");
        return;
    }

    const int prefetch_buffer_size = 3;
    FrameData fd(ggd,prefetch_buffer_size);
    vtkCamera* currentGuiCamera = this->renderer->GetActiveCamera();
    fd.SetUpOffscreenRender(this->frameData, currentGuiCamera); // Sets up canonical camera for fd.renderer

    int totalOperations = totalFramesInRange * visOptsToRender.size();
    QProgressDialog progress("Rendering all frames...", "Abort", 0, totalOperations, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.show();
    QCoreApplication::processEvents(); // Ensure dialog is shown

    int operationCount = 0;
    bool abortRendering = false;

    qDebug() << "Starting batch rendering. Total frames:" << totalFramesInRange << "Total VisOpts:" << visOptsToRender.size();

    // Outer loop: Iterate through frames
    for (int frameNum = frameFrom; frameNum <= frameTo; ++frameNum) {
        if (abortRendering) {
            break; // Exit outer loop if aborted
        }

        progress.setLabelText(QString("Processing Frame %1/%2...")
                                  .arg(frameNum - frameFrom + 1)
                                  .arg(totalFramesInRange));
        QCoreApplication::processEvents();

        qDebug() << "--- Loading HDF5 data for frame:" << frameNum << "---";
        fd.UpdateQueue(frameNum, frameTo); // Load frame data ONCE per frame

        // Inner loop: Iterate through visualization options for the current frame
        for (VTKVisualization::VisOpt currentVisOpt : visOptsToRender) {
            if (progress.wasCanceled()) {
                qDebug() << "Batch rendering aborted by user during VisOpt loop.";
                abortRendering = true;
                break; // Exit inner (VisOpt) loop
            }
            if (abortRendering) { // Double check, though outer loop break should handle it
                break;
            }

            operationCount++;
            QMetaEnum qme = QMetaEnum::fromType<VTKVisualization::VisOpt>();
            QString visOptName = qme.valueToKey(static_cast<int>(currentVisOpt));

            progress.setValue(operationCount);
            progress.setLabelText(QString("Rendering Frame %1 (%2/%3) - Vis: %4 (Op %5/%6)")
                                      .arg(frameNum)
                                      .arg(frameNum - frameFrom + 1)
                                      .arg(totalFramesInRange)
                                      .arg(visOptName)
                                      .arg(operationCount)
                                      .arg(totalOperations));
            QCoreApplication::processEvents(); // Keep GUI responsive

            qDebug() << "Rendering frame" << frameNum << "with VisOpt" << visOptName;
            fd.RenderFrame(currentVisOpt); // Render current VisOpt for the loaded frame
                // RenderFrame calls representation.ChangeVisualizationOption -> SynchronizeValues

            // QCoreApplication::processEvents(); // Process events after each render might be too frequent
            // but good for very slow single renders.
            // The one at the start of the inner loop is usually enough.
        }
        qDebug() << "--- Finished all VisOpts for frame:" << frameNum << "---";
        QCoreApplication::processEvents(); // Process events after all VisOpts for a frame are done
    }

    progress.setValue(totalOperations); // Ensure progress bar shows 100% if not aborted
    if (abortRendering) {
        QMessageBox::information(this, "Rendering Aborted", "Batch rendering was aborted by the user.");
    } else {
        QMessageBox::information(this, "Rendering Complete", "All selected frames and visualization types have been rendered.");
    }
    qDebug() << "PPMainWindow::render_all_triggered() finished.";
}

void PPMainWindow::generate_ffmpeg_script()
{
    const int frames = ggd.countFrames;
    const int fps = 30;
    std::filesystem::create_directories("output/raster");
    const std::string filename = "output/raster/genvideo.sh";
    std::ofstream scriptFile(filename);

    const std::string fmtStr = R"(ffmpeg -y -r {} -f image2 -start_number 1 -i "{}" -vframes {} -vcodec libx264 -vf "scale=1920:1080:force_original_aspect_ratio=decrease, pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white" -crf 21 -pix_fmt yuv420p "{}")";

    std::string cmd_P = fmt::format(fmt::runtime(fmtStr), fps, R"(P/%05d.jpg)", frames, R"(P.mp4)");
    std::string cmd_Q = fmt::format(fmt::runtime(fmtStr), fps, R"(Q/%05d.jpg)", frames, R"(Q.mp4)");
    std::string cmd_Jp_inv = fmt::format(fmt::runtime(fmtStr), fps, R"(Jpinv/%05d.jpg)", frames, R"(Jp_inv.mp4)");
    std::string cmd_colors = fmt::format(fmt::runtime(fmtStr), fps, R"(colors/%05d.jpg)", frames, R"(colors.mp4)");
    std::string cmd_Ridges = fmt::format(fmt::runtime(fmtStr), fps, R"(Ridges/%05d.jpg)", frames, R"(Ridges.mp4)");
    std::string cmd_vel = fmt::format(fmt::runtime(fmtStr), fps, R"(velocity/%05d.jpg)", frames, R"(velocity.mp4)");

    scriptFile << cmd_P << '\n' << cmd_Q << '\n' << cmd_Jp_inv << '\n' << cmd_colors << '\n' << cmd_Ridges << '\n' << cmd_vel;
    scriptFile.close();
    int result = std::system(("chmod +x " + filename).c_str());

    //std::string concat = R"(ffmpeg -i P.mp4 -i Q.mp4 -i Jp_inv.mp4 -i colors.mp4 -filter_complex "[0:v:0][1:v:0][2:v:0][3:v:0]concat=n=4:v=1[outv]" -map "[outv]" output.mp4)";
//    std::string cmd1 = fmt::format("ffmpeg -y -r {} -f image2 -start_number 1 -i \"P/%05d.png\" -vframes {} -vcodec libx264 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -crf 21  -pix_fmt yuv420p \"P.mp4\"\n", fps, frames);
}

