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
#include "vtk_representation.h"


PPMainWindow::~PPMainWindow() {delete ui;}

PPMainWindow::PPMainWindow(QWidget *parent)
    : QMainWindow(parent), frameData(ggd)
    , ui(new Ui::PPMainWindow)
{
    ui->setupUi(this);

    scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    setCentralWidget(scrollArea);
    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);
    scrollArea->setWidget(qt_vtk_widget);


    renderer->SetBackground(1.0,1.0,1.0);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(interactor);


    renderer->AddActor(representation.textBgActor);
    renderer->AddActor(representation.scalarBarBgActor);

    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.scalarBar);
    renderer->AddActor(representation.actorText);


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

    qdsbTransparency = new QDoubleSpinBox();
    qdsbTransparency->setRange(0, 1);
    qdsbTransparency->setValue(0);
    qdsbTransparency->setDecimals(1);
    qdsbTransparency->setSingleStep(0.1);
    ui->toolBar->addWidget(qdsbTransparency);

    ui->toolBar->addSeparator();

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
    QMetaEnum qme = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>();
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
            memcpy(representation.ranges, ba.constData(), ba.size());
        }

        var = settings.value("transparency_coeffs");
        if(!var.isNull())
        {
            QByteArray ba = var.toByteArray();
            memcpy(representation.transparency_coeffs, ba.constData(), ba.size());
        }

        var = settings.value("vis_option");
        if(!var.isNull())
        {
            comboBox_visualizations->setCurrentIndex(var.toInt());
            qdsbValRange->setValue(representation.ranges[var.toInt()]);
        }
    }
    else
    {
        cameraReset_triggered();
    }

    connect(ui->action_camera_reset, &QAction::triggered, this, &PPMainWindow::cameraReset_triggered);
    connect(ui->actionRender_Frame, &QAction::triggered, this, &PPMainWindow::render_frame_triggered);
    connect(ui->actionRender_All, &QAction::triggered, this, &PPMainWindow::render_all_triggered);
    connect(qdsbValRange,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &PPMainWindow::limits_changed);
    connect(qdsbTransparency,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &PPMainWindow::limits_changed);

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

    QByteArray ranges((char*)representation.ranges, sizeof(representation.ranges));
    settings.setValue("visualization_ranges", ranges);

    QByteArray transparency_coeffs((char*)representation.transparency_coeffs, sizeof(representation.transparency_coeffs));
    settings.setValue("transparency_coeffs", transparency_coeffs);

    settings.setValue("vis_option", comboBox_visualizations->currentIndex());

    event->accept();
}


void PPMainWindow::limits_changed(double val_)
{
    qDebug() << "limits_changed";
    int idx = (int)representation.VisualizingVariable;
    representation.ranges[idx] = qdsbValRange->value();
    representation.transparency_coeffs[idx] = qdsbTransparency->value();
    representation.SynchronizeTopology();
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

    if(ggd.countFrames <= 0)
    {
        slider2->setEnabled(false);
        QMessageBox::warning(this, "No Frames Found", "The specified directory does not contain any valid frame files.");
        return;
    }

    slider2->setMaximum(ggd.countFrames);

    qsbFrameTo->setMaximum(ggd.countFrames-1);
    qsbFrameTo->setValue(ggd.countFrames-1);

    qsbFrameFrom->setMaximum(ggd.countFrames-1);
    qsbFrameFrom->setValue(0);
    slider2->setValue(ggd.countFrames-1);
}

void PPMainWindow::sliderValueChanged(int val)
{
    qDebug() << "slider set to " << val;

    bool success = frameData.LoadFrame(val);
    if(!success)
    {
        LOGR("Failed to load and display frame {}.", val);
        return;
    }
    frameData.LinkRepresentation(this->representation);
    this->representation.SynchronizeTopology();
    renderWindow->Render();
}


void PPMainWindow::comboboxIndexChanged_visualizations(int index)
{
    representation.ChangeVisualizationOption(index);

    qdsbValRange->blockSignals(true);
    qdsbTransparency->blockSignals(true);
    qdsbValRange->setValue(representation.ranges[index]);
    qdsbTransparency->setValue(representation.transparency_coeffs[index]);
    qdsbValRange->blockSignals(false);
    qdsbTransparency->blockSignals(false);

    renderWindow->Render();
}



void PPMainWindow::render_frame_triggered()
{
    qDebug() << "render_frame_triggered()";
    std::string visName = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>().valueToKey(representation.VisualizingVariable);
    const int selectedFrame = slider2->value();
    std::string filename = visName + "_" + std::to_string(selectedFrame) + ".jpg";

    const QSizePolicy originalPolicy = qt_vtk_widget->sizePolicy();

    scrollArea->setWidgetResizable(false);
    qt_vtk_widget->setFixedSize(1920, 1080);
    QCoreApplication::processEvents();

    windowToImageFilter->SetInput(renderWindow);
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());

    renderWindow->DoubleBufferOff();
    renderWindow->Render();
    windowToImageFilter->SetInputBufferTypeToRGB();
    renderWindow->WaitForCompletion();

    writer->SetFileName(filename.c_str());
    writer->Write();

    renderWindow->DoubleBufferOn();

    qt_vtk_widget->setMinimumSize(0, 0);
    qt_vtk_widget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    qt_vtk_widget->setSizePolicy(originalPolicy);
    scrollArea->setWidgetResizable(true);
    QCoreApplication::processEvents();
}


void PPMainWindow::render_all_triggered()
{
    qDebug() << "render_all_triggered() started.";

    // --- 1. Define the Batch Job & Validate Inputs ---
    const int frameFrom = qsbFrameFrom->value();
    const int frameTo = qsbFrameTo->value();
    generate_ffmpeg_script(frameFrom, frameTo);


    if (frameTo < frameFrom) {
        QMessageBox::warning(this, "Invalid Range", "Frame 'To' must be greater than or equal to Frame 'From'.");
        return;
    }
    const int totalFrames = (frameTo - frameFrom) + 1;

    // --- 2. Set Up Progress Dialog ---
    const int totalOperations = totalFrames * m_visOptsToRender.size();
    QProgressDialog progress("Rendering all frames...", "Abort", 0, totalOperations, this);
    progress.setWindowModality(Qt::WindowModal);
    QCoreApplication::processEvents();

    // --- 3. Prepare for Batch Rendering (Done ONCE) ---
    const QSizePolicy originalPolicy = qt_vtk_widget->sizePolicy();
    scrollArea->setWidgetResizable(false);
    qt_vtk_widget->setFixedSize(1920, 1080);
    QCoreApplication::processEvents();

    renderWindow->DoubleBufferOff();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetInputBufferTypeToRGB();
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());

    // --- 4. Main Batch Processing Loop ---
    int operationCount = 0;
    for (int frameNum = frameFrom; frameNum <= frameTo; ++frameNum) {
        if (progress.wasCanceled()) break;

        progress.setLabelText(QString("Loading frame %1...").arg(frameNum));
        QCoreApplication::processEvents();

        if (!frameData.LoadFrame(frameNum)) {
            LOGR("Failed to load frame {}, skipping.", frameNum);
            operationCount += m_visOptsToRender.size();
            progress.setValue(operationCount);
            continue;
        }
        frameData.LinkRepresentation(this->representation);

        for (const auto& visOpt : m_visOptsToRender) {
            if (progress.wasCanceled()) break;
            operationCount++;

            const QMetaEnum metaEnum = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>();
            const QString visName = metaEnum.valueToKey(visOpt);

            progress.setValue(operationCount);
            progress.setLabelText(QString("Frame %1/%2 : Rendering %3")
                                      .arg(frameNum - frameFrom + 1).arg(totalFrames).arg(visName));
            QCoreApplication::processEvents();

            // --- CRITICAL FIX: Update state and render BEFORE capturing ---
            representation.ChangeVisualizationOption(visOpt);
            renderWindow->Render();
            windowToImageFilter->Modified(); // This is extra important

            // --- Construct Correct Output Path ---
            QDir frameDir(QString::fromStdString(ggd.frameDirectory));
            frameDir.cdUp(); // Go up to the parent (e.g., .../cb2k/)
            const QString rasterBasePath = frameDir.filePath("raster");
            const QString subDir = QString("%1/%2").arg(rasterBasePath).arg(visName);
            QDir().mkpath(subDir); // Use QDir to create the full path
            const QString outputPath = QString("%1/%2.jpg").arg(subDir).arg(frameNum, 5, 10, QChar('0'));

            // --- Capture and Write ---
            writer->SetFileName(outputPath.toStdString().c_str());
            writer->Write();
        }
    }

    // --- 5. Restore GUI State (Done ONCE) ---
    qt_vtk_widget->setMinimumSize(0, 0);
    qt_vtk_widget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    qt_vtk_widget->setSizePolicy(originalPolicy);
    scrollArea->setWidgetResizable(true);
    renderWindow->DoubleBufferOn();
    QCoreApplication::processEvents();

    // Refresh the final interactive view to match the last rendered item
    renderWindow->Render();

    // --- 6. Finalize ---
    progress.setValue(totalOperations);
    QMessageBox::information(this, "Rendering Complete", "Batch rendering has finished or was canceled.");
}



// --- Core Logic Function (Rewritten) ---
// This is the clean, reusable implementation.
void PPMainWindow::generate_ffmpeg_script(int frameFrom, int frameTo)
{
    qDebug() << "Generating ffmpeg script for frames" << frameFrom << "to" << frameTo;

    const int totalFrames = (frameTo - frameFrom) + 1;
    if (totalFrames <= 0) {
        qDebug() << "Invalid frame range provided to generate_ffmpeg_script.";
        return;
    }
    const int fps = 30;

    // Correctly determine the output "raster" directory path.
    QDir frameDir(QString::fromStdString(ggd.frameDirectory));
    frameDir.cdUp(); // Go from ".../output/cb2k/frames" to ".../output/cb2k/"
    const QString rasterPath = frameDir.filePath("raster");
    QDir().mkpath(rasterPath); // Ensure the raster directory exists
    const std::string scriptFilename = QDir(rasterPath).filePath("genvideo.sh").toStdString();

    std::ofstream scriptFile(scriptFilename);
    if (!scriptFile.is_open()) {
        LOGR("Failed to open script file for writing: {}", scriptFilename);
        return;
    }

    // Write a standard, robust script header.
    scriptFile << "#!/bin/bash\n";
    scriptFile << "# This script will generate mp4 videos from the rendered JPG frames.\n";
    scriptFile << "# It is designed to be run from within the 'raster' directory.\n";
    scriptFile << "cd \"$(dirname \"$0\")\"\n\n";

    // Define the ffmpeg command template.
    // Using -1 for padding automatically centers the image.
    const std::string fmtStr = R"(ffmpeg -y -r {0:} -f image2 -start_number {1:} -i "{2:}" -vframes {3:} -vcodec libx264 -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1:white" -crf 21 -pix_fmt yuv420p "{4:}")";
    const QMetaEnum metaEnum = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>();

    // Loop through the class member list to generate a command for each visualization.
    for (const auto& visOpt : m_visOptsToRender) {
        const std::string visName = metaEnum.valueToKey(visOpt);
        const std::string inputFilePattern = visName + "/%05d.jpg";
        const std::string outputVideoFile = visName + ".mp4";

        std::string command = fmt::format(fmt::runtime(fmtStr),
                                          fps,
                                          frameFrom,
                                          inputFilePattern,
                                          totalFrames,
                                          outputVideoFile);

        scriptFile << "# Generate video for " << visName << "\n";
        scriptFile << command << "\n\n";
    }

    scriptFile.close();
    // Make the script executable.
    int result = std::system(("chmod +x " + scriptFilename).c_str());

    ui->statusbar->showMessage(QString("Generated genvideo.sh in %1").arg(rasterPath), 5000);
}
