#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QCloseEvent>
#include <QStringList>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>
#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::~MainWindow() {delete ui;}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    params = new ParamsWrapper(&model.prms);
    representation.model = &model;
    worker = new BackgroundWorker(&model);

    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);

    renderer->SetBackground(1.0,1.0,1.0);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(interactorStyle);

    // property browser
    pbrowser = new ObjectPropertyBrowser(this);

    // splitter
    splitter = new QSplitter(Qt::Orientation::Horizontal);
    splitter->addWidget(pbrowser);
    splitter->addWidget(qt_vtk_widget);
    splitter->setSizes(QList<int>({100, 500}));
    setCentralWidget(splitter);

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

    qsbIntentionalSlowdown = new QSpinBox();
    qsbIntentionalSlowdown->setRange(0,1000);
    qsbIntentionalSlowdown->setValue(0);
    ui->toolBar->addWidget(qsbIntentionalSlowdown);

    // statusbar
    statusLabel = new QLabel();
    labelElapsedTime = new QLabel();
    labelStepCount = new QLabel();
    labelWindSpeed = new QLabel();
    labelWindDirection = new QLabel();

    QSizePolicy sp;
    const int status_width = 90;
    sp.setHorizontalPolicy(QSizePolicy::Fixed);
    labelStepCount->setSizePolicy(sp);
    labelStepCount->setFixedWidth(status_width);
    labelElapsedTime->setSizePolicy(sp);
    labelElapsedTime->setFixedWidth(status_width);
    labelWindSpeed->setSizePolicy(sp);
    labelWindSpeed->setFixedWidth(status_width);
    labelWindDirection->setSizePolicy(sp);
    labelWindDirection->setFixedWidth(status_width);

    ui->statusbar->addWidget(statusLabel);
    ui->statusbar->addPermanentWidget(labelWindSpeed);
    ui->statusbar->addPermanentWidget(labelWindDirection);
    ui->statusbar->addPermanentWidget(labelElapsedTime);
    ui->statusbar->addPermanentWidget(labelStepCount);

// anything that includes the Model
    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.scalarBar);

    // populate combobox
    QMetaEnum qme = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>();
    for(int i=0;i<qme.keyCount();i++) comboBox_visualizations->addItem(qme.key(i));

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_visualizations(index); });


    // slider
    slider1 = new QSlider(Qt::Horizontal);
    ui->toolBar->addWidget(slider1);
    slider1->setTracking(true);
    slider1->setMinimum(0);
    slider1->setMaximum(10000);
    connect(slider1, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));


    // read/restore saved settings
    settingsFileName = QDir::currentPath() + "/cm.ini";
    QFileInfo fi(settingsFileName);

    if(fi.exists())
    {
        QSettings settings(settingsFileName,QSettings::IniFormat);
        QVariant var;

        vtkCamera* camera = renderer->GetActiveCamera();
        renderer->ResetCamera();
        camera->ParallelProjectionOn();

        var = settings.value("camData");
        if(!var.isNull())
        {
            double *vec = (double*)var.toByteArray().constData();
            camera->SetClippingRange(1e-1,1e4);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetPosition(vec[0],vec[1],vec[2]);
            camera->SetFocalPoint(vec[3],vec[4],vec[5]);
            camera->SetParallelScale(vec[6]);
            camera->Modified();
        }

        var = settings.value("visualization_ranges");
        if(!var.isNull())
        {
            QByteArray ba = var.toByteArray();
            memcpy(representation.ranges, ba.constData(), ba.size());
        }

        var = settings.value("take_screenshots");
        if(!var.isNull())
        {
            bool b = var.toBool();
            ui->actionTake_Screenshots->setChecked(b);
        }

        comboBox_visualizations->setCurrentIndex(settings.value("vis_option").toInt());

        var = settings.value("splitter_size_0");
        if(!var.isNull())
        {
            int sz1 = var.toInt();
            int sz2 = settings.value("splitter_size_1").toInt();
            splitter->setSizes(QList<int>({sz1, sz2}));
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


    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetScale(1); // image quality
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    windowToImageFilter->ReadFrontBufferOn(); // read from the back buffer
    writerPNG->SetInputConnection(windowToImageFilter->GetOutputPort());

    connect(ui->action_quit, &QAction::triggered, this, &MainWindow::quit_triggered);
    connect(ui->action_camera_reset, &QAction::triggered, this, &MainWindow::cameraReset_triggered);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::open_snapshot_triggered);
    connect(ui->actionStart_Pause, &QAction::triggered, this, &MainWindow::simulation_start_pause);
    connect(ui->actionLoad_Parameters, &QAction::triggered, this, &MainWindow::load_parameter_triggered);
    connect(ui->actionPrint_Camera_Params, &QAction::triggered, this, &MainWindow::print_camera_params);
    connect(ui->actionView_ScalarBar, &QAction::triggered, this, &MainWindow::toggle_scalarbar);

    connect(qdsbValRange,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);
    connect(qsbIntentionalSlowdown,QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::spinbox_slowdown_value_changed);

    connect(worker, SIGNAL(workerPaused()), SLOT(background_worker_paused()));
    connect(worker, SIGNAL(stepCompleted()), SLOT(simulation_data_ready()));

    connect(params, SIGNAL(propertyChanged()), SLOT(parameters_updated()));

    pbrowser->setActiveObject(params);
    qDebug() << "MainWindow constructor done";
}


void MainWindow::closeEvent(QCloseEvent* event)
{
    quit_triggered();
    event->accept();
}


void MainWindow::quit_triggered()
{
    qDebug() << "MainWindow::quit_triggered() ";
    worker->Finalize();
    // save settings and stop simulation
    QSettings settings(settingsFileName,QSettings::IniFormat);
    qDebug() << "MainWindow: closing main window; " << settings.fileName();

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

    settings.setValue("vis_option", comboBox_visualizations->currentIndex());


    QList<int> szs = splitter->sizes();
    settings.setValue("splitter_size_0", szs[0]);
    settings.setValue("splitter_size_1", szs[1]);

    settings.setValue("take_screenshots", ui->actionTake_Screenshots->isChecked());
    QApplication::quit();
}



void MainWindow::comboboxIndexChanged_visualizations(int index)
{
    representation.ChangeVisualizationOption(index);
    qdsbValRange->setValue(representation.ranges[index]);
    renderWindow->Render();
}

void MainWindow::limits_changed(double val_)
{
    int idx = (int)representation.VisualizingVariable;
    representation.ranges[idx] = val_;
    representation.SynchronizeTopology();
    renderWindow->Render();
}

void MainWindow::cameraReset_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);
    camera->SetFocalPoint(0, 0., 0.);
    camera->SetPosition(0.0, 0.0, 50.0);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(2.5);

    camera->Modified();
    renderWindow->Render();
}


void MainWindow::open_snapshot_triggered()
{
    QString defaultPath = QDir::currentPath() + "/_data/snapshots";

    // Check if the directory exists
    if (!QDir(defaultPath).exists()) {
        // Fall back to the current path if the default directory does not exist
        defaultPath = QDir::currentPath();
    }

    QString qFileName = QFileDialog::getOpenFileName(this, "Open Simulation Snapshot", defaultPath, "HDF5 Files (*.h5)");
    if(qFileName.isNull())return;
    OpenSnapshot(qFileName);
}

void MainWindow::OpenSnapshot(QString fileName)
{
/*
    model.snapshot.ReadSnapshot(fileName.toStdString());
    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    updateGUI();
*/
}


void MainWindow::load_parameter_triggered()
{
    QString qFileName = QFileDialog::getOpenFileName(this, "Load Parameters", QDir::currentPath(), "JSON Files (*.json)");
    if(qFileName.isNull())return;
    QString tmp;
    LoadParameterFile(qFileName, tmp);
}



void MainWindow::simulation_data_ready()
{
    updateGUI();
    if(ui->actionTake_Screenshots->isChecked())
        screenshot();
    model.UnlockCycleMutex();
}


void MainWindow::updateGUI()
{
    LOGV("updateGUI");
    labelStepCount->setText(QString::number(model.prms.SimulationStep));
    labelElapsedTime->setText(QString("%1 s").arg(model.prms.SimulationTime,0,'f',0));
//    labelWindSpeed->setText(QString("%1 m/s").arg(model.windSpeed,0,'f',2));
//    labelWindDirection->setText(QString("%1 deg").arg(model.windAngle,0,'f',0));

    // display date
    int64_t display_date = (int64_t)model.prms.SimulationTime + model.prms.SimulationStartUnixTime;
    std::time_t unix_time = display_date;
    std::tm* tm_time = std::gmtime(&unix_time);
    // Format the time
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S UTC", tm_time);
    representation.actorText->SetInput(buffer);

    //statusLabel->setText(QString("per cycle: %1 ms").arg(model.compute_time_per_cycle,0,'f',3));

    if(model.SyncTopologyRequired)
    {
        model.SyncTopologyRequired = false;
        representation.SynchronizeTopology();
    }
    else
    {
        representation.SynchronizeTopology();
    }
    renderWindow->Render();

    worker->visual_update_requested = false;
}

void MainWindow::simulation_start_pause(bool checked)
{
    if(!worker->running && checked)
    {
        qDebug() << "starting simulation via GUI";
        statusLabel->setText("starting simulation");
        worker->Resume();
    }
    else if(worker->running && !checked)
    {
        qDebug() << "pausing simulation via GUI";
        statusLabel->setText("pausing simulation");
        worker->Pause();
        ui->actionStart_Pause->setEnabled(false);
    }
}

void MainWindow::background_worker_paused()
{
    ui->actionStart_Pause->blockSignals(true);
    ui->actionStart_Pause->setEnabled(true);
    ui->actionStart_Pause->setChecked(false);
    ui->actionStart_Pause->blockSignals(false);
    statusLabel->setText("simulation stopped");
}

void MainWindow::screenshot()
{
    if(model.prms.SimulationStep % model.prms.UpdateEveryNthStep) return;
    QString outputPath = QDir::currentPath()+ "/" + screenshot_directory.c_str() + "/" +
                         QString::number(model.prms.AnimationFrameNumber()).rightJustified(5, '0') + ".png";

    QDir pngDir(QDir::currentPath()+ "/"+ screenshot_directory.c_str());
    if(!pngDir.exists()) pngDir.mkdir(QDir::currentPath()+ "/"+ screenshot_directory.c_str());

    renderWindow->DoubleBufferOff();
    renderWindow->Render();
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    renderWindow->WaitForCompletion();

    windowToImageFilter->Update();
    windowToImageFilter->Modified();

    writerPNG->Modified();
    writerPNG->SetFileName(outputPath.toUtf8().constData());
    writerPNG->Write();
    renderWindow->DoubleBufferOn();
}


void MainWindow::LoadParameterFile(QString qFileName, QString resumeSnapshot)
{
    qDebug() << "MainWindow::LoadParameterFile " << qFileName << " ;resume file: " << resumeSnapshot;
    model.LoadParameterFile(qFileName.toStdString(), resumeSnapshot.toStdString());

    this->setWindowTitle(qFileName);

    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    updateGUI();
}


void MainWindow::spinbox_slowdown_value_changed(int val)
{
    model.intentionalSlowdown = val;
}



void MainWindow::sliderValueChanged(int val)
{
    LOGR("sliderValueChanged {}", val);
//    model.fluent_interpolatror.SetTime((double)val);
//    representation.SynchronizeValues();
//    renderWindow->Render();

    //model.fluent_interpolatror.LoadDataFrame(val);

 /*
    float b_val = (float)val/1000.;
    float max_val = 11.0*24*3600;
    float set_val = b_val*max_val;

    representation.wind_visualization_time = b_val*max_val;

    int64_t display_date = (int64_t)set_val + model.prms.SimulationStartUnixTime;

    std::time_t unix_time = display_date;
    std::tm* tm_time = std::gmtime(&unix_time);
    // Format the time
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S UTC", tm_time);
    representation.actorText->SetInput(buffer);
*/
}


void MainWindow::parameters_updated()
{
    qDebug() << "MainWindow::parameters_updated(); ptsize " << model.prms.ParticleViewSize;
    representation.SynchronizeTopology();
    renderWindow->Render();
}

void MainWindow::print_camera_params()
{
    // --- Basic Checks (Assuming renderer, camera, qt_vtk_widget are valid) ---
    vtkCamera* camera = renderer->GetActiveCamera(); // Assuming renderer is accessible member
    if (!camera) {
        qWarning() << "print_camera_params: Active camera is null.";
        return;
    }
    if (!qt_vtk_widget) {
        qWarning() << "print_camera_params: QVTKOpenGLNativeWidget is null.";
        return;
    }

    QSize viewWindowSize = qt_vtk_widget->size();
    int viewportWidthPixels = viewWindowSize.width();
    int viewportHeightPixels = viewWindowSize.height();

    qDebug() << "Viewport Size (Pixels): " << viewportWidthPixels << "x" << viewportHeightPixels;

    if (viewportWidthPixels <= 0 || viewportHeightPixels <= 0) {
        qWarning() << "print_camera_params: Invalid viewport dimensions.";
        return;
    }

    // --- Get Relevant Camera Parameters ---
    double parallelScale = camera->GetParallelScale(); // This is HALF the actual visible height in world units
    double camPos[3];
    camera->GetPosition(camPos);
    double viewCenterXWorld = camPos[0];
    double viewCenterYWorld = camPos[1];

    qDebug() << "Camera Parallel Scale: " << parallelScale;
    qDebug() << "Camera Position (View Center): (" << viewCenterXWorld << "," << viewCenterYWorld << ")";

    // --- Calculate ACTUAL Visible Dimensions based on Camera and Viewport ---
    double actualVisibleHeightWorld = 2.0 * parallelScale;
    double actualAspectRatio = static_cast<double>(viewportWidthPixels) / static_cast<double>(viewportHeightPixels);
    double actualVisibleWidthWorld = actualVisibleHeightWorld * actualAspectRatio;

    // --- Calculate Target 16:9 Dimensions, Matching the ACTUAL Visible Width ---
    const double targetAspectRatio = 16.0 / 9.0;

    // Use the actual visible width as the target width
    double targetVisibleWidthWorld = actualVisibleWidthWorld;

    // Calculate the corresponding height for a 16:9 ratio
    double targetVisibleHeightWorld = targetVisibleWidthWorld / targetAspectRatio;

    // --- Calculate Bottom-Left Offset for the TARGET 16:9 region ---
    // The offset is still centered around the camera's position (view center)
    // but uses the TARGET dimensions.
    double offsetXWorld = viewCenterXWorld - (targetVisibleWidthWorld / 2.0);
    double offsetYWorld = viewCenterYWorld - (targetVisibleHeightWorld / 2.0);

    // --- Output ---
    // Printing the offset and size that represent a 16:9 region,
    // centered like the current view, and scaled so its width matches
    // the width currently visible in the viewport.
    qDebug() << "--- Target 16:9 Raster Region (World Coordinates) ---";
    qDebug() << "  (Based on matching current view width)";
    qDebug() << "  Offset (Bottom-Left): (" << offsetXWorld << "," << offsetYWorld << ")";
    qDebug() << "  Size: (" << targetVisibleWidthWorld << "x" << targetVisibleHeightWorld << ")";
    // Optional: print ratio check
    // qDebug() << "  Calculated Ratio Check:" << (targetVisibleWidthWorld / targetVisibleHeightWorld);
}


void MainWindow::toggle_scalarbar(bool checked)
{
    representation.scalarBar->SetVisibility(checked);
    renderWindow->Render();
}
