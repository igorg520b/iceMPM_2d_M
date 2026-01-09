#include "mainwindow.h"
#include "visual_representation.h"
#include "flowfieldgenerator.h"
#include "fluentflowimporter.h"
#include "simulation/parameters_sim.h"
#include <QFileDialog>
#include <QMetaEnum>
#include <QVBoxLayout>
#include <QToolBar>
#include <QLabel>
#include <QMenu>
#include <QAction>
#include <QSettings>
#include <QDir>
#include <QFileInfo>
#include <spdlog/spdlog.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), representation(hsd)
{
    setWindowTitle("Preparer");
    setGeometry(100, 100, 1200, 800);

    // Setup settings file
    settingsFileName = QDir::currentPath() + "/preparer.ini";

    // Setup UI programmatically
    setupUI();
    createMenuBar();

    // Load saved settings (visualization mode and camera state)
    loadSettings();
    loadCameraState();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    // Create central VTK widget
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);
    setCentralWidget(qt_vtk_widget);

    // Configure renderer
    renderer->SetBackground(0.9, 0.9, 0.85);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(interactorStyle);

    // Create toolbar with visualization controls
    QToolBar *toolBar = addToolBar("Visualization");

    // Combobox for visualization options
    comboBox_visualizations = new QComboBox();
    toolBar->addWidget(comboBox_visualizations);

    // Range label and spinbox
    QLabel *lbl1 = new QLabel("range:");
    toolBar->addWidget(lbl1);

    qdsbValRange = new QDoubleSpinBox();
    qdsbValRange->setRange(-10, 10);
    qdsbValRange->setValue(-2);
    qdsbValRange->setDecimals(2);
    qdsbValRange->setSingleStep(0.25);
    toolBar->addWidget(qdsbValRange);

    // Transparency label and spinbox
    QLabel *lbl2 = new QLabel("tr:");
    toolBar->addWidget(lbl2);

    qdsbTransparency = new QDoubleSpinBox();
    qdsbTransparency->setRange(0, 1);
    qdsbTransparency->setValue(0);
    qdsbTransparency->setDecimals(1);
    qdsbTransparency->setSingleStep(0.1);
    toolBar->addWidget(qdsbTransparency);

    // Flow time slider
    QLabel *lbl3 = new QLabel("flow time:");
    toolBar->addWidget(lbl3);

    flowTimeSlider = new QSlider(Qt::Horizontal);
    flowTimeSlider->setRange(0, 1000);  // 0 to 1000 subdivisions representing 0 to 1000 seconds
    flowTimeSlider->setValue(0);
    flowTimeSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    toolBar->addWidget(flowTimeSlider);

    // Add representation actors to renderer
    renderer->AddActor(representation.textBgActor);
    renderer->AddActor(representation.scalarBarBgActor);
    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.actor_region_boundary);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.actorTextTitle);
    renderer->AddActor(representation.scalarBar);

    // Create status bar with point count (left side) and time display (right side)
    statusLabel = new QLabel("Points: 0");
    statusBar()->addWidget(statusLabel);

    // Add permanent widget on the right side of status bar for time display
    timeLabel = new QLabel("Time: 0.0 s");
    timeLabel->setFixedWidth(150);
    timeLabel->setAlignment(Qt::AlignRight);
    statusBar()->addPermanentWidget(timeLabel);

    // Populate visualization options combobox
    QMetaEnum qme = QMetaEnum::fromType<VisualRepresentation::VisOpt>();
    for (int i = 0; i < qme.keyCount(); i++) {
        comboBox_visualizations->addItem(qme.key(i));
    }

    // Connect signals
    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::comboboxIndexChanged_visualizations);

    connect(qdsbValRange, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::limits_changed);

    connect(qdsbTransparency, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::limits_changed);

    connect(flowTimeSlider, QOverload<int>::of(&QSlider::valueChanged),
            this, &MainWindow::flowTimeSliderChanged);
}

void MainWindow::createMenuBar()
{
    // File menu
    QMenu *fileMenu = menuBar()->addMenu("&File");

    QAction *openJsonAction = fileMenu->addAction("&Open JSON");
    openJsonAction->setShortcut(QKeySequence::Open);
    connect(openJsonAction, &QAction::triggered, this, &MainWindow::openJsonFile_triggered);

    fileMenu->addSeparator();

    QAction *exitAction = fileMenu->addAction("E&xit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    // View menu
    QMenu *viewMenu = menuBar()->addMenu("&View");

    QAction *resetCameraAction = viewMenu->addAction("&Reset Camera");
    resetCameraAction->setShortcut(Qt::Key_R);
    connect(resetCameraAction, &QAction::triggered, this, &MainWindow::resetCamera_triggered);
}

void MainWindow::openJsonFile_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open JSON Configuration", "",
                                                    "JSON Files (*.json);;All Files (*)");
    if (!fileName.isEmpty()) {
        LoadParameterFile(fileName);
    }
}

void MainWindow::resetCamera_triggered()
{
    spdlog::debug("MainWindow::resetCamera_triggered()");
    vtkCamera *camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1, 1e3);
    camera->SetFocalPoint(0, 0., 0.);
    camera->SetPosition(0.0, 0.0, 50.0);
    camera->SetViewUp(0.0, 1.0, 0.0);

    camera->Modified();
    renderWindow->Render();
}

void MainWindow::LoadParameterFile(QString fileName)
{
    params.LoadParamsFile(fileName.toStdString());

    try {
        // Construct file paths from config file directory (images are in same dir as JSON)
        std::string configDir = params.ConfigFileDirectory;
        // Land mask is optional - only construct path if provided
        std::string landmaskPath = params.ImageLandMask.empty() ? "" : (configDir + "/" + params.ImageLandMask);
        std::string colorPath = configDir + "/" + params.ImageColor;
        std::string icemaskPath = configDir + "/" + params.ImageIceMask;
        // Crushed mask is optional - only construct path if provided
        std::string crushedmaskPath = params.ImageCrushedMask.empty() ? "" : (configDir + "/" + params.ImageCrushedMask);
        // Cracked mask is optional - only construct path if provided
        std::string crackedmaskPath = params.ImageCrackedMask.empty() ? "" : (configDir + "/" + params.ImageCrackedMask);
        // Thickness mask is optional - only construct path if provided
        std::string thicknessmaskPath = params.ImageThicknessMask.empty() ? "" : (configDir + "/" + params.ImageThicknessMask);
        std::string projectDir = params.ProjectDirectory;

        // Unified grid and points preparation (loads images once, flips them, then processes)
        hsd.PrepareGridAndPoints(landmaskPath, colorPath, icemaskPath, crushedmaskPath, crackedmaskPath,
                                 projectDir, params.DimensionHorizontal, params.PointsPerCell,
                                 params.ThicknessFrom, params.ThicknessTo, thicknessmaskPath);

        spdlog::info("Preparer: Grid and Points prepared successfully");

        // Update status bar with point count (with thousands separator)
        unsigned numPoints = hsd.hssoa.size;
        QLocale locale = QLocale::English;
        QString pointCountStr = locale.toString((int)numPoints);
        statusLabel->setText(QString("Points: %1").arg(pointCountStr));

        // --- Try loading simulation.json for Projection Params & ERA5 Data ---
        std::filesystem::path configPath(fileName.toStdString());
        std::filesystem::path simConfigDir = configPath.parent_path();
        std::filesystem::path simJsonPath = simConfigDir / "simulation.json";

        if (std::filesystem::exists(simJsonPath)) {
            spdlog::info("Found simulation.json, parsing for projection parameters & ERA5...");
            // We use hsd.prms.ParseFile which returns map of paths
            std::map<std::string, std::string> simParseResult = hsd.prms.ParseFile(simJsonPath.string());
            
            // If simulation.json had ERA5Data, initialize WACI with it
            // NOTE: This might enable UseWindData, but we will override it below based on prepare.json
            if (simParseResult.count("ERA5Data")) {
                 std::string era5File = simParseResult["ERA5Data"];
                 // Check if path is absolute or relative
                 std::filesystem::path era5Path(era5File);
                 if (era5Path.is_relative()) {
                     era5Path = simConfigDir / era5Path;
                 }

                 if (std::filesystem::exists(era5Path)) {
                     hsd.waci.SetEra5Path(era5Path.string());
                     spdlog::info("ERA5 Wind Data loaded from: {}", era5Path.string());
                 } else {
                     spdlog::warn("Warning: ERA5Data path in simulation.json not found: {}", era5Path.string());
                 }
            }
        } else {
            spdlog::info("simulation.json not found in project directory - wind visualization may not work correctly.");
        }

        // --- Initialize Wind Data (from prepare.json) overrides selection ---
        // If WindData is explicitly provided in prepare.json, use it.
        // If NOT provided, DISABLE wind, even if simulation.json populated it.
        if (!params.WindData.empty()) {
             std::string windPath = params.ConfigFileDirectory + "/" + params.WindData;
             hsd.prms.UseWindData = true;
             hsd.waci.SetEra5Path(windPath);
             spdlog::info("Preparer: Loaded ERA5 Wind Data from {}", windPath);
        } else {
            hsd.prms.UseWindData = false;
        }

        // Generate flow field if specified in JSON
        if (!params.FlowType.empty()) {
            FlowFieldGenerator flowGen;
            flowGen.GenerateFlow(params,
                                hsd.prms.GridXTotal, hsd.prms.GridYTotal, hsd.prms.cellsize,
                                hsd.prms.ModeledRegionOffsetX, hsd.prms.ModeledRegionOffsetY,
                                hsd.prms.InitializationImageSizeX, hsd.prms.InitializationImageSizeY,
                                projectDir);
            // Set HDF5 path in WACI
            hsd.waci.SetHDF5Path(projectDir + "/grid_flow.h5");
            // Initialize WACI with first frame at time t=0
            hsd.waci.SetTime(0.0);
            spdlog::info("Preparer: Flow field generated successfully and initialized to t=0");
        }

        // Update visualization
        representation.SynchronizeTopology();
        renderWindow->Render();
    } catch (const std::exception &e) {
        spdlog::error("Preparer error: {}", e.what());
    }
}

void MainWindow::comboboxIndexChanged_visualizations(int index)
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

void MainWindow::limits_changed(double val)
{
    int idx = (int)representation.VisualizingVariable;
    representation.ranges[idx] = qdsbValRange->value();
    representation.transparency_coeffs[idx] = qdsbTransparency->value();
    representation.SynchronizeTopology();
    renderWindow->Render();
}

void MainWindow::flowTimeSliderChanged(int value)
{
    // Convert slider value (0-1000) to actual time (0-TimeScale seconds)
    // TimeScale parameter controls the time range (default=1000, so slider maps to 0-1000 seconds)
    double time_t = (static_cast<double>(value) / 1000.0) * params.TimeScale;

    LOGR("MainWindow::flowTimeSliderChanged: slider_value={}, time_t={}, TimeScale={}", value, time_t, params.TimeScale);

    // Update WACI with new time
    auto [ocean_changed, wind_changed] = hsd.waci.SetTime(time_t);
    LOGR("MainWindow::flowTimeSliderChanged: ocean_changed={}; wind_changed", ocean_changed, wind_changed);

    // Update visualization time in representation


    // Update time display in status bar
    timeLabel->setText(QString("Time: %1 s").arg(time_t, 0, 'f', 1));

    // Redraw with new flow field frame
    LOGR("MainWindow::flowTimeSliderChanged: Calling SynchronizeTopology and Render");
    representation.simulationTime = time_t;
    representation.SynchronizeTopology();
    renderWindow->Render();
    LOGR("MainWindow::flowTimeSliderChanged: Done");
}

void MainWindow::loadSettings()
{
    QFileInfo fi(settingsFileName);
    if (!fi.exists()) {
        return;  // File doesn't exist yet, use defaults
    }

    QSettings settings(settingsFileName, QSettings::IniFormat);

    QVariant vis_option = settings.value("vis_option");
    if (!vis_option.isNull()) {
        int option = vis_option.toInt();
        comboBox_visualizations->blockSignals(true);
        comboBox_visualizations->setCurrentIndex(option);
        comboBox_visualizations->blockSignals(false);

        qdsbValRange->blockSignals(true);
        qdsbValRange->setValue(representation.ranges[option]);
        qdsbValRange->blockSignals(false);

        qdsbTransparency->blockSignals(true);
        qdsbTransparency->setValue(representation.transparency_coeffs[option]);
        qdsbTransparency->blockSignals(false);

        representation.ChangeVisualizationOption(option);
        renderWindow->Render();
    }
}

void MainWindow::saveSettings()
{
    QSettings settings(settingsFileName, QSettings::IniFormat);
    settings.setValue("vis_option", comboBox_visualizations->currentIndex());
}

void MainWindow::loadCameraState()
{
    QFileInfo fi(settingsFileName);
    if (!fi.exists()) {
        return;  // File doesn't exist yet, use default camera
    }

    QSettings settings(settingsFileName, QSettings::IniFormat);

    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();

    QVariant var = settings.value("camData");
    if (!var.isNull()) {
        double vec[7];
        const double* data = (const double*)var.toByteArray().constData();

        // Copy data safely
        for (int i = 0; i < 7; i++) {
            vec[i] = data[i];
        }

        camera->SetClippingRange(1e-1, 1e4);
        camera->SetViewUp(0.0, 1.0, 0.0);
        camera->SetPosition(vec[0], vec[1], vec[2]);
        camera->SetFocalPoint(vec[3], vec[4], vec[5]);
        camera->SetParallelScale(vec[6]);
        camera->Modified();

        spdlog::debug("Camera state loaded: pos ({}, {}, {}), focal ({}, {}, {}), scale {}",
                      vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6]);
    }
}

void MainWindow::saveCameraState()
{
    QSettings settings(settingsFileName, QSettings::IniFormat);

    double data[7];
    renderer->GetActiveCamera()->GetPosition(&data[0]);
    renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
    data[6] = renderer->GetActiveCamera()->GetParallelScale();

    QByteArray arr((char*)data, sizeof(data));
    settings.setValue("camData", arr);

    spdlog::debug("Camera state saved: pos ({}, {}, {}), focal ({}, {}, {}), scale {}",
                  data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    saveSettings();
    saveCameraState();
    QMainWindow::closeEvent(event);
}
