#include "mainwindow.h"
#include "visual_representation.h"
#include "flowfieldgenerator.h"
#include "fluentflowimporter.h"
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
    setWindowTitle("Plate Preparer");
    setGeometry(100, 100, 1200, 800);

    // Setup settings file
    settingsFileName = QDir::currentPath() + "/preparer.ini";

    // Enable debug grid in preparer (for none mode visualization)
    representation.enableDebugGrid = true;

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
    renderer->AddActor(representation.actor_debug_grid);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.scalarBar);

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
        std::string landmaskPath = configDir + "/" + params.ImageLandMask;
        std::string colorPath = configDir + "/" + params.ImageColor;
        std::string icemaskPath = configDir + "/" + params.ImageIceMask;
        std::string crushedmaskPath = configDir + "/" + params.ImageCrushedMask;
        std::string projectDir = params.ProjectDirectory;

        // Unified grid and points preparation (loads images once, flips them, then processes)
        hsd.PrepareGridAndPoints(landmaskPath, colorPath, icemaskPath, crushedmaskPath,
                                 projectDir, params.DimensionHorizontal, params.PointsPerCell,
                                 params.ThicknessFrom, params.ThicknessTo);

        spdlog::info("Preparer: Grid and Points prepared successfully");

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
    // Convert slider value (0-1000) to actual time (0-1000 seconds)
    // Slider range is [0, 1000], directly representing seconds
    double time_t = static_cast<double>(value);

    // Update WACI with new time
    bool frames_changed = hsd.waci.SetTime(time_t);

    // Update visualization time in representation
    representation.wind_visualization_time = time_t;

    // Redraw with new flow field frame
    representation.SynchronizeTopology();
    renderWindow->Render();
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
