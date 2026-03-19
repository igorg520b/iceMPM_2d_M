// pp_mainwindow.cpp

#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QCloseEvent>
#include <QStringList>
#include <QCoreApplication>
#include <QMessageBox>
#include <QProgressDialog>
#include <QMenuBar>

#include <algorithm>
#include <filesystem>
#include <map>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "visualizer_mainwindow.h"
#include "visual_representation.h"
#include "frame_utils.h"


PPMainWindow::~PPMainWindow() {}

PPMainWindow::PPMainWindow(QWidget *parent)
    : QMainWindow(parent), representation(hsd)
{
    // Memory Optimizations:
    hsd.SetVisualizationMode(true);
    hsd.waci.SetEnabled(false);
    
    setWindowTitle("MPM Post-Processor");

    // Create central widget with scroll area for VTK rendering
    scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    setCentralWidget(scrollArea);

    // Setup VTK rendering widget
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);
    scrollArea->setWidget(qt_vtk_widget);

    // Configure VTK renderer
    renderer->SetBackground(1.0, 1.0, 1.0);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(interactor);

    // Add VTK actors to renderer
    renderer->AddActor(representation.textBgActor);
    renderer->AddActor(representation.scalarBarBgActor);
    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.actor_region_boundary);
    
    // Flow Vis Actors
    renderer->AddActor(representation.actor_vectors);
    renderer->AddActor(representation.actor_streamlines);

    renderer->AddActor(representation.scalarBar);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.actorTextTitle);


    // Create toolbar
    toolBar = addToolBar("Controls");

    // Visualization type selector
    comboBox_visualizations = new QComboBox();
    toolBar->addWidget(comboBox_visualizations);

    // Value range control
    qdsbValRange = new QDoubleSpinBox();
    qdsbValRange->setRange(-10, 10);
    qdsbValRange->setValue(-2);
    qdsbValRange->setDecimals(2);
    qdsbValRange->setSingleStep(0.25);
    toolBar->addWidget(qdsbValRange);

    // Transparency spinner
    toolBar->addWidget(new QLabel(" Transparency:"));
    qdsbTransparency = new QDoubleSpinBox(this);
    qdsbTransparency->setRange(0.0, 1000.0);
    qdsbTransparency->setSingleStep(0.1);
    qdsbTransparency->setValue(1.0);
    qdsbTransparency->setDecimals(1);
    toolBar->addWidget(qdsbTransparency);
    
    toolBar->addSeparator();

    // Frame range selection
    qsbFrameFrom = new QSpinBox();
    qsbFrameTo = new QSpinBox();
    toolBar->addWidget(qsbFrameFrom);
    toolBar->addWidget(qsbFrameTo);

    toolBar->addSeparator();
    qsbCurrentFrame = new QSpinBox();
    toolBar->addWidget(qsbCurrentFrame);
    btnSubmitFrame = new QPushButton("Submit");
    toolBar->addWidget(btnSubmitFrame);
    connect(btnSubmitFrame, &QPushButton::clicked, this, &PPMainWindow::submitFrame_triggered);

    // Frame slider
    slider2 = new QSlider(Qt::Horizontal);
    toolBar->addWidget(slider2);
    slider2->setTracking(false);
    slider2->setMinimum(1);
    slider2->setMaximum(10000);
    connect(slider2, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));

    // Populate visualization options
    QMetaEnum qme = QMetaEnum::fromType<VisualRepresentation::VisOpt>();
    for(int i = 0; i < qme.keyCount(); i++) {
        comboBox_visualizations->addItem(qme.key(i));
    }

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_visualizations(index); });

    // Create status bar to show current frame
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusBar->showMessage("Ready");

    // Setup camera
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();

    // Create render selector dialog (hidden by default)
    m_renderSelectorDialog = new RenderSelectorDialog(this);
    m_renderSelectorDialog->hide();

    // Load and restore user settings
    settingsFileName = QDir::currentPath() + "/ppcm.ini";
    QFileInfo fi(settingsFileName);

    if(fi.exists())
    {
        QSettings settings(settingsFileName, QSettings::IniFormat);
        QVariant var;

        // Restore camera position
        var = settings.value("camData");
        if(!var.isNull())
        {
            double *vec = (double*)var.toByteArray().constData();
            camera->SetClippingRange(1e-1, 1e4);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetPosition(vec[0], vec[1], vec[2]);
            camera->SetFocalPoint(vec[3], vec[4], vec[5]);
            camera->SetParallelScale(vec[6]);
            camera->ParallelProjectionOn();
            camera->Modified();
        }



        // Restore visualization option
        var = settings.value("vis_option");
        if(!var.isNull())
        {
            comboBox_visualizations->setCurrentIndex(var.toInt());
        }
        // Restore scroll tracking preference
        bool scrollTracking = settings.value("scroll_tracking", false).toBool();
        slider2->setTracking(scrollTracking);
    }
    else
    {
        cameraReset_triggered();
    }

    // Load render selector saved selections
    m_renderSelectorDialog->loadSelections(settingsFileName);
    int idx = comboBox_visualizations->currentIndex();
    qdsbValRange->setValue(representation.ranges[idx]);
    qdsbTransparency->setValue(representation.transparency_coeffs[idx]);

    // ========== MENUS ==========

    // File menu
    QMenu *fileMenu = menuBar()->addMenu("&File");

    QAction *openProjectAction = fileMenu->addAction("&Open Project...");
    openProjectAction->setShortcut(QKeySequence::Open);
    connect(openProjectAction, &QAction::triggered, this, &PPMainWindow::openProject_triggered);

    QAction *openFramesAction = fileMenu->addAction("&Open Frames...");
    openFramesAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_O));
    connect(openFramesAction, &QAction::triggered, this, &PPMainWindow::openFrames_triggered);

    fileMenu->addSeparator();

    QAction *exitAction = fileMenu->addAction("E&xit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    // Tools menu
    QMenu *toolsMenu = menuBar()->addMenu("&Tools");

    QAction *resetCameraAction = toolsMenu->addAction("&Reset Camera");
    resetCameraAction->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_R));
    connect(resetCameraAction, &QAction::triggered, this, &PPMainWindow::cameraReset_triggered);

    QAction *renderFrameAction = toolsMenu->addAction("&Render Frame");
    renderFrameAction->setShortcut(QKeySequence((int)Qt::CTRL | (int)Qt::Key_F));
    connect(renderFrameAction, &QAction::triggered, this, &PPMainWindow::render_frame_triggered);

    QAction *renderAllAction = toolsMenu->addAction("&Render All");
    renderAllAction->setShortcut(QKeySequence(Qt::Key_F5));
    connect(renderAllAction, &QAction::triggered, this, &PPMainWindow::render_all_triggered);

    // View menu
    QMenu *viewMenu = menuBar()->addMenu("&View");

    // Camera menu
    QMenu *cameraMenu = menuBar()->addMenu("&Camera");
    for(int i = 0; i < 5; ++i) {
        QAction *saveAction = cameraMenu->addAction(QString("Save Pos %1").arg(i+1));
        connect(saveAction, &QAction::triggered, this, [this, i](){
            captureCameraToSlot(i);
        });
    }
    cameraMenu->addSeparator();
    for(int i = 0; i < 5; ++i) {
        QAction *useAction = cameraMenu->addAction(QString("Use Slot %1").arg(i+1));
        useAction->setCheckable(true);
        connect(useAction, &QAction::toggled, this, [this, i](bool checked){
            m_useCameraSlot[i] = checked;
        });
    }

    cameraMenu->addSeparator();

    showTitleAction = cameraMenu->addAction("Show Title");
    showTitleAction->setCheckable(true);
    showTitleAction->setChecked(true); // Default is shown
    connect(showTitleAction, &QAction::toggled, this, [this](bool checked){
        representation.textBgActor->SetVisibility(checked);
        representation.actorText->SetVisibility(checked);
        representation.actorTextTitle->SetVisibility(checked);
        renderWindow->Render();
    });

    highResSnapshotAction = cameraMenu->addAction("Snapshot High-Res");
    highResSnapshotAction->setCheckable(true);
    highResSnapshotAction->setChecked(false); // Default is low-res
    connect(highResSnapshotAction, &QAction::toggled, this, [this](bool checked){
        m_highResSnapshot = checked;
    });

    // NOW we can apply the saved settings to the actions
    if(fi.exists()) {
        QSettings settings(settingsFileName, QSettings::IniFormat);
        bool showTitle = settings.value("show_title", true).toBool();
        showTitleAction->setChecked(showTitle);
        
        bool highRes = settings.value("high_res_snapshot", false).toBool();
        highResSnapshotAction->setChecked(highRes);
    }

    QAction *renderSelectorAction = viewMenu->addAction("&Render Selector");
    renderSelectorAction->setCheckable(true);
    renderSelectorAction->setChecked(false);
    connect(renderSelectorAction, &QAction::triggered, this, &PPMainWindow::toggleRenderSelector);

    QAction *loadFlowAction = viewMenu->addAction("Load Flow Data");
    loadFlowAction->setCheckable(true);
    loadFlowAction->setChecked(false);
    connect(loadFlowAction, &QAction::toggled, this, [&](bool checked){
        hsd.waci.SetEnabled(checked);
        // Force update of WACI state
        hsd.waci.SetTime(hsd.prms.SimulationTime);
        representation.SynchronizeTopology();
        renderWindow->Render();
    });



    // Connect value range changes
    connect(qdsbValRange, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &PPMainWindow::limits_changed);
    connect(qdsbTransparency, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &PPMainWindow::limits_changed);



    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &PPMainWindow::comboboxIndexChanged_visualizations);

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



    settings.setValue("vis_option", comboBox_visualizations->currentIndex());
    settings.setValue("scroll_tracking", slider2->hasTracking());
    
    if (showTitleAction) {
        settings.setValue("show_title", showTitleAction->isChecked());
    }
    if (highResSnapshotAction) {
        settings.setValue("high_res_snapshot", highResSnapshotAction->isChecked());
    }

    // Save render selector state
    m_renderSelectorDialog->saveSelections(settingsFileName);

    if (!currentProjectDirectory.empty()) {
        QString frameFile = QString::fromStdString(currentProjectDirectory) + "/last_frame.txt";
        QFile file(frameFile);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << slider2->value();
            file.close();
        }
    }

    event->accept();
}


void PPMainWindow::limits_changed(double val_)
{
    qDebug() << "limits_changed";
    int idx = (int)representation.VisualizingVariable;
    representation.ranges[idx] = qdsbValRange->value();
    representation.transparency_coeffs[idx] = qdsbTransparency->value();
    // Frame will be redrawn only after pressing 'submit' button
}

void PPMainWindow::submitFrame_triggered()
{
    int targetFrame = qsbCurrentFrame->value();
    if (targetFrame != slider2->value()) {
        slider2->setValue(targetFrame); // This triggers sliderValueChanged, loading and rendering
    } else {
        // Redraw frame manually, e.g. if the limits or representations changed
        this->representation.SynchronizeTopology();
        renderWindow->Render();
    }
}



void PPMainWindow::cameraReset_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);

    const double dx = hsd.prms.cellsize*hsd.prms.InitializationImageSizeX/2;
    const double dy = hsd.prms.cellsize*hsd.prms.InitializationImageSizeY/2;

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
    namespace fs = std::filesystem;

    // Extract the directory containing the JSON file
    fs::path jsonPath(fileName.toStdString());
    fs::path jsonFileDir = jsonPath.parent_path();
    if (jsonFileDir.empty()) {
        jsonFileDir = ".";
    }

    LOGR("\n=== Loading project configuration ===");
    LOGR("JSON file: {}", jsonPath.string());
    LOGR("Project directory: {}", jsonFileDir.string());

    // Remember this project directory for loading frames later
    currentProjectDirectory = jsonFileDir.string();

    // Read the JSON configuration file to find where the grid data is stored
    LOGR("Parsing JSON configuration...");
    std::map<std::string, std::string> parseResult = hsd.prms.ParseFile(jsonPath.string());
    LOGR("Project title: {}", parseResult["SimulationTitle"]);

    // Load the grid structure that was created by plate_preparer
    // This contains grid dimensions, cell size, and other geometric information
    LOGR("Loading grid structure...");
    fs::path gridPath = jsonFileDir / parseResult["GridData"];
    LOGR("Grid file: {}", gridPath.string());
    if (!fs::exists(gridPath)) {
        throw std::runtime_error(fmt::format("Grid file not found: {}", gridPath.string()));
    }
    hsd.LoadGridDataFromFile(gridPath.string());
    LOGR("Grid loaded successfully");

    // Load flow field data (for v_eta and v_norm visualization)
    // Load flow field data (for v_eta and v_norm visualization)


    // Load wind data if available (for wind visualization)
    if (parseResult.count("ERA5Data") && !parseResult["ERA5Data"].empty()) {
        fs::path windPath = jsonFileDir / parseResult["ERA5Data"];
        LOGR("Loading wind data: {}", windPath.string());
        if (fs::exists(windPath)) {
            hsd.waci.SetEra5Path(windPath.string());
            LOGR("Wind data loaded successfully");
        } else {
            LOGR("Warning: Wind data file not found: {}", windPath.string());
        }
    }

    // Load GLO12 data if available
    if (parseResult.count("GLO12Data") && !parseResult["GLO12Data"].empty()) {
        fs::path glo12Path = jsonFileDir / parseResult["GLO12Data"];
        LOGR("Loading GLO12 data: {}", glo12Path.string());
        if (fs::exists(glo12Path)) {
            hsd.waci.SetGLO12Path(glo12Path.string());
            LOGR("GLO12 data loaded successfully");
        } else {
            LOGR("Warning: GLO12 data file not found: {}", glo12Path.string());
        }
    }

    // Note: We only visualize grid data, not particle data, so we skip loading snapshots

    // Set up the output directory where simulation frames are saved
    // This is where we'll find f00000.h5, f00001.h5, etc.
    LOGR("Checking for simulation output directory...");
    fs::path outputDir = jsonFileDir / "output";
    fs::path framesDir = outputDir / "frames";
    if (fs::exists(framesDir)) {
        LOGR("Found frames directory: {}", framesDir.string());
    }
    hsd.output_directory = outputDir.string();

    LOGR("Project configuration loaded successfully\n");
}

void PPMainWindow::LoadFramesDirectory(QString framesDirectory)
{
    qDebug() << "PPMainWindow::LoadFramesDirectory: " << framesDirectory;
    currentFrameDirectory = framesDirectory.toStdString();

    int countFrames = frame_utils::ScanFrameDirectory(currentFrameDirectory);

    if(countFrames <= 0)
    {
        slider2->setEnabled(false);
        QMessageBox::warning(this, "No Frames Found", "The specified directory does not contain any valid frame files.");
        return;
    }

    // Configure slider
    slider2->setEnabled(true);
    slider2->setMaximum(countFrames);

    // Configure slider tracking based on grid size
    if (hsd.prms.GridXTotal > GRID_SIZE_TRACKING_THRESHOLD) {
        slider2->setTracking(false);
        LOGR("Grid width {} > {} pixels, slider tracking disabled for performance",
             hsd.prms.GridXTotal, GRID_SIZE_TRACKING_THRESHOLD);
    } else {
        slider2->setTracking(true);
        LOGR("Grid width {} <= {} pixels, slider tracking enabled",
             hsd.prms.GridXTotal, GRID_SIZE_TRACKING_THRESHOLD);
    }

    // Setup frame range controls
    qsbFrameTo->setMaximum(countFrames);
    qsbFrameTo->setValue(countFrames);

    qsbFrameFrom->setMaximum(countFrames);
    qsbFrameFrom->setValue(1);

    qsbCurrentFrame->setMaximum(countFrames);
    qsbCurrentFrame->setValue(countFrames);

    int initialFrame = countFrames;
    if (!currentProjectDirectory.empty()) {
        QString frameFile = QString::fromStdString(currentProjectDirectory) + "/last_frame.txt";
        QFile file(frameFile);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&file);
            int savedFrame = 0;
            in >> savedFrame;
            if (savedFrame >= 1 && savedFrame <= countFrames) {
                initialFrame = savedFrame;
            }
            file.close();
        }
    }

    // Load and display the initial frame as a starting point
    slider2->setValue(initialFrame);

    statusBar->showMessage(QString("Loaded %1 frames").arg(countFrames));
}

void PPMainWindow::sliderValueChanged(int val)
{
    qsbCurrentFrame->blockSignals(true);
    qsbCurrentFrame->setValue(val);
    qsbCurrentFrame->blockSignals(false);
    
    try {
        // Load the frame data from disk
        // Load the frame data from disk
        // std::string framePath = frame_utils::GetFramePath(currentFrameDirectory, val);
        hsd.LoadFrameData(val, currentFrameDirectory);

        representation.simulationTime = hsd.prms.SimulationTime;
        qDebug() << "time set to " << hsd.prms.SimulationTime;

        // Update WACI time to match current frame (for v_eta and v_norm visualization)
        hsd.waci.SetTime(hsd.prms.SimulationTime);

        // Update the visualization with new frame data
        this->representation.SynchronizeTopology();


        // Update status bar with current frame number
        statusBar->showMessage(QString("Frame %1 | Time: %2").arg(val).arg(hsd.prms.SimulationTime, 0, 'f', 2));

        // Render the updated frame
        renderWindow->Render();
    } catch (const std::exception& e) {
        LOGR("Failed to load and display frame {}: {}", val, e.what());
    }
}


void PPMainWindow::comboboxIndexChanged_visualizations(int index)
{
    // Lazy Allocation Logic
    VisualRepresentation::VisOpt opt = (VisualRepresentation::VisOpt)index;

    // Auto-enable flow interpolation if a flow visualization is selected
    if (opt == VisualRepresentation::VisOpt::v_ocean_norm || 
        opt == VisualRepresentation::VisOpt::v_wind_norm ||
        opt == VisualRepresentation::VisOpt::ocean_streamlines || 
        opt == VisualRepresentation::VisOpt::wind_streamlines) 
    {
        if (!hsd.waci.interpolation_enabled) {
            hsd.waci.SetEnabled(true);
            hsd.waci.SetTime(hsd.prms.SimulationTime);
        }
    }
    
    // Safety check: Don't attempt allocation if grid dimensions are not loaded yet
    if (hsd.prms.GridXTotal == 0 || hsd.prms.GridYTotal == 0) {
        representation.ChangeVisualizationOption(index);
        return;
    }

    std::vector<int> requiredArrays = representation.GetRequiredGridArrays(opt);

    bool needsReload = false;
    for(int arrayIdx : requiredArrays) {
        if(!hsd.IsGridArrayAllocated(arrayIdx)) {
            hsd.AllocateGridArray(arrayIdx);
            needsReload = true;
        }
    }

    if(needsReload) {
        int currentFrame = slider2->value(); // Assuming slider tracks current frame
        
        statusBar->showMessage("Allocating and reloading data...");
        QCoreApplication::processEvents();
        try {
             hsd.LoadFrameData(currentFrame, currentFrameDirectory);
             // Ensure WACI time is correct if we reloaded
             hsd.waci.SetTime(hsd.prms.SimulationTime);
        } catch (const std::exception& e) {
            LOGR("Lazy load failed: {}", e.what());
        }
        statusBar->showMessage("Ready");
    }

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
    statusBar->showMessage("Rendering frame...");
    QCoreApplication::processEvents();

    std::string visName = QMetaEnum::fromType<VisualRepresentation::VisOpt>().valueToKey(representation.VisualizingVariable);
    const int selectedFrame = slider2->value();
    std::string filename = visName + "_" + std::to_string(selectedFrame) + ".jpg";

    const QSizePolicy originalPolicy = qt_vtk_widget->sizePolicy();

    scrollArea->setWidgetResizable(false);
    
    if (m_highResSnapshot) {
        qt_vtk_widget->setFixedSize(19200, 10800);
    } else {
        qt_vtk_widget->setFixedSize(1920, 1080);
    }
    
    QCoreApplication::processEvents();

    windowToImageFilter->SetInput(renderWindow);
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());

    renderWindow->DoubleBufferOff();
    renderWindow->Render();
    windowToImageFilter->SetInputBufferTypeToRGB();
    renderWindow->WaitForCompletion();

    windowToImageFilter->Modified();
    writer->SetFileName(filename.c_str());
    writer->Write();

    renderWindow->DoubleBufferOn();

    qt_vtk_widget->setMinimumSize(0, 0);
    qt_vtk_widget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    qt_vtk_widget->setSizePolicy(originalPolicy);
    scrollArea->setWidgetResizable(true);
    QCoreApplication::processEvents();

    statusBar->showMessage(QString("Rendered frame %1 as %2").arg(selectedFrame).arg(QString::fromStdString(visName)), 3000);
}



void PPMainWindow::captureCameraToSlot(int index)
{
    if (index < 0 || index >= 5) return;
    vtkCamera* cam = renderer->GetActiveCamera();
    cam->GetPosition(&m_cameraSlots[index][0]);
    cam->GetFocalPoint(&m_cameraSlots[index][3]);
    m_cameraSlots[index][6] = cam->GetParallelScale();
    statusBar->showMessage(QString("Camera position saved to slot %1").arg(index + 1), 2000);
}

void PPMainWindow::render_all_triggered()
{
    qDebug() << "render_all_triggered() started.";

    // Get the frame range from the UI controls
    const int frameFrom = qsbFrameFrom->value();
    const int frameTo = qsbFrameTo->value();

    // Get selected visualization options from dialog
    std::vector<VisualRepresentation::VisOpt> visOptsToRender =
        m_renderSelectorDialog->getSelectedOptions();

    // Check if any flow visualization is selected
    bool needFlow = false;
    for (const auto& opt : visOptsToRender) {
        if (opt == VisualRepresentation::VisOpt::v_ocean_norm || 
            opt == VisualRepresentation::VisOpt::v_wind_norm ||
            opt == VisualRepresentation::VisOpt::ocean_streamlines || 
            opt == VisualRepresentation::VisOpt::wind_streamlines) 
        {
            needFlow = true;
            break;
        }
    }
    
    if (needFlow && !hsd.waci.interpolation_enabled) {
        hsd.waci.SetEnabled(true);
    }

    // Validate that at least one option is selected
    if (visOptsToRender.empty()) {
        QMessageBox::warning(this, "No Visualizations Selected",
            "Please select at least one visualization option in the Render Selector (View → Render Selector).");
        return;
    }

    // Pre-allocate arrays for all selected visualizations
    // This ensures that LoadFrameData will load the necessary data
    for (const auto& visOpt : visOptsToRender) {
        std::vector<int> requiredArrays = representation.GetRequiredGridArrays(visOpt);
        for(int arrayIdx : requiredArrays) {
            hsd.AllocateGridArray(arrayIdx);
        }
    }

    // Identify active camera slots (including default)
    struct RenderTarget {
        std::string dirName;
        int slotIndex; // -1 for default
    };
    std::vector<RenderTarget> targets;
    targets.push_back({"raster", -1}); // Default
    for(int i = 0; i < 5; ++i) {
        if(m_useCameraSlot[i]) {
            targets.push_back({"raster_" + std::to_string(i+1), i});
        }
    }

    // Validate that the frame range is valid
    if (frameTo < frameFrom) {
        QMessageBox::warning(this, "Invalid Range", "Frame 'To' must be greater than or equal to Frame 'From'.");
        return;
    }
    const int totalFrames = (frameTo - frameFrom) + 1;

    statusBar->showMessage(QString("Rendering %1 frames...").arg(totalFrames));

    // Create a progress dialog to show rendering status
    const int totalOperations = totalFrames * visOptsToRender.size() * targets.size();
    QProgressDialog progress("Rendering all frames...", "Abort", 0, totalOperations, this);
    progress.setWindowModality(Qt::WindowModal);
    QCoreApplication::processEvents();

    // Set up the rendering window to a fixed size for consistent output
    const QSizePolicy originalPolicy = qt_vtk_widget->sizePolicy();
    scrollArea->setWidgetResizable(false);
    qt_vtk_widget->setFixedSize(1920, 1080);
    QCoreApplication::processEvents();

    // Configure VTK rendering and image capture
    renderWindow->DoubleBufferOff();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetInputBufferTypeToRGB();
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());

    // Save initial camera state to restore later
    vtkCamera* cam = renderer->GetActiveCamera();
    std::array<double, 7> initialCamState;
    cam->GetPosition(&initialCamState[0]);
    cam->GetFocalPoint(&initialCamState[3]);
    initialCamState[6] = cam->GetParallelScale();

    // Main loop: process each frame
    int operationCount = 0;
    for (int frameNum = frameFrom; frameNum <= frameTo; ++frameNum) {
        if (progress.wasCanceled()) break;

        progress.setLabelText(QString("Loading frame %1...").arg(frameNum));
        QCoreApplication::processEvents();

        // Load the frame data from HDF5 file (ONCE per frame)
        try {
            hsd.LoadFrameData(frameNum, currentFrameDirectory);
            representation.simulationTime = hsd.prms.SimulationTime;
            // Update WACI time to match current frame
            hsd.waci.SetTime(hsd.prms.SimulationTime);
        } catch (const std::exception& e) {
            LOGR("Failed to load frame {}: {}", frameNum, e.what());
            operationCount += visOptsToRender.size() * targets.size();
            progress.setValue(operationCount);
            continue;
        }

        // Render each visualization option for this frame
        for (const auto& visOpt : visOptsToRender) {
            if (progress.wasCanceled()) break;

            const QMetaEnum metaEnum = QMetaEnum::fromType<VisualRepresentation::VisOpt>();
            const QString visName = metaEnum.valueToKey(visOpt);

            // Update visualization (ONCE per VisOpt per frame)
            representation.ChangeVisualizationOption(visOpt);

            // Render for each camera position
            for (const auto& target : targets) {
                 if (progress.wasCanceled()) break;
                 operationCount++;

                 progress.setValue(operationCount);
                 progress.setLabelText(QString("Frame %1/%2 : %3 [%4]")
                                           .arg(frameNum - frameFrom + 1)
                                           .arg(totalFrames)
                                           .arg(visName)
                                           .arg(QString::fromStdString(target.dirName)));
                 
                 // Apply camera state
                 if (target.slotIndex == -1) {
                     // Default: use initial state (or should we track the "live" camera? 
                     // The requirement says "The current version works well... current camera state is used".
                     // So we should restore the state the user had before clicking Render.
                     cam->SetPosition(initialCamState[0], initialCamState[1], initialCamState[2]);
                     cam->SetFocalPoint(initialCamState[3], initialCamState[4], initialCamState[5]);
                     cam->SetParallelScale(initialCamState[6]);
                 } else {
                     // Apply slot state
                     const auto& slot = m_cameraSlots[target.slotIndex];
                     cam->SetPosition(slot[0], slot[1], slot[2]);
                     cam->SetFocalPoint(slot[3], slot[4], slot[5]);
                     cam->SetParallelScale(slot[6]);
                 }
                 cam->Modified();
                 
                 // Render
                 renderWindow->Render();
                 windowToImageFilter->Modified();

                 // Create output directory structure and save rendered image
                 QDir frameDir(QString::fromStdString(currentFrameDirectory));
                 frameDir.cdUp(); // Navigate to parent directory (output/)
                 const QString rasterBasePath = frameDir.filePath(QString::fromStdString(target.dirName));
                 const QString subDir = QString("%1/%2").arg(rasterBasePath).arg(visName);
                 QDir().mkpath(subDir); // Create full directory path
                 const QString outputPath = QString("%1/%2.jpg").arg(subDir).arg(frameNum, 5, 10, QChar('0'));

                 // Capture and write the image to disk
                 writer->SetFileName(outputPath.toStdString().c_str());
                 writer->Write();
            }
        }
    }

    // Restore the GUI to its original state
    cam->SetPosition(initialCamState[0], initialCamState[1], initialCamState[2]);
    cam->SetFocalPoint(initialCamState[3], initialCamState[4], initialCamState[5]);
    cam->SetParallelScale(initialCamState[6]);
    cam->Modified();

    qt_vtk_widget->setMinimumSize(0, 0);
    qt_vtk_widget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    qt_vtk_widget->setSizePolicy(originalPolicy);
    scrollArea->setWidgetResizable(true);
    renderWindow->DoubleBufferOn();
    QCoreApplication::processEvents();

    // Generate scripts for all populated targets
    for(const auto& target : targets) {
        generate_ffmpeg_script(frameFrom, frameTo, target.dirName);
    }

    // Refresh the display to show the last rendered frame
    renderWindow->Render();

    // Show completion message
    progress.setValue(totalOperations);
    QMessageBox::information(this, "Rendering Complete", "Batch rendering has finished or was canceled.");
}



// --- Core Logic Function (Rewritten) ---
// This is the clean, reusable implementation.
void PPMainWindow::generate_ffmpeg_script(int frameFrom, int frameTo, std::string dirName)
{
    qDebug() << "Generating ffmpeg script for frames" << frameFrom << "to" << frameTo << "in" << QString::fromStdString(dirName);

    const int totalFrames = (frameTo - frameFrom) + 1;
    if (totalFrames <= 0) return;

    // Correctly determine the output "raster" directory path.
    QDir frameDir(QString::fromStdString(currentFrameDirectory));
    frameDir.cdUp(); // Go from ".../output/frames" to ".../output/"
    const QString rasterPath = frameDir.filePath(QString::fromStdString(dirName));
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
    scriptFile << "# It is designed to be run from within the '" << dirName << "' directory.\n";
    scriptFile << "cd \"$(dirname \"$0\")\"\n\n";

    // Define variables
    scriptFile << "FPS=30\n";
    scriptFile << "START_FRAME=" << frameFrom << "\n";
    scriptFile << "LAST_FRAME=" << frameTo << "\n";
    scriptFile << "NUM_FRAMES=$((LAST_FRAME - START_FRAME + 1))\n\n";

    // Define the ffmpeg command template.
    // We use shell variables $FPS, $START_FRAME, $NUM_FRAMES directly.
    // Arguments to fmt: {0} = input pattern, {1} = output filename
    const std::string fmtStr = R"(ffmpeg -y -r $FPS -f image2 -start_number $START_FRAME -i "{0:}" -vframes $NUM_FRAMES -vcodec libx264 -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1:white" -crf 21 -pix_fmt yuv420p "{1:}")";
    const QMetaEnum metaEnum = QMetaEnum::fromType<VisualRepresentation::VisOpt>();

    // Get selected visualization options from dialog
    std::vector<VisualRepresentation::VisOpt> visOptsToRender =
        m_renderSelectorDialog->getSelectedOptions();

    // Loop through the selected visualizations to generate a command for each
    for (const auto& visOpt : visOptsToRender) {
        const std::string visName = metaEnum.valueToKey(visOpt);
        const std::string inputFilePattern = visName + "/%05d.jpg";
        const std::string outputVideoFile = visName + ".mp4";

        std::string command = fmt::format(fmt::runtime(fmtStr),
                                          inputFilePattern,
                                          outputVideoFile);

        scriptFile << "# Generate video for " << visName << "\n";
        scriptFile << command << "\n\n";
    }

    scriptFile.close();
    // Make the script executable.
    int result = std::system(("chmod +x " + scriptFilename).c_str());
    (void)result;
}


void PPMainWindow::toggleScrollTracking(bool checked)
{
    qDebug() << "tracking toggled: " << checked;
    slider2->setTracking(checked);
}


void PPMainWindow::toggleRenderSelector()
{
    if (m_renderSelectorDialog->isVisible()) {
        m_renderSelectorDialog->hide();
    } else {
        m_renderSelectorDialog->show();
        m_renderSelectorDialog->raise();
        m_renderSelectorDialog->activateWindow();
    }
}


void PPMainWindow::TryLoadDefaultFrames()
{
    if(currentProjectDirectory.empty())
    {
        LOGR("No project directory set, cannot auto-load frames");
        return;
    }

    // Try to load frames from default location: [project_dir]/output/frames
    std::string defaultFramesPath = currentProjectDirectory + "/output/frames";

    if(!std::filesystem::exists(defaultFramesPath))
    {
        LOGR("Default frames directory does not exist: {}", defaultFramesPath);
        return;
    }

    LOGR("Found default frames directory, auto-loading from: {}", defaultFramesPath);
    LoadFramesDirectory(QString::fromStdString(defaultFramesPath));
}


void PPMainWindow::openProject_triggered()
{
    qDebug() << "openProject_triggered()";

    QString fileName = QFileDialog::getOpenFileName(this, "Open Project Configuration", "",
                                                    "JSON Files (*.json);;All Files (*)");
    if(!fileName.isEmpty())
    {
        LoadParametersFile(fileName);
        // After loading project, try to auto-load frames from default location
        TryLoadDefaultFrames();
    }
}


void PPMainWindow::openFrames_triggered()
{
    qDebug() << "openFrames_triggered()";

    if(currentProjectDirectory.empty())
    {
        QMessageBox::warning(this, "No Project Loaded",
                            "Please open a project first before selecting frames directory.");
        return;
    }

    QString framesDir = QFileDialog::getExistingDirectory(this, "Select Frames Directory",
                                                         QString::fromStdString(currentProjectDirectory));
    if(!framesDir.isEmpty())
    {
        LoadFramesDirectory(framesDir);
    }
}

void PPMainWindow::toggleContours(bool checked)
{
    qDebug() << "toggleContours: " << checked;
    // representation.ShowEtaContours = checked;
    representation.SynchronizeTopology(); // Will update visibility and data
    renderWindow->Render();
}

void PPMainWindow::contourIntervalChanged(double val)
{
    // representation.ContourInterval = val;
    representation.SynchronizeTopology(); // Trigger update if contours are shown
    renderWindow->Render();
}
