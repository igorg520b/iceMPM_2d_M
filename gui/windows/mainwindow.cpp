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
    , representation(model.sim_data)
{
    ui->setupUi(this);

    params = new ParamsWrapper(&model.prms);
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

    QLabel *lbl1 = new QLabel("range:");
    ui->toolBar->addWidget(lbl1);

    // double spin box
    qdsbValRange = new QDoubleSpinBox();
    qdsbValRange->setRange(-10, 10);
    qdsbValRange->setValue(-2);
    qdsbValRange->setDecimals(2);
    qdsbValRange->setSingleStep(0.25);
    ui->toolBar->addWidget(qdsbValRange);

    QLabel *lbl2 = new QLabel("tr:");
    ui->toolBar->addWidget(lbl2);

    qdsbTransparency = new QDoubleSpinBox();
    qdsbTransparency->setRange(0, 1000);
    qdsbTransparency->setValue(0);
    qdsbTransparency->setDecimals(1);
    qdsbTransparency->setSingleStep(0.1);
    ui->toolBar->addWidget(qdsbTransparency);

    QLabel *lbl3 = new QLabel("sldn:");
    ui->toolBar->addWidget(lbl3);

    qsbIntentionalSlowdown = new QSpinBox();
    qsbIntentionalSlowdown->setRange(0,1000);
    qsbIntentionalSlowdown->setValue(0);
    ui->toolBar->addWidget(qsbIntentionalSlowdown);

    // statusbar
    statusLabel = new QLabel();
    labelElapsedTime = new QLabel();
    labelStepCount = new QLabel();

    QSizePolicy sp;
    const int status_width = 90;
    sp.setHorizontalPolicy(QSizePolicy::Fixed);
    labelStepCount->setSizePolicy(sp);
    labelStepCount->setFixedWidth(status_width);
    labelElapsedTime->setSizePolicy(sp);
    labelElapsedTime->setFixedWidth(status_width);

    ui->statusbar->addWidget(statusLabel);
    ui->statusbar->addPermanentWidget(labelElapsedTime);
    ui->statusbar->addPermanentWidget(labelStepCount);

// anything that includes the Model
    renderer->AddActor(representation.textBgActor);
    renderer->AddActor(representation.scalarBarBgActor);

    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.actor_region_boundary);

    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.actorTextTitle);
    renderer->AddActor(representation.scalarBar);

    // populate combobox
    QMetaEnum qme = QMetaEnum::fromType<VisualRepresentation::VisOpt>();
    for(int i=0;i<qme.keyCount();i++) comboBox_visualizations->addItem(qme.key(i));

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_visualizations(index); });


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

    connect(ui->action_quit, &QAction::triggered, this, &MainWindow::quit_triggered);
    connect(ui->action_camera_reset, &QAction::triggered, this, &MainWindow::cameraReset_triggered);
    connect(ui->actionStart_Pause, &QAction::triggered, this, &MainWindow::simulation_start_pause);
    connect(ui->actionLoad_Parameters, &QAction::triggered, this, &MainWindow::load_parameter_triggered);
    connect(ui->actionView_ScalarBar, &QAction::triggered, this, &MainWindow::toggle_scalarbar);

    connect(qdsbValRange, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);
    connect(qdsbTransparency, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);
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

    settings.setValue("vis_option", comboBox_visualizations->currentIndex());

    QList<int> szs = splitter->sizes();
    settings.setValue("splitter_size_0", szs[0]);
    settings.setValue("splitter_size_1", szs[1]);

    QApplication::quit();
}



void MainWindow::comboboxIndexChanged_visualizations(int index)
{
    if (model.prms.GridXTotal <= 0) return;

    VisualRepresentation::VisOpt opt = (VisualRepresentation::VisOpt)index;
    auto required = representation.GetRequiredGridArrays(opt);

    bool missing = false;
    for (int req : required) {
        if (!model.sim_data.IsGridArrayAllocated(req)) {
            missing = true;
            break;
        }
    }

    if (missing) {
        if (model.sim_data.isVisualizationMode) {
             // Allocate missing arrays
             for (int req : required) {
                 model.sim_data.AllocateGridArray(req);
             }
             
             // Reload frame data to populate the newly allocated arrays
             // We use AnimationFrameNumber which should track the currently visualized frame
             int currentFrame = model.prms.AnimationFrameNumber();
             
             std::string framesDir = model.sim_data.output_directory;
             if(framesDir.empty()) framesDir = "output";
             framesDir += "/frames";
             
             // We need to reload data. 
             // Note: LoadFrameData is thread-safe? locked below anyway.
             // But we are accessing model.sim_data which needs locking?
             // The lock is acquired below. We should acquire it here too?
             // Actually, the Original code acquired lock inside the scope. 
             // We should act safely.
             std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
             model.sim_data.LoadFrameData(currentFrame, framesDir);
        } 
        // else: Simulation mode. We expect dense allocation. 
        // If missing, it's an anomaly but we won't crash here. 
        // Rendering might fail/skip if VisualRepresentation checks pointers (which it does).
    }

    {
        std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
        representation.ChangeVisualizationOption(index);
    }
    qdsbValRange->blockSignals(true);
    qdsbTransparency->blockSignals(true);
    qdsbValRange->setValue(representation.ranges[index]);
    qdsbTransparency->setValue(representation.transparency_coeffs[index]);
    qdsbValRange->blockSignals(false);
    qdsbTransparency->blockSignals(false);
    renderWindow->Render();
}

void MainWindow::limits_changed(double val_)
{
    int idx = (int)representation.VisualizingVariable;
    representation.ranges[idx] = qdsbValRange->value();
    representation.transparency_coeffs[idx] = qdsbTransparency->value();
    std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
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

    const double dx = model.prms.cellsize * model.prms.InitializationImageSizeX/2;
    const double dy = model.prms.cellsize * model.prms.InitializationImageSizeY/2;

    camera->SetPosition(dx, dy, 50.);
    camera->SetFocalPoint(dx, dy, 0.);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(std::min(dx,dy)*1.1);

    camera->Modified();
    renderWindow->Render();
}




void MainWindow::load_parameter_triggered()
{
    QString qFileName = QFileDialog::getOpenFileName(this, "Load Parameters", QDir::currentPath(), "JSON Files (*.json)");
    if(qFileName.isNull())return;
    LoadParameterFile(qFileName);
}



void MainWindow::simulation_data_ready()
{
    updateGUI();
}


void MainWindow::updateGUI()
{
    //LOGV("updateGUI");
    labelStepCount->setText(QString::number(model.prms.SimulationStep));
    labelElapsedTime->setText(QString("%1 s").arg(model.prms.SimulationTime,0,'f',0));

    //statusLabel->setText(QString("per cycle: %1 ms").arg(model.compute_time_per_cycle,0,'f',3));

    {
        std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
        model.SyncTopologyRequired = false;
        representation.simulationTime = model.prms.SimulationTime;
        representation.UpdateTimeText();
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


void MainWindow::LoadParameterFile(QString qFileName)
{
    qDebug() << "MainWindow::LoadParameterFile " << qFileName;
    model.LoadParameterFile(qFileName.toStdString());

    this->setWindowTitle(qFileName);
    pbrowser->setActiveObject(params);

    // Apply the selected visualization option now that the model is loaded
    comboboxIndexChanged_visualizations(comboBox_visualizations->currentIndex());

    updateGUI();
}


void MainWindow::spinbox_slowdown_value_changed(int val)
{
    model.intentionalSlowdown = val;
}



void MainWindow::parameters_updated()
{
    qDebug() << "MainWindow::parameters_updated(); ptsize " << model.prms.ParticleViewSize;
    std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
    representation.SynchronizeTopology();
    renderWindow->Render();
}


void MainWindow::toggle_scalarbar(bool checked)
{
    representation.scalarBar->SetVisibility(checked);
    renderWindow->Render();
}
