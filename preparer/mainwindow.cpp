#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QImageReader>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    QLabel *lblScale = new QLabel("s:");
    QLabel *lblX = new QLabel("x:");
    QLabel *lblY = new QLabel("y:");

    ui->toolBar->addWidget(lblScale);

    // double spin box
    qdsbScale = new QDoubleSpinBox();
    qdsbScale->setRange(0, 1000000);
    qdsbScale->setDecimals(6);
    qdsbScale->setSingleStep(0.000001);
    qdsbScale->setValue(params.FLUENT_Scale);
    ui->toolBar->addWidget(qdsbScale);

    ui->toolBar->addWidget(lblX);

    qdsbOffsetX = new QDoubleSpinBox();
    qdsbOffsetX->setRange(0, 1000000);
    qdsbOffsetX->setValue(params.FLUENT_OffsetX);
    qdsbOffsetX->setDecimals(0);
    qdsbOffsetX->setSingleStep(1);
    ui->toolBar->addWidget(qdsbOffsetX);

    ui->toolBar->addWidget(lblY);

    qdsbOffsetY = new QDoubleSpinBox();
    qdsbOffsetY->setRange(0, 1000000);
    qdsbOffsetY->setValue(params.FLUENT_OffsetY);
    qdsbOffsetY->setDecimals(1);
    qdsbOffsetY->setSingleStep(1);
    ui->toolBar->addWidget(qdsbOffsetY);


    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);
    setCentralWidget(qt_vtk_widget);


    renderer->SetBackground(0.9,0.9,0.85);
    renderWindow->AddRenderer(renderer);

    renderWindow->GetInteractor()->SetInteractorStyle(interactorStyle);

    renderer->AddActor(mii.actor);
    renderer->AddActor(mii.fdp.actor);




    connect(ui->actionRender_Flow_Field, &QAction::triggered, this, &MainWindow::render_flow_data);
    connect(ui->actionRender_Boundary_Conditions, &QAction::triggered, this, &MainWindow::render_boundary_conditions);
    connect(ui->actionApply_Transform, &QAction::triggered, this, &MainWindow::apply_transform);
    connect(ui->action_Rasterize_Flow, &QAction::triggered, this, &MainWindow::rasterize_flow);
    connect(ui->actionRender_Vn, &QAction::triggered, this, &MainWindow::render_vn);
    connect(ui->actionRender_Vx, &QAction::triggered, this, &MainWindow::render_vx);
    connect(ui->actionRender_Vy, &QAction::triggered, this, &MainWindow::render_vy);

    connect(ui->actionRender_Thickness, &QAction::triggered, this, &MainWindow::render_thickness);
    connect(ui->actionRender_Original, &QAction::triggered, this, &MainWindow::render_original);

    connect(qdsbScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::value_changed);
    connect(qdsbOffsetX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::value_changed);
    connect(qdsbOffsetY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::value_changed);

}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::LoadParameterFile(QString fileName, bool createDXFOnly)
{
    params.LoadParamsFile(fileName.toStdString());

    if(createDXFOnly)
    {
        qDebug() << "creating DXF file";
        // make DXF file from the main simulation area
        mii.sip.LoadSVG(params.fileNameSVG, params.width, params.height, params.MainPathID, params.RectanglePathID,
                        params.FluentPathID, params.ProjectDirectory);

        //mii.sip.bezierPaths.front().exportToAnsysDiscovery("mainpath_ansys_script.py",3);
        return;
    }


    if(!params.fileNamePNG.empty())
    {
        qDebug() << "loading PNG";
        mii.LoadImage(params.fileNamePNG, params.width, params.height);

        mii.sip.LoadSVG(params.fileNameSVG, params.width, params.height, params.MainPathID, params.RectanglePathID,
                        params.FluentPathID, params.ProjectDirectory);
        mii.sip.RasterizeSVG();
        ui->actionRender_Boundary_Conditions->setChecked(true);

        mii.IdentifyIceThickness(params);
        mii.SaveAsHDF5(params.ProjectDirectory);

        if(!params.fileNameFluentCAS.empty())
        {
            mii.fdp.LoadFluentResult(params.fileNameFluentDAT, params.fileNameFluentCAS, params.width, params.height);
//            mii.fdp.ApplyTransform(qdsbScale->value(), qdsbOffsetX->value(), qdsbOffsetY->value());
            mii.fdp.ApplyTransform(mii.sip.extents_fluent);
            mii.fdp.Rasterize();
            mii.fdp.ApplyDiffusion(mii.sip.path_indices, 4, 0.02);
            mii.fdp.SaveAsHDF5(params.ProjectDirectory);
            ui->actionRender_Flow_Field->setChecked(true);
        }
        else if(params.ConstFlow)
        {
            mii.fdp.PopulateWithConstValue(params.width, params.height, params.FlowX, params.FlowY);
            mii.fdp.SaveAsHDF5(params.ProjectDirectory);
            ui->actionRender_Flow_Field->setChecked(true);
        }
    }

    mii.Render(ui->actionRender_Boundary_Conditions->isChecked(), ui->actionRender_Flow_Field->isChecked());
    renderWindow->Render();
}




void MainWindow::render_flow_data(bool checked)
{
    mii.fdp.actor->SetVisibility(checked);
//    mii.Render(ui->actionRender_Boundary_Conditions->isChecked(), ui->actionRender_Flow_Field->isChecked());
    renderWindow->Render();
}

void MainWindow::render_boundary_conditions(bool checked)
{
    mii.Render(ui->actionRender_Boundary_Conditions->isChecked(), ui->actionRender_Flow_Field->isChecked());
    renderWindow->Render();
}

void MainWindow::apply_transform()
{
//    mii.actor->VisibilityOff();
//    mii.fdp.ApplyTransform(qdsbScale->value(), qdsbOffsetX->value(), qdsbOffsetY->value());
    renderWindow->Render();

}

void MainWindow::value_changed(double newValue)
{
    apply_transform();
}



void MainWindow::rasterize_flow()
{
    mii.fdp.Rasterize();
}

void MainWindow::render_vn()
{
    mii.RenderV_n();
    renderWindow->Render();
}

void MainWindow::render_vx()
{
    mii.RenderV_x();
    renderWindow->Render();
}

void MainWindow::render_vy()
{
    mii.RenderV_y();
    renderWindow->Render();
}


void MainWindow::render_thickness()
{
    mii.RenderIceThickness();
    renderWindow->Render();
}

void MainWindow::render_original()
{
    mii.RenderOriginal();
    renderWindow->Render();
}
