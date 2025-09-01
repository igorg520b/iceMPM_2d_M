#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QSizePolicy>
#include <QPushButton>
#include <QSplitter>
#include <QLabel>
#include <QVBoxLayout>
#include <QTreeWidget>
#include <QProgressBar>
#include <QMenu>
#include <QList>
#include <QDebug>
#include <QComboBox>
#include <QMetaEnum>
#include <QDir>
#include <QString>
#include <QCheckBox>
#include <QFile>
#include <QTextStream>
#include <QIODevice>
#include <QSettings>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QFileInfo>

#include <QVTKOpenGLNativeWidget.h>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkNew.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkInteractorStyleImage.h>

//#include "SpecialSelector2D.h"


#include "objectpropertybrowser.h"
#include "vtk_representation.h"
#include "model.h"
#include "parameters_wrapper.h"
#include "backgroundworker.h"
#include "snapshotmanager.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>

#include <spdlog/spdlog.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    Ui::MainWindow *ui;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent( QCloseEvent* event ) override;
    icy::Model model;
    void LoadParameterFile(QString qFileName, QString resumeSnapshot);    // return file name of the point cloud

private Q_SLOTS:
    void quit_triggered();

    void background_worker_paused();
    void simulation_data_ready();

    void simulation_start_pause(bool checked);
    void cameraReset_triggered();
    void open_snapshot_triggered();
    void load_parameter_triggered();

    void comboboxIndexChanged_visualizations(int index);
    void limits_changed(double val);
    void sliderValueChanged(int val);
    void spinbox_slowdown_value_changed(int val);

    void parameters_updated();
    void print_camera_params();
    void toggle_scalarbar(bool checked);

private:
    void updateGUI();   // when simulation is started/stopped or when a step is advanced
    void updateActorText();
    void save_binary_data();

    void OpenSnapshot(QString fileName);

    BackgroundWorker *worker;
    icy::VisualRepresentation representation;
    ParamsWrapper *params;

    QString settingsFileName;       // includes current dir
    QLabel *statusLabel;                    // statusbar
    QLabel *labelElapsedTime;
    QLabel *labelStepCount;
    QLabel *labelWindSpeed, *labelWindDirection;
    QComboBox *comboBox_visualizations;
    QDoubleSpinBox *qdsbValRange;   // high and low limits for value scale
    QSpinBox *qsbIntentionalSlowdown;

    ObjectPropertyBrowser *pbrowser;    // to show simulation settings/properties
    QSplitter *splitter;
    QSlider *slider1;

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;

    // other
    const std::string outputDirectory = "default_output";
//    vtkNew<SpecialSelector2D> specialSelector2D;
    vtkNew<vtkInteractorStyleImage> interactorStyle;


    // screenshots
    const std::string screenshot_directory = "screenshots";
    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    vtkNew<vtkPNGWriter> writerPNG;
    void screenshot();

    friend class SpecialSelector2D;

};
#endif // MAINWINDOW_H
