#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QSlider>
#include <QMenuBar>
#include <QSettings>

#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkNew.h>
#include <vtkInteractorStyleImage.h>

#include "visual_representation.h"

#include "parameterparser.h"
#include "host_side_data.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void LoadParameterFile(QString fileName);

private Q_SLOTS:
    void openJsonFile_triggered();
    void resetCamera_triggered();
    void comboboxIndexChanged_visualizations(int index);
    void limits_changed(double val);
    void flowTimeSliderChanged(int value);
    void closeEvent(QCloseEvent *event) override;

private:
    void setupUI();
    void createMenuBar();
    void loadSettings();
    void saveSettings();
    void loadCameraState();
    void saveCameraState();

    ParameterParser params;
    HostSideData hsd;
    VisualRepresentation representation;

    // Visualization controls
    QComboBox *comboBox_visualizations;
    QDoubleSpinBox *qdsbValRange;
    QDoubleSpinBox *qdsbTransparency;
    QSlider *flowTimeSlider;              // slider for flow field time control

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkInteractorStyleImage> interactorStyle;

    QString settingsFileName;
};
#endif // MAINWINDOW_H
