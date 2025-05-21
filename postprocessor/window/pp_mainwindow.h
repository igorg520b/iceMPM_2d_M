// pp_mainwindow.h

#ifndef PPMAINWINDOW_H
#define PPMAINWINDOW_H

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
#include <vtkInteractorStyleImage.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include "vtk_visualization.h"
#include "framedata.h"


QT_BEGIN_NAMESPACE
namespace Ui { class PPMainWindow; }
QT_END_NAMESPACE

class PPMainWindow : public QMainWindow
{
    Q_OBJECT
private:
    Ui::PPMainWindow *ui;

public:
    PPMainWindow(QWidget *parent = nullptr);
    ~PPMainWindow();

    void closeEvent( QCloseEvent* event ) override;

    void LoadParametersFile(QString fileName);
    void LoadFramesDirectory(QString framesDirectory);


private Q_SLOTS:
    void cameraReset_triggered();
    void comboboxIndexChanged_visualizations(int index);
    void limits_changed(double val);

    void sliderValueChanged(int val);

    void render_frame_triggered();
    void render_all_triggered();

//    void open_frame_triggered();
//    void load_selected_frame_triggered();
//    void toggle_wind_visualization(bool checked);

private:
    GeneralGridData ggd;
    FrameData frameData;


    void OpenSnapshot(QString fileName);
    void generate_ffmpeg_script();

    QString settingsFileName;       // includes current dir
    QComboBox *comboBox_visualizations;
    QDoubleSpinBox *qdsbValRange;   // high and low limits for value scale
    QSpinBox *qsbFrameFrom, *qsbFrameTo;
    QSlider *slider2;

    // VTK
    vtkRenderWindow *renderWindow;
//    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;

    // other
    vtkNew<vtkInteractorStyleImage> interactor;
};
#endif
