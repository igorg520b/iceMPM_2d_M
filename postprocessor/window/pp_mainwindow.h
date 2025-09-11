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
#include <QScrollArea>

#include <QVTKOpenGLNativeWidget.h>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkNew.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRendererCollection.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkJPEGWriter.h>

#include "vtk_representation.h"
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

private:
    GeneralGridData ggd;
    FrameData frameData;
    icy::VisualRepresentation representation;

    QString settingsFileName;       // includes current dir
    QComboBox *comboBox_visualizations;
    QDoubleSpinBox *qdsbValRange;   // high and low limits for value scale
    QDoubleSpinBox *qdsbTransparency;

    QSpinBox *qsbFrameFrom, *qsbFrameTo;
    QSlider *slider2;
    QScrollArea *scrollArea;

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;

    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    vtkNew<vtkJPEGWriter> writer;
    // other
    vtkNew<vtkInteractorStyleImage> interactor;


    void generate_ffmpeg_script(int frameFrom, int frameTo);
    // list which categories we render
    const std::vector<icy::VisualRepresentation::VisOpt> m_visOptsToRender = {
        icy::VisualRepresentation::VisOpt::grid_colors,
        icy::VisualRepresentation::VisOpt::grid_mass,
        icy::VisualRepresentation::VisOpt::grid_Jpinv,
        icy::VisualRepresentation::VisOpt::grid_P,
        icy::VisualRepresentation::VisOpt::grid_Q,
        icy::VisualRepresentation::VisOpt::grid_vnorm,
        icy::VisualRepresentation::VisOpt::str_vonMises
    };
};
#endif
