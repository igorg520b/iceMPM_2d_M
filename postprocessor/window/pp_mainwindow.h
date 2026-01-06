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
#include <QStatusBar>
#include <QSlider>
#include <QToolBar>

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

#include "visual_representation.h"
#include "host_side_data.h"
#include "render_selector_dialog.h"


class PPMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    PPMainWindow(QWidget *parent = nullptr);
    ~PPMainWindow();

    void closeEvent( QCloseEvent* event ) override;
    void LoadParametersFile(QString fileName);
    void LoadFramesDirectory(QString framesDirectory);
    void TryLoadDefaultFrames();  // Auto-load frames from default location if they exist

private Q_SLOTS:
    void cameraReset_triggered();
    void comboboxIndexChanged_visualizations(int index);
    void limits_changed(double val);
    void sliderValueChanged(int val);
    void render_frame_triggered();
    void render_all_triggered();
    void toggleScrollTracking(bool checked);
    void openProject_triggered();
    void openFrames_triggered();
    void toggleRenderSelector();
    void toggleContours(bool checked);
    void contourIntervalChanged(double val);

private:
    HostSideData hsd;
    VisualRepresentation representation;
    std::string currentFrameDirectory;   // Directory containing frame files
    std::string currentProjectDirectory; // Directory containing the JSON project file

    QString settingsFileName;       // includes current dir
    QComboBox *comboBox_visualizations;
    QDoubleSpinBox *qdsbValRange;   // high and low limits for value scale
    QDoubleSpinBox *qdsbTransparency;
    QDoubleSpinBox *qdsbContourInterval;

    QSpinBox *qsbFrameFrom, *qsbFrameTo;
    QSlider *slider2;
    QScrollArea *scrollArea;
    QToolBar *toolBar;
    QStatusBar *statusBar;

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;

    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    vtkNew<vtkJPEGWriter> writer;
    // other
    vtkNew<vtkInteractorStyleImage> interactor;

    // Render selector dialog
    RenderSelectorDialog* m_renderSelectorDialog;

    void generate_ffmpeg_script(int frameFrom, int frameTo);

    // Grid size threshold for slider tracking behavior
    static constexpr int GRID_SIZE_TRACKING_THRESHOLD = 4000;
};
#endif
