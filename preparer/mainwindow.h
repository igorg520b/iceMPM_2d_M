#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QString>

#include <vtkInteractorStyleRubberBand2D.h>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkNew.h>
#include <vtkInteractorStyleImage.h>

#include "mainimageimporter.h"
#include "parameterparser.h"


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void LoadParameterFile(QString fileName, bool createDXFOnly);

private Q_SLOTS:
    void render_flow_data(bool checked);
    void render_boundary_conditions(bool checked);
    void apply_transform();
    void value_changed(double newValue);
    void rasterize_flow();
    void render_vn();
    void render_vx();
    void render_vy();
    void render_thickness();
    void render_original();

private:
    Ui::MainWindow *ui;
    MainImageImporter mii;
    ParameterParser params;


    QDoubleSpinBox *qdsbScale, *qdsbOffsetX, *qdsbOffsetY;
    QSlider *slider1;

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkInteractorStyleImage> interactorStyle;

};
#endif // MAINWINDOW_H
