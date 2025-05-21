#include "mainwindow.h"

#include <QApplication>
#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
//    w.setWindowState(Qt::WindowMaximized); // Set the state explicitly
//    w.showMaximized();
    w.resize(1400, 900);
    w.show();
//    QTimer::singleShot(2, &w, [&w]() { w.showMaximized(); }); // Maximize after event loop starts
    return a.exec();
}
