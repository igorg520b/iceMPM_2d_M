#include "mainwindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QDir>
#include <iostream>
#include <omp.h>

int main(int argc, char *argv[])
{
    std::cout << "num_threads " << omp_get_max_threads() << std::endl;
    std::cout << "testing threads" << std::endl;
    int nthreads, tid;
#pragma omp parallel
    { std::cout << omp_get_thread_num(); }
    std::cout << std::endl;


    QApplication a(argc, argv);
    QApplication::setApplicationName("iceMPM");
    QApplication::setApplicationVersion("1.2");

    QCommandLineParser parser;
    parser.setApplicationDescription("MPM simulation of ice with GUI");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("parameters", QCoreApplication::translate("main", "JSON parameter file"));

    parser.process(a);

    const QStringList args = parser.positionalArguments();
    MainWindow w;

    if(args.size() >= 1)
    {
        QString parameters_file = args[0];

        // Check if the provided path is a directory
        QFileInfo fileInfo(parameters_file);
        if (fileInfo.isDir()) {
            // If it's a directory, append default filename "simulation.json"
            parameters_file = QDir(parameters_file).filePath("simulation.json");
        }

        w.LoadParameterFile(parameters_file);
    }

    w.resize(1800,1000);
    w.move(0,0);
    w.show();
//    w.showMaximized();
    return a.exec();
}
