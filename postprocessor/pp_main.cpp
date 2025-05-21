#include "pp_mainwindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QCommandLineParser>
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
    QApplication::setApplicationName("MPM Postprocessor");
    QApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Post-processing the HDF5 simulation output");
    parser.addPositionalArgument("parameters", QCoreApplication::translate("main", "JSON parameter file"));

    QCommandLineOption framesDirectoryOption(
        QStringList() << "f" << "frames",
        QCoreApplication::translate("main", "Directory where frames are located"),
        QCoreApplication::translate("main", "directory"));

    parser.addOption(framesDirectoryOption);
    parser.process(a);

    const QStringList args = parser.positionalArguments();
    PPMainWindow w;

    if(args.size() >= 1)
    {
        QString parametersFile = args[0];
        w.LoadParametersFile(parametersFile);

        if (parser.isSet(framesDirectoryOption))
        {
            QString directory = parser.value(framesDirectoryOption);
            std::cout << "main, framesDirectory " << directory.toStdString();
            w.LoadFramesDirectory(directory);
        }
    }

    w.resize(1400,900);
    w.show();
//    w.showMaximized();
    return a.exec();
}
