#include "mainwindow.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QTimer>
#include <QFileInfo>
#include <QDir>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);


    QCommandLineParser parser;
    parser.setApplicationDescription("Prepating the input files for MPM simulation");
    parser.addPositionalArgument("parameters", QCoreApplication::translate("main", "JSON parameter file"));

    parser.process(a);

    const QStringList args = parser.positionalArguments();

    MainWindow w;

    if(args.size() >= 1)
    {
        QString parametersFile = args[0];

        // Check if the provided path is a directory
        QFileInfo fileInfo(parametersFile);
        if (fileInfo.isDir()) {
            // If it's a directory, append default filename "prepare.json"
            parametersFile = QDir(parametersFile).filePath("prepare.json");
        }

        w.LoadParameterFile(parametersFile);
    }

    w.resize(1400, 900);
    w.show();
    return a.exec();

    return 0;
}
