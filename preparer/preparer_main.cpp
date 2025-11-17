#include "mainwindow.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QTimer>

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
        w.LoadParameterFile(parametersFile);
    }

    w.resize(1400, 900);
    w.show();
    return a.exec();

    return 0;
}
