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

    QCommandLineOption dxfOption(
        QStringList() << "d" << "dxf",
        QCoreApplication::translate("main", "Only create dxf file.")
        );
    parser.addOption(dxfOption); // Add the option to the parser

    parser.process(a);

    const QStringList args = parser.positionalArguments();

    MainWindow w;

    if(args.size() >= 1)
    {
        QString parametersFile = args[0];

        bool dxfOnly = parser.isSet(dxfOption);
        w.LoadParameterFile(parametersFile, dxfOnly);

        if(!dxfOnly)
        {
            w.resize(1400, 900);
            w.show();
            return a.exec();
        }
    }
    else
    {
        throw std::runtime_error("parameter file required");
    }

    return 0;
}
