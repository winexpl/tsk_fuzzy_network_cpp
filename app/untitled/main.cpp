#include "mainwindow.h"

#include <QApplication>
#include <QLocale>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QLocale::setDefault(QLocale::C);
    QString test = "5.123";
    qDebug() << "Test conversion:" << test.toDouble();
    MainWindow w;
    w.show();
    return a.exec();
}
