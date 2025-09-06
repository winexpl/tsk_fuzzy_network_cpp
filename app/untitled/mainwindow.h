#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QGraphicsScene>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include "state.h"
#include <QtCharts>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    // Состояние приложения
    State m_state;
    double m_maxMse;

    // Поток для выполнения обучения
    QThread *m_workerThread;

    QChartView *chartViewMse;
    QChartView *chartViewAccuracy;
    QChart *m_mseChart;
    QChart *m_accuracyChart;
    QLineSeries *m_mseSeries;
    QLineSeries *m_accuracySeries;
    QList<QPointF> m_mseData;
    QList<QPointF> m_accuracyData;
    double m_mseYScale = 0.0;
    double m_accuracyYScale = 1.0;

    void setupCharts();
    void drawGraph(QGraphicsScene* scene, const QVector<QPointF>& data, const QString& title);
    void updateGraphs(int epoch, double mse, double accuracy);

    // Методы инициализации
    void setupConnections();
    void updateUIState();

signals:
    void startLearning();
    void addNoise(double noiseLevel);
    void resetNoise();
private slots:
    // Слоты для кнопок UI
    void onBtnSearchPathClicked();
    void onBtnInitWeigthClicked();
    void onBtnShuffleDatasetClicked();
    void onBtnLearningClicked();
    void onBtnPredictClicked();
    void onSaveModel();
    void onDeleteModel();
    void onLoadModel();
    void onBtnAddNoise();
    void onBtnResetNoise();

    // Слоты для обработки сигналов от State
    void onMetricsUpdated(int epoch, double mse, double accuracy);
    void onMetrics2StepUpdated(int epoch, double mse, double accuracy);
    void onPredicted(metric::Metric metric);
    void onLearningStarted();
    void onLearningStopped();
    void onLearningCompleted();
    void onErrorOccurred(const QString& message);

};

#endif // MAINWINDOW_H
