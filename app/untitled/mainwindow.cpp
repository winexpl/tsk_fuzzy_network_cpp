// mainwindow.cpp
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QLineSeries>
#include <QFileDialog>
#include <QMessageBox>
#include <QGraphicsLineItem>
#include <QGraphicsTextItem>

MainWindow::~MainWindow() {
    if (m_workerThread->isRunning()) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
    delete ui;
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    setupCharts();
    setupConnections();
    updateUIState();
}
void MainWindow::setupCharts()
{
    // Настройка таблиц (оставляем без изменений)
    ui->tableWidgetMetrics->setColumnCount(3);
    ui->tableWidgetMetrics->setHorizontalHeaderLabels({"Эпоха", "MSE", "Accuracy"});
    ui->tableWidgetPredict->setColumnCount(3);
    ui->tableWidgetPredict->setHorizontalHeaderLabels({"Эпоха", "MSE", "Accuracy"});

    // Создание и настройка виджетов графиков
    chartViewMse = new QChartView(this);
    chartViewAccuracy = new QChartView(this);
    ui->verticalLayout_6->addWidget(chartViewAccuracy);
    ui->verticalLayout_6->addWidget(chartViewMse);

    // График MSE (красный)
    m_mseChart = new QChart();
    m_mseSeries = new QLineSeries();
    m_mseSeries->setColor(Qt::red);  // Устанавливаем красный цвет

    // Настройка пера для более толстой линии
    QPen pen(Qt::red);
    pen.setWidth(2);
    m_mseSeries->setPen(pen);

    m_mseChart->addSeries(m_mseSeries);
    m_mseChart->setTitle("MSE (Mean Squared Error)");

    // Создаем оси и настраиваем их
    QValueAxis *axisXMse = new QValueAxis;
    QValueAxis *axisYMse = new QValueAxis;

    axisXMse->setTitleText("Эпоха");
    axisYMse->setTitleText("MSE");

    m_mseChart->addAxis(axisXMse, Qt::AlignBottom);
    m_mseChart->addAxis(axisYMse, Qt::AlignLeft);

    m_mseSeries->attachAxis(axisXMse);
    m_mseSeries->attachAxis(axisYMse);

    m_mseChart->legend()->hide();
    chartViewMse->setChart(m_mseChart);
    chartViewMse->setRenderHint(QPainter::Antialiasing);

    // График Accuracy (синий)
    m_accuracyChart = new QChart();
    m_accuracySeries = new QLineSeries();
    m_accuracySeries->setColor(Qt::blue);  // Синий цвет для точности

    m_accuracyChart->addSeries(m_accuracySeries);
    m_accuracyChart->setTitle("Accuracy");

    // Создаем оси для Accuracy
    QValueAxis *axisXAcc = new QValueAxis;
    QValueAxis *axisYAcc = new QValueAxis;

    axisXAcc->setTitleText("Эпоха");
    axisYAcc->setTitleText("Accuracy");
    axisYAcc->setRange(0, 1.0);  // Accuracy всегда от 0 до 1

    m_accuracyChart->addAxis(axisXAcc, Qt::AlignBottom);
    m_accuracyChart->addAxis(axisYAcc, Qt::AlignLeft);

    m_accuracySeries->attachAxis(axisXAcc);
    m_accuracySeries->attachAxis(axisYAcc);

    m_accuracyChart->legend()->hide();
    chartViewAccuracy->setChart(m_accuracyChart);
    chartViewAccuracy->setRenderHint(QPainter::Antialiasing);

    // Инициализация переменной для хранения максимального MSE
    m_maxMse = 0.0;
}

void MainWindow::onBtnAddNoise()
{
    m_state.addNoise(ui->spinNoiseLevel->value());
}

void MainWindow::onBtnResetNoise()
{
    m_state.resetNoise();
}

void MainWindow::updateGraphs(int epoch, double mse, double accuracy)
{
    // Обновляем данные
    m_mseSeries->append(epoch, mse);
    m_accuracySeries->append(epoch, accuracy);

    // Обновляем максимальное значение MSE
    if (mse > m_maxMse || epoch == 1) {
        m_maxMse = mse;
    }

    // Настраиваем диапазоны осей
    m_mseChart->axisX()->setRange(1, epoch);
    m_mseChart->axisY()->setRange(0, m_maxMse * 1.1);  // Верхняя граница = maxMSE + 10%

    m_accuracyChart->axisX()->setRange(1, epoch);
    // Для Accuracy оставляем фиксированный диапазон 0-1

    // Обновляем отображение
    chartViewMse->repaint();
    chartViewAccuracy->repaint();
}

void MainWindow::onMetricsUpdated(int epoch, double mse, double accuracy)
{
    int row = ui->tableWidgetMetrics->rowCount();
    ui->tableWidgetMetrics->insertRow(row);
    ui->tableWidgetMetrics->setItem(row, 0, new QTableWidgetItem(QString::number(epoch)));
    ui->tableWidgetMetrics->setItem(row, 1, new QTableWidgetItem(QString::number(mse, 'e', 3)));
    ui->tableWidgetMetrics->setItem(row, 2, new QTableWidgetItem(QString::number(accuracy, 'f', 4)));
    ui->tableWidgetMetrics->scrollToBottom();

    m_mseData.append(QPointF(epoch, mse));
    m_accuracyData.append(QPointF(epoch, accuracy));

    m_mseYScale = qMax(mse * 1.1, m_mseYScale);
    m_accuracyYScale = 1.0; // Accuracy всегда от 0 до 1

    updateGraphs(epoch, mse, accuracy);
}

void MainWindow::setupConnections() {
    // Кнопки интерфейса
    connect(ui->btnSearchPath, &QPushButton::clicked, this, &MainWindow::onBtnSearchPathClicked);
    connect(ui->btnInitWeigth, &QPushButton::clicked, this, &MainWindow::onBtnInitWeigthClicked);
    connect(ui->btnShuffleDataset, &QPushButton::clicked, this, &MainWindow::onBtnShuffleDatasetClicked);
    connect(ui->btnLearning, &QPushButton::clicked, this, &MainWindow::onBtnLearningClicked);
    connect(ui->btnPredict, &QPushButton::clicked, this, &MainWindow::onBtnPredictClicked);
    connect(ui->btnSaveModel, &QAction::triggered, this, &MainWindow::onSaveModel);
    connect(ui->btnLoadModel, &QAction::triggered, this, &MainWindow::onLoadModel);
    connect(ui->btnResetNoisy, &QPushButton::clicked, this, &MainWindow::onBtnResetNoise);
    connect(ui->btnNoisy, &QPushButton::clicked, this, &MainWindow::onBtnAddNoise);


    // Сигналы от State
    connect(&m_state, &State::metricsUpdated, this, &MainWindow::onMetricsUpdated);
    connect(&m_state, &State::metrics2StepUpdated, this, &MainWindow::onMetrics2StepUpdated);
    connect(&m_state, &State::learningStarted, this, &MainWindow::onLearningStarted);
    connect(&m_state, &State::learningStopped, this, &MainWindow::onLearningStopped);
    connect(&m_state, &State::learningCompleted, this, &MainWindow::onLearningCompleted);
    connect(&m_state, &State::errorOccurred, this, &MainWindow::onErrorOccurred);
    connect(&m_state, &State::predicted, this, &MainWindow::onPredicted);
}

void MainWindow::updateUIState() {
    bool datasetLoaded = m_state.isDatasetLoaded();
    bool modelInitialized = m_state.isModelInitialized();
    bool isTraining = m_state.isTraining();

    ui->btnInitWeigth->setEnabled(datasetLoaded && !isTraining);
    ui->btnShuffleDataset->setEnabled(datasetLoaded && !isTraining);
    ui->btnLearning->setEnabled(modelInitialized);
    ui->btnLearning->setText(isTraining ? "Остановить" : "Обучить");
}

void MainWindow::onBtnSearchPathClicked() {
    QString filePath = QFileDialog::getOpenFileName(
        this, "Select Dataset", QDir::homePath(),
        "CSV Files (*.csv);;All Files (*.*)");

    if (!filePath.isEmpty()) {
        try {
            std::string str = filePath.toStdString();
            m_state.loadDataset(str);
            DatasetModel* dm = new DatasetModel(this);
            m_state.datasetModel = dm;
            ui->tableViewDataset->setModel(m_state.datasetModel);
            ui->tableViewDataset->doItemsLayout();
            dm->setData(m_state.dataset);

            updateUIState();
            // QMessageBox::information(this, "Success", "Dataset loaded successfully");
        } catch (const std::exception& e) {
            // QMessageBox::critical(this, "Error", QString("Failed to load dataset: %1").arg(e.what()));
        }
    }
}

void MainWindow::onSaveModel()
{
    QString filePath = QFileDialog::getSaveFileName(
        this,
        tr("Сохранить модель"),
        QDir::homePath(),
        tr("Файлы модели (*.model)")
    );

    if (filePath.isEmpty()) {
        return;
    }

    try {
        std::ofstream ofs(filePath.toStdString());
        if (!ofs.is_open()) {
            QMessageBox::critical(this, tr("Ошибка"), tr("Не удалось открыть файл"));
            return;
        }

        boost::archive::text_oarchive out(ofs);
        m_state.saveModel(out);
        ofs.close();
    } catch (const std::exception& e) {

    }
}

void MainWindow::onDeleteModel()
{

}

void MainWindow::onLoadModel()
{
    QString filePath = QFileDialog::getOpenFileName(
        this, "Select Dataset", QDir::homePath(),
        "Model Files (*.model);;All Files (*.*)");

    if (!filePath.isEmpty()) {
        try {
            std::string str = filePath.toStdString();
            m_state.loadModel(str);
            updateUIState();
            // QMessageBox::information(this, "Success", "Dataset loaded successfully");
        } catch (const std::exception& e) {
            // QMessageBox::critical(this, "Error", QString("Failed to load dataset: %1").arg(e.what()));
        }
    }
}

void MainWindow::onBtnInitWeigthClicked() {
    try {
        m_state.initTSK(
            ui->spinCMeansClusters->value(),
            ui->spinCMeansFuzziness->value(),
            ui->spinCMeansEpsilon->value()
        );
        updateUIState();
        // QMessageBox::information(this, "Success", "Weights initialized successfully");
    } catch (const std::exception& e) {
        // QMessageBox::critical(this, "Error", QString("Failed to initialize weights: %1").arg(e.what()));
    }
}

void MainWindow::onBtnShuffleDatasetClicked() {
    m_state.shuffleDataset();
    // QMessageBox::information(this, "Success", "Dataset shuffled successfully");
}

void MainWindow::onBtnLearningClicked() {
    if (m_state.isTraining()) {
        m_state.stopLearning();
    } else {
        learning::TrainingConfig config;
        config.grad_clip = ui->spinGradClip->value();
        config.nu_c = ui->spinNuC->value();
        config.nu_sigma = ui->spinNuSigma->value();
        config.nu_b = ui->spinNuB->value();

        m_state.configureLearning(
            ui->spinEpochsCount->value(),
            ui->spinIter2Step->value(),
            config,
            ui->checkBoxIsGpu->isChecked(),
            ui->spinSepDataset->value()
            );

        m_workerThread = new QThread{};
        m_state.moveToThread(m_workerThread);
        connect(m_workerThread, &QThread::started, &m_state, &State::startLearning);
        m_workerThread->start();
    }
    updateUIState();
}


void MainWindow::onMetrics2StepUpdated(int epoch, double mse, double accuracy) {
    onMetricsUpdated(epoch, mse, accuracy);
}

void MainWindow::onLearningStopped() {
    ui->statusbar->showMessage("Training stopped", 3000);
    updateUIState();
}

void MainWindow::onLearningCompleted() {
    // QMessageBox::information(this, "Success", "Training completed successfully");
    onLearningStopped();
}

void MainWindow::onLearningStarted() {
    // QMessageBox::information(this, "Success", "Training started successfully");
    onLearningStopped();
}

void MainWindow::onErrorOccurred(const QString& message) {
    // QMessageBox::critical(this, "Error", message);
    onLearningStopped();
}

void MainWindow::onBtnPredictClicked()
{
    learning::TrainingConfig config;
    config.grad_clip = ui->spinGradClip->value();
    config.nu_c = ui->spinNuC->value();
    config.nu_sigma = ui->spinNuSigma->value();
    config.nu_b = ui->spinNuB->value();

    m_state.configureLearning(
        ui->spinEpochsCount->value(),
        ui->spinIter2Step->value(),
        config,
        ui->checkBoxIsGpu->isChecked(),
        ui->spinSepDataset->value()
    );


    m_state.predict();
}

void MainWindow::onPredicted(metric::Metric metric)
{
    int row = ui->tableWidgetPredict->rowCount();
    ui->tableWidgetPredict->insertRow(row);
    ui->tableWidgetPredict->setItem(row, 0, new QTableWidgetItem(QString::number(m_state.m_currentEpoch / 2)));
    ui->tableWidgetPredict->setItem(row, 1, new QTableWidgetItem(QString::number(metric.mse, 'e', 3)));
    ui->tableWidgetPredict->setItem(row, 2, new QTableWidgetItem(QString::number(metric.accuracy, 'f', 4)));
    ui->tableWidgetPredict->scrollToBottom();

}
