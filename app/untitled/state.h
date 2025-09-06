// state.h
#ifndef STATE_H
#define STATE_H

#include <QObject>
#include <memory>
#include "metric.h"
#include "datasetmodel.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include "tsk_fuzzy_network/c_means.h"
#include "tsk_fuzzy_network/tsk.h"
#include "tsk_fuzzy_network/learning_algorithms.h"
#include <utility>

class State : public QObject {
    Q_OBJECT
public:
    explicit State(QObject *parent = nullptr);
    ~State();
    // Методы для работы с данными
    void loadDataset(std::string& filePath);
    void shuffleDataset();
    void initTSK(int m, double fuzziness, double epsilon);
    void configureLearning(int epochs, int count2stepIter,
                           const learning::TrainingConfig& config, bool isGpu, double sep);
    void saveModel(boost::archive::text_oarchive&);
    void loadModel(std::string& filePath);

    // Проверки состояния
    bool isDatasetLoaded() const { return dataset != nullptr; }
    bool isModelInitialized() const { return tsk != nullptr; }
    bool isTraining() const { return m_isLearning; }
    int m_currentEpoch = 0;

    void resetNoise();
    void addNoise(double noiseLevel);

    Dataset* dataset = nullptr;
    DatasetModel* datasetModel = nullptr;
    std::pair<Dataset, Dataset> trainAndTest;
    std::string file;

signals:
    void metricsUpdated(int epoch, double mse, double accuracy);
    void metrics2StepUpdated(int epoch, double mse, double accuracy);
    void learningStarted();
    void learningStopped();
    void learningCompleted();
    void errorOccurred(const QString& message);
    void predicted(metric::Metric);

public slots:

    void startLearning();
    void stopLearning();
    void predict();

private:
    std::unique_ptr<tsk::TSK> tsk;
    std::unique_ptr<learning::HybridAlgorithm> halg;
    learning::TrainingConfig m_config;
    int m_count2stepIter;
    bool m_isGpu;
    bool m_isLearning = false;
    double sep = 0;
    int m_epochs = 0;
};

#endif // STATE_H
