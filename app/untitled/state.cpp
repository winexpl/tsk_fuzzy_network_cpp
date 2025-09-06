// state.cpp
#include "state.h"
#include <QThread>
#include <QDebug>
#include <QString>

State::State(QObject *parent) : QObject(parent) {}

State::~State()
{
    if(dataset) delete dataset;
}
void State::initTSK(int m, double fuzziness, double epsilon) {
    if (!isDatasetLoaded()) {
        emit errorOccurred(tr("Dataset not loaded"));
        return;
    }

    try {
        tsk = std::make_unique<tsk::TSK>(dataset->vectorSize, m);
        tsk::CMeans cmeans(m, fuzziness, epsilon);
        cmeans.fit(dataset->getX());
        tsk->setC(cmeans.getCentroids());
        tsk->setSigma(cmeans.getSigma());
        qDebug() << "TSK model initialized successfully";
    } catch (const std::exception& e) {
        emit errorOccurred(tr("TSK initialization failed: %1").arg(e.what()));
    }
}

void State::configureLearning(int epochs, int count2stepIter,
                              const learning::TrainingConfig& config, bool isGpu, double sep) {
    m_epochs = epochs;
    m_count2stepIter = count2stepIter;
    m_config = config;
    m_isGpu = isGpu;
    this->sep = sep;
}

void State::startLearning() {
    if (!isModelInitialized()) {
        emit errorOccurred(tr("Model not initialized"));
        return;
    }

    m_isLearning = true;
    std::pair<Dataset, Dataset> trainAndTest = dataset->splitDatasetOnTrainAndTest(sep);
    halg = std::make_unique<learning::HybridAlgorithm>(tsk.get(), trainAndTest.first);

    emit learningStarted();

    auto callback = [this](metric::Metric metric) {
        emit metricsUpdated(m_currentEpoch / 2 + 1, metric.mse, metric.accuracy);
        m_currentEpoch++;
        return !m_isLearning;
    };

    auto callback2step = [this](metric::Metric metric) {
        qDebug() << metric.accuracy << " " << metric.mse;
    };

    try {
        halg->learning(dataset->dim, m_epochs, m_count2stepIter,
                       m_config, m_isGpu, callback, callback2step, m_epochs < 2? 0.001: 0);
        emit learningCompleted();
    } catch (const std::exception& e) {
        emit errorOccurred(tr("Training failed: %1").arg(e.what()));
        emit learningStopped();
    }
    m_isLearning = false;
    emit learningStopped();
}

void State::loadDataset(std::string& filePath) {
    file = filePath;
    dataset = new Dataset(filePath);
}

void State::shuffleDataset() {
    dataset->shuffle();
}

void State::stopLearning() {
    if (m_isLearning) {
        m_isLearning = false;
        qDebug() << "Training stopped by user";
    }
}

void State::saveModel(boost::archive::text_oarchive& stream) {

    stream << *tsk;
}

void State::loadModel(std::string& filePath)  // const ссылка, т.к. файл не изменяется
{
    std::ifstream ifs(filePath, std::ios::binary);  // Открываем в бинарном режиме
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    try {
        tsk::TSK loadedModel;
        {
            boost::archive::text_iarchive ia(ifs);
            ia >> loadedModel;
        }

        ifs.close();
        if(tsk) {
            tsk.release();
        }
        tsk = std::make_unique<tsk::TSK>(std::move(loadedModel));
    } catch (const boost::archive::archive_exception& e) {
        ifs.close();
        throw std::runtime_error("Boost archive error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        ifs.close();
        throw std::runtime_error("Model loading failed: " + std::string(e.what()));
    }
}

void State::predict()
{
    std::pair<Dataset, Dataset> trainAndTest = dataset->splitDatasetOnTrainAndTest(sep);
    metric::Metric metric = tsk->evaluate(trainAndTest.second.getX(), trainAndTest.second.getD(), trainAndTest.second.getClassCount());
    emit predicted(metric);
}

void State::resetNoise()
{
    if(dataset) delete dataset;
    dataset = new Dataset(file);
}

void State::addNoise(double noiseLevel)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(-noiseLevel, noiseLevel);

    boost::multi_array<double, 2> noisyData(dataset->getX());
    auto *data_ptr = noisyData.data();
    const size_t num_elements = noisyData.num_elements();

    for (size_t i = 0; i < num_elements; ++i)
    {
        data_ptr[i] += dist(gen);
    }

    dataset->x = noisyData;
}
