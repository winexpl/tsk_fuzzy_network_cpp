#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include "tsk_fuzzy_network/learning_algorithms.h"
#include "tsk_fuzzy_network/dataset.h"
#include "tsk_fuzzy_network/c_means.h"
#include "metric.h"
#include <gtest/gtest.h>
#include <random>
#include <fstream>
#include <memory>
#include <iomanip>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class JsonTestListener : public testing::EmptyTestEventListener
{
    json test_results;
    std::ofstream json_file;
    std::string current_test_name;

public:
    explicit JsonTestListener(const std::string &filename) : json_file(filename)
    {
        if (!json_file.is_open())
        {
            throw std::runtime_error("Failed to open JSON file");
        }
        test_results["tests"] = json::array();
    }

    ~JsonTestListener()
    {
        json_file << std::setw(4) << test_results << std::endl;
    }

    void OnTestStart(const testing::TestInfo &test_info) override
    {
        current_test_name = std::string(test_info.test_suite_name()) + "." + test_info.name();
        test_results["tests"].push_back({{"name", current_test_name},
                                         {"status", "RUNNING"},
                                         {"details", json::object()}});
    }

    void OnTestEnd(const testing::TestInfo &test_info) override
    {
        for (auto &test : test_results["tests"])
        {
            if (test["name"] == current_test_name)
            {
                test["status"] = test_info.result()->Passed() ? "PASSED" : "FAILED";
                test["time_ms"] = test_info.result()->elapsed_time();
                break;
            }
        }
    }

    void AddTestDetails(const std::string &test_name, const json &details)
    {
        for (auto &test : test_results["tests"])
        {
            if (test["name"] == test_name)
            {
                test["details"].update(details);
                break;
            }
        }
    }
};

auto *listener = new JsonTestListener("test_tictactoc.json");

class TicTacTocClassificationTest : public ::testing::Test
{
public:
    static Dataset dataset;
    static std::pair<Dataset, Dataset> datasetPair;

protected:
    static void SetUpTestSuite()
    {
        dataset.shuffle();
        datasetPair = dataset.splitDatasetOnTrainAndTest(0.8);
    }

    static boost::multi_array<double, 2> addNoise(const boost::multi_array<double, 2> &data, double noiseLevel)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(-noiseLevel, noiseLevel);

        boost::multi_array<double, 2> noisyData(data);
        auto *data_ptr = noisyData.data();
        const size_t num_elements = noisyData.num_elements();

        for (size_t i = 0; i < num_elements; ++i)
        {
            data_ptr[i] += dist(gen);
        }

        return noisyData;
    }

    static std::unique_ptr<tsk::TSK> createAndTrainModel(int m, int epoch)
    {
        tsk::CMeans cmeans(m, 0, 0.0001);
        cmeans.fit(datasetPair.first.getX());

        auto tskModel = std::make_unique<tsk::TSK>(datasetPair.first.getX().shape()[1], m);
        tskModel->setC(cmeans.getCentroids());
        tskModel->setSigma(cmeans.getSigma());
        learning::HybridAlgorithm hybridAlg(tskModel.get(), datasetPair.first);
        learning::TrainingConfig config;
        hybridAlg.learning(datasetPair.first.getCountVectors(), epoch, 20, config);
        return tskModel;
    }
};

std::string filename = "resource/old/old-tic-tac-toc.csv";
Dataset TicTacTocClassificationTest::dataset = Dataset::readFromCsv(filename);
std::pair<Dataset, Dataset> TicTacTocClassificationTest::datasetPair = dataset.splitDatasetOnTrainAndTest(0.8);

TEST_F(TicTacTocClassificationTest, BasicClassification)
{
    const int m = 8;
    const int epochs = 3;
    const int count_of_iterations = 5;
    json results = json::array();

    for (int i = 0; i < count_of_iterations; ++i)
    {
        dataset.shuffle();

        // Запуск таймера
        auto start = std::chrono::high_resolution_clock::now();
        auto model = createAndTrainModel(m, epochs);
        auto end = std::chrono::high_resolution_clock::now();

        // Вычисление времени обучения
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        auto predictions = model->predict(datasetPair.second.getX());
        double accuracy = metric::Metric::calculateAccuracy(
            datasetPair.second.getD(), predictions, dataset.getClassCount());
        double mse = metric::Metric::calculateMSE(
            datasetPair.second.getD(), predictions, dataset.getClassCount());

        results.push_back({
            {"accuracy", accuracy},
            {"mse", mse},
            {"training_time_ms", duration} // Записываем время обучения
        });

        EXPECT_GT(accuracy, 0.95);
        EXPECT_LT(mse, 0.1);
    }

    listener->AddTestDetails(
        "TicTacTocClassificationTest.BasicClassification",
        {{"parameters", {{"epochs", epochs}, {"clusters", m}}},
         {"results", results}});
}

TEST_F(TicTacTocClassificationTest, NoisyDataClassification)
{
    const std::vector<int> clusterCounts = {6, 8, 10};
    const int max_epochs = 15;
    json noise_results = json::array();
    const std::vector<double> noiseLevels = {0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5};

    for (int m : clusterCounts)
    {
        json cluster_epoch_results = json::array();

        for (int epoch = 10; epoch <= max_epochs; epoch+=1)
        {
            auto model = createAndTrainModel(m, epoch);
            json epoch_noise_results = json::array();

            for (double noise : noiseLevels)
            {
                auto noisyX = addNoise(datasetPair.second.getX(), noise);
                auto predictions = model->predict(noisyX);
                double accuracy = metric::Metric::calculateAccuracy(
                    datasetPair.second.getD(), predictions, dataset.getClassCount());

                epoch_noise_results.push_back({{"noise_level", noise}, {"accuracy", accuracy}});
                EXPECT_GT(accuracy, 0.95);
            }

            cluster_epoch_results.push_back({{"epoch", epoch}, {"clusters", m}, {"noise_tests", epoch_noise_results}});
        }

        noise_results.push_back({{"cluster_count", m}, {"epoch_results", cluster_epoch_results}});
    }

    listener->AddTestDetails(
        "TicTacTocClassificationTest.NoisyDataClassification",
        {{"cluster_experiments", noise_results}});
}

TEST_F(TicTacTocClassificationTest, DifferentClusterCounts)
{
    const std::vector<int> clusterCounts = {4, 6, 8, 10, 12};
    int epochs = 3;
    json cluster_results = json::array();

    for (int m : clusterCounts)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto model = createAndTrainModel(m, epochs);
        auto end = std::chrono::high_resolution_clock::now();

        auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        auto predictions = model->predict(datasetPair.second.getX());
        double accuracy = metric::Metric::calculateAccuracy(
            datasetPair.second.getD(), predictions, dataset.getClassCount());

        cluster_results.push_back({{"clusters", m},
                                   {"accuracy", accuracy},
                                   {"training_time_ms", training_duration}});

        EXPECT_GT(accuracy, 0.95);
    }

    listener->AddTestDetails(
        "TicTacTocClassificationTest.DifferentClusterCounts",
        {{"parameters", {{"epochs", epochs}}},
         {"cluster_tests", cluster_results}});
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    testing::UnitTest &unit_test = *testing::UnitTest::GetInstance();
    testing::TestEventListeners &listeners = unit_test.listeners();

    // Удаляем стандартный вывод
    delete listeners.Release(listeners.default_result_printer());

    // Добавляем наш JSON-лисенер
    listeners.Append(listener);

    return RUN_ALL_TESTS();
}