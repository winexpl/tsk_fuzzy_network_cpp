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
#include "fuzzy_api.h"

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

auto *listener = new JsonTestListener("test_performance.json");

class PerformanceTest : public ::testing::Test
{
protected:
    static std::unique_ptr<tsk::TSK> createAndTrainModel(auto& datasetPair, int m, int epoch, bool use_gpu)
    {
        tsk::CMeans cmeans(m, 0, 0.001);
        cmeans.fit(datasetPair.first.getX());

        auto tskModel = std::make_unique<tsk::TSK>(datasetPair.first.getX().shape()[1], m);
        tskModel->setC(cmeans.getCentroids());
        tskModel->setSigma(cmeans.getSigma());

        learning::HybridAlgorithm hybridAlg(tskModel.get(), datasetPair.first);
        learning::TrainingConfig config;
        config.grad_clip = 1;
        hybridAlg.learning(datasetPair.first.getCountVectors(), epoch, 50, config, use_gpu);
        return tskModel;
    }
};

TEST_F(PerformanceTest, IrisDatasetCPUvsGPU)
{
    std::string filename = "/mnt/masha/projects/tsk-fuzzy-network-cpp/resource/irises.csv";
    Dataset dataset(filename);
    auto datasetPair = dataset.splitDatasetOnTrainAndTest(0.7);
    const int m = 2; // Оптимальное количество кластеров для Iris
    const int epochs = 20; // Увеличиваем для более точного измерения производительности
    
    json performance_results = json::array();

    // Тестирование CPU
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto model = createAndTrainModel(datasetPair, m, epochs, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        auto predictions = model->predict(datasetPair.second.getX());
        double accuracy = metric::Metric::calculateAccuracy(
            datasetPair.second.getD(), predictions, dataset.getClassCount());

        performance_results.push_back({
            {"device", "CPU"},
            {"accuracy", accuracy},
            {"training_time_ms", duration},
            {"throughput_samples_per_sec", datasetPair.first.getCountVectors() * epochs * 50.0 / duration}
        });
    }

    // Тестирование GPU
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto model = createAndTrainModel(datasetPair, m, epochs, true);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        auto predictions = model->predict(datasetPair.second.getX());
        double accuracy = metric::Metric::calculateAccuracy(
            datasetPair.second.getD(), predictions, dataset.getClassCount());

        performance_results.push_back({
            {"device", "GPU"},
            {"accuracy", accuracy},
            {"training_time_ms", duration},
            {"throughput_samples_per_sec", datasetPair.first.getCountVectors() * epochs * 50.0 / duration}
        });
    }

    listener->AddTestDetails(
        "PerformanceTest.IrisDatasetCPUvsGPU",
        {{"parameters", {{"epochs", epochs}, {"clusters", m}}},
         {"performance", performance_results}});
}

TEST_F(PerformanceTest, LargeDatasetCPUvsGPU)
{
    std::string filename = "/mnt/masha/projects/tsk-fuzzy-network-cpp/resource/irises.csv";
    Dataset dataset(filename);
    auto datasetPair = dataset.splitDatasetOnTrainAndTest(0.7);
    const int m = 2; // Больше кластеров для сложного датасета
    const int epochs = 50;
    
    json performance_results = json::array();

    // CPU
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto model = createAndTrainModel(datasetPair, m, epochs, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        performance_results.push_back({
            {"device", "CPU"},
            {"training_time_ms", duration},
            {"throughput_samples_per_sec", datasetPair.first.getCountVectors() * epochs * 50.0 / duration}
        });
    }

    // GPU
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto model = createAndTrainModel(datasetPair, m, epochs, true);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        performance_results.push_back({
            {"device", "GPU"},
            {"training_time_ms", duration},
            {"throughput_samples_per_sec", datasetPair.first.getCountVectors() * epochs * 50.0 / duration}
        });
    }

    listener->AddTestDetails(
        "PerformanceTest.LargeDatasetCPUvsGPU",
        {{"parameters", {{"epochs", epochs}, {"clusters", m}}},
         {"performance", performance_results}});
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