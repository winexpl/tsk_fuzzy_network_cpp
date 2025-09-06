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

auto *listener = new JsonTestListener("test_noisy.json");

class NoisyTest : public ::testing::Test
{
protected:
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

    static std::unique_ptr<tsk::TSK> createAndTrainModel(auto &datasetPair, int m, int epoch)
    {
        tsk::CMeans cmeans(m, 0, 0.001);
        cmeans.fit(datasetPair.first.getX());

        auto tskModel = std::make_unique<tsk::TSK>(datasetPair.first.getX().shape()[1], m);
        tskModel->setC(cmeans.getCentroids());
        tskModel->setSigma(cmeans.getSigma());

        learning::HybridAlgorithm hybridAlg(tskModel.get(), datasetPair.first);
        learning::TrainingConfig config;
        config.grad_clip = 1;
        hybridAlg.learning(datasetPair.first.getCountVectors(), epoch, 1000, config);
        return tskModel;
    }
};

TEST_F(NoisyTest, IrisDataset)
{
    std::string filename = "/mnt/masha/projects/tsk-fuzzy-network-cpp/resource/irises.csv";
    Dataset dataset(filename);
    auto datasetPair = dataset.splitDatasetOnTrainAndTest(0.7);
    addNoise(datasetPair.first.getX(), 0.1);
    addNoise(datasetPair.second.getX(), 0.1);
    const std::vector<int> clusterCounts = {2};
    int epochs = 5;
    json cluster_results = json::array();

    for (int m : clusterCounts)
    {
        for (size_t i = 0; i < epochs; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto model = createAndTrainModel(datasetPair, m, i);
            auto end = std::chrono::high_resolution_clock::now();

            auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            auto predictions = model->predict(datasetPair.second.getX());
            double accuracy = metric::Metric::calculateAccuracy(
                datasetPair.second.getD(), predictions, dataset.getClassCount());

            cluster_results.push_back({{"clusters", m},
                                       {"epoch", i},
                                       {"accuracy", accuracy},
                                       {"training_time_ms", training_duration}});
        }
    }

    listener->AddTestDetails(
        "NoisyTest.IrisDataset",
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