#ifndef TSK_LEARNING_H
#define TSK_LEARNING_H

#include "tsk_fuzzy_network/tsk.h"
#include <eigen3/Eigen/SVD>
#include "dataset.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include "metric.h"
#include <optional>

namespace learning
{

    struct TrainingConfig
    {
        double nu_c = 0.05;
        double nu_sigma = 0.01;
        double nu_b = 0.005;
        double momentum = 0.9;
        double grad_clip = 2;
    };
    using CallbackTrain = std::function<bool(metric::Metric metric)>;
    using CallbackGetMetrics2Step = std::function<void(metric::Metric metric)>;
    using Train2StepFunction = std::function<void(int startIndex, int endIndex,
                                                  learning::TrainingConfig config, learning::CallbackGetMetrics2Step callbackGetMetrics2Step)>;

    struct HybridAlgorithm;

};

struct learning::HybridAlgorithm
{
    int batchSize;
    tsk::TSK *tsk;
    Dataset dataset;

    HybridAlgorithm(tsk::TSK *tsk, Dataset dataset);
    metric::Metric calculateMetrics() const;
    void logMetrics(int epoch, const std::string &step, metric::Metric &metric) const;

    void learning(int batchSize, int epochCount, int countSecondStepIter, learning::TrainingConfig config,
                  bool isGpu = true,
                  std::optional<CallbackTrain> callbackTrain = std::nullopt,
                  std::optional<CallbackGetMetrics2Step> callbackGetMetrics2Step = std::nullopt, double error = 0);

private:
    Eigen::MatrixXd _d;

    double dE(double e, const auto &x, int paramNum, const std::function<double(double, double, double, double)> &dnuFunction);

    double dW(int paramNum, int roleNum, const auto &x, const std::function<double(double, double, double, double)> &dnuFunction);

    void learningTskBatchFirstStep(int startIndex, int endIndex);

    void learningTskBatchSecondStepGPU(int startIndex, int endIndex, learning::TrainingConfig config,
                                       std::optional<CallbackGetMetrics2Step> callbackGetMetrics2Step = std::nullopt, double error = 0);
    void learningTskBatchSecondStep(int startIndex, int endIndex, learning::TrainingConfig config,
                                    std::optional<CallbackGetMetrics2Step> callbackGetMetrics2Step = std::nullopt);

    double m(const auto x);

    double l(const auto x, int i);

    int deltaKronecker(int i, int j);

    Eigen::MatrixXd inverse(Eigen::MatrixXd &);
};

#endif