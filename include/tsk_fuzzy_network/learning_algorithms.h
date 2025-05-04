#ifndef TSK_LEARNING_H
#define TSK_LEARNING_H

#include "tsk_fuzzy_network/tsk.h"
#include "dataset.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace learning
{
    struct HybridAlgorithm;
};

struct learning::HybridAlgorithm
{
    int batchSize;
    tsk::TSK *tsk;
    Dataset dataset;

    HybridAlgorithm(tsk::TSK *tsk, Dataset dataset);
    std::pair<double, double> calculateMetrics() const;
    void learning(int batchSize, int epochCount, int countSecondStepIter = 100, double nu = 0.01);
    void logMetrics(int epoch, const std::string &step, double accuracy, double mse) const;

    private : Eigen::MatrixXd _d;

    double dE(double e, const auto &x, int paramNum, const std::function<double(double, double, double, double)> &dnuFunction);

    double dW(int paramNum, int roleNum, const auto &x, const std::function<double(double, double, double, double)> &dnuFunction);

    void learningTskBatchFirstStep(int startIndex, int endIndex);
    void learningTskBatchSecondStepGPU(int startIndex, int endIndex, double nu);
    void learningTskBatchSecondStep(int startIndex, int endIndex, double nu = 0.001);

    double m(const auto x);

    double l(const auto x, int i);

    int deltaKronecker(int i, int j);

    Eigen::MatrixXd inverse(Eigen::MatrixXd &);
};

#endif