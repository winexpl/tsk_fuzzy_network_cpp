#include "tsk_fuzzy_network/learning_algorithms.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <math.h>
#include "metric.h"
#include <cuda_runtime.h>
#include "tsk_fuzzy_network/cuda_learning_algorithm.h"


Eigen::MatrixXd boostMultiArrayToEigenMatrix(const boost::multi_array<double, 2> &boostMultiArray)
{
    const size_t rows = boostMultiArray.shape()[0];
    const size_t cols = boostMultiArray.shape()[1];
    Eigen::MatrixXd eigenMatrix(rows, cols);

    for (size_t i = 0; i < rows; ++i)
    {
        eigenMatrix.row(i) = Eigen::Map<const Eigen::RowVectorXd>(boostMultiArray[i].origin(), cols);
    }
    return eigenMatrix;
}

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<double> &vec)
{
    Eigen::MatrixXd matrix(vec.size(), 1);
    for (int i = 0; i < vec.size(); ++i)
    {
        matrix(i, 0) = vec[i];
    }
    return matrix;
}

learning::HybridAlgorithm::HybridAlgorithm(tsk::TSK *tsk, Dataset dataset)
    : tsk(tsk), dataset(dataset), _d(vectorToEigenMatrix(dataset.getD())) {}

void learning::HybridAlgorithm::learningTskBatchFirstStep(int startIndex, int endIndex)
{
    int n = tsk->getN();
    int m = tsk->getM();
    auto x = dataset.getX();

    boost::multi_array<double, 2> oldP = tsk->getP();
    boost::multi_array<double, 2> p(boost::extents[endIndex - startIndex][(n + 1) * m]);

    for (int i = 0; i < endIndex - startIndex; i++)
    {
        auto xi = x[startIndex + i];
        auto w = tsk->getRoleMultipleLayerOut(xi);
        double wSum = 0.0;
        for (int k = 0; k < m; ++k)
        {
            wSum += w[k];
        }
        for (int j = 0; j < m; j++)
        {
            p[i][j * (n + 1)] = w[j] / wSum;
            for (int k = 1; k < n + 1; k++)
            {
                p[i][j * (n + 1) + k] = w[j] / wSum * xi[k - 1];
            }
        }
    }
    Eigen::MatrixXd dEigen = _d.block(startIndex, 0, endIndex - startIndex, 1);
    Eigen::MatrixXd oldPEigen = boostMultiArrayToEigenMatrix(p);
    Eigen::MatrixXd oldPInverseEigen = inverse(oldPEigen);
    Eigen::MatrixXd newPEigen = oldPInverseEigen * dEigen;
    tsk->updateP(newPEigen);
}

void learning::HybridAlgorithm::learning(int batchSize, int epochCount, int countSecondStepIter, double nu)
{
    const int countOfLearningVectors = dataset.getCountVectors();
    const int classCount = dataset.getClassCount();

    for (int epoch = 0; epoch < epochCount; ++epoch)
    {
        for (int startIndex = 0; startIndex < countOfLearningVectors; startIndex += batchSize)
        {
            const int endIndex = std::min(startIndex + batchSize, countOfLearningVectors);

            learningTskBatchFirstStep(startIndex, endIndex);
            auto [accuracy1, mse1] = calculateMetrics();
            logMetrics(epoch + 1, "first step", accuracy1, mse1);

            for (int j = 0; j < countSecondStepIter; ++j)
            {
                learningTskBatchSecondStep(startIndex, endIndex, nu);
            }

            auto [accuracy2, mse2] = calculateMetrics();
            logMetrics(epoch + 1, "second step", accuracy2, mse2);

            if (std::isnan(mse2))
            {
                tsk->clearFuzzyLayer();
            }
        }
    }
}

std::pair<double, double> learning::HybridAlgorithm::calculateMetrics() const
{
    const auto predictedValues = tsk->predict(dataset.getX());
    const double accuracy = metric::Metric::calculateAccuracy(
        dataset.getD(), predictedValues, dataset.getClassCount());
    const double mse = metric::Metric::calculateMSE(
        dataset.getD(), predictedValues, dataset.getClassCount());
    return {accuracy, mse};
}

void learning::HybridAlgorithm::logMetrics(int epoch, const std::string &step,
                                           double accuracy, double mse) const
{
    std::ostringstream logMessage;
    logMessage << "Epoch " << epoch << " after " << step
               << " - Accuracy: " << accuracy
               << ", MSE: " << mse;
    auto &logger = Logger::getInstance();
    logger.logInfo(logMessage.str());
}

void learning::HybridAlgorithm::learningTskBatchSecondStep(int startIndex, int endIndex, double nu)
{
    const int n = tsk->getN();
    const int m = tsk->getM();
    const int totalParams = n * m;
    const auto &p = tsk->getP();
    const auto &vectors = dataset.getX();
    const auto &targets = dataset.getD();

    const double nuC = nu;
    const double nuSigma = nu;
    const double nuB = nu;

    for (int i = startIndex; i < endIndex; ++i)
    {
        const auto &currentTrainVector = vectors[i];
        const double target = targets[i];
        const double output = tsk->predict1(currentTrainVector);
        const double error = output - target;

        for (int paramNum = 0; paramNum < totalParams; ++paramNum)
        {
            double currentC = tsk->getC()[paramNum];
            double currentSigma = tsk->getSigma()[paramNum];
            double currentB = tsk->getB()[paramNum];

            double dEdC = dE(error, p, currentTrainVector, paramNum,
                             [](double x, double sigma, double c, double b)
                             {
                                 const double t = (x - c) / sigma;
                                 const double t_pow = std::pow(t, 2 * b - 1);
                                 return 2.0 * b / sigma * t_pow / std::pow(1 + t * t, 2);
                             });

            double dEdSigma = dE(error, p, currentTrainVector, paramNum,
                                 [](double x, double sigma, double c, double b)
                                 {
                                     const double t = (x - c) / sigma;
                                     const double t_pow = std::pow(t, 2 * b);
                                     return 2.0 * b / sigma * t_pow / std::pow(1 + t * t, 2);
                                 });

            double dEdB = dE(error, p, currentTrainVector, paramNum,
                             [](double x, double sigma, double c, double b)
                             {
                                 const double t = (x - c) / sigma;
                                 if (std::abs(t) < 1e-10)
                                     return 0.0;
                                 const double t_pow = std::pow(t, 2 * b);
                                 return -2.0 * t_pow * std::log(std::abs(t)) / std::pow(1 + t * t, 2);
                             });

            auto safeUpdate = [](double &param, double gradient, double learningRate)
            {
                if (!std::isnan(gradient) && !std::isinf(gradient))
                {
                    param -= learningRate * gradient;
                }
            };

            safeUpdate(currentC, dEdC, nuC);
            safeUpdate(currentSigma, dEdSigma, nuSigma);
            safeUpdate(currentB, dEdB, nuB);

            tsk->setSigma(currentSigma, paramNum);
            tsk->setC(currentC, paramNum);
            tsk->setB(std::round(currentB), paramNum);
        }
    }
}

double learning::HybridAlgorithm::dE(double error, const boost::multi_array<double, 2> &params,
                                     const auto &inputVector, int paramNum, const std::function<double(double, double, double, double)> &gradientFunc)
{
    /**
     * e - ошибка
     * p - двумерный массив параметров
     * x - текущий вектор параметров
     * paramNum - номер изменяемого параметра
     * dnu - функция dW (реализуется через std::function или другую функцию)
     */
    const size_t numRoles = params.shape()[0];
    const size_t paramSize = params.shape()[1];
    double weightedSum = 0.0;

    const int inputOffset = paramNum / tsk->getM();
    const double x_val = inputVector[inputOffset];

    for (size_t roleIdx = 0; roleIdx < numRoles; ++roleIdx)
    {
        double linearCombination = params[roleIdx][0];

        for (size_t paramIdx = 1; paramIdx < paramSize; ++paramIdx)
        {
            linearCombination += params[roleIdx][paramIdx] * inputVector[paramIdx - 1];
        }

        const double weightDerivative = dW(paramNum, roleIdx, inputVector, gradientFunc);
        weightedSum += linearCombination * weightDerivative;
    }
    return error * weightedSum;
}

double learning::HybridAlgorithm::dW(int paramNum, int roleNum, const auto &inputVector,
                                     const std::function<double(double, double, double, double)> &gradientFunc)
{
    const int N = tsk->getN();
    const int M = tsk->getM();
    const int paramRoleIdx = paramNum % M;
    const int featureIdx = paramNum / M;

    const double m_val = m(inputVector);
    const double l_val = l(inputVector, roleNum);
    const double denominator = m_val * m_val;

    const double numerator = (deltaKronecker(roleNum, paramRoleIdx) * m_val - l_val);

    double membershipProduct = 1.0;
    const double *sigma = tsk->getSigma().data();
    const double *c = tsk->getC().data();
    const double *b = tsk->getB().data();

    for (int j = 0; j < N; ++j)
    {
        if (j != featureIdx)
        {
            const int idx = j * M + paramRoleIdx;
            membershipProduct *= tsk->applyFuzzyFunction(
                inputVector[j],
                sigma[idx],
                c[idx],
                b[idx]);
        }
    }

    const double x = inputVector[featureIdx];
    const double currentSigma = sigma[paramNum];
    const double currentC = c[paramNum];
    const double currentB = b[paramNum];
    const double derivative = gradientFunc(x, currentSigma, currentC, currentB);

    return (numerator / denominator) * membershipProduct * derivative;
}

double learning::HybridAlgorithm::m(const auto x)
{
    double m = tsk->getM();
    double n = tsk->getN();
    double res = 0;
    for (int i = 0; i < m; ++i)
    {
        double temp = 1.0;
        for (int j = 0; j < n; ++j)
        {
            temp *= tsk->applyFuzzyFunction(x[j], tsk->getSigma()[j * m + i], tsk->getC()[j * m + i], tsk->getB()[j * m + i]);
        }
        res += temp;
    }
    return res;
}

double learning::HybridAlgorithm::l(const auto x, int i)
{
    double m = tsk->getM();
    double n = tsk->getN();
    double temp = 1.0;
    for (int j = 0; j < n; ++j)
    {
        temp *= tsk->applyFuzzyFunction(x[j], tsk->getSigma()[j * m + i], tsk->getC()[j * m + i], tsk->getB()[j * m + i]);
    }
    return temp;
}

int learning::HybridAlgorithm::deltaKronecker(int i, int j)
{
    return i == j ? 1 : 0;
}

Eigen::MatrixXd learning::HybridAlgorithm::inverse(Eigen::MatrixXd &p)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(p, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto &singularValues = svd.singularValues();
    Eigen::MatrixXd singularValuesInv = singularValues.array().inverse().matrix().asDiagonal();
    Eigen::MatrixXd pseudoInverseM = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
    boost::multi_array<double, 2> pseudoInverse(boost::extents[pseudoInverseM.rows()][pseudoInverseM.cols()]);
    return pseudoInverseM;
}

void learning::HybridAlgorithm::learningTskBatchSecondStepGPU(int startIndex, int endIndex, double nu)
{
    const int n = tsk->getN();
    const int m = tsk->getM();
    const int totalParams = n * m;
    const auto &p = tsk->getP();
    const auto &vectors = dataset.getX();
    const auto &targets = dataset.getD();
    const int num_features = vectors[0].size();
    const int numRoles = p.shape()[0];
    const int paramSize = p.shape()[1];

    std::vector<double> initial_c = tsk->getC();
    std::vector<double> initial_sigma = tsk->getSigma();
    std::vector<double> initial_b = tsk->getB();

    double *d_c, *d_sigma, *d_b;
    double *d_vectors, *d_targets, *d_params;

    cudaMalloc(&d_c, totalParams * sizeof(double));
    cudaMalloc(&d_sigma, totalParams * sizeof(double));
    cudaMalloc(&d_b, totalParams * sizeof(double));
    cudaMalloc(&d_vectors, vectors.size() * num_features * sizeof(double));
    cudaMalloc(&d_targets, targets.size() * sizeof(double));
    cudaMalloc(&d_params, numRoles * paramSize * sizeof(double));

    cudaMemcpy(d_c, tsk->getC().data(), totalParams * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, tsk->getSigma().data(), totalParams * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, tsk->getB().data(), totalParams * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> flat_vectors;
    for (const auto &v : vectors)
    {
        flat_vectors.insert(flat_vectors.end(), v.begin(), v.end());
    }
    cudaMemcpy(d_vectors, flat_vectors.data(), flat_vectors.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> flat_params(numRoles * paramSize);
    for (int i = 0; i < numRoles; ++i)
    {
        for (int j = 0; j < paramSize; ++j)
        {
            flat_params[i * paramSize + j] = p[i][j];
        }
    }
    cudaMemcpy(d_params, flat_params.data(), flat_params.size() * sizeof(double), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = endIndex - startIndex;

    learningKernelWrapper(gridSize, gridSize, d_c, d_sigma, d_b,
        d_vectors, d_targets, d_params,
        numRoles, paramSize, n, m, num_features,
        startIndex, endIndex, nu, nu, nu);

    cudaMemcpy(tsk->getC().data(), d_c, totalParams * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsk->getSigma().data(), d_sigma, totalParams * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsk->getB().data(), d_b, totalParams * sizeof(double), cudaMemcpyDeviceToHost);

    bool c_changed = false, sigma_changed = false, b_changed = false;
    const double epsilon = 1e-10; // Порог для сравнения чисел с плавающей точкой
    
    for (int i = 0; i < totalParams; ++i) {
        if (fabs(initial_c[i] - tsk->getC()[i]) > epsilon) c_changed = true;
        if (fabs(initial_sigma[i] - tsk->getSigma()[i]) > epsilon) sigma_changed = true;
        if (fabs(initial_b[i] - tsk->getB()[i]) > epsilon) b_changed = true;
    }

    cudaFree(d_c);
    cudaFree(d_sigma);
    cudaFree(d_b);
    cudaFree(d_vectors);
    cudaFree(d_targets);
    cudaFree(d_params);
}