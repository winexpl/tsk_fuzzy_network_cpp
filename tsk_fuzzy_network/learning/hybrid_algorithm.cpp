#include "tsk_fuzzy_network/learning_algorithms.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <math.h>
#include "metric.h"
#include <cuda_runtime.h>
#include "tsk_fuzzy_network/cuda_learning_algorithm.h"
#include <omp.h>

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
                learningTskBatchSecondStepGPU(startIndex, endIndex, nu);
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

            double dEdC = dE(error, currentTrainVector, paramNum,
                             [](double x, double sigma, double c, double b)
                             {
                                 const double t = (x - c) / sigma;
                                 const double t_pow = std::pow(t, 2 * b - 1);
                                 return 2.0 * b / sigma * t_pow / std::pow(1 + std::pow(t, 2 * b), 2);
                             });

            double dEdSigma = dE(error, currentTrainVector, paramNum,
                                 [](double x, double sigma, double c, double b)
                                 {
                                     const double t = (x - c) / sigma;
                                     const double t_pow = std::pow(t, 2 * b);
                                     return 2.0 * b / sigma * t_pow / std::pow(1 + t_pow, 2);
                                 });

            double dEdB = dE(error, currentTrainVector, paramNum,
                             [](double x, double sigma, double c, double b)
                             {
                                 const double t = (x - c) / sigma;
                                 const double t_pow = std::pow(t, 2 * b);
                                 return -2.0 * t_pow * std::log(std::abs(t)) / std::pow(1 + t_pow, 2);
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

double learning::HybridAlgorithm::dE(double e, const auto &x,
    int paramNum, const std::function<double(double, double, double, double)> &dnuFunction) {
    /**
    * e - ошибка
    * p - двумерный массив параметров
    * x - текущий вектор параметров
    * paramNum - номер изменяемого параметра
    * dnu - функция dW (реализуется через std::function или другую функцию)
    */
    double sum_p = 0.0;
    const auto &p = tsk->getP();
    for (size_t roleNum = 0; roleNum < p.shape()[0]; ++roleNum) {
        double temp = p[roleNum][0];
        for (size_t xIndexInPolynom = 1; xIndexInPolynom < p.shape()[1]; ++xIndexInPolynom) {
            temp += p[roleNum][xIndexInPolynom] * x[xIndexInPolynom - 1];
        }
        double dw = dW(paramNum, roleNum, x, dnuFunction);
        temp *= dw;
        sum_p += temp;
    }
    return e * sum_p;
}

double learning::HybridAlgorithm::dW(int paramNum, int roleNum, const auto& x, const std::function<double(double, double, double, double)>& dnuFunction)
{
    /**
     * paramNum - [cij] (=i*j) i-правило, j-параметр входного вектора
     * roleNum - [wk] (=k)
     */
    int vectorLength = tsk->getN(); // n
    int countOfRole = tsk->getM(); // m

    int paramRoleNum = paramNum % countOfRole; // i for changedParam
    int paramVectorIndexNum = paramNum / countOfRole; // j for changedParam


    double m_res = m(x);
    double l_res = l(x, roleNum);


    double res = (deltaKronecker(roleNum, paramRoleNum) * m_res - l_res) / std::pow(m_res, 2);

    double multiple = 1;
    for(int j = 0; j < vectorLength; ++j)
    {
        if(paramVectorIndexNum != j)
            multiple *= tsk->applyFuzzyFunction(x[j], tsk->getSigma()[j*countOfRole+paramRoleNum], tsk->getC()[j*countOfRole+paramRoleNum], tsk->getB()[j*countOfRole+paramRoleNum]);
    }
    res *= multiple;
    double dnu = dnuFunction(x[paramVectorIndexNum], tsk->getSigma()[paramNum], tsk->getC()[paramNum], tsk->getB()[paramNum]);
    res *= dnu;
    return res;
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
    auto &p = tsk->getP();
    const int dim = dataset.getCountVectors();
    const auto &vectors = dataset.getX();
    const auto &targets = dataset.getD();
    auto &c = tsk->getC();
    auto &sigma = tsk->getSigma();
    auto &b = tsk->getB();

    learningKernelWrapper(vectors, targets, startIndex, endIndex, nu, n, m, c, sigma, b, p, tsk);
}
