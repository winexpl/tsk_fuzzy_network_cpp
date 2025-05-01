#include "tsk_fuzzy_network/learning_algorithms.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <math.h>
#include "metric.h"

Eigen::MatrixXd boostMultiArrayToEigenMatrix(const boost::multi_array<double, 2>& boostMultiArray)
{
    // std::cout << "boostMultiArrayToEigenMatrix " << "boostMultiArray.size()=" << boostMultiArray.size() << std::endl;
    auto eigenMatrix = Eigen::MatrixXd(boostMultiArray.shape()[0], boostMultiArray.shape()[1]);
    for (size_t i = 0; i < boostMultiArray.shape()[0]; ++i) {
        for (size_t j = 0; j < boostMultiArray.shape()[1]; ++j) {
            // std::cout << "boostMultiArrayToEigenMatrix " << "boostMultiArray[" << i << "][" << j << "]=" << boostMultiArray.size() << std::endl;
            eigenMatrix(i, j) = boostMultiArray[i][j];
            // std::cout << "eigenMatrix(i, j) " << eigenMatrix(i, j) << " boostMultiArray[i][j] " << boostMultiArray[i][j] << std::endl;
        }
    }
    return eigenMatrix;
}

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<double>& vec)
{
    Eigen::MatrixXd matrix(vec.size(),1);
    for (int i = 0; i < vec.size(); ++i) {
        matrix(i, 0) = vec[i];
    }
    return matrix;
}

void learning::HybridAlgorithm::learning(int batchSize, int epochCount, int countSecondStepIter, double nu)
{
    int m = tsk->getM();
    int n = tsk->getN();
    int countOfLearningVectors = dataset.getCountVectors();
    
    for(int i = 0; i < epochCount; ++i)
    {
        for(int startIndex = 0; startIndex < countOfLearningVectors; startIndex += batchSize)
        {
            int endIndex = startIndex + batchSize;
            if(endIndex > countOfLearningVectors) endIndex = countOfLearningVectors;

            this->learningTskBatchFirstStep(startIndex, endIndex);

            std::vector<double> predictedValues = tsk->predict(dataset.getX());

            double accuracy = metric::Metric::calculateAccuracy(dataset.getD(), predictedValues, dataset.getClassCount());
            double mse = metric::Metric::calculateMSE(dataset.getD(), predictedValues, dataset.getClassCount());
            std::cout << std::endl << "epoch " << i+1 << " after first step "
                << "accuracy: " << accuracy << " "
                << "mse: " << mse << std::endl;
            
            // многократное повторение 2 этапа
            for(int j = 0; j < countSecondStepIter; ++j)
            {
                this->learningTskBatchSecondStep(startIndex, endIndex, nu);
            }

            predictedValues = tsk->predict(dataset.getX());
            accuracy = metric::Metric::calculateAccuracy(dataset.getD(), predictedValues, dataset.getClassCount());
            mse = metric::Metric::calculateMSE(dataset.getD(), predictedValues, dataset.getClassCount());
            std::cout << std::endl << "epoch " << i+1 << " after second step "
                << "accuracy: " << accuracy << " "
                << "mse: " << mse << std::endl;
            // сброс если есть nan
            if(std::isnan(mse)) tsk->clearFuzzyLayer();
        }
    }
}

learning::HybridAlgorithm::HybridAlgorithm(tsk::TSK* tsk, Dataset &dataset) :
tsk{tsk}, dataset{dataset}
{
    _d = vectorToEigenMatrix(dataset.getD());
};

void learning::HybridAlgorithm::learningTskBatchFirstStep(int startIndex, int endIndex)
{
    int n = tsk->getN();
    int m = tsk->getM();
    auto x = dataset.getX();

    boost::multi_array<double, 2> oldP = tsk->getP();
    boost::multi_array<double, 2> p(boost::extents[endIndex - startIndex][(n+1)*m]);

    for(int i = 0; i < endIndex-startIndex; i++)
    {
        auto xi = x[startIndex + i];
        auto w = tsk->getRoleMultipleLayerOut(xi);
        double wSum = 0.0;
        for(int k = 0; k < m; ++k)
        {
            wSum += w[k];
        }
        for(int j = 0; j < m; j++)
        {
            p[i][j*(n+1)] = w[j]/wSum;
            for(int k = 1; k < n+1; k++)
            {
                p[i][j*(n+1)+k] = w[j]/wSum * xi[k-1];
            }
        }
    }
    Eigen::MatrixXd dEigen = _d.block(startIndex, 0, endIndex-startIndex, 1);
    Eigen::MatrixXd oldPEigen = boostMultiArrayToEigenMatrix(p);
    Eigen::MatrixXd oldPInverseEigen = inverse(oldPEigen);
    Eigen::MatrixXd newPEigen = oldPInverseEigen * dEigen;
    tsk->updateP(newPEigen);
}

void learning::HybridAlgorithm::learningTskBatchSecondStep(int startIndex, int endIndex, double nu) {
    int n = tsk->getN();
    int m = tsk->getM();

    const auto p = tsk->getP();
    double nuC = nu;
    double nuSigma = nu;
    double nuB = nu;

    auto vectors = dataset.getX();

    for(int i = startIndex; i < endIndex; i++)
    {
        auto currentTrainVector = vectors[i];
        double d = dataset.getD()[i];

        for(int paramNum = 0; paramNum < n*m; ++paramNum)
        {
            const double out = tsk->predict1(currentTrainVector);
            double e = (out - d);

            double newC = tsk->getC()[paramNum];
            double newSigma = tsk->getSigma()[paramNum];
            double newB = tsk->getB()[paramNum];

            double dEdC = dE(e, p, currentTrainVector, paramNum,
            [](double x, double sigma, double c, double b) -> double {
                double dnu = 2.0*b/sigma * std::pow( (x-c)/sigma, 2*b-1.0);
                dnu /= std::pow( 1 + std::pow( (x-c)/sigma, 2*b ), 2);
                return dnu;
            });

            double dEdSigma = dE(e, p, currentTrainVector, paramNum,
            [](double x, double sigma, double c, double b) -> double {
                double dnu = 2.0*b/sigma * std::pow( (x-c)/sigma, 2.0*b);
                dnu /= std::pow( 1.0+std::pow( (x-c)/sigma, 2*b ), 2);
                return dnu;
            });

            double dEdB = dE(e, p, currentTrainVector, paramNum,
            [](double x, double sigma, double c, double b) -> double {
                // FIXME potentially error in the std::log()!
                double dnu = -2.0 * std::pow( (x-c)/sigma, 2.0*b ) * std::log( std::abs((x-c)/sigma) );
                dnu /= std::pow( 1.0+std::pow( (x-c)/sigma, 2.0*b ), 2);
                return dnu;
            });
            
            if( !std::isnan(dEdB) && !std::isinf(dEdB)) newB -= nuB * dEdB;
            if( !std::isnan(dEdC) && !std::isinf(dEdC)) newC -= nuC * dEdC;
            if( !std::isnan(dEdSigma) && !std::isinf(dEdSigma)) newSigma -= nuSigma * dEdSigma;

            tsk->setSigma(newSigma, paramNum);
            tsk->setC(newC, paramNum);
            tsk->setB(std::round(newB), paramNum);
        }
    }
}

double learning::HybridAlgorithm::dE(double e, const  boost::multi_array<double, 2> &p, auto x,
    int paramNum, std::function<double(double, double, double, double)> dnuFunction) {
    /**
    * e - ошибка
    * p - двумерный массив параметров
    * x - текущий вектор параметров
    * paramNum - номер изменяемого параметра
    * dnu - функция dW (реализуется через std::function или другую функцию)
    */
    double sum_p = 0.0;

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

double learning::HybridAlgorithm::dW(int paramNum, int roleNum, auto x, std::function<double(double, double, double, double)> dnuFunction)
{
    /**
     * paramNum - [cij] (=i*j) i-правило, j-параметр входного вектора
     * roleNum - [wk] (=k)
     * dnuFuction - производная для параметра (c, b, sigma) ф. Гаусса
     */
    int vectorLength = tsk->getN(); // n
    int countOfRole = tsk->getM(); // m

    int paramRoleNum = paramNum % countOfRole; // i for changed param
    int paramVectorIndexNum = paramNum / countOfRole; // j for changed param

    double m_res = m(x);
    double l_res = l(x, roleNum);


    double res =  (deltaKronecker(roleNum, paramRoleNum) * m_res - l_res) / std::pow(m_res, 2);

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


double learning::HybridAlgorithm::m(auto x) {
    double m = tsk->getM();
    double n = tsk->getN();
    double res = 0;
    for(int i = 0; i < m; ++i)
    {
        double temp = 1.0;
        for(int j = 0; j < n; ++j)
        {
            temp *= tsk->applyFuzzyFunction(x[j], tsk->getSigma()[j*m+i], tsk->getC()[j*m+i], tsk->getB()[j*m+i]);
        }
        res += temp;
    }
    return res;
}

double learning::HybridAlgorithm::l(auto x, int i) {
    double m = tsk->getM();
    double n = tsk->getN();
    double temp = 1.0;
    for(int j = 0; j < n; ++j)
    {
        temp *= tsk->applyFuzzyFunction(x[j], tsk->getSigma()[j*m+i], tsk->getC()[j*m+i], tsk->getB()[j*m+i]);
    }
    return temp;
}

int learning::HybridAlgorithm::deltaKronecker(int i, int j) {
    return i==j? 1:0;
}

Eigen::MatrixXd learning::HybridAlgorithm::inverse(Eigen::MatrixXd &p)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(p, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& singularValues = svd.singularValues();
    Eigen::MatrixXd singularValuesInv = singularValues.array().inverse().matrix().asDiagonal();
    Eigen::MatrixXd pseudoInverseM = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
    boost::multi_array<double, 2> pseudoInverse(boost::extents[pseudoInverseM.rows()][pseudoInverseM.cols()]);
    return pseudoInverseM;
}
