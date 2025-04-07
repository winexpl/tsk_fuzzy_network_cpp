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

    int countOfLearningVectors = dataset.getCountVectors();
    
    for(int i = 0; i < epochCount; ++i)
    {
        for(int startIndex = 0; startIndex < 1; startIndex += batchSize)
        {
            int endIndex = startIndex + batchSize;
            if(endIndex > countOfLearningVectors) endIndex = countOfLearningVectors - 1;
            this->learningTskBatchFirstStep(startIndex, endIndex);
            std::vector<double> predictedValues = tsk->predict(dataset.getX());
            std::cout << ".";
            std::cout << std::endl << "epoch " << i << "after first step "
                << "accuracy: " << metric::Metric::calculateAccuracy(dataset.getD(), predictedValues, dataset.classesCount) << std::endl
                << "mse: " << metric::Metric::calculateMSE(dataset.getD(), predictedValues, dataset.classesCount) << std::endl;
            for(int j = 0; j < countSecondStepIter; ++j)
            {
                this->learningTskBatchSecondStep(startIndex, endIndex, nu);
            }
            predictedValues = tsk->predict(dataset.getX());
            double accuracy = metric::Metric::calculateAccuracy(dataset.getD(), predictedValues, dataset.classesCount);
            double mse = metric::Metric::calculateMSE(dataset.getD(), predictedValues, dataset.classesCount);
            std::cout << std::endl << "epoch " << i << "after second step "
                << "accuracy: " << accuracy << std::endl
                << "mse: " << mse << std::endl;
            if(std::isnan(mse)) tsk->clearFuzzyLayer();
        }
    }

}

learning::HybridAlgorithm::HybridAlgorithm(std::shared_ptr<tsk::TSK> tsk, Dataset &dataset) :
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
        // double wSum = std::accumulate(w.cbegin(), w.cend(), 0.0);
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
    for(int i = startIndex; i < endIndex; i++)
    {
        std::vector<double> oldC = tsk->getC();
        std::vector<double> oldSigma = tsk->getSigma();
        std::vector<double> oldB = tsk->getB();
        // this->oldC=oldC;
        // this->oldB=oldB;
        // this->oldSigma=oldSigma;
        std::vector<double> newC(oldC);
        std::vector<double> newSigma(oldSigma);
        std::vector<double> newB(oldB);


        for(int paramNum = 0; paramNum < n*m; ++paramNum)
        {
            oldC = tsk->getC();
            oldSigma = tsk->getSigma();
            oldB = tsk->getB();
            double d = dataset.getD()[i];
            auto currentTrainVector = dataset.getX()[i];
            double out = tsk->predict1(currentTrainVector);
            double e = (out - d);

            newC[paramNum] = oldC[paramNum];
            newSigma[paramNum] = oldSigma[paramNum];
            newB[paramNum] = oldB[paramNum];

            double dEdC = dE(e, p, currentTrainVector, paramNum,
            [](double x, double sigma, double c, double b) -> double {
                double dnu = 2.0*b/sigma * std::pow( (x-c)/sigma, 2*(int)b-1.0);
                dnu /= std::pow( 1 + std::pow( (x-c)/sigma, 2*(int)b ), 2);
                return dnu;
            });

            double dEdSigma = dE(e, p, currentTrainVector, paramNum,
            [](double x, double sigma, double c, double b) -> double {
                double dnu = 2.0*b/sigma * std::pow( (x-c)/sigma, 2.0*(int)b);
                dnu /= std::pow( 1.0+std::pow( (x-c)/sigma, 2*(int)b ), 2);
                return dnu;
            });

            double dEdB = dE(e, p, currentTrainVector, paramNum,
            [](double x, double sigma, double c, double b) -> double {
                // FIXME potentially error in the std::log()!
                double dnu = -2.0 * std::pow( (x-c)/sigma, 2.0*(int)b ) * std::log( (x-c)/sigma );
                dnu /= std::pow( 1.0+std::pow( (x-c)/sigma, 2.0*(int)b ), 2);
                return dnu;
            });
            
            if( !std::isnan(dEdB) && !std::isinf(dEdB)) newB[paramNum] -= nuB * dEdB;
            if( !std::isnan(dEdC) && !std::isinf(dEdC)) newC[paramNum] -= nuC * dEdC;
            if( !std::isnan(dEdSigma) && !std::isinf(dEdSigma)) newSigma[paramNum] -= nuSigma * dEdSigma;
            // tsk->setSigma(newSigma);
            // tsk->setC(newC);
            // tsk->setB(newB);

            tsk->setSigma(newSigma[paramNum], paramNum);
            tsk->setC(newC[paramNum], paramNum);
            tsk->setB(newB[paramNum], paramNum);

            // if(newC > 100000) newC = oldC[paramNum];
            // if(newSigma > 100000) newSigma = oldSigma[paramNum];
            // if(newB > 100000) newB = oldB[paramNum];
            // std::cout << dEdC << "-" << newC[paramNum] << " " << dEdSigma << "-" << newSigma[paramNum] << " " << dEdB << "-" << newB[paramNum] << " | ";
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
     */
    int vectorLength = tsk->getN(); // n
    int countOfRole = tsk->getM(); // m

    int paramRoleNum = paramNum % countOfRole; // i for changedParam
    int paramVectorIndexNum = paramNum / countOfRole; // j for changedParam

    double res = deltaKronecker(roleNum, paramRoleNum);

    double m_res = m(x);
    res *= m_res;
    double l_res = l(x, paramRoleNum);
    res -= l_res;
    res /= std::pow(m_res, 2);
    auto c = tsk->getC();
    auto b = tsk->getB();
    auto sigma = tsk->getSigma();

    double multiple = 1;
    
    for (int l = paramRoleNum; l < countOfRole * vectorLength; l+=countOfRole) {
        if(paramVectorIndexNum != (l-paramRoleNum)/countOfRole) multiple *= tsk->applyFuzzyFunction(x[paramVectorIndexNum], sigma[l], c[l], b[l]);
    }
    res *= multiple;
    res *= dnuFunction(x[paramVectorIndexNum], sigma[paramNum], c[paramNum], b[paramNum]);
    return res;
}


double learning::HybridAlgorithm::m(auto x) {
    auto w = tsk->getRoleMultipleLayerOut(x);
    double res = 0;
    for (int i = 0; i < w.size(); i++) {
        res += w[i];
    }
    return res;
}

double learning::HybridAlgorithm::l(auto x, int i) {
    auto w = tsk->getRoleMultipleLayerOut(x);
    return w[i];
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
