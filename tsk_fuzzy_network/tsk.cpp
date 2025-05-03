#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include "metric.h"
#include <sstream>

tsk::TSK::TSK(int n, int m, int out)
    : _fuzzyLayer{tsk::layers::FuzzyLayer(n, m*n)},
    _roleMultipleLayer{tsk::layers::RoleMultipleLayer(m*n, m)},
    _multipleLayer{tsk::layers::MultipleLayer(m, m*out, n)},
    _sumLayer{tsk::layers::SumLayer(m*out, out)},
    _n{n}, _m{m}, _out{out}
{
    std::ostringstream logStream;
    logStream << "Модель создана.\n"
            << "\t_fuzzyLayer in=" << n << " out=" << m * n << "\n"
            << "\t_roleMultipleLayer in=" << m * n << " out=" << m << "\n"
            << "\t_multipleLayer in=" << m << " out=" << m * out << "\n"
            << "\t_sumLayer in=" << m * out << " out=" << out;

    Logger::getInstance().logInfo(logStream.str());
}

double tsk::TSK::applyFuzzyFunction(double x, double sigma, double c, double b)
{
    return _fuzzyLayer.fuzzyFunction(x, sigma, c, b);
}

void tsk::TSK::updateP(Eigen::MatrixXd &p)
{
    auto& oldP = _multipleLayer.p;
    auto oldPShape = oldP.shape();
    boost::multi_array<double, 2> newP(boost::extents[oldPShape[0]][oldPShape[1]]);
    for (int i = 0; i < oldPShape[0]*oldPShape[1]; ++i) {
        newP[i/(oldPShape[1])][i%(oldPShape[1])] = p(i,0);
    }
    std::cout << std::endl;

    _multipleLayer.p = newP;
}

boost::multi_array<double, 2>& tsk::TSK::getP()
{
    return this->_multipleLayer.p;
};

std::vector<double>& tsk::TSK::getSigma()
{
    return this->_fuzzyLayer.sigma;
}

void tsk::TSK::setSigma(double sigma, int index)
{
    _fuzzyLayer.sigma[index]=sigma;
}

void tsk::TSK::setC(double c, int index)
{
    _fuzzyLayer.c[index]=c;
}

void tsk::TSK::setB(double b, int index)
{
    _fuzzyLayer.b[index]=b;
}

void tsk::TSK::setSigma(std::vector<double> sigma)
{
    _fuzzyLayer.sigma=sigma;
}

void tsk::TSK::setC(std::vector<double> c)
{
    _fuzzyLayer.c=c;
}

void tsk::TSK::setB(std::vector<double> b)
{
    _fuzzyLayer.b=b;
}

std::vector<double>& tsk::TSK::getB()
{
    return this->_fuzzyLayer.b;
}

std::vector<double>& tsk::TSK::getC()
{
    return this->_fuzzyLayer.c;
}

int tsk::TSK::getN()
{
    return _n;
}

int tsk::TSK::getM()
{
    return _m;
}

std::vector<double> tsk::TSK::predict(const boost::multi_array<double,2>& x) const
{
    std::vector<double> predict;
    for(int i = 0; i < x.shape()[0]; i++) {
        auto xi = x[i];
        double out = predict1(xi);
        predict.push_back(out);
    }
    return predict;
}

std::vector<double> tsk::TSK::evaluate(const boost::multi_array<double,2>& x, const std::vector<double>& y, int classesCount) const
{
    /**
     * x - входные векторы
     * y - ожидаемые значения
     * x.size() == y.size()
     * return значения которые дает модель
     */

    std::vector<double> predictedValues = predict(x);
    
    std::cout << metric::Metric::calculateAccuracy(y, predictedValues, classesCount);
    std::cout << metric::Metric::calculateMSE(y, predictedValues, classesCount);
    
    return predictedValues;
}