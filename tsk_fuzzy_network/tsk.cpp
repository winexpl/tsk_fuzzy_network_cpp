#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"

tsk::TSK::TSK(int n, int m, int out)
    : _fuzzyLayer{tsk::layers::FuzzyLayer(n, m*n)},
    _roleMultipleLayer{tsk::layers::RoleMultipleLayer(m*n, m)},
    _multipleLayer{tsk::layers::MultipleLayer(m, m*out, n)},
    _sumLayer{tsk::layers::SumLayer(m*out, out)}
{
    std::cout << "Модель создана." << "\n"
        << "_fuzzyLayer in=" << n << " out=" << m*n << "\n"
        << "_roleMultipleLayer in=" << m*n << " out=" << m << "\n"
        << "_multipleLayer in=" << m << " out=" << m*out << "\n"
        << "_sumLayer in=" << m*out << " out=" << out << std::endl;
}

std::ostream& operator<<(std::ostream& os, boost::multi_array<double,2>& x) {
    for(int i = 0; i < x.shape()[1]; i++) {
        for(int j = 0; j < x.shape()[0]; j++) {
            os << "x[" << i << "][" << j << "] = " << x[i][j] << "\n";
        }
    }
    os.flush();
    return os;
}


void tsk::TSK::updateP(Eigen::MatrixXd &p) {
    auto& oldP = _multipleLayer.p;
    auto oldPShape = oldP.shape();
    boost::multi_array<double, 2> newP(boost::extents[oldPShape[0]][oldPShape[1]]);
    std::cout << "update " << p.size() << " " << oldPShape[0] << " " << oldPShape[1] << std::endl;
    for (int i = 0; i < oldPShape[0]*oldPShape[1]; ++i) {
        std::cout << "i=" << i << std::endl;
        newP[i/(oldPShape[1])][i%(oldPShape[1])] = p(i,0);
    }
    std::cout << "end update" << std::endl;

    std::cout << newP << std::endl;
    _multipleLayer.p = newP;
}

boost::multi_array<double, 2>& tsk::TSK::getP() {
    return this->_multipleLayer.p;
};
std::vector<double>& tsk::TSK::getSigma() {
    return this->_fuzzyLayer.sigma;
}
std::vector<double>& tsk::TSK::getB() {
    return this->_fuzzyLayer.b;
}
std::vector<double>& tsk::TSK::getC() {
    return this->_fuzzyLayer.c;
}

std::vector<double> tsk::TSK::predict(boost::multi_array<double,2>& x) {
    std::cout << "predict " << x.shape()[0] <<std::endl;
    std::vector<double> predict;
    for(int i = 0; i < x.shape()[0]; i++) {
        std::cout << "i=" << i << " predict" <<std::endl;
        auto xi = x[i];
        double out = predict1(xi);
        predict.push_back(out);
    }
    return predict;
}

std::vector<double> tsk::TSK::evaluate(boost::multi_array<double,2>& x, std::vector<double>& y) {
    /**
     * x - входные векторы
     * t - ожидаемые значения
     * x.size() == t.size()
     * return значения которые дает модель
     */

}