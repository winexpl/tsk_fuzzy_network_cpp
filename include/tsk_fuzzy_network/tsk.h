#ifndef TSK_MODEL
#define TSK_MODEL

#include "tsk_fuzzy_network/layers.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/SVD>

namespace tsk {
    struct TSK;
}

struct tsk::TSK {
    TSK(int N, int M, int out=1);

    void updateP(Eigen::MatrixXd&);
    
    std::vector<double> predict(boost::multi_array<double,2>& x);

    std::vector<double> evaluate(boost::multi_array<double,2>& x, std::vector<double>& y);
    
    template <is_indexed T>
    double predict1(T& x) {
        std::vector<double> y1 = _fuzzyLayer.get(x);
        std::vector<double> y2 = _roleMultipleLayer.get(y1);
        std::vector<double> y3 = _multipleLayer.get(y2, x);
        double y4 = _sumLayer.get(y3, y2);
        return y4;
    }

    boost::multi_array<double, 2>& getP();
    std::vector<double>& getSigma();
    std::vector<double>& getB();
    std::vector<double>& getC();

private:
    tsk::layers::FuzzyLayer _fuzzyLayer;
    tsk::layers::RoleMultipleLayer _roleMultipleLayer;
    tsk::layers::MultipleLayer _multipleLayer;
    tsk::layers::SumLayer _sumLayer;
    
    int _n; // число параметров
    int _m; // число правил
    int _out; // число выходов
};



#endif