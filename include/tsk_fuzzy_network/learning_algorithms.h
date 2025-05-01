#ifndef TSK_LEARNING_H
#define TSK_LEARNING_H

#include "tsk_fuzzy_network/tsk.h"
#include "dataset.h"

namespace learning
{
    struct HybridAlgorithm;
};

struct learning::HybridAlgorithm
{
    int batchSize;
    tsk::TSK* tsk;
    Dataset dataset;

    HybridAlgorithm(tsk::TSK* tsk, Dataset &dataset);
    
    void learning(int batchSize, int epochCount, int countSecondStepIter=100, double nu=0.01);
private:
    Eigen::MatrixXd _d;
    
    double dE(double e, const  boost::multi_array<double, 2> &p, auto x,
        int paramNum, std::function<double(double, double, double, double)> dnuFunction);

    double dW(int paramNum, int roleNum, auto x, std::function<double(double, double, double, double)> dnuFunction);

    void learningTskBatchFirstStep(int startIndex, int endIndex);

    void learningTskBatchSecondStep(int startIndex, int endIndex, double nu);

    double m(auto x);
    
    double l(auto x, int i);
    
    int deltaKronecker(int i, int j);

    Eigen::MatrixXd inverse(Eigen::MatrixXd&);
};

#endif