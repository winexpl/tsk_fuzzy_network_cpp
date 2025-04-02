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
    std::shared_ptr<tsk::TSK> tsk;
    Dataset dataset;

    HybridAlgorithm(std::shared_ptr<tsk::TSK> tsk, Dataset &dataset, int batchSize);
    void learning();
private:
    Eigen::MatrixXd _d;

    void learningTskBatchFirstStep(int startIndex, int endIndex);
    Eigen::MatrixXd inverse(Eigen::MatrixXd&);
};

#endif