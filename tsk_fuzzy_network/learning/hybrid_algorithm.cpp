#include "tsk_fuzzy_network/learning_algorithms.h"

Eigen::MatrixXd boostMultiArrayToEigenMatrix(const boost::multi_array<double, 2>& boostMultiArray) {
    std::cout << "convert1" << std::endl;
    auto eigenMatrix = Eigen::MatrixXd(boostMultiArray.shape()[0], boostMultiArray.shape()[1]);
    for (size_t i = 0; i < boostMultiArray.shape()[0]; ++i) {
        for (size_t j = 0; j < boostMultiArray.shape()[1]; ++j) {
            std::cout << "convert" << i << j << std::endl;

            eigenMatrix(i, j) = boostMultiArray[i][j];
        }
    }
    return eigenMatrix;
}

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<double>& vec) {
    Eigen::MatrixXd matrix(1,vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        matrix(0, i) = vec[i];
    }
    return matrix;
}

void learning::HybridAlgorithm::learning() {
    for(int startIndex = 0; startIndex < dataset.getCountXVectors(); startIndex += batchSize) {
        int endIndex = startIndex + batchSize;
        if(endIndex > dataset.getCountXVectors()) endIndex = dataset.getCountXVectors() - 1;
        this->learningTskBatchFirstStep(startIndex, endIndex);
    }
}

learning::HybridAlgorithm::HybridAlgorithm(std::shared_ptr<tsk::TSK> tsk, Dataset &dataset, int batchSize) :
tsk{tsk}, dataset{dataset}, batchSize{batchSize}
{
    _d = vectorToEigenMatrix(dataset.getD());
};

void learning::HybridAlgorithm::learningTskBatchFirstStep(int startIndex, int endIndex)
{
    boost::multi_array<double, 2> oldP = tsk->getP();

    Eigen::MatrixXd oldPEigen = boostMultiArrayToEigenMatrix(oldP);
    Eigen::MatrixXd oldPInverseEigen = inverse(oldPEigen);
    Eigen::MatrixXd newPEigen = oldPInverseEigen * _d;

    tsk->updateP(newPEigen);
}

Eigen::MatrixXd learning::HybridAlgorithm::inverse(Eigen::MatrixXd &p) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(p, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& singularValues = svd.singularValues();
    Eigen::MatrixXd singularValuesInv = singularValues.array().inverse().matrix().asDiagonal();

    Eigen::MatrixXd pseudoInverseM = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
    boost::multi_array<double, 2> pseudoInverse(boost::extents[pseudoInverseM.rows()][pseudoInverseM.cols()]);

    return pseudoInverseM;
}
