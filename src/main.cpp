#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include "tsk_fuzzy_network/learning_algorithms.h"
#include <iostream>
#include "dataset.h"
#include "out.h"
#include "metric.h"
#include "tsk_fuzzy_network/c_means.h"

std::ostream& operator<<(std::ostream& os, std::vector<double>& x) {
    for(int i = 0; i < x.size(); i++) {
        os << "x[" << i << "] = " << x[i] << "\n";
    }
    os.flush();
    return os;
}

std::ostream& operator<<(std::ostream& os, boost::multi_array<double,2>& x)
{
    std::cout << x.shape()[0] << " " << x.shape()[1] << std::endl;
    for(int i = 0; i < x.shape()[0]; i++)
    {
        for(int j = 0; j < x.shape()[1]; j++)
        {
            os << "x[" << i << "][" << j << "] = " << x[i][j] << "\n";
        }
    }
    os.flush();
    return os;
}

int main(int argc, char* argv[]) {
    int m = 4;
    std::string filename = "resource/new-magic-gamma-telescope.csv";
    Dataset dataset = readDataset(filename);
    dataset.shuffle();
    std::pair<Dataset, Dataset> datasetPair = dataset.splitDatasetOnTrainAndTest(0.8);

    tsk::CMeans cmeans(m, 0, 0.0001);
    cmeans.fit(datasetPair.first.getX());
    std::vector<double> c = cmeans.getCentroids();
    std::vector<double> sigma = cmeans.getSigma();

    std::cout << sigma;
    std::cout << c;

    tsk::TSK tsk(dataset.getX().shape()[1], m);
    tsk.setC(c);
    tsk.setSigma(sigma);
    std::shared_ptr<tsk::TSK> tsk_shptr = std::make_shared<tsk::TSK>(tsk);
    learning::HybridAlgorithm hybridAlg(tsk_shptr, datasetPair.first);
    std::string input;

    do {
        hybridAlg.learning(datasetPair.first.getCountVectors(), 4, 10, 0.001);
        auto predict = tsk_shptr->predict(datasetPair.second.getX());
        
        std::cout << "accuracy: " << metric::Metric::calculateAccuracy(datasetPair.second.getD(), predict, dataset.classesCount) << std::endl
            << "mse: " << metric::Metric::calculateMSE(datasetPair.second.getD(), predict, dataset.classesCount) << std::endl;
        std::cout << "Нажмите Enter для продолжения (или введите что-то для выхода): ";
        std::getline(std::cin, input);
    } while (input.empty());

    auto p = tsk_shptr->getP();
    auto predict = tsk_shptr->predict(datasetPair.second.getX());
    for(int i = 0; i < predict.size(); i++) {
        std::cout << "predict[" << i << "] = " << predict[i] << " "
                    << "y[" << i << "] = " << datasetPair.second.getD()[i] << "\t"
                    << predict[i]/dataset.getD()[i] << "\n";
    }
    std::cout << "accuracy: " << metric::Metric::calculateAccuracy(datasetPair.second.getD(), predict, dataset.classesCount) << std::endl
    << "mse: " << metric::Metric::calculateMSE(datasetPair.second.getD(), predict, dataset.classesCount) << std::endl;
    return 0;
}

