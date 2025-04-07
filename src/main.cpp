#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include "tsk_fuzzy_network/learning_algorithms.h"
#include <iostream>
#include "dataset.h"
#include "out.h"
#include "metric.h"

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
    std::string filename = "resource/new-irises.csv";
    Dataset dataset = readDataset(filename);
    dataset.shuffle();
    std::pair<Dataset, Dataset> datasetPair = dataset.splitDatasetOnTrainAndTest(0.8);
    tsk::TSK tsk(dataset.getX().shape()[1], 10);
    std::shared_ptr<tsk::TSK> tsk_shptr = std::make_shared<tsk::TSK>(tsk);
    learning::HybridAlgorithm hybridAlg(tsk_shptr, datasetPair.first);

    std::string input;

    do {
        // Ваша основная логика
        hybridAlg.learning(dataset.getCountVectors(), 1, 10, 0.005);

        std::cout << "Нажмите Enter для продолжения (или введите что-то для выхода): ";
        std::getline(std::cin, input);
    } while (input.empty()); // Продолжаем, пока строка пуста (Enter)

    auto p = tsk_shptr->getP();
    auto predict = tsk_shptr->predict(dataset.getX());
    std::cout << "accuracy: " << metric::Metric::calculateAccuracy(dataset.getD(), predict, dataset.classesCount) << std::endl
                << "mse: " << metric::Metric::calculateMSE(dataset.getD(), predict, dataset.classesCount) << std::endl;
    std::cout << predict;
    return 0;
}

