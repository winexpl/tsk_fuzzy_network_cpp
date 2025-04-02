#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include "tsk_fuzzy_network/learning_algorithms.h"
#include <iostream>
#include "dataset.h"
#include "out.h"

std::ostream& operator<<(std::ostream& os, std::vector<double>& x) {
    for(int i = 0; i < x.size(); i++) {
        os << "x[" << i << "] = " << x[i] << "\n";
    }
    os.flush();
    return os;
}

int main(int argc, char* argv[]) {
    std::string filename = "resource/new-irises.csv";
    Dataset dataset = readDataset(filename);
    tsk::TSK tsk(4,1);
    std::cout << dataset.getCountXVectors() << " " << dataset.getX().shape()[1] << std::endl;

    std::shared_ptr<tsk::TSK> tsk_shptr = std::make_shared<tsk::TSK>(tsk);
    learning::HybridAlgorithm hybridAlg(tsk_shptr, dataset, 16);
    hybridAlg.learning();
    auto predict = tsk_shptr->predict(dataset.getX());
    std::cout << predict << std::endl;
    return 0;
}

