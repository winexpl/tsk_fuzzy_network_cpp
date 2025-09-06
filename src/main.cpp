#include "tsk_fuzzy_network/dataset.h"
#include "tsk_fuzzy_network/tsk.h"
#include <boost/archive/text_iarchive.hpp>

int main()
{
    std::string path = "/mnt/masha/projects/tsk-fuzzy-network-cpp/resource/tic-tac-toc.csv";
    Dataset *dataset = new Dataset(path);
    tsk::TSK tsk;

    std::string modelPath = "/mnt/masha/projects/tsk-fuzzy-network-cpp/models/tictactoe3rolesNoise.model";
    std::ifstream ifs(modelPath, std::ios::binary);

    {
        boost::archive::text_iarchive ia(ifs);
        ia >> tsk;
    }

    double noiseLevel = 0.05;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(-noiseLevel, noiseLevel);
    
    boost::multi_array<double, 2> noisyData(dataset->getX());
    auto *data_ptr = noisyData.data();
    const size_t num_elements = noisyData.num_elements();

    for (size_t i = 0; i < num_elements; ++i)
    {
        data_ptr[i] += dist(gen);
    }

    
    auto predicts = tsk.predict(noisyData);
    delete dataset;

    for(int i = 0; i < predicts.size(); ++i) {
    std::cout << dataset->d[i] << " " << predicts[i] << std::endl;
    }
    tsk.print(std::cout);
}