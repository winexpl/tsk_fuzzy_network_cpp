#ifndef DATASET
#define DATASET

#include <boost/multi_array.hpp>
#include <utility>

struct Dataset
{
    Dataset(boost::multi_array<double, 2> x, std::vector<double> d, int classesCount);

    std::vector<double>& getD() {
        return d;
    }

    boost::multi_array<double, 2>& getX() {
        return x;
    }

    int getCountVectors() {
        return x.shape()[0];
    }

    void shuffle();

    std::pair<Dataset, Dataset> splitDatasetOnTrainAndTest(double separationCoefficient);
    int classesCount;

private:
    boost::multi_array<double, 2> x;
    std::vector<double> d;
};

Dataset readDataset(std::string&);

#endif