#ifndef DATASET
#define DATASET

#include <boost/multi_array.hpp>

struct Dataset
{
    Dataset(boost::multi_array<double, 2> x, std::vector<double> d);

    std::vector<double>& getD() {
        return d;
    }

    boost::multi_array<double, 2>& getX() {
        return x;
    }

    int getCountXVectors() {
        return x.shape()[0];
    }

private:
    boost::multi_array<double, 2> x;
    std::vector<double> d;
};

Dataset readDataset(std::string&);

#endif