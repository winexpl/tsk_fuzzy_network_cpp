#ifndef DATASET
#define DATASET

#include <boost/multi_array.hpp>
#include <utility>
#include <map>

struct Dataset
{
    Dataset() {}
    Dataset(boost::multi_array<double, 2> x,
        std::vector<double> d,
        int dim,
        int classCount,
        std::vector<std::string> paramNames,
        std::vector<std::string> classNames,
        std::map<int, std::map<std::string, double>> columnsMap,
        std::map<std::string, double> classMap);

    static Dataset readFromCsv(std::string &filename);

    std::vector<double>& getD();
    boost::multi_array<double, 2>& getX();
    int getClassCount();
    int getCountVectors();

    void shuffle();

    std::pair<Dataset, Dataset> splitDatasetOnTrainAndTest(double separationCoefficient);

    boost::multi_array<double, 2> x;
    std::vector<double> d;
    int dim;
    int classCount;
    std::vector<std::string> paramNames;
    std::vector<std::string> classNames;
    std::map<int, std::map<std::string, double>> columnsMap;
    std::map<std::string, double> classMap;
private:

};

#endif