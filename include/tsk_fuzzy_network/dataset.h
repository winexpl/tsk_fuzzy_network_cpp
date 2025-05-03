#ifndef DATASET
#define DATASET

#include <boost/multi_array.hpp>
#include <utility>
#include <map>

#include <boost/serialization/access.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

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

    Dataset(boost::multi_array<double, 2> x,
            std::vector<double> d,
            int dim,
            int classCount);

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & dim;
        ar & classCount;
        ar & d;

        int rows = x.shape()[0];
        int cols = x.shape()[1];

        ar & rows;
        ar & cols;

        std::vector<double> flatData(rows * cols);

        if (Archive::is_saving::value)
        {
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    flatData[i * cols + j] = x[i][j];
        }

        ar & flatData;

        if (Archive::is_loading::value)
        {
            x.resize(boost::extents[rows][cols]);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    x[i][j] = flatData[i * cols + j];
        }
    }

    static Dataset readFromCsv(std::string &filename);

    const std::vector<double> &getD() const;
    const boost::multi_array<double, 2> &getX() const;
    int getClassCount() const;
    int getCountVectors() const;

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