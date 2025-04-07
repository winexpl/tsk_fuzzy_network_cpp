#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <boost/multi_array.hpp>
#include <random>
#include "dataset.h"
#include "out.h"

Dataset::Dataset(boost::multi_array<double, 2> x, std::vector<double> d, int classesCount) :
    classesCount{classesCount}, d{d}, x{x}
    {
        if(x.shape()[0] != d.size()) {
            throw std::runtime_error("The number of parameter lines must match the number of outputs.");
        }
    }


std::vector<double> split(const std::string& str, char delimiter) {
    std::vector<double> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(atof(token.c_str()));
    }

    return tokens;
}

void Dataset::shuffle()
{
    int k = getCountVectors();
    std::vector<int> arr(k);
    std::iota(arr.begin(), arr.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(arr.begin(), arr.end(), g);

    boost::multi_array<double, 2> shuffledX(boost::extents[x.shape()[0]][x.shape()[1]]);
    std::vector<double> shuffledD;

    for(int i = 0; i < k; i++)
    {
        shuffledX[i] = x[arr[i]];
        shuffledD.push_back(d[arr[i]]);
    }
    x = shuffledX;
    d = shuffledD;
}

Dataset readDataset(std::string &filename) {
    std::ifstream file{filename};
    if(!file.is_open()) {
        throw std::runtime_error("file not found or not permission");
    }
    
    std::vector<std::vector<double>> x;
    std::vector<double> d;
    std::string line;
    std::string token;
    std::vector<double> xi;
    std::set<double> classes;
    std::getline(file, line, '\n');
    
    while (std::getline(file, line, '\n')) {
        xi = split(line, ',');
        double di = xi[xi.size()-1];
        xi.pop_back();
        classes.insert(di);
        d.push_back(di);
        x.emplace_back(xi);
    }
    size_t rows = x.size();
    size_t cols = x[0].size();
    // std::cout << "readDataset(): " << "rows="<<rows<<" cols="<<cols<<std::endl;

    boost::multi_array<double, 2> boost_x(boost::extents[rows][cols]);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            boost_x[i][j] = x[i][j];
        }
    }
    return Dataset(boost_x, d, classes.size());
}

std::pair<Dataset, Dataset> Dataset::splitDatasetOnTrainAndTest(double separationCoefficient)
{
    int k = getCountVectors();
    int trainLength = std::round(k * separationCoefficient);
    int testLength = k - trainLength;

    std::vector<double> trainY(d.begin(), d.begin() + trainLength);
    std::vector<double> testY(d.begin() + trainLength, d.end());

    boost::multi_array<double, 2> trainX(boost::extents[trainLength][x.shape()[1]]);
    boost::multi_array<double, 2> testX(boost::extents[testLength][x.shape()[1]]);

    for(int i = 0; i < trainLength; ++i)
    {
        trainX[i] = x[i];
    }

    for(int j = trainLength; j < k; ++j)
    {
        testX[j-trainLength] = x[j];
    }

    Dataset trainDataset(trainX, trainY, classesCount);
    Dataset testDataset(testX, testY, classesCount);

    return std::make_pair(trainDataset, testDataset);
}