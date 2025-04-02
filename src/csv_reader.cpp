#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>
#include "dataset.h"
#include "out.h"

Dataset::Dataset(boost::multi_array<double, 2> x, std::vector<double> d) :
    d{d}, x{x}
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
    std::getline(file, line, '\n');
    
    while (std::getline(file, line, '\n')) {
        xi = split(line, ',');
        double di = xi[xi.size()-1];
        xi.pop_back();
        d.push_back(di);
        x.emplace_back(xi);
    }
    std::cout << d.size() << std::endl;
    size_t rows = x.size();
    size_t cols = x[0].size();
    std::cout << "rows="<<rows<<"cols="<<cols<<std::endl;

    boost::multi_array<double, 2> boost_x(boost::extents[rows][cols]);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            boost_x[i][j] = x[i][j];
        }
    }
    std::cout << boost_x.shape()[0] << std::endl;
    return Dataset(boost_x, d);
}