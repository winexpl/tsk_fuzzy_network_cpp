#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <boost/multi_array.hpp>
#include <random>
#include <map>
#include <hash_map>
#include "tsk_fuzzy_network/dataset.h"
#include "logger.h"
#include <sstream>

bool isConvertibleToDouble(const std::string &str)
{
    try
    {
        std::stod(str); // Пробуем преобразовать
        return true;    // Конвертация успешна
    }
    catch (std::invalid_argument &)
    {
        return false; // Ошибка: не число
    }
    catch (std::out_of_range &)
    {
        return false; // Ошибка: число слишком большое/маленькое
    }
}

Dataset::Dataset(boost::multi_array<double, 2> x,
                 std::vector<double> d,
                 int dim,
                 int classCount,
                 std::vector<std::string> paramNames,
                 std::vector<std::string> classNames,
                 std::map<int, std::map<std::string, double>> columnsMap,
                 std::map<std::string, double> classMap) : x{x},
                                                           d{d},
                                                           dim{dim},
                                                           classCount{classCount},
                                                           paramNames{paramNames},
                                                           classNames{classNames},
                                                           columnsMap{columnsMap},
                                                           classMap{classMap}
{
    if (x.shape()[0] != d.size())
    {
        throw std::runtime_error("The number of parameter lines must match the number of outputs.");
    }
    std::ostringstream logStream;
    logStream << "Датасет создан.\n"
              << "\tdim=" << dim << "\n"
              << "\tclassCount=" << classCount;
    Logger::getInstance().logInfo(logStream.str());
}

Dataset::Dataset(boost::multi_array<double, 2> x,
                 std::vector<double> d,
                 int dim,
                 int classCount) : x{x},
                                   d{d},
                                   dim{dim},
                                   classCount{classCount}
{
    if (x.shape()[0] != d.size())
    {
        throw std::runtime_error("The number of parameter lines must match the number of outputs.");
    }
    std::ostringstream logStream;
    logStream << "Датасет создан.\n"
              << "\tdim=" << dim << "\n"
              << "\tclassCount=" << classCount << "\n";
    Logger::getInstance().logInfo(logStream.str());
}

std::vector<std::string> split(const std::string &str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter))
    {
        tokens.push_back(token.c_str());
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

    for (int i = 0; i < k; i++)
    {
        shuffledX[i] = x[arr[i]];
        shuffledD.push_back(d[arr[i]]);
    }
    x = shuffledX;
    d = shuffledD;
}

Dataset Dataset::readFromCsv(std::string &filename)
{
    std::ifstream file{filename};
    if (!file.is_open())
    {
        throw std::runtime_error("file not found or not permission");
    }

    std::vector<std::vector<double>> x;
    std::vector<std::vector<std::string>> xString;
    std::vector<double> d;
    std::vector<std::string> dString;
    std::vector<std::string> paramNames;
    std::vector<std::string> classNames;

    std::string line;
    std::string token;
    std::vector<std::string> xiString;

    std::map<std::string, double> classMap;
    std::map<int, std::map<std::string, double>> columnsMap;

    std::getline(file, line, '\n');
    paramNames = split(line, ',');

    double classNum{0};
    while (std::getline(file, line, '\n'))
    {
        xiString = split(line, ',');
        std::string diString = xiString[xiString.size() - 1];
        xiString.pop_back();

        if (!classMap.count(diString))
        {
            classMap.emplace(diString, classNum++);
        }
        dString.emplace_back(diString);
        xString.emplace_back(xiString);
    }
    size_t rows = xString.size();
    size_t cols = xString[0].size();
    double classCount = classMap.size();

    boost::multi_array<double, 2> boost_x(boost::extents[rows][cols]);

    for (size_t j = 0; j < cols; ++j)
    {
        if (isConvertibleToDouble(xString[0][j]))
        {
            for (size_t i = 0; i < rows; ++i)
            {
                boost_x[i][j] = std::stod(xString[i][j]);
            }
        }
        else
        {
            std::map<std::string, double> columnMap;
            int paramNum{0};

            for (size_t i = 0; i < rows; ++i)
            {
                if (!columnMap.count(xString[i][j]))
                {
                    columnMap.emplace(xString[i][j], paramNum++);
                }
            }

            for (size_t i = 0; i < rows; ++i)
            {
                boost_x[i][j] = columnMap[xString[i][j]];
            }
            columnsMap.emplace(j, columnMap);
        }
    }

    if (isConvertibleToDouble(dString[0]))
    {
        for (int i = 0; i < dString.size(); ++i)
        {
            d.emplace_back(std::stod(dString[i]));
        }
    }
    else
    {
        for (int i = 0; i < dString.size(); ++i)
        {
            d.emplace_back(classMap[dString[i]] / classCount);
        }
    }

    return Dataset(boost_x, d, boost_x.shape()[0], classMap.size(), paramNames, classNames, columnsMap, classMap);
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

    for (int i = 0; i < trainLength; ++i)
    {
        trainX[i] = x[i];
    }

    for (int j = trainLength; j < k; ++j)
    {
        testX[j - trainLength] = x[j];
    }

    Dataset trainDataset(trainX, trainY, trainLength, classCount, paramNames, classNames, columnsMap, classMap);
    Dataset testDataset(testX, testY, testLength, classCount, paramNames, classNames, columnsMap, classMap);

    return std::make_pair(trainDataset, testDataset);
}

int Dataset::getClassCount() const
{
    return classCount;
}

int Dataset::getCountVectors() const
{
    return dim;
}

const std::vector<double> &Dataset::getD() const
{
    return d;
}

const boost::multi_array<double, 2> &Dataset::getX() const
{
    return x;
}