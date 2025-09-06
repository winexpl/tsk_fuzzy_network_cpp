#ifndef DATASET
#define DATASET

#include <boost/multi_array.hpp>
#include <utility>
#include <map>
#include <iostream>
#include <boost/serialization/access.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <iomanip>

struct Dataset
{
    int dim;
    int classCount;
    int vectorSize;
    boost::multi_array<double, 2> x;
    std::vector<double> d;
    std::vector<std::string> paramNames;
    std::vector<std::string> classNames;

    Dataset() {}

    Dataset(std::string &filename);
    
    Dataset(boost::multi_array<double, 2> x,
            std::vector<double> d,
            int dim,
            int classCount,
            int vectorSize,
            std::vector<std::string> paramNames,
            std::vector<std::string> classNames);

    Dataset(boost::multi_array<double, 2> x,
            std::vector<double> d,
            int dim,
            int classCount,
            int vectorSize);

    
    void print() 
    {
        using namespace std;
        
        // Вывод основных параметров
        cout << "=== Dataset Parameters ===\n"
             << "Dimension (dim): " << dim << "\n"
             << "Class count: " << classCount << "\n"
             << "Vector size: " << vectorSize << "\n\n";
    
        // Вывод названий параметров
        if(!paramNames.empty()) {
            cout << "=== Feature Names ===\n";
            for(size_t i = 0; i < paramNames.size(); ++i) {
                cout << "[" << i << "] " << paramNames[i] << "\n";
            }
            cout << "\n";
        }
    
        // Вывод названий классов
        if(!classNames.empty()) {
            cout << "=== Class Names ===\n";
            for(size_t i = 0; i < classNames.size(); ++i) {
                cout << "[" << i << "] " << classNames[i] << "\n";
            }
            cout << "\n";
        }
    
        // Вывод матрицы данных
        if(x.shape()[0] > 0 && x.shape()[1] > 0) {
            cout << "=== Data Matrix (" << x.shape()[0] 
                 << "x" << x.shape()[1] << ") ===\n";
                 
            // Заголовок таблицы
            cout << left << setw(8) << "Row";
            for(size_t col = 0; col < x.shape()[1]; ++col) {
                string colName = (col < paramNames.size()) ? 
                                paramNames[col] : "Feature " + to_string(col);
                cout << setw(12) << colName;
            }
            cout << "Class\n";
            
            // Данные
            cout << fixed << setprecision(4);
            for(size_t row = 0; row < x.shape()[0]; ++row) {
                cout << setw(8) << row;
                for(size_t col = 0; col < x.shape()[1]; ++col) {
                    cout << setw(12) << x[row][col];
                }
                // Вывод класса
                if(row < d.size()) {
                    int class_id = static_cast<int>(round(d[row]*(classCount-1)));
                    string className = (class_id < static_cast<int>(classNames.size())) ? 
                                      classNames[class_id] : "Unknown";
                    cout << className << " " << d[row];
                }
                cout << "\n";
            }
            cout << "\n";
        }
    }

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        std::cout << "start serialize...\n";
        ar & dim;
        std::cout << "set dim\n";
        ar & classCount;
        std::cout << "set class count\n";
        ar & d;
        std::cout << "set d\n";
        int rows = x.shape()[0];
        int cols = x.shape()[1];

        ar & rows;
        ar & cols;

        std::cout << "set r c\n";
        std::vector<double> flatData(rows * cols);

        if (Archive::is_saving::value)
        {
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    flatData[i * cols + j] = x[i][j];
        }

        ar & flatData;
        std::cout << "set x\n";

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
    const std::vector<std::string> &getClassNames() const;
    const std::vector<std::string> &getParamNames() const;
    int getClassCount() const;
    int getCountVectors() const;

    void shuffle();

    std::pair<Dataset, Dataset> splitDatasetOnTrainAndTest(double separationCoefficient);
};

#endif