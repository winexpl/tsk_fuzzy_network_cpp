#ifndef TSK_MODEL
#define TSK_MODEL

#include "tsk_fuzzy_network/layers.h"
#include "metric.h"
#include <sstream>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/SVD>
#include <iomanip>

namespace tsk
{
    struct TSK;
}

struct tsk::TSK
{
    TSK(int N, int M, int out = 1);
    TSK() {}

    void updateP(Eigen::MatrixXd &);

    std::vector<double> predict(const boost::multi_array<double, 2> &x) const;

    metric::Metric evaluate(const boost::multi_array<double, 2> &x, const std::vector<double> &y, int classesCount) const;

    template <is_indexed T>
    double predict1(const T &x) const
    {
        // Получаем промежуточные значения
        std::vector<double> y1 = _fuzzyLayer.get(x);
        std::vector<double> y2 = _roleMultipleLayer.get(y1);
        std::vector<double> y3 = _multipleLayer.get(y2, x);
        double y4 = _sumLayer.get(y3, y2);

        // Создаем поток для форматированного вывода
        // std::ostringstream oss;
        // oss << std::fixed << std::setprecision(4); // Фиксируем 4 знака после запятой

        // // Вывод с красивым форматированием
        // oss << "\n=== Neural Network Layer Outputs ===\n";

        // // Вывод y1 (Fuzzy Layer)
        // oss << "Fuzzy Layer (y1):\n";
        // for (size_t i = 0; i < y1.size(); ++i)
        // {
        //     oss << "  Neuron " << i + 1 << ": " << std::setw(8) << y1[i];
        //     if ((i + 1) % 3 == 0)
        //         oss << "\n"; // Перенос строки каждые 3 значения
        // }
        // if (y1.size() % 3 != 0)
        //     oss << "\n";

        // // Вывод y2 (Role Multiple Layer)
        // oss << "Role Multiple Layer (y2):\n";
        // for (size_t i = 0; i < y2.size(); ++i)
        // {
        //     oss << "  Rule " << i + 1 << " weight: " << std::setw(8) << y2[i] << "\n";
        // }

        // // Вывод y3 (Multiple Layer)
        // oss << "Multiple Layer (y3):\n";
        // for (size_t i = 0; i < y3.size(); ++i)
        // {
        //     oss << "  Output " << i + 1 << ": " << std::setw(8) << y3[i];
        //     if ((i + 1) % 2 == 0)
        //         oss << "\n"; // Перенос строки каждые 2 значения
        // }
        // if (y3.size() % 2 != 0)
        //     oss << "\n";

        // // Вывод финального результата
        // oss << "Sum Layer (y4) - Final Output:\n";
        // oss << "  " << std::string(30, '-') << "\n";
        // oss << "  Result: " << std::setw(10) << y4 << "\n";
        // oss << std::string(50, '=') << "\n";

        // // Выводим в консоль
        // std::cout << oss.str();

        return y4;
    }

    boost::multi_array<double, 2> &getP();
    std::vector<double> &getSigma();
    std::vector<double> &getB();
    std::vector<double> &getC();

    void setSigma(double sigma, int index);
    void setC(double c, int index);
    void setB(double b, int index);

    double applyFuzzyFunction(double x, double c, double sigma, double b);

    int getN();
    int getM();

    template <is_indexed T>
    std::vector<double> getFuzzyLayerOut(T &x) const
    {
        std::vector<double> y1 = _fuzzyLayer.get(x);
        return y1;
    }

    template <is_indexed T>
    std::vector<double> getRoleMultipleLayerOut(T &x) const
    {
        std::vector<double> y1 = _fuzzyLayer.get(x);
        std::vector<double> y2 = _roleMultipleLayer.get(y1);
        return y2;
    }

    template <is_indexed T>
    std::vector<double> getMultipleLayerOut(T &x) const
    {
        std::vector<double> y1 = _fuzzyLayer.get(x);
        std::vector<double> y2 = _roleMultipleLayer.get(y1);
        std::vector<double> y3 = _multipleLayer.get(y2, x);
        return y3;
    }

    void clearFuzzyLayer()
    {
        _fuzzyLayer = tsk::layers::FuzzyLayer(_n, _m * _n);
    }

    void setSigma(std::vector<double> sigma);
    void setC(std::vector<double> c);
    void setB(std::vector<double> b);

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & _n;
        ar & _m;
        ar & _out;
        ar & _fuzzyLayer;
        ar & _roleMultipleLayer;
        ar & _multipleLayer;
        ar & _sumLayer;
    }

    void print(std::ostream &os = std::cout) const;

private:
    tsk::layers::FuzzyLayer _fuzzyLayer;
    tsk::layers::RoleMultipleLayer _roleMultipleLayer;
    tsk::layers::MultipleLayer _multipleLayer;
    tsk::layers::SumLayer _sumLayer;

    int _n;   // число параметров
    int _m;   // число правил
    int _out; // число выходов
};

#endif