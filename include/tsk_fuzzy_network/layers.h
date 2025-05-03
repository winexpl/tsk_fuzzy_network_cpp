#ifndef TSK_LAYERS
#define TSK_LAYERS

#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <utility>
#include <boost/multi_array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "logger.h"


namespace tsk {
    template <typename T>
    concept is_indexed = requires (T a, int i) {
        { a[i] } -> std::convertible_to<typename T::value_type>;
        requires !requires { a[i][0]; };
    };

    template <typename T>
    concept is_double_indexed = requires (T a, int i, int j) {
        { a[i][j] } -> std::convertible_to<typename std::remove_reference<decltype(a[i][j])>::type>;
    };

    template <is_indexed T, is_indexed Y>
    bool is_same_length(const T& a1, const Y& a2) {
        return (a1.cend() - a1.cbegin()) == (a2.cend() - a2.cbegin());
    }
    namespace layers {
        struct Layer;
        struct FuzzyLayer;
        struct MultipleLayer;
        struct RoleMultipleLayer;
        struct SumLayer;

        double generalGaussian(double x, double sigma, double c, double b);

        static int write(std::ostream& os, Layer& layer);

        inline std::random_device rd;
    }
}

struct tsk::layers::Layer {
    Layer(int dimInput, int dimOutput);
    int dimInput;
    int dimOutput;

    Layer() {}
};

struct tsk::layers::FuzzyLayer : public tsk::layers::Layer {
    /**
     * first layer
     */
    std::function<double(double,double,double,double)> fuzzyFunction;

    std::vector<double> sigma;
    std::vector<double> c;
    std::vector<double> b;

    FuzzyLayer(int dimInput, int dimOutput);
    FuzzyLayer() {}
    template <tsk::is_indexed T>
    std::vector<double> get(T&) const;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & dimInput;
        ar & dimOutput;
        ar & sigma;
        ar & c;
        ar & b;
        if(Archive::is_loading::value)
        {
            fuzzyFunction=generalGaussian;
        }
    }
};

struct tsk::layers::MultipleLayer : tsk::layers::Layer {
    /**
     * third layer
     */
    boost::multi_array<double, 2> p;
    
    MultipleLayer(int dimInput, int dimOutput, int N);
    MultipleLayer() {}
    template <tsk::is_indexed T, tsk::is_indexed Y>
    std::vector<double> get(T&, Y&) const;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & dimInput;
        ar & dimOutput;
        
        int rows = p.shape()[0];
        int cols = p.shape()[1];

        ar & rows;
        ar & cols;

        std::vector<double> flatData(rows * cols);

        if (Archive::is_saving::value) {
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    flatData[i * cols + j] = p[i][j];
        }

        ar & flatData;

        if (Archive::is_loading::value) {
            p.resize(boost::extents[rows][cols]);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    p[i][j] = flatData[i * cols + j];
        }

    }
};

struct tsk::layers::RoleMultipleLayer : tsk::layers::Layer {
    /**
     * second layer
     */
    RoleMultipleLayer(int dimInput, int dimOutput);
    RoleMultipleLayer() {}
    template <tsk::is_indexed T>
    std::vector<double> get(T&) const;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & dimInput;
        ar & dimOutput;
    }
};

struct tsk::layers::SumLayer : tsk::layers::Layer {
    /**
     * fourth layer
     */
    SumLayer(int dimInput, int dimOutput);
    SumLayer() {}
    template <tsk::is_indexed T, tsk::is_indexed Y>
    double get(T&, Y&) const;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & dimInput;
        ar & dimOutput;
    }
};

template <tsk::is_indexed T>
std::vector<double> tsk::layers::FuzzyLayer::get(T& x) const {
    if(x.size() != dimInput)
        throw std::runtime_error("the size of the input vector is not equal to the dimension of the fuzzification layer");
    std::vector<double> y(dimOutput);
    
    int M = dimOutput / dimInput;
    for(int i = 0; i < dimInput; i++) {
        for(int j = 0; j < M; j++) {
            int k = i*M+j;
            y[k] = fuzzyFunction(x[i], sigma[k], c[k], b[k]);
        }
    }
    return y;
}

template <tsk::is_indexed T, tsk::is_indexed Y>
std::vector<double> tsk::layers::MultipleLayer::get(T& v, Y& x) const {
    if(v.size() != dimInput)
        throw std::runtime_error("the size of the input vector is not equal to the dimension of the multiplication layer");
    
    std::vector<double> y(dimOutput);
    int numOfOut = dimOutput / dimInput;
    int numOfIn = x.size();
    for(int i = 0; i < dimInput; i++) {
        for(int j = 0; j < numOfOut; j++) {
            int l = i*numOfOut+j;
            double temp = p[l][0];
            for(int k = 0; k < numOfIn; k++) {
                temp += p[l][k+1] * x[k];
            }
            temp *= v[i];
            y[l] = temp;
        }
    }
    return std::move(y);
}

template <tsk::is_indexed T>
std::vector<double> tsk::layers::RoleMultipleLayer::get(T& x) const {
    if(x.size() != dimInput)
        throw std::runtime_error("the size of the input vector is not equal to the dimension of the multiplication layer");
    
    std::vector<double> y(dimOutput);
    int N = dimInput / dimOutput;
    for(int i = 0; i < dimOutput; i++) {
        y[i] = 1;
        for(int j = 0; j < N; j++) {
            int k = i+j*dimOutput;
            y[i] *= x[k];
        }
    }

    return std::move(y);
}

template <tsk::is_indexed T, tsk::is_indexed Y>
double tsk::layers::SumLayer::get(T& x, Y& v) const {
    /**
     * x - выход предыдущего слоя
     * v - выход слоя role_multiple
     * методом поддерживается только модель с одним выходом
     */
    double y = std::accumulate(x.begin(), x.end(), 0.0);
    double sum = std::accumulate(v.begin(), v.end(), 0.0);

    return y/sum;
}

#endif