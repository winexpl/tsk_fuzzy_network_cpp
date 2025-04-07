#ifndef TSK_LAYERS
#define TSK_LAYERS

#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <utility>
#include "boost/multi_array.hpp"


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

    template <tsk::is_indexed T>
    std::vector<double> get(T&);
};

struct tsk::layers::MultipleLayer : tsk::layers::Layer {
    /**
     * third layer
     */
    boost::multi_array<double, 2> p;
    
    MultipleLayer(int dimInput, int dimOutput, int N);

    template <tsk::is_indexed T, tsk::is_indexed Y>
    std::vector<double> get(T&, Y&);
};

struct tsk::layers::RoleMultipleLayer : tsk::layers::Layer {
    /**
     * second layer
     */
    RoleMultipleLayer(int dimInput, int dimOutput);

    template <tsk::is_indexed T>
    std::vector<double> get(T&);
};

struct tsk::layers::SumLayer : tsk::layers::Layer {
    /**
     * fourth layer
     */
    SumLayer(int dimInput, int dimOutput);

    template <tsk::is_indexed T, tsk::is_indexed Y>
    double get(T&, Y&);
};

template <tsk::is_indexed T>
std::vector<double> tsk::layers::FuzzyLayer::get(T& x) {
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
std::vector<double> tsk::layers::MultipleLayer::get(T& v, Y& x) {
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
std::vector<double> tsk::layers::RoleMultipleLayer::get(T& x) {
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
double tsk::layers::SumLayer::get(T& x, Y& v) {
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