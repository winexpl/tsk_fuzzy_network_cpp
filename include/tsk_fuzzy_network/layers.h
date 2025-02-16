#ifndef TSK_LAYERS
#define TSK_LAYERS

#include <functional>
#include <vector>
#include <cmath>
#include <random>
#include "boost/multi_array.hpp"


namespace tsk {
    template <typename T>
    concept is_indexed = requires (T a, int i) {
        { a[i] } -> std::convertible_to<typename T::value_type>;
        requires !requires { a[i][0]; };
    };

    template <typename T>
    concept is_double_indexed = requires (T a, int i, int j) {
        { a[i][j] } -> std::convertible_to<typename T::value_type>;
        requires !requires { a[i][j][0]; };
    };

    template <is_indexed T, is_indexed Y>
    bool is_same_length(const T& a1, const Y& a2) {
        return (a1.cend() - a1.cbegin()) == (a2.cend() - a2.cbegin());
    }
    namespace layers {
        struct layer;
        struct fuzzy_layer;
        struct multiple_layer;
        struct role_multiple_layer;
        struct sum_layer;

        double general_gaussian(double x, double sigma, double c, double b);

        static int write(std::ostream& os, layer& layer);

        inline std::random_device rd;
    }
}

struct tsk::layers::layer {
    layer(int dim_input, int dim_output);
    int dim_input;
    int dim_output;
};

struct tsk::layers::fuzzy_layer : public tsk::layers::layer {
    /**
     * first layer
     */
    std::function<double(double,double,double,double)> fuzzy_function;

    std::vector<double> sigma;
    std::vector<double> c;
    std::vector<double> b;

    fuzzy_layer(int dim_input, int dim_output);

    template <tsk::is_indexed T>
    std::vector<double> get(T&);
};

struct tsk::layers::multiple_layer : tsk::layers::layer {
    /**
     * third layer
     */
    boost::multi_array<double, 2> p;
    
    multiple_layer(int dim_input, int dim_output, int N);

    template <tsk::is_indexed T, tsk::is_indexed Y>
    std::vector<double> get(T&, Y&);
};

struct tsk::layers::role_multiple_layer : tsk::layers::layer {
    /**
     * second layer
     */
    role_multiple_layer(int dim_input, int dim_output);

    template <tsk::is_indexed T>
    std::vector<double> get(T&);
};

struct tsk::layers::sum_layer : tsk::layers::layer {
    /**
     * fourth layer
     */
    sum_layer(int dim_input, int dim_output);

    template <tsk::is_indexed T, tsk::is_indexed Y>
    double get(T&, Y&);
};

template <tsk::is_indexed T>
std::vector<double> tsk::layers::fuzzy_layer::get(T& x) {
    if(x.size() != dim_input)
        throw std::runtime_error("the size of the input vector is not equal to the dimension of the fuzzification layer");
    std::vector<double> y(dim_output);
    
    int M = dim_output / dim_input;
    for(int i = 0; i < dim_input; i++) {
        for(int j = 0; j < M; j++) {
            int k = i*M+j;
            y[k] = fuzzy_function(x[i], sigma[k], c[k], b[k]);
        }
    }
    return std::move(y);
}

template <tsk::is_indexed T, tsk::is_indexed Y>
std::vector<double> tsk::layers::multiple_layer::get(T& v, Y& x) {
    if(v.size() != dim_input)
        throw std::runtime_error("the size of the input vector is not equal to the dimension of the multiplication layer");
    
    std::vector<double> y(dim_output);
    int num_of_out = dim_output / dim_input;
    int num_of_in = x.size();
    for(int i = 0; i < dim_input; i++) {
        for(int j = 0; j < num_of_out; j++) {
            int l = i*num_of_out+j;
            double temp = p[l][0];
            for(int k = 0; k < num_of_in; k++) {
                temp += p[l][k+1] * x[k];
            }
            temp *= v[i];
            y[l] = temp;
        }
    }
    return std::move(y);
}

template <tsk::is_indexed T>
std::vector<double> tsk::layers::role_multiple_layer::get(T& x) {
    if(x.size() != dim_input)
        throw std::runtime_error("the size of the input vector is not equal to the dimension of the multiplication layer");
    
    std::vector<double> y(dim_output);
    int N = dim_input / dim_output;
    for(int i = 0; i < dim_output; i++) {
        y[i] = 1;
        for(int j = 0; j < N; j++) {
            int k = i+j*dim_output;
            y[i] *= x[k];
        }
    }

    return std::move(y);
}

template <tsk::is_indexed T, tsk::is_indexed Y>
double tsk::layers::sum_layer::get(T& x, Y& v) {
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