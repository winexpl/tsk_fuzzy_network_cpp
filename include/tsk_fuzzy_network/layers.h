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


#endif