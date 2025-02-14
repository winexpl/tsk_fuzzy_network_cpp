#include "tsk_fuzzy_network/layers.h"
#include <exception>
#include <algorithm>
#include <execution>

double tsk::layers::general_gaussian(double x, double sigma, double c, double b) {
    return 1 / (1 + pow( (x-c)/sigma, 2*b));
};

tsk::layers::fuzzy_layer::fuzzy_layer(int dim_input, int dim_output) :
    layer(dim_input, dim_output),
    fuzzy_function(tsk::layers::general_gaussian),
    sigma(std::vector<double>(dim_output)),
    c(std::vector<double>(dim_output)),
    b(std::vector<double>(dim_output))
{
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_sigma(0.1, 1);
    std::uniform_int_distribution<> dis_c(0.1, 1);
    std::uniform_int_distribution<> dis_b(1, 5);

    std::generate(std::execution::par, sigma.begin(), sigma.end(), [&]() { return dis_sigma(gen); });
    std::generate(std::execution::par, c.begin(), c.end(), [&]() { return dis_c(gen); });
    std::generate(std::execution::par, b.begin(), b.end(), [&]() { return dis_b(gen); });
}

std::vector<double> &&tsk::layers::fuzzy_layer::get(std::vector<double>& x) {
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


