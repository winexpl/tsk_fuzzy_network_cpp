#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
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
    std::uniform_real_distribution<> dis_sigma(0.1, 1);
    std::uniform_real_distribution<> dis_c(0.1, 1);
    std::uniform_real_distribution<> dis_b(1, 5);

    std::generate(std::execution::par, sigma.begin(), sigma.end(), [&]() { return dis_sigma(gen); });
    std::generate(std::execution::par, c.begin(), c.end(), [&]() { return dis_c(gen); });
    std::generate(std::execution::par, b.begin(), b.end(), [&]() { return dis_b(gen); });
}

