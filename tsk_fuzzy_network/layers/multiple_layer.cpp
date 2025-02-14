#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include <iostream>
#include <execution>


tsk::layers::multiple_layer::multiple_layer(int dim_input, int dim_output, int N) :
    layer(dim_input, dim_output),
    p(boost::multi_array<double, 2>(boost::extents[dim_input][N+1]))
{
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::mt19937 gen(rd());

    auto begin = p.data();
    auto end = p.data() + p.num_elements();

    std::generate(std::execution::par, begin, end, [&]() { return dis(gen); });
}

template <tsk::is_indexed T, tsk::is_indexed Y>
std::vector<double> tsk::layers::multiple_layer::get(T& v, Y& x) {
    if(x.size() != dim_input)
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