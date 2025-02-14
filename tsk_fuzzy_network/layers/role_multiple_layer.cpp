#include "tsk_fuzzy_network/layers.h"

tsk::layers::role_multiple_layer::role_multiple_layer(int dim_input, int dim_output) :
    layer(dim_input, dim_output) { }

std::vector<double> &&tsk::layers::role_multiple_layer::get(std::vector<double>& x) {
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