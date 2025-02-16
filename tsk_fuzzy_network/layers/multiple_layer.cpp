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