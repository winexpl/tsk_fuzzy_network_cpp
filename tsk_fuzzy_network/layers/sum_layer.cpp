#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include <algorithm>

tsk::layers::sum_layer::sum_layer(int dim_input, int dim_output) :
    layer(dim_input, dim_output) { }

template <tsk::is_indexed T, tsk::is_indexed Y>
double tsk::layers::sum_layer::get(T& x, Y& v) {
    /**
     * x - выход предыдущего слоя
     * v - выход слоя role_multiple
     * методом поддерживается только модель с одним выходом
     */
    double y = std::accumulate(x.begin(), x.end(), 0);
    double sum = std::accumulate(v.begin(), v.end(), 0);

    return y/sum;
}