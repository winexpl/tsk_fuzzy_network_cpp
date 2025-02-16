#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include <algorithm>

tsk::layers::sum_layer::sum_layer(int dim_input, int dim_output) :
    layer(dim_input, dim_output) { }