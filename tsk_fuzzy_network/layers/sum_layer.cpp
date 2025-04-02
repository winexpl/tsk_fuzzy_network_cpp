#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include <algorithm>

tsk::layers::SumLayer::SumLayer(int dimInput, int dimOutput) :
    Layer(dimInput, dimOutput) { }