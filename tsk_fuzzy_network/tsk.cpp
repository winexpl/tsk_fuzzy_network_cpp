#include "tsk_fuzzy_network/tsk.h"
#include "tsk_fuzzy_network/layers.h"

tsk::TSK::TSK(int N, int M, int out) 
    : fuzzy_layer{tsk::layers::fuzzy_layer(N, M)},
    role_multiple_layer{tsk::layers::role_multiple_layer(M*N, M)},
    multiple_layer{tsk::layers::multiple_layer(M, M*out, N)},
    sum_layer{tsk::layers::sum_layer(M*out, out)}
{ }

void tsk::TSK::update_p(boost::multi_array<double, 2>&& p) {
    multiple_layer.p = p;
}

std::vector<double> tsk::TSK::predict(boost::multi_array<double, 2>& x) {
    
}

std::vector<double> tsk::TSK::evaluate(boost::multi_array<double, 2>& x) {
    
}

double tsk::TSK::predict(std::vector<double>& x) {
    std::vector<double>&& y1 = fuzzy_layer.get(x);
    std::vector<double>&& y2 = role_multiple_layer.get(y1);
    std::vector<double>&& y3 = multiple_layer.get(y2, x);
    double y4 = sum_layer.get(y3, y2);
}