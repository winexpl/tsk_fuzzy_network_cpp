#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"


tsk::TSK::TSK(int N, int M, int out)
    : fuzzy_layer{tsk::layers::fuzzy_layer(N, M*N)},
    role_multiple_layer{tsk::layers::role_multiple_layer(M*N, M)},
    multiple_layer{tsk::layers::multiple_layer(M, M*out, N)},
    sum_layer{tsk::layers::sum_layer(M*out, out)}
{ }

void tsk::TSK::update_p(boost::multi_array<double, 2>&& p) {
    multiple_layer.p = p;
}