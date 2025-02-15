#ifndef TSK_MODEL
#define TSK_MODEL

#include "tsk_fuzzy_network/layers.h"


namespace tsk {
    struct TSK;
}

struct tsk::TSK {
    TSK(int N, int M, int out=1);

    void update_p(boost::multi_array<double, 2>&& p);
    
    template <is_double_indexed T>
    std::vector<double> predict(T&);

    template <is_double_indexed T, is_indexed Y>
    std::vector<double> evaluate(T&, Y&);
    
    template <is_indexed T>
    double predict(T&);
private:
    tsk::layers::fuzzy_layer fuzzy_layer;
    tsk::layers::role_multiple_layer role_multiple_layer;
    tsk::layers::multiple_layer multiple_layer;
    tsk::layers::sum_layer sum_layer;
    
    int N; // число параметров
    int M; // число правил
    int out; // число выходов
};

#endif