#ifndef TSK_MODEL
#define TSK_MODEL

#include "layers.h"

using namespace tsk::layers;

namespace tsk {
    struct TSK;
}

struct tsk::TSK {
    TSK(int N, int M, int out=1);
    tsk::layers::fuzzy_layer fuzzy_layer;

    void update_p(boost::multi_array<double, 2>&& p);
    
    std::vector<double> predict(boost::multi_array<double, 2>&);
    std::vector<double> evaluate(boost::multi_array<double, 2>&);
    double predict(std::vector<double>&);
private:
    
    tsk::layers::role_multiple_layer role_multiple_layer;
    tsk::layers::multiple_layer multiple_layer;
    tsk::layers::sum_layer sum_layer;
    
    int N; // число параметров
    int M; // число правил
    int out; // число выходов
};

#endif