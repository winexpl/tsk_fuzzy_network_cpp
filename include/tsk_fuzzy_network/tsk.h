#ifndef TSK_MODEL
#define TSK_MODEL

#include "tsk_fuzzy_network/layers.h"
#include <iostream>


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

template <tsk::is_double_indexed T>
std::vector<double> tsk::TSK::predict(T& x) {
    for(int i = 0; i < x.size(); i++) {
        auto xi = x[i];
        predict(xi);
    }
}

template <tsk::is_double_indexed T, tsk::is_indexed Y>
std::vector<double> tsk::TSK::evaluate(T& x, Y& t) {
    /**
     * x - входные векторы
     * t - ожидаемые значения
     * x.size() == t.size()
     * return значения которые дает модель
     */

}

template <tsk::is_indexed T>
double tsk::TSK::predict(T& x) {
    std::vector<double> y1 = fuzzy_layer.get(x);
    std::vector<double> y2 = role_multiple_layer.get(y1);
    std::vector<double> y3 = multiple_layer.get(y2, x);
    double y4 = sum_layer.get(y3, y2);
    return y4;
}

#endif