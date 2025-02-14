#ifndef TSK_MODEL
#define TSK_MODEL

#include "layers.h"

namespace tsk {
    struct TSK;

    template <typename T>
    concept is_indexed = requires (T a, int i) {
        { a[i] } -> std::convertible_to<typename T::value_type>;
        requires !requires { a[i][0]; };
    };

    template <typename T>
    concept is_double_indexed = requires (T a, int i, int j) {
        { a[i][j] } -> std::convertible_to<typename T::value_type>;
        requires !requires { a[i][j][0]; };
    };

    template <is_indexed T, is_indexed Y>
    bool is_same_length(const T& a1, const Y& a2) {
        return (a1.cend() - a1.cbegin()) == (a2.cend() - a2.cbegin());
    }
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