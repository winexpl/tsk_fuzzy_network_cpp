#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include <exception>
#include <algorithm>
#include <execution>

double tsk::layers::generalGaussian(double x, double sigma, double c, double b) {
    double value = 1.0 / (1.0 + std::pow( (x-c)/sigma, 2*(int)b));
    return value;
};

tsk::layers::FuzzyLayer::FuzzyLayer(int dimInput, int dimOutput) :
    Layer(dimInput, dimOutput),
    fuzzyFunction(tsk::layers::generalGaussian),
    sigma(std::vector<double>(dimOutput)),
    c(std::vector<double>(dimOutput)),
    b(std::vector<double>(dimOutput))
{
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disSigma(0, 10);
    std::uniform_real_distribution<> disC(0, 10);
    std::uniform_real_distribution<> disB(1, 5);

    std::generate(std::execution::par, sigma.begin(), sigma.end(), [&]() { return disSigma(gen); });
    std::generate(std::execution::par, c.begin(), c.end(), [&]() { return disC(gen); });
    std::generate(std::execution::par, b.begin(), b.end(), [&]() { return disB(gen); });
}

