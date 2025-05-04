#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <boost/multi_array.hpp>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include "tsk_fuzzy_network/learning_algorithms.h"
#include "tsk_fuzzy_network/tsk.h"

typedef double (*GradientFunc)(double, double, double, double);

namespace
{
    constexpr int WARP_SIZE = 32;
    constexpr int OPTIMAL_BLOCK_SIZE = 256;
    constexpr double EPSILON = 1e-100;

    __device__ __forceinline__ double safe_pow(double a, double b)
    {
        return (a <= 0.0) ? 0.0 : __expf(b * __logf(a));
    }

    __device__ __forceinline__ double safe_divide(double a, double b)
    {
        return (fabs(b) < EPSILON) ? 0.0 : __fdividef(a, b);
    }
}

__device__ __forceinline__ double fuzzyFunction(double x, double sigma, double c, double b)
{
    const double t = __fdividef(x - c, sigma);
    const double t_sq = t * t;
    const double denominator = __fadd_rn(1.0, safe_pow(t_sq, b));
    return safe_divide(1.0, denominator);
}

__device__ __forceinline__ int deltaKronecker(int i, int j)
{
    return (i == j) ? 1 : 0;
}

__device__ double compute_m(const double *x, const double *c, const double *sigma,
                            const double *b, int n, int m)
{
    double res = 0.0;
    for (int i = 0; i < m; ++i)
    {
        double temp = 1.0;
        for (int j = 0; j < n; ++j)
        {
            int idx = j * m + i;
            temp = __fmul_rn(temp, fuzzyFunction(x[j], sigma[idx], c[idx], b[idx]));
        }
        res = __fadd_rn(res, temp);
    }
    return fmax(res, EPSILON);
}

__device__ double compute_l(const double *x, int i, const double *c,
                            const double *sigma, const double *b, int n, int m)
{
    double temp = 1.0;
    for (int j = 0; j < n; ++j)
    {
        int idx = j * m + i;
        temp = __fmul_rn(temp, fuzzyFunction(x[j], sigma[idx], c[idx], b[idx]));
    }
    return fmax(temp, EPSILON);
}

__device__ double dWdSigma(double x, double sigma, double c, double b)
{
    const double t = safe_divide(x - c, sigma);
    const double t_sq = t * t;
    const double t_pow = safe_pow(t_sq, b);
    const double denom = safe_pow(1.0 + t_sq, 2.0);
    return safe_divide(2.0 * b * t_pow, __fmul_rn(sigma, denom));
}

__device__ double dWdC(double x, double sigma, double c, double b)
{
    const double t = safe_divide(x - c, sigma);
    const double t_sq = t * t;
    const double t_pow = safe_pow(t_sq, b - 0.5);
    const double denom = safe_pow(1.0 + t_sq, 2.0);
    return safe_divide(2.0 * b * t * t_pow, __fmul_rn(sigma, denom));
}

__device__ double dWdB(double x, double sigma, double c, double b)
{
    const double t = safe_divide(x - c, sigma);
    if (fabs(t) < EPSILON)
        return 0.0;
    const double t_sq = t * t;
    const double t_pow = safe_pow(t_sq, b);
    const double denom = safe_pow(1.0 + t_sq, 2.0);
    const double log_t = __logf(fabs(t));
    return safe_divide(-2.0 * t_pow * log_t, denom);
}

__device__ void compute_gradients(const double *x, const double *sigma, const double *c,
                                  const double *b, const double *p, int n, int m,
                                  double *dEdC, double *dEdSigma, double *dEdB)
{
    const int paramNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (paramNum >= n * m)
        return;

    const int featureIdx = paramNum / m;
    const int clusterIdx = paramNum % m;

    double weightedSumC = 0.0;
    double weightedSumSigma = 0.0;
    double weightedSumB = 0.0;

    const double m_val = compute_m(x, c, sigma, b, n, m);

    for (int roleIdx = 0; roleIdx < m; ++roleIdx)
    {
        double linearCombination = p[roleIdx * (n + 1)];
#pragma unroll 4
        for (int paramIdx = 1; paramIdx < n + 1; ++paramIdx)
        {
            linearCombination = __fma_rn(p[roleIdx * (n + 1) + paramIdx], x[paramIdx - 1], linearCombination);
        }

        const double l_val = compute_l(x, roleIdx, c, sigma, b, n, m);
        const double denominator = __fmul_rn(m_val, m_val);
        const double numerator = __fsub_rn(__fmul_rn(deltaKronecker(roleIdx, clusterIdx), m_val), l_val);

        double membershipProduct = 1.0;
        for (int j = 0; j < n; ++j)
        {
            if (j != featureIdx)
            {
                const int idx = j * m + clusterIdx;
                membershipProduct = __fmul_rn(membershipProduct,
                                              fuzzyFunction(x[j], sigma[idx], c[idx], b[idx]));
            }
        }

        const double commonFactor = safe_divide(
            __fmul_rn(numerator, membershipProduct),
            denominator);

        const double x_val = x[featureIdx];
        const double currentSigma = sigma[paramNum];
        const double currentC = c[paramNum];
        const double currentB = b[paramNum];

        weightedSumC = __fadd_rn(weightedSumC,
                                 __fmul_rn(linearCombination,
                                           __fmul_rn(commonFactor, dWdC(x_val, currentSigma, currentC, currentB))));

        weightedSumSigma = __fadd_rn(weightedSumSigma,
                                     __fmul_rn(linearCombination,
                                               __fmul_rn(commonFactor, dWdSigma(x_val, currentSigma, currentC, currentB))));

        weightedSumB = __fadd_rn(weightedSumB,
                                 __fmul_rn(linearCombination,
                                           __fmul_rn(commonFactor, dWdB(x_val, currentSigma, currentC, currentB))));
    }

    *dEdC = weightedSumC;
    *dEdSigma = weightedSumSigma;
    *dEdB = weightedSumB;
}

__device__ double device_predict(const double *x, const double *sigma, const double *c,
                                 const double *b, const double *p, int n, int m)
{
    double final_product = 0.0;
    double sum_activation = 0.0;

    for (int i = 0; i < m; ++i)
    {
        double activation = 1.0;
        for (int j = 0; j < n; ++j)
        {
            int idx = j * m + i;
            activation = __fmul_rn(activation, fuzzyFunction(x[j], sigma[idx], c[idx], b[idx]));
        }

        double rule_output = p[i * (n + 1)];
        for (int k = 0; k < n; ++k)
        {
            rule_output = __fma_rn(p[i * (n + 1) + k + 1], x[k], rule_output);
        }

        final_product = __fadd_rn(final_product, __fmul_rn(rule_output, activation));
        sum_activation = __fadd_rn(sum_activation, activation);
    }

    return safe_divide(final_product, sum_activation);
}

__global__ void learningKernel(double *c, double *sigma, double *b,
                               const double *vector, const double target,
                               const double *params, int n, int m, int dim,
                               double nu, double *delta_c, double *delta_sigma, double *delta_b)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_params = n * m;

    if (tid >= total_params)
        return;

    double output = device_predict(vector, sigma, c, b, params, n, m);
    double error = __fsub_rn(output, target);

    double dEdC, dEdSigma, dEdB;
    compute_gradients(vector, sigma, c, b, params, n, m, &dEdC, &dEdSigma, &dEdB);

    if (!isnan(dEdC))
        delta_c[tid] = __fmul_rn(-nu, __fmul_rn(error, dEdC));
    if (!isnan(dEdSigma))
        delta_sigma[tid] = __fmul_rn(-nu, __fmul_rn(error, dEdSigma));
    if (!isnan(dEdB))
        delta_b[tid] = __double2int_rn(__fsub_rn(delta_b[tid], __fmul_rn(nu, __fmul_rn(error, dEdB))));
}
cudaError_t learningKernelWrapper(
    const boost::multi_array<double, 2> &vectors, const std::vector<double> &targets, int startIndex, int endIndex,
    double nu, int n, int m, std::vector<double> &c, std::vector<double> &sigma, std::vector<double> &b,
    boost::multi_array<double, 2> &p, tsk::TSK *tsk)
{
    const int total_params = n * m;

    // Проверка размеров
    if (vectors.shape()[0] < endIndex || targets.size() < endIndex)
    {
        return cudaErrorInvalidValue;
    }

    thrust::device_vector<double> d_c(c.begin(), c.end());
    thrust::device_vector<double> d_sigma(sigma.begin(), sigma.end());
    thrust::device_vector<double> d_b(b.begin(), b.end());
    thrust::device_vector<double> d_params(p.data(), p.data() + m * (n + 1));

    thrust::device_vector<double> d_delta_c(total_params, 0.0);
    thrust::device_vector<double> d_delta_sigma(total_params, 0.0);
    thrust::device_vector<double> d_delta_b(total_params, 0.0);

    const int numElements = n * m;
    const int numBlocks = (total_params + numElements - 1) / numElements;
    size_t sharedMemSize = (n * m + 2 * m) * sizeof(double);

    for (int i = startIndex; i < endIndex; ++i)
    {
        const double *x = vectors[i].origin();
        const double target = targets[i];

        thrust::device_vector<double> d_x(x, x + n);

        learningKernel<<<numBlocks, numElements, sharedMemSize>>>(
            thrust::raw_pointer_cast(d_c.data()),
            thrust::raw_pointer_cast(d_sigma.data()),
            thrust::raw_pointer_cast(d_b.data()),
            thrust::raw_pointer_cast(d_x.data()),
            target,
            thrust::raw_pointer_cast(d_params.data()),
            n, m, vectors.shape()[1], nu,
            thrust::raw_pointer_cast(d_delta_c.data()),
            thrust::raw_pointer_cast(d_delta_sigma.data()),
            thrust::raw_pointer_cast(d_delta_b.data()));

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            return err;

        thrust::transform(d_c.begin(), d_c.end(), d_delta_c.begin(), d_c.begin(), thrust::plus<double>());
        thrust::transform(d_sigma.begin(), d_sigma.end(), d_delta_sigma.begin(), d_sigma.begin(), thrust::plus<double>());
        thrust::transform(d_b.begin(), d_b.end(), d_delta_b.begin(), d_b.begin(), thrust::plus<double>());
    }

    thrust::copy(d_c.begin(), d_c.end(), c.begin());
    thrust::copy(d_sigma.begin(), d_sigma.end(), sigma.begin());
    thrust::copy(d_b.begin(), d_b.end(), b.begin());

    tsk->setC(c);
    tsk->setSigma(sigma);
    tsk->setB(b);

    return cudaSuccess;
}