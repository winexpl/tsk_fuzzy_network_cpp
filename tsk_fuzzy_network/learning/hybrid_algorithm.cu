#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <boost/multi_array.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "metric.h"
#include "tsk_fuzzy_network/learning_algorithms.h"
#include "tsk_fuzzy_network/tsk.h"
#include <iomanip>

#define CHECK_CUDA_ERROR(err)                                                            \
    if (err != cudaSuccess)                                                              \
    {                                                                                    \
        printf("CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                                                         \
    }

namespace
{
    constexpr int WARP_SIZE = 32;
    constexpr int OPTIMAL_BLOCK_SIZE = 256;
    constexpr double EPSILON = 1e-8;
    constexpr int UNROLL_FACTOR = 4;

    // Динамические буферы моментума
    __device__ double *momentum_c = nullptr;
    __device__ double *momentum_sigma = nullptr;
    __device__ double *momentum_b = nullptr;
}

__device__ __forceinline__ double fuzzy_membership(double x, double sigma, double c, double b)
{
    const double t = (x - c) / sigma;
    const double t_sq = t * t;
    return 1.0 / (1.0 + pow(t_sq, b));
}

__device__ __forceinline__ void clip_gradient(double *grad, double clip_value)
{
    double grad_norm = sqrt(*grad * *grad);
    if (grad_norm > clip_value)
    {
        *grad = (*grad) * clip_value / (grad_norm + EPSILON);
    }
}

__global__ void fuzzy_learning_kernel(
    double *__restrict__ c,
    double *__restrict__ sigma,
    double *__restrict__ b,
    const double *__restrict__ input,
    const double target_error,
    const double *__restrict__ params,
    int num_features,
    int num_rules,
    learning::TrainingConfig config)
{

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_features * num_rules)
        return;

    const int feature_idx = tid / num_rules;
    const int rule_idx = tid % num_rules;

    double grad_c = 0, grad_sigma = 0, grad_b = 0;
    const double x = input[feature_idx];
    const double current_sigma = sigma[tid];
    const double current_c = c[tid];
    const double current_b = b[tid];

    // Вычисления градиентов
    double m_val = 0.0;
    for (int rule = 0; rule < num_rules; ++rule)
    {
        double product = 1.0;
        for (int f = 0; f < num_features; ++f)
        {
            const int idx = f * num_rules + rule;
            product *= fuzzy_membership(input[f], sigma[idx], c[idx], b[idx]);
        }
        m_val += product;
    }

    double inv_m_sq = 1.0 / (m_val * m_val);

    for (int k = 0; k < num_rules; ++k)
    {
        double linear_comb = params[k * (num_features + 1)];
        for (int j = 1; j <= num_features; ++j)
        {
            linear_comb += params[k * (num_features + 1) + j] * input[j - 1];
        }

        double l_val = 1;

        for (int f = 0; f < num_features; ++f)
        {
            const int idx = f * num_rules + k;
            l_val *= fuzzy_membership(input[f], sigma[idx], c[idx], b[idx]);
        }

        double membership_product = 1.0;
        for (int j = 0; j < num_features; ++j)
        {
            if (j != feature_idx)
            {
                membership_product *= fuzzy_membership(input[j],
                                                       sigma[j * num_rules + rule_idx],
                                                       c[j * num_rules + rule_idx],
                                                       b[j * num_rules + rule_idx]);
            }
        }

        double common_factor = ((k == rule_idx) * m_val - l_val) *
                               membership_product * inv_m_sq;

        double t = (x - current_c) / current_sigma;
        double t_sq = t * t;
        double t_pow_2b = pow(t_sq, current_b);
        double inv_denom = 1.0 / pow(1.0 + t_pow_2b, 2);

        // Градиенты
        grad_c = linear_comb * common_factor *
                  (2 * current_b * pow(t_sq, current_b - 0.5) / current_sigma * inv_denom);

        grad_sigma = linear_comb * common_factor *
                      (2 * current_b * t_pow_2b * t / current_sigma) * inv_denom;

        grad_b = linear_comb * common_factor *
                  (-2 * t_pow_2b * log(fabs(t)) * inv_denom);
    }

    grad_c *= target_error;
    grad_sigma *= target_error;
    grad_b *= target_error;

    __syncthreads();
    // Обработка градиентов
    clip_gradient(&grad_c, config.grad_clip);
    clip_gradient(&grad_sigma, config.grad_clip);
    clip_gradient(&grad_b, config.grad_clip);

    double newC = c[tid] - config.nu_c * grad_c;
    double newB = b[tid] - config.nu_b * grad_b;
    double newSigma = sigma[tid] - config.nu_sigma * grad_sigma;
    if(!std::isnan(newC)) c[tid] = newC;
    if(!std::isnan(newB)) b[tid] = newB;
    if(!std::isnan(newSigma)) sigma[tid] = newSigma;
}

void print_parameter_changes(
    const std::vector<double> &old_params,
    const std::vector<double> &new_params,
    const std::string &param_name,
    int max_to_print = 5)
{
    std::cout << "\n=== " << param_name << " Parameter Changes ===\n";
    for (size_t i = 0; i < std::min(old_params.size(), new_params.size()); ++i)
    {
        if (i < max_to_print || i >= new_params.size() - max_to_print)
        {
            std::cout << std::fixed << std::setprecision(5)
                      << "Param " << i << ": "
                      << old_params[i] << " → " << new_params[i]
                      << " (Δ=" << new_params[i] - old_params[i] << ")\n";
        }
        if (i == max_to_print)
        {
            std::cout << "...\n";
        }
    }
}

cudaError_t learningKernelWrapper(
    const boost::multi_array<double, 2> &input_vectors,
    const std::vector<double> &targets,
    int classesCount,
    int start_idx, int end_idx,
    learning::TrainingConfig config,
    int num_features, int num_rules,
    std::vector<double> &c_params,
    std::vector<double> &sigma_params,
    std::vector<double> &b_params,
    boost::multi_array<double, 2> &linear_params,
    tsk::TSK *fuzzy_model,
    std::optional<learning::CallbackGetMetrics2Step> callbackGetMetrics2Step, double default_error)
{
    cudaError_t err = cudaSuccess;

    // Перенос данных на устройство
    thrust::device_vector<double> d_c(c_params.begin(), c_params.end());
    thrust::device_vector<double> d_sigma(sigma_params.begin(), sigma_params.end());
    thrust::device_vector<double> d_b(b_params.begin(), b_params.end());
    thrust::device_vector<double> d_p(linear_params.data(),
                                      linear_params.data() + num_rules * (num_features + 1));
    thrust::device_vector<double> d_x(num_features);

    // Конфигурация ядра
    const int total_params = num_features * num_rules;
    const int block_size = std::min(OPTIMAL_BLOCK_SIZE, total_params);
    const int grid_size = (total_params + block_size - 1) / block_size;
    const size_t shared_mem_size = num_features * sizeof(double);

    double total_error = 0.0;
    double total_accuracy = 0.0;
    std::vector<double> predictes;

    std::vector<double> old_c = c_params;
    std::vector<double> old_sigma = sigma_params;
    std::vector<double> old_b = b_params;

    for (int i = start_idx; i < end_idx; ++i)
    {
        const double *x = input_vectors[i].origin();
        thrust::copy(x, x + num_features, d_x.begin());

        double predict = fuzzy_model->predict1(input_vectors[i]);
        double error = predict - targets[i];
        // error = error != 0 ? error: default_error;
        double accuracy = predict;

        fuzzy_learning_kernel<<<grid_size, block_size, shared_mem_size>>>(
            thrust::raw_pointer_cast(d_c.data()),
            thrust::raw_pointer_cast(d_sigma.data()),
            thrust::raw_pointer_cast(d_b.data()),
            thrust::raw_pointer_cast(d_x.data()),
            error,
            thrust::raw_pointer_cast(d_p.data()),
            num_features,
            num_rules,
            config);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        // Обновление модели
        if (fuzzy_model != nullptr)
        {
            thrust::copy(d_c.begin(), d_c.end(), c_params.begin());
            thrust::copy(d_sigma.begin(), d_sigma.end(), sigma_params.begin());
            thrust::copy(d_b.begin(), d_b.end(), b_params.begin());

            fuzzy_model->setC(c_params);
            fuzzy_model->setSigma(sigma_params);
            fuzzy_model->setB(b_params);
        }

        predict = fuzzy_model->predict1(input_vectors[i]);
        double current_error = predict - targets[i];
        total_error += current_error * current_error;
        predictes.push_back(predict);
    }

    total_error /= (end_idx - start_idx);
    total_accuracy = metric::Metric::calculateAccuracy(targets, predictes, classesCount);

    metric::Metric metric{total_accuracy, total_error};
    if(callbackGetMetrics2Step.has_value()) (*callbackGetMetrics2Step)(metric);
    // Print training summary
    // std::cout << "\n=== Training Summary ===";
    // print_parameter_changes(old_c, c_params, "Center (c)");
    // print_parameter_changes(old_sigma, sigma_params, "Width (σ)");
    // print_parameter_changes(old_b, b_params, "Shape (b)");
    // std::cout << "\nAverage RMSE: " << sqrt(total_error) << "\n";
    return cudaSuccess;
}