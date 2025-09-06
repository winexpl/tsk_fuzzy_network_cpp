#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include "tsk_fuzzy_network/learning_algorithms.h"

cudaError_t learningKernelWrapper(
    const boost::multi_array<double, 2>& input_vectors,
    const std::vector<double>& targets,
    int classesCount,
    int start_idx, int end_idx,
    learning::TrainingConfig config,
    int num_features, int num_rules,
    std::vector<double>& c_params,
    std::vector<double>& sigma_params,
    std::vector<double>& b_params,
    boost::multi_array<double, 2>& linear_params,
    tsk::TSK* fuzzy_model,
    std::optional<learning::CallbackGetMetrics2Step> callbackGetMetrics2Step = std::nullopt, double error = 0);