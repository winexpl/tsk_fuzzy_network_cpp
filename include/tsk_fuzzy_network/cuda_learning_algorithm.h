#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

cudaError_t learningKernelWrapper(const boost::multi_array<double, 2> &vectors, const std::vector<double> &targets, int startIndex, int endIndex,
    double nu, int n, int m, std::vector<double> &c, std::vector<double> &sigma, std::vector<double>& b,
    boost::multi_array<double, 2>& p, tsk::TSK* tsk);