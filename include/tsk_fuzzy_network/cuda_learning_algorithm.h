// learning/cuda_kernels.h
#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void CUDA_CALLABLE learningKernel(
    double *c, double *sigma, double *b,
    const double *vectors, const double *targets,
    const double *params, int numRoles, int paramSize,
    int n, int m, int num_features, int startIndex,
    int endIndex, double nuC, double nuSigma, double nuB,
    int gridSize, int blockSize);

void learningKernelWrapper(int gridSize, int blockSize, double *c, double *sigma, double *b,
                           const double *vectors, const double *targets,
                           const double *params, int numRoles, int paramSize,
                           int n, int m, int num_features, int startIndex,
                           int endIndex, double nuC, double nuSigma, double nuB);