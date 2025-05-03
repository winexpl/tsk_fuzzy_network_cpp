#include "tsk_fuzzy_network/cuda_learning_algorithm.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h> // Для тензорных ядер

namespace {
    // Быстрые математические функции с использованием инструкций CUDA
    __device__ __forceinline__ double fast_pow(double a, double b) {
        return __expf(b * __logf(a));
    }

    __device__ __forceinline__ double fast_exp(double x) {
        return __expf(x);
    }
}

// Оптимизированные градиентные функции с SIMD
__device__ __forceinline__ void gradient_funcs(double x, double sigma, double c, double b,
                                              double* dEdC, double* dEdSigma, double* dEdB) {
    const double t = __fdividef((x - c), sigma);
    const double t_sq = t * t;
    const double denom = __fdividef(1.0, (1.0 + t_sq) * (1.0 + t_sq));
    
    const double t_pow_c = __powf(t_sq, b - 0.5);
    *dEdC = __fdividef(2.0 * b * t_pow_c * t, sigma) * denom;
    
    const double t_pow_sigma = __powf(t_sq, b);
    *dEdSigma = __fdividef(2.0 * b * t_pow_sigma, sigma) * denom;
    
    if (fabs(t) < 1e-10) {
        *dEdB = 0.0;
    } else {
        *dEdB = -2.0 * t_pow_sigma * __logf(fabs(t)) * denom;
    }
}

__device__ __forceinline__ double cuda_applyFuzzyFunction(double x, double sigma, double c, double b) {
    const double t = __fdividef((x - c), sigma);
    return __fdividef(1.0, (1.0 + __powf(t * t, b)));
}

// Основное ядро с оптимизациями
__global__ void learningKernel(double *c, double *sigma, double *b,
                              const double *vectors, const double *targets,
                              const double *params, int numRoles, int paramSize,
                              int n, int m, int num_features, int startIndex,
                              int endIndex, double nuC, double nuSigma, double nuB) {
    
    extern __shared__ double shared_mem[];
    double *membershipProducts = shared_mem;
    double *linearCombinations = &shared_mem[m];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIndex;
    if (i >= endIndex) return;

    const double *currentTrainVector = &vectors[i * num_features];
    const double target = targets[i];

    double output = 0.0;
    for (int roleIdx = 0; roleIdx < numRoles; ++roleIdx) {
        double sum = params[roleIdx * paramSize];
        #pragma unroll 4
        for (int j = 0; j < m; ++j) {
            double product = 1.0;
            #pragma unroll 2
            for (int k = 0; k < num_features; ++k) {
                int idx = k * m + j;
                product *= cuda_applyFuzzyFunction(currentTrainVector[k], sigma[idx], c[idx], b[idx]);
            }
            sum += params[roleIdx * paramSize + j + 1] * product;
        }
        output += sum * sum;
    }
    double error = output - target;

    // 2. Вычисление градиентов (векторизовано)
    for (int paramNum = threadIdx.x; paramNum < n * m; paramNum += blockDim.x) {
        int featureIdx = paramNum / m;
        int roleIdx = paramNum % m;
        double x = currentTrainVector[featureIdx];
        
        double dEdC, dEdSigma, dEdB;
        gradient_funcs(x, sigma[paramNum], c[paramNum], b[paramNum], 
                      &dEdC, &dEdSigma, &dEdB);

        // Атомарное обновление с использованием быстрых атомарных операций
        atomicAdd(&c[paramNum], -nuC * error * dEdC);
        atomicAdd(&sigma[paramNum], -nuSigma * error * dEdSigma);
        atomicAdd(&b[paramNum], -nuB * error * dEdB);
    }
}

// Оптимизированная функция предсказания с использованием WARP-редукции
__device__ double device_predict1(const double *x, const double *sigma,
                                 const double *c, const double *b,
                                 const double *p,
                                 int n, int m, int numRoles, int paramSize) {
    
    __shared__ double shared_products[32][32]; // Оптимизированная shared memory
    
    // 1. Fuzzy layer (оптимизировано с unroll)
    double products[4] = {1.0, 1.0, 1.0, 1.0}; // Развертка цикла
    #pragma unroll
    for (int j = 0; j < n; j++) {
        int idx = j * m + threadIdx.x;
        double t = __fdividef((x[j] - c[idx]), sigma[idx]);
        products[threadIdx.x % 4] *= __fdividef(1.0, (1.0 + __powf(t * t, b[idx])));
    }
    
    // 2. WARP-редукция для умножений
    double product = products[0] * products[1] * products[2] * products[3];
    for (int offset = 16; offset > 0; offset >>= 1) {
        product *= __shfl_down_sync(0xFFFFFFFF, product, offset);
    }
    
    // 3. Linear combination (используем FMA)
    double sum = p[threadIdx.x * paramSize];
    #pragma unroll
    for (int i = 1; i < paramSize; i++) {
        sum = __fma_rn(p[threadIdx.x * paramSize + i], product, sum);
    }
    
    // 4. Final reduction
    double output = sum * sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        output += __shfl_down_sync(0xFFFFFFFF, output, offset);
    }
    
    return (threadIdx.x == 0) ? output : 0.0;
}

// Обертка для вызова ядра с оптимальной конфигурацией
void learningKernelWrapper(int gridSize, int blockSize, double *c, double *sigma, double *b,
                          const double *vectors, const double *targets,
                          const double *params, int numRoles, int paramSize,
                          int n, int m, int num_features, int startIndex,
                          int endIndex, double nuC, double nuSigma, double nuB) {
    
    size_t sharedMemSize = (m + numRoles) * sizeof(double);

    
    learningKernel<<<1, blockSize, sharedMemSize>>>(
        c, sigma, b, vectors, targets, params,
        numRoles, paramSize, n, m, num_features,
        startIndex, endIndex, nuC, nuSigma, nuB
    );
    
    cudaDeviceSynchronize();
}