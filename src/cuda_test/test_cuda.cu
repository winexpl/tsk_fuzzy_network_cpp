#include "test_cuda.h"
#include <stdio.h>


__global__ void test() {
    printf("hi from thread#%d", threadIdx.x);
}

void testWrapper() {
    int N = 1;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    test<<<numBlocks, threadsPerBlock>>>();
}
