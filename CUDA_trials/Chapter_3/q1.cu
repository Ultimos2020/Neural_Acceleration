#include <cstdio>
#include <cuda_runtime.h>

__global__ void col_calculation_kernel(float *M, float *N, int width, int height) {


    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width) {
        //for (int row = 0; row < height; ++row) {
            float Pvalue = 0;
            for (int k = 0; k < width; ++k) {
             Pvalue += M[k * width + col] * N[k * width + col];
            }
            N[row * width + col] = Pvalue;
        //}
    }

}
