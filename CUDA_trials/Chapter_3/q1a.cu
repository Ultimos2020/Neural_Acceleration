#include <cstdio>
#include <cuda_runtime.h>

__global__ void row_calculation_kernel(float *M, float *N, float *P, int width, int height) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height) {
        for (int col = 0; col < width; ++col) {
            float Pvalue = 0;
            for (int k = 0; k < width; ++k) {
                Pvalue += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = Pvalue;
        }
    }

}

int main() {
    const int width = 4;
    const int height = 4;

    float h_P[width * height] = {0};

    float h_N[width * height] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    int number_of_threads = width;
    float h_M[width * height] = {
       1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10, 11, 12,
       13, 14, 15, 16
    };

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, width * height * sizeof(float));
    cudaMalloc((void**)&d_N, width * height * sizeof(float));
    cudaMalloc((void**)&d_P, width * height * sizeof(float));

    cudaMemcpy(d_M, h_M, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, width * height * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 16;
    int gridSize = (number_of_threads + blockSize - 1) / blockSize;

    row_calculation_kernel<<<gridSize, blockSize>>>(d_M, d_N, d_P,width, height);

    cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result matrix N after column-wise calculation:\n");
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%f ", h_P[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}