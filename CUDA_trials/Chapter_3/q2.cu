#include <cstdio>
#include <cuda_runtime.h>

__global__ void matrix_vec_mul (float *B, float *C, float *A, int width, int height){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height) {
        float sum = 0;
        for (int col = 0; col < width; col++) {
            sum += B[row * width + col] * C[col];
        }
        A[row] = sum;
    }
}

int main() {
    const int width = 4;
    const int height = 4;

    float h_A[height] = {0};
    float h_B[width * height] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
 
    float h_C[width] = {1,1,1,1};

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, height * sizeof(float));
    cudaMalloc((void**)&d_B, width * height * sizeof(float));
    cudaMalloc((void**)&d_C, width * sizeof(float));

    //cudaMemcpy(d_A, h_A, height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, width * sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 16;
    int threadsPerBlock = height;
    int numBlocks = (height + blocksize - 1) / blocksize;

    matrix_vec_mul<<<numBlocks, threadsPerBlock>>>(d_B, d_C, d_A, width, height);

    cudaMemcpy(h_A, d_A, height * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; i++) {
        printf("A[%d] = %f\n", i, h_A[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}