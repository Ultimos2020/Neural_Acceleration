#include <cstdio>
#include <cuda_runtime.h>

// GPU kernel: adds A[i] + B[i] → C[i]
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1<<20;               // 1M elements
    const size_t bytes = N * sizeof(float);

    // 1) Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // 2) Initialize inputs
    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(i) * 2.0f;
    }

    // 3) Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 4) Copy data host → device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 5) Launch kernel with enough blocks to cover N threads
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 6) Copy result device → host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 7) Verify
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            ok = false;
            printf("Mismatch at %d: %f vs %f\n", i, h_C[i], h_A[i]+h_B[i]);
            break;
        }
    }
    printf("Result %s\n", ok ? "OK" : "FAILED");

    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // 8) Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return ok ? 0 : 1;
}

