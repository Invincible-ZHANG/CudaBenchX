// ========================================
// Filename : vector_add.cu
// Author   : Zijian Zhang
// Date     : 2025
// Purpose  : Demonstrates CUDA parallel addition: C[i] = A[i] + B[i]
// Compile  : nvcc -o vector_add src/vector_add.cu
// Run      : ./vector_add
// ========================================

#include <iostream>
#include <cuda_runtime.h>

// ========================================
// CUDA kernel: vectorAdd
// Each thread computes one element of output vector C[i] = A[i] + B[i]
//
// Parameters:
//   A - pointer to input vector A on device (GPU memory)
//   B - pointer to input vector B on device
//   C - pointer to output vector C on device
//   N - total number of elements in vectors
// ========================================
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (i < N) C[i] = A[i] + B[i];                 // Element-wise addition
}

int main() {
    int N = 1 << 20;                  // Vector length = 2^20 = 1,048,576
    //How much video memory the GPU allocates
    size_t size = N * sizeof(float); // Total memory size in bytes

    // ========================================
    // Allocate and initialize host memory (CPU-side)
    // h_A, h_B - input vectors on host
    // h_C      - output vector on host
    // ========================================
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // Each element is set to 1.0
        h_B[i] = 2.0f;
    }

    // ========================================
    // Allocate device memory (GPU-side)
    // d_A, d_B - input vectors on device
    // d_C      - output vector on device
    // ========================================
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // ========================================
    // Copy data from host to device
    // Copies h_A → d_A and h_B → d_B
    // ========================================
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ========================================
    // Configure kernel launch parameters
    // blockSize  - number of threads per block
    // numBlocks  - number of thread blocks
    // ========================================
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // ========================================
    // Launch CUDA kernel on GPU
    // Launches 'numBlocks' blocks with 'blockSize' threads per block
    // Each thread computes one element of C
    // ========================================
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // ========================================
    // Copy result vector back from device to host
    // d_C → h_C
    // ========================================
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the first result element (expected: 3.0)
    std::cout << "Result[0] = " << h_C[0] << std::endl;

    // ========================================
    // Free allocated memory on both host and device
    // ========================================
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
