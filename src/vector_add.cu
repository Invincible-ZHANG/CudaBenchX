// ========================================
// 文件名：vector_add.cu
// 作者：Zijian Zhang
// 日期：2025
// 功能：使用 CUDA 并行执行向量加法 C = A + B
// 编译方法（在终端执行）：
//     nvcc -o vector_add src/vector_add.cu
// 运行方法：
//     ./vector_add
// ========================================

#include <iostream>
#include <cuda_runtime.h>

// ========================================
// GPU 核函数：每个线程计算一个 C[i] = A[i] + B[i]
// ========================================
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    if (i < N) C[i] = A[i] + B[i];                 // 计算每个元素
}

int main() {
    int N = 1 << 20; // 向量长度为 2^20 = 1048576
    size_t size = N * sizeof(float); // 所有数组大小（字节）

    // ========================================
    // 分配并初始化主机端内存（CPU 上）
    // ========================================
    float* h_A = new float[N], * h_B = new float[N], * h_C = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // A 全部为 1.0
        h_B[i] = 2.0f; // B 全部为 2.0
    }

    // ========================================
    // 分配设备端内存（GPU 上）
    // ========================================
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // ========================================
    // 数据从 CPU 拷贝到 GPU
    // ========================================
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ========================================
    // 配置 CUDA kernel 网格和线程块
    // ========================================
    int blockSize = 256; // 每个线程块 256 个线程
    int numBlocks = (N + blockSize - 1) / blockSize; // 共需要的线程块数（向上取整）

    // ========================================
    // 启动 CUDA 核函数
    // ========================================
    vectorAdd << <numBlocks, blockSize >> > (d_A, d_B, d_C, N);

    // ========================================
    // 将结果从 GPU 拷贝回 CPU
    // ========================================
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印第一个计算结果：期望值 = 1.0 + 2.0 = 3.0
    std::cout << "Result[0] = " << h_C[0] << std::endl;

    // ========================================
    // 释放内存资源
    // ========================================
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
