// ========================================
// �ļ�����vector_add.cu
// ���ߣ�Zijian Zhang
// ���ڣ�2025
// ���ܣ�ʹ�� CUDA ����ִ�������ӷ� C = A + B
// ���뷽�������ն�ִ�У���
//     nvcc -o vector_add src/vector_add.cu
// ���з�����
//     ./vector_add
// ========================================

#include <iostream>
#include <cuda_runtime.h>

// ========================================
// GPU �˺�����ÿ���̼߳���һ�� C[i] = A[i] + B[i]
// ========================================
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // ȫ���߳�����
    if (i < N) C[i] = A[i] + B[i];                 // ����ÿ��Ԫ��
}

int main() {
    int N = 1 << 20; // ��������Ϊ 2^20 = 1048576
    size_t size = N * sizeof(float); // ���������С���ֽڣ�

    // ========================================
    // ���䲢��ʼ���������ڴ棨CPU �ϣ�
    // ========================================
    float* h_A = new float[N], * h_B = new float[N], * h_C = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // A ȫ��Ϊ 1.0
        h_B[i] = 2.0f; // B ȫ��Ϊ 2.0
    }

    // ========================================
    // �����豸���ڴ棨GPU �ϣ�
    // ========================================
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // ========================================
    // ���ݴ� CPU ������ GPU
    // ========================================
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ========================================
    // ���� CUDA kernel ������߳̿�
    // ========================================
    int blockSize = 256; // ÿ���߳̿� 256 ���߳�
    int numBlocks = (N + blockSize - 1) / blockSize; // ����Ҫ���߳̿���������ȡ����

    // ========================================
    // ���� CUDA �˺���
    // ========================================
    vectorAdd << <numBlocks, blockSize >> > (d_A, d_B, d_C, N);

    // ========================================
    // ������� GPU ������ CPU
    // ========================================
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // ��ӡ��һ��������������ֵ = 1.0 + 2.0 = 3.0
    std::cout << "Result[0] = " << h_C[0] << std::endl;

    // ========================================
    // �ͷ��ڴ���Դ
    // ========================================
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
