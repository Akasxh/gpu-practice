// Run using 

// nsys profile --stats=true ./gemm 1024 1024 1024
// sudo /usr/local/cuda/bin/ncu -f -o report_kernel --set full ./gemm 1024 1024 1024


// GTX 1650 is a Turing architecture card (TU117 chip), 
// with 896 CUDA cores, 4GB GDDR5, and a memory bandwidth around 128 GB/s ( not sure )

// C = α * A * B + β * C, 

// where 
// A is M x K, 
// B is K x N, 
// C is M x N matrices. 

// For simplicity, we'll start with α=1, β=0, and assume row-major storage

#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Basic GEMM kernel
__global__ void basic_gemm(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M N K" << std::endl;
        return 1;
    }
    
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // Fill with random data
    // https://www.geeksforgeeks.org/cpp/stdmt19937-class-in-cpp/
    // about mt19937: https://www.cplusplus.com/reference/random/mt19937/
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) h_A[i] = dist(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dist(gen);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    // Launch Kernel (asynchronous)
    basic_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    // Implicit sync: This will block until the kernel is done
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    
    return 0;
}