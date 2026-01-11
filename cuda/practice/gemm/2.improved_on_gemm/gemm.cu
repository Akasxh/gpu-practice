#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Tiled GEMM: Load tiles of A and B into shared mem to reuse data, reduce global accesses.

#define BLOCK_SIZE 16

__global__ void tiled_gemm(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along K dimension.
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tileA whr each thread loads one element if within bounds. A is row-major, so coalesced if threads in x-dim load consecutive.
        
        if (row < M && (t * BLOCK_SIZE + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tileB where transposed access—threads in y-dim for coalescing.
        if (col < N && (t * BLOCK_SIZE + threadIdx.y) < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // Ensure tiles loaded before compute. This is needed

        // accumulate operations from shared
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();  // Reuse shared for next tile.
    }

    if (row < M && col < N) {
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

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch Kernel (asynchronous)
    tiled_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    // Implicit sync: This will block until the kernel is done
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}