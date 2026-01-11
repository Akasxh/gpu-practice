# Basic GEMM (Naive Implementation)

This folder contains a naive implementation of Matrix Multiplication (`C = A * B`) using CUDA. It serves as a baseline to demonstrate the performance bottlenecks of accessing global memory without optimizations.

## The Code
The kernel `basic_gemm` assigns one thread to compute one element of the output matrix `C`. 
```cpp
// Iterate over K to calculate dot product
float sum = 0.0f;
for (int i = 0; i < K; ++i) {
    sum += A[row * K + i] * B[i * N + col];
}
C[row * N + col] = sum;
```
**Issue**: For every multiply-add operation, the thread must fetch two values (`A` and `B`) from global memory. This results in **Accessing Global Memory 2*K times** for just **2*K FLOPs**, leading to a very low arithmetic intensity.

## Performance Analysis (Nsight Compute)

The following analysis helps explain the "bad" performance seen in the Nsight Compute reports (`nsight1.png` and `nsight2.png`).

### 1. Memory Bound (Likely visible in `nsight1.png`)
*   **What it shows**: You would see high **GPU Speed Of Light (SOL) Memory** usage (closer to the peak bandwidth of the card, e.g., ~60-80%+) but extremely low **SOL Compute** usage (<5%).
*   **Why it's "bad"**: The GPU computes infinitely faster than it can read data. The execution units are sitting idle/starved because they are waiting for data to arrive from DRAM. We are not utilizing the available TFLOPS.

### 2. Warp Stalls (Likely visible in `nsight2.png` or Source View)
*   **What it shows**: High percentage of "Long Scoreboard" or "Stall Wait" for the inner loop line `sum += ...`.
*   **Why it's "bad"**: The 99.57% stall rate mentioned typically means threads are blocked waiting for global memory requests to return.
    *   **Latency**: Global memory access takes ~400-800 cycles.
    *   **Throughput**: Bandwidth is saturated.
*   **Instruction Mix**: Heavy on Load/Store (LD/ST) vs Compute (FMA).

## Conclusion
*   **Bottleneck**: **Global Memory Bandwidth**.
*   **Theoretical Limit**: The performance is capped by the card's memory bandwidth (e.g., ~128 GB/s on a GTX 1650), far below its compute potential.

## What Could Be Done Better?
To improve performance, we need to increase **Arithmetic Intensity** (do more math per byte loaded).

1.  **Shared Memory Tiling**:
    *   Load small blocks (tiles) of A and B into **Shared Memory** (on-chip cache).
    *   Threads synchronize and compute using these cached values.
    *   **Benefit**: drastically reduces global memory accesses (by a factor of the tile size, e.g., 16x fewer loads).

2.  **Memory Coalescing**:
    *   Ensure all threads in a warp access continuous memory addresses. (Note: `B` is already coalesced here, but `A` is not ideally accessed if row-major).

3.  **Vectorization**:
    *   Use `float4` to load data to utilize wider bus transactions.