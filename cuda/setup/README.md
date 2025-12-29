# CUDA Setup

This directory contains setup instructions and resources for CUDA development.

## Overview

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables developers to use GPU acceleration for general-purpose computing.

## Getting Started

This document records the system configuration, software versions, and specific installation steps required to reproduce the backend deep learning environment on the **ASUS ROG Zephyrus G14** running Ubuntu.

## 1. System Environment & Versions

| Component | Version | Notes |
| :--- | :--- | :--- |
| **Hardware** | ASUS ROG Zephyrus G14 | Nvidia GTX 1650 |
| **OS** | Ubuntu 24.04 LTS | Kernel 6.8+ (Inferred from GCC version) |
| **Python** | 3.12.3 | System default |
| **GCC** | 13.3.0 | `x86_64-linux-gnu` |
| **NVIDIA Driver** | 580.95.05 | Existing system driver (via Ubuntu repo) |
| **CUDA Toolkit** | 13.1 | Build `V13.1.80` |

### Initial Verification
Before installation, current system tools were verified:
```bash
python3 --version  # Output: Python 3.12.3
gcc --version      # Output: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
nvidia-smi         # Verified Driver Version: 580.95.05

```

---

## 2. CUDA Toolkit Installation Guide

**Objective:** Install CUDA Toolkit 13.1 *without* overwriting the working display driver (580.95).

### Step 1: Download Official Runfile

Used the `runfile (local)` method from the [NVIDIA Developer Website](https://developer.nvidia.com/cuda-downloads).

```bash
wget [https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run](https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run)

```

### Step 2: Run Installer

**Crucial:** The `--override` flag is required to bypass strict GCC version checks on Ubuntu 24.04.

```bash
sudo sh cuda_13.1.0_590.44.01_linux.run --override

```

### Step 3: Interactive Configuration (The "Magic Sauce")

When the ncurses (text) menu appears, follow these exact selections to ensure safety:

1. **EULA:** Type `accept`.
2. **Selection Menu:**
* `[ ] Driver` -> **UNCHECK** (Press Spacebar). *Prevent overwriting the stable 580.95 driver.*
* `[X] CUDA Toolkit 13.1` -> **CHECK**.
* `[ ] Kernel Objects` -> **UNCHECK** (specifically `nvidia-fs`). *Not required for this laptop configuration.*
* `[X] CUDA Documentation` -> Optional (can leave checked).


3. **Action:** Select `Install` and press Enter.

*Note: The installer may warn about a missing driver version 590.xx. This can be ignored due to NVIDIA Minor Version Compatibility (13.x driver works with 13.x toolkit).*

---

## 3. Post-Installation Setup

Configure the shell to locate the new compiler and libraries.

**Add to `~/.bashrc`:**

```bash
export PATH=/usr/local/cuda-13.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

**Reload Shell:**

```bash
source ~/.bashrc

```

**Final Version Check:**

```bash
nvcc --version

```

*Expected Output:* `Cuda compilation tools, release 13.1, V13.1.80`

---

## 4. Verification (Test Script)

To ensure the GPU is actually accessible despite the driver version mismatch warning, compile and run this test kernel.

Create a **File:** `test_cuda.cu`

```Bash

nano test_cuda.cu
```

Copy and paste the below code:

```cpp
#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // 1. Launch kernel
    helloFromGPU<<<1, 5>>>();
    
    // 2. Force CPU to wait for GPU (synchronize)
    cudaError_t err = cudaDeviceSynchronize();

    // 3. Check for errors
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Success! GPU is communicating with the CPU.\n");
    return 0;
}

```

(Save and exit: Ctrl+O, Enter, Ctrl+X)

**Compilation & Execution:**

```bash
nvcc test_cuda.cu -o test_cuda
./test_cuda

```

**Success Output:**

```text
Hello World from GPU thread 0!
...
Hello World from GPU thread 4!
Success! GPU is communicating with the CPU.

```

HI
