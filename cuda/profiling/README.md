# CUDA Profiling flow

This guide documents the standard workflow for profiling CUDA applications using NVIDIA's modern toolchain: **Nsight Systems (Macro View)** and **Nsight Compute (Micro View)**.

## Quick Reference (Cheat Sheet)

*Run these commands in order.*

```bash
# 1. Compile with Line Information (Crucial for Source View)
nvcc -lineinfo -o vector_add vector_add.cu

# 2. Macro Profile (Timeline & System View)
# Generates: report1.nsys-rep
nsys profile --stats=true ./vector_add

# 3. Micro Profile (Kernel Efficiency & Stalls)
# Generates: report_kernel.ncu-rep
sudo /usr/local/cuda/bin/ncu -f -o report_kernel --set full ./vector_add

# 4. Open GUIs
nsys-ui report1.nsys-rep
ncu-ui report_kernel.ncu-rep

```

---

## 🛠 Prerequisites & Setup

**Permission Fix (Avoid typing sudo every time):**
To avoid `ERR_NVGPUCTRPERM`, allow user access to GPU counters:

```bash
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-profiler.conf
# Reboot your machine after running this

```

---

## Phase 1: The Macro View (Nsight Systems)

<img width="1376" height="810" alt="image" src="https://github.com/user-attachments/assets/bf773182-fb1c-42d8-83e6-37a3fbd8a187" />

**Goal:** Answer "Is my application CPU bound or GPU bound?" and "Am I spending too much time moving memory?"

### Command

```bash
nsys profile --stats=true ./vector_add

```

### Key Terminal Output

The `--stats=true` flag provides a summary immediately in the terminal.

```text
[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)        Name          
 --------  ---------------  ---------  ------------  ----------------------
     89.6      159,619,785          3  53,206,595.0  cudaMalloc            
     10.1       17,919,163          3   5,973,054.3  cudaMemcpy            
      0.1          149,183          1     149,183.0  cudaLaunchKernel      

```

### Analysis

* **Observation:** The kernel launch (`cudaLaunchKernel`) is only **0.1%** of the API time. `cudaMalloc` and `cudaMemcpy` dominate.
* **Conclusion:** This application is dominated by memory management overhead. Optimizing the math in the kernel will have negligible impact on the *total* runtime unless we process more data.

**Visual Analysis:**
Open the report (`nsys-ui report1.nsys-rep`) to see the timeline bubbles.

---

## Phase 2: The Micro View (Nsight Compute)

<img width="1857" height="1049" alt="image" src="https://github.com/user-attachments/assets/d6337289-643e-489a-baaf-2b583899bb7c" />

**Goal:** Answer "Why is my specific kernel slow?" (e.g., Memory Bandwidth vs Compute).

### Command

*Note: We use `-f` to overwrite existing files and `--set full` to capture all stall reasons.*

```bash
sudo /usr/local/cuda/bin/ncu -f -o report_kernel --set full ./vector_add

```

### Key Terminal Output

If you run without `--set full`, you get a "Speed of Light" summary in the terminal:

```text
  vectorAdd(const float *, const float *, float *, int) (19532, 1, 1)x(256, 1, 1)
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Memory Throughput                 %        93.44  <-- CRITICAL
    Compute (SM) Throughput           %        19.55
    ----------------------- ----------- ------------
    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.

```

### Analysis

* **Memory Throughput (93.44%):** The GPU memory controller is saturated. We are **Memory Bound**.
* **Compute Throughput (19.55%):** The math cores (SMs) are idle 80% of the time waiting for data.

---

## Phase 3: Root Cause Analysis (GUI)

<img width="1456" height="674" alt="image" src="https://github.com/user-attachments/assets/20ddf9c4-0845-4283-8a71-e3f074a5250b" />

### 1. The Summary View

Open the report: `ncu-ui report_kernel.ncu-rep`
Look at the **"Speed of Light"** chart. It visually confirms the terminal data.

### 2. The Source Code View

This is the most important step for developers.

* Click the **Source** tab.
* Select **View: PTX or SASS or SOURCE**.

**What to look for:**
If you see high "Stall" percentages (blue bars) next to memory instructions (`LDG`, `STG`), it confirms the threads are blocked waiting for RAM.

### Common Fixes based on Profiling

1. **High Memory Throughput + Low Compute:** Use `float4` (vectorized loads) or Shared Memory tiling.
2. **High Compute + Low Memory:** Simplify math logic or use fast math intrinsics (`__sinf` vs `sin`).
3. **Low Occupancy:** Adjust `threadsPerBlock` (usually 128 or 256 is sweet spot).

---

## Troubleshooting Notes

* **Command Not Found (sudo):**
* *Symptom:* `sudo: ncu: command not found`
* *Solution:* Use the full path: `sudo /usr/local/cuda/bin/ncu`.
