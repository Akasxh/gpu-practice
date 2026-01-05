

## What is memory bandwidth?

Memory bandwidth is the maximum rate at which data can be transferred between different levels of the memory hierarchy .

It represents the theoretical maximum achievable throughput for moving data in bytes per second. It determines the slope of the "memory roof" in a roofline model of the hardware.

There are many memory bandwidths in a complete system — one between each level of the memory hierarchy .

The most important bandwidth is that between the GPU RAM and the register files of the Streaming Multiprocessors (SMs) , because the working sets

of most kernels only fit in GPU RAM , not anywhere higher up in the memory hierarchy . It is for this reason that that bandwidth is the primary one used in roofline modeling of GPU kernel performance.

Contemporary GPUs have memory bandwidths measured in terabytes per second. For example, B200 GPUs

have a (bidirectional) memory bandwidth of 8 TB/sec to their HBM3e memory. This is much lower than the arithmetic bandwidth of the Tensor Cores in these GPUs, leading to increased ridge point arithmetic intensity .

Representative bandwidth numbers for NVIDIA data center GPUs between the Ampere and Blackwell Streaming Multiprocessor architectures are listed in the table below

<img width="1258" height="189" alt="image" src="https://github.com/user-attachments/assets/4bfdfbb5-8fa5-474c-9910-cf0324088706" />

This table is describing the **Roofline Model** characteristics of different GPUs. 

<img width="839" height="554" alt="image" src="https://github.com/user-attachments/assets/2ff4cff7-ddcc-47ce-be95-fded25411ac2" />

### 1. What is the "Ridge Point"?

The Ridge Point (often called **Machine Balance**) is a simple ratio:

It answers this specific question:

> **"How many calculations (FLOPs) must I perform for every single Byte of data I fetch from memory to keep this GPU 100% busy?"**

### 2. How to Read the Numbers (The "Idea")

Think of the GPU as a kitchen:

* **Compute (TFLOPs):** The Chef chopping vegetables.
* **Memory (TB/s):** The Runner bringing vegetables from the fridge.

The **Ridge Point** tells you how fast the Chef is compared to the Runner.

* **A100 (Ridge Point: 156):**
* For every byte of data the runner brings, the Chef needs to do **156** chopping motions.
* If you do *less* than 156 ops/byte, the Chef waits (Memory Bound).
* If you do *more*, the Runner waits (Compute Bound).


* **B200 FP4 (Ridge Point: 1125):**
* This Chef is **insanely fast**.
* For every byte the runner brings, the Chef must do **1125** chopping motions just to stay busy.
* This is much "harder" to satisfy. Most simple algorithms don't do 1125 math operations on a single piece of data.



### 3. The "Ideal" Situation

There is no "ideal" number for the hardware (it is just a physical fact of the chip), but there is an ideal goal for **your code**:

**You want your kernel's Arithmetic Intensity to be HIGHER than the Ridge Point.**

* **If your kernel > Ridge Point:** You are **Compute Bound**. You are getting the maximum TFLOPs the chip advertising. This is the "ideal" state for matrix multiplication.
* **If your kernel < Ridge Point:** You are **Memory Bound**. The chip is idle, waiting for data. This is common for activation functions, simple additions, or naive attention mechanisms.

### 4. Why this matters for you (Optimization)

Look at the trend in your image:

* A100: **156** FLOPs/byte
* H100: **295** FLOPs/byte
* B200 (FP4): **1125** FLOPs/byte

**The Gap is Widening.** Compute is getting faster *much* more quickly than Memory Bandwidth.

This explains why techniques like **Flash Attention** are necessary. Flash Attention doesn't just "go faster"; it drastically increases the arithmetic intensity (doing more math in SRAM/Shared Memory) to overcome these massive Ridge Points. On a B200, if you can't reuse data 1000+ times once you load it, you will never see those 9000 TFLOPs.

### Summary

* **Low Ridge Point (156):** Easier to get max performance.
* **High Ridge Point (1125):** Very difficult to get max performance; requires heavy optimization and data reuse (Tiling/Shared Memory).
