# ESWIN EIC7700X NPU, DSP & GPU Operations and Kernel Catalog

## 1. Architectural Overview & Compute Engines

The ESWIN EIC7700X (and dual-die EIC7702 on the DC-ROMA RISC-V Mainboard II / FML13V03) integrates a heterogeneous computing cluster on each die:

- **NPU Subsystem**: NVDLA-derived deep learning accelerator running at 1040 MHz (up to 1500 MHz at 1.05V).
  - Die 0: 9.99 TOPS (INT8) / 4.99 TFLOPS (FP16).
  - Die 1: 9.99 TOPS (INT8) / 4.99 TFLOPS (FP16).
  - Total Compute: **19.99 TOPS INT8 / 9.98 TFLOPS FP16**.
  - Local Memory: 4 MiB on-chip SPRAM per die (`llc_spram0`, `llc_spram1`).
  - Management Core: Embedded SiFive E31 32-bit RISC-V core managing the NPU command queue and LLC.
- **DSP Subsystem**: 4x Cadence Vision Q7 DSP cores per die (8 DSP cores total, `/dev/es-dsp0`..`/dev/es-dsp7`).
  - Xtensa instruction set with 64KB TIE `ivp32` vector extensions.
  - Local SRAM: 108 KB (`dram0`) and 120 KB (`dram1`) per DSP core with IDMA descriptors.
- **HAE Subsystem (Hardware Acceleration Engine)**: Specialized fixed-function 2D image processing hardware (`/dev/es_hae`, `/dev/es_hae1`) for color space conversion, resize, crop, and normalize.
- **GPU Subsystem**: Imagination AXM-8-256 (IMG B-Series architecture).
  - Compute: 0.25 TFLOPS (250 GFLOPS FP32) / 1.0 TOPS INT8.
  - Fillrate: 8 GPixels/sec.
  - APIs: OpenCL 3.0, Vulkan 1.3, OpenGL ES 3.2, EGL 1.5.

---

## 2. NPU Hardware Execution Engines (NVDLA-Derived)

From DWARF symbols in `EsAAC` and `EsSimulator`, the NPU execution pipeline consists of 8 specialized hardware engines programmed via KMD/UMD descriptors:

| Engine | Full Name | Capabilities & Supported Operations | Precision | Layout |
| :--- | :--- | :--- | :--- | :--- |
| **CDMA** | Convolution DMA | Fetches weight tensors and feature maps from DRAM/SPRAM into internal buffers. Supports compression decompression (weight sparse decompression). | INT8, FP16 | NCHW (C0-aligned) |
| **CSC** | Convolution Sequence Controller | Orchestrates convolution scheduling, dilation, padding, and MAC array feed. Handles split-conv across channels. | - | - |
| **CMAC** | Convolution MAC Array | 2D systolic dot-product array. Executes 2D Convolution, 1x1 Pointwise Conv (Linear GEMM), Grouped Conv, Depthwise Conv. | INT8, FP16 | NCHW, $1\times 1$ |
| **CACC** | Convolution Accumulator | Accumulates partial sums from CMAC across reduction steps, converts to 32-bit intermediate representation. | INT32, FP32 accum | - |
| **SDP** | Single Data Point Engine | Elementwise operations following Conv or standalone: Bias Add, Batch Normalization, Scaling, PReLU, ReLU, LeakyReLU, Sigmoid/Tanh (via 256/512-entry LUTs), Elementwise Add/Mul (Eltwise), Precision conversion (FP16 $\to$ INT8, INT8 $\to$ FP16). | INT8, FP16, FP32 | Elementwise |
| **PDP** | Planar Data Point Engine | Spatial pooling: MaxPool 2D, AveragePool 2D with arbitrary stride and padding. | INT8, FP16 | NCHW |
| **CDP** | Channel Data Point Engine | Local Response Normalization (LRN) across adjacent channels with programmable exponent and scale. | INT8, FP16 | NCHW |
| **RUBIK** | Data Reshaping Engine | Hardware tensor reordering: Transpose, Reshape, Slice, Concat, De-blocking, PixelShuffle (Depth-to-Space / Space-to-Depth). | INT8, FP16 | NCHW |
| **BDMA** | Bridge DMA Engine | High-throughput asynchronous memory copies: DRAM $\leftrightarrow$ SPRAM (SRAM), DRAM $\leftrightarrow$ DRAM with 64-byte alignment. | Byte-level | Arbitrary |

---

## 3. DSP Kernel Catalog (Cadence Vision Q7 Xtensa)

When an operation is non-linear, dynamic, or outside the NVDLA fixed pipeline, `EsAAC` partitions the graph and dispatches sub-graphs to the Cadence Vision Q7 DSPs.
There are **372 distinct compiled DSP kernel plugins** located in `rootfs-esssim/home/eswin/dsp_kernels/*.so` (and mirrored in firmware).

### 3.1 LLM & Transformer Kernels

| Kernel Base Name | Data Types | Channel Variants / Flags | Description |
| :--- | :--- | :--- | :--- |
| `libes_onnx_rms_normalization` | FP16/FP16, FP16/INT8 | `_c32`, `_c64`, `_c96`, `_c128`, `_c160`, `_c192`, `_c256`, `_c384`, `_c512`, `_c768`, `_c1024`, `_c2048`, `_c2736`, `_c3584`, `_c4096`, `_c8192` | Highly optimized RMSNorm with dedicated vectorized channel loops matching Llama/Qwen/SmolLM hidden dimensions. |
| `libes_rms_norm` | FP16/FP16, FP16/INT8 | Generic channel lengths | Fallback RMSNorm. |
| `libes_onnx_layer_normalization` | FP16, INT16, INT8 | Standard and `_affine` | Full LayerNorm with learnable $\gamma, \beta$. |
| `libes_rotary_position_embedding` | INT8/FP16, INT8/INT8 | Standard RoPE | Applies rotary embeddings to query/key tensors. |
| `libes_qk_rotary_position_embedding`| FP16/FP16, INT8/FP16, INT8/INT8 | Fused Q & K RoPE | Fused Rotary Position Embedding applied concurrently to Query and Key projections. |
| `libes_position_embedding` | INT8/FP16 | Absolute Positional | Standard absolute positional embedding add. |
| `libes_multi_head_attention` | INT8/INT8 | `_with_bias`, `_without_bias`, `_without_bias_large` | Fused MHA for small attention sequences. |
| `libes_attn_score` | INT8/INT8 | Dot product attention score | $Q \times K^T$ scale computation. |
| `libes_onnx_softmax` / `libes_softmax` | FP16, INT16, INT8, FP32 | `axis0`, `axis1`, `axis2`, `axis3`, `_c32`, `_v2` | Numerically stable softmax with online max-subtraction. |
| `libes_onnx_topk` / `libes_topk` | FP16/FP16, INT8/INT8, INT16/FP16, INT16/INT16 | Top-K sorting | Top-K sampling / beam search extraction. |

### 3.2 Non-Linear Activation & Mathematical Kernels

| Operation Family | Data Types | Kernels Available | Implementation Notes |
| :--- | :--- | :--- | :--- |
| **GELU** | FP16, FP32 | `libes_onnx_gelu.so` | Exact Gaussian Error Linear Unit (used in Gemma / BERT). |
| **SiLU / Swish** | FP16, INT8 | Handled via `libes_lut_act_*.so` / `libes_onnx_hard_swish.so` | Used in Llama / Qwen / SmolLM SwiGLU. |
| **Sigmoid / Tanh** | FP16, INT8 | `libes_onnx_sigmoid.so`, `libes_lut_act_*.so` | Piecewise polynomial or 512-entry LUT table. |
| **Mish / ELU / SELU** | FP16, INT8 | `libes_onnx_mish.so`, `libes_onnx_elu.so`, `libes_onnx_selu.so` | Advanced activation functions. |
| **Exp / Log / Sqrt** | FP16, INT8 | `libes_onnx_exp.so`, `libes_onnx_log_*.so`, `libes_onnx_sqrt_*.so` | Elementary transcendentals. |
| **Sin / Cos** | FP16 | `libes_onnx_sin.so`, `libes_onnx_cos.so` | Used for dynamic RoPE frequency cache calculation. |
| **Abs / Ceil / Floor / Neg** | FP16, INT32 | `libes_onnx_abs_*.so`, `libes_onnx_ceil_*.so`, `libes_onnx_floor_*.so`, `libes_onnx_neg_*.so` | Elementwise unary math. |

### 3.3 Tensor Manipulation & Indexing Kernels

| Operation Family | Data Types | Variants | Description |
| :--- | :--- | :--- | :--- |
| **Gather / GatherElements** | FP16, INT16, INT8 | `libes_gather_*.so`, `libes_onnx_gather_elements.so` | Token embedding lookup ($W_e[tokens]$) and cross-entropy indexing. |
| **ScatterND** | FP16, INT16, INT8 | `libes_onnx_scatternd_*.so` | Tensor scatter / selective update (KV cache update). |
| **Slice / Concat** | FP16, INT16, INT8 | `libes_slice.so`, `libes_onnx_slice.so`, `libes_concat_s8s8.so` | Strided tensor slicing and concatenation. |
| **Transpose** | FP16, INT16, INT8 | `libes_transpose.so`, `libes_onnx_transpose.so` | Arbitrary 4D/5D dimension permutation. |
| **Where / Equal / Less / Greater** | FP16, INT32, INT16, INT8 | `libes_onnx_where_*.so`, `libes_onnx_equal_*.so`, `libes_onnx_less_*.so` | Conditional masking (causal attention masking). |
| **Reduce (Sum/Mean/Max/Min)**| FP16, INT16, INT8 | `libes_onnx_reduce_sum_*.so`, `libes_onnx_reduce_mean_*.so`, etc. | Reductions along specified axes. |

### 3.4 Vision & Object Detection Kernels

| Operation Family | Variants Count | Kernels | Description |
| :--- | :--- | :--- | :--- |
| **Resize / Interpolation** | 36 plugins | `libes_resize_*.so`, `libes_onnx_resize_*.so` | Nearest & Bilinear, CHW & HWC, FP16/INT16/INT8/UINT8. |
| **Detection Output / NMS** | 41 plugins | `libes_detection_out_*.so`, `libes_nms_*.so` | YOLOv3, YOLOv5, YOLOv6, YOLOv7, Faster-RCNN, SSD post-processing. |
| **ROI Align / ROI Pool** | 4 plugins | `libes_roi_align_f16f16.so`, `libes_roi_pool_*.so` | Two-stage object detector feature extraction. |
| **WarpAffine / Perspective** | 4 plugins | `libes_warp_affine_*.so`, `libes_perspective_affine_*.so` | Geometric transformations (OCR, license plate alignment). |

---

## 4. ONNX Operator to Engine Mapping Matrix

Based on symbol analysis of `EsAAC` (`mlir::eaac::EHLOToEIRPass` and `eaac::*Emitter`) and empirical compilation tests:

| ONNX Operator | Target Engine | Fallback / Condition | Notes |
| :--- | :--- | :--- | :--- |
| `Conv` (2D) | **NPU CMAC + SDP** | DSP if kernel size $> 7\times 7$ or dilation $> 2$ | Primary accelerated op. |
| `Gemm` / `MatMul` | **NPU CMAC** (lowered to $1\times 1$ Conv) | DSP `libes_onnx_gemm_*.so` | Must be 4D shape $[1, C_{in}, 1, 1] \to [1, C_{out}, 1, 1]$. |
| `Relu` / `Clip` | **NPU SDP** | None (hardware native) | Fused directly into previous Conv where possible. |
| `Sigmoid` / `Tanh` | **NPU SDP** (LUT mode) | DSP `libes_onnx_sigmoid.so` | 256/512 point piecewise linear approximation. |
| `Add` / `Sub` / `Mul` | **NPU SDP** (Eltwise) | DSP `libes_onnx_add_*.so`, `libes_mul_*.so` | Requires identical shapes or broadcast along channel. |
| `MaxPool` / `AveragePool` | **NPU PDP** | DSP if kernel size $> 8\times 8$ | Up to $8\times 8$ window native. |
| `BatchNormalization` | **NPU SDP** | DSP `libes_onnx_batch_normalization_*.so`| Fused into Conv scale/bias weights at compile time. |
| `Reshape` / `Flatten` | **NPU RUBIK** or Metadata | None | Zero-copy when strides permit. |
| `Transpose` | **NPU RUBIK** | DSP `libes_transpose.so` | Fast memory DMA transpositions. |
| `Concat` | **NPU RUBIK** / Blit | DSP `libes_concat_s8s8.so` | Concatenates along C, H, or W. |
| `Softmax` | **DSP** | NPU unsupported | Dispatched to `libes_onnx_softmax.so`. |
| `LayerNormalization` | **DSP** | NPU unsupported | Dispatched to `libes_onnx_layer_normalization_*.so`. |
| `RMSNormalization` | **DSP** | NPU unsupported | Dispatched to `libes_onnx_rms_normalization_*.so`. |
| `Gather` | **DSP** | NPU unsupported | Dispatched to `libes_onnx_gather_*.so`. |
| `ScatterND` | **DSP** | NPU unsupported | Dispatched to `libes_onnx_scatternd_*.so`. |
| `GELU` | **DSP** | NPU unsupported | Dispatched to `libes_onnx_gelu.so`. |
| `Resize` | **HAE** or **DSP** | NPU unsupported | Hardware HAE handles bilinear/nearest; DSP handles complex. |

---

## 5. Precision & Quantization Support

### 5.1 Supported Precisions
- **INT8 (Signed, 8-bit)**:
  - Input/Output: `int8_t` range $[-128, 127]$.
  - Accumulator: `int32_t` inside CACC.
  - Scale factor: 32-bit fixed point multiplier + shift or FP32 scale.
  - Zero-point: Symmetric ($zp = 0$) preferred; asymmetric supported via SDP bias correction.
- **INT16 (Signed, 16-bit)**:
  - Supported on SDP, PDP, and select DSP kernels for high-dynamic-range intermediate activations.
- **FP16 (IEEE 754 Half-Precision, 16-bit)**:
  - 1 sign bit, 5 exponent bits, 10 mantissa bits.
  - Supported across all NPU engines (`--gen-fp16-model=yes`) and DSP kernels.
  - Accumulator: FP32 internal accumulation.
- **FP32 (Single Precision, 32-bit)**:
  - Supported on CPU and GPU (AXM-8-256).
  - Supported as input/output in DSP post-processing kernels (`_s8f32`, `_f16f32`).

### 5.2 Quantization Schemes
1. **Per-Tensor Symmetric**: Single FP32 scale $S$ per tensor, $q = \text{clip}(\text{round}(x / S), -128, 127)$. Standard for activations.
2. **Per-Channel Symmetric (Weights)**: Separate scale $S_c$ for each output channel $c \in [0, C_{out}-1]$. Supported in CMAC convolution.
3. **KL-Divergence / MinMax Calibration**: Calibration data evaluated by `EsQuant` producing `quant_stats.json`.

---

## 6. Layout & Alignment Constraints

1. **4D Static Shape Requirement**:
   - The NPU hardware microcode strictly operates on 4D tensors: $[N, C, H, W]$.
   - 1D/2D vectors (e.g. LLM hidden states $[1, D]$) **must** be reshaped to $[1, D, 1, 1]$.
   - Dynamic dimensions are rejected by `EsAAC` (`getBatchFromInputShape failed`). All dimensions must be fixed at compile time.
2. **Channel Alignment ($C_0$)**:
   - The CMAC array processes data in atomic vector blocks of $C_0 = 16$ or $C_0 = 32$ elements.
   - Channel dimensions should ideally be multiples of 16 (or 32) to prevent hardware zero-padding waste.
3. **Memory Alignment**:
   - DMA and SMMU buffers must be aligned to **4096 bytes** (4KB page boundary) or **2097152 bytes** (2MB huge page boundary) via `ES_SYS_MemAlloc`.

---

## 7. Imagination AXM-8-256 GPU Capabilities

The onboard GPU is an Imagination AXM-8-256 core:
- **Compute Power**: 0.25 TFLOPS FP32 (250 GFLOPS) / 1.0 TOPS INT8.
- **Fillrate**: 8 GPixels/s.
- **Driver Architecture**:
  - Kernel driver: `pvrsrvkm.ko` (PowerVR Services Kernel Module).
  - Userspace ICD: `libGLESv2.so`, `libEGL.so`, `libOpenCL.so`, `libvulkan.so`.
- **Compute Offload Potential**:
  - Well-suited for non-NPU workloads: Embedding lookups, custom token sampling, Whisper audio mel-spectrogram FFT preprocessing, and vision rendering.
  - Shares the unified DDR4 system memory with CPU and NPU (zero-copy dmabuf sharing possible via `EGL_EXT_image_dma_buf_import` and `clCreateFromGLTexture` / DMA-buf OpenCL extensions).
