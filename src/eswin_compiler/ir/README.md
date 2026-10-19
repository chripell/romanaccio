# ESWIN Intermediate Representation (EIR)

EIR is a graph-level Intermediate Representation tailored for the dual-die ESWIN EIC7700X NPU platform.

## Key Design Principles & Invariants

1. **Strict 4D Tensors $[N, C, H, W]$**:
   - The hardware engines (CMAC, SDP, PDP, RUBIK) and the `.model` format strictly operate on 4D spatial tensors.
   - Any 1D, 2D, or 3D tensor from ONNX or safetensors is canonically reshaped to $[N, C, H, W]$ using `normalize_to_4d`:
     - 1D $[C] \rightarrow [1, C, 1, 1]$
     - 2D $[M, K] \rightarrow [M, K, 1, 1]$
     - 3D $[B, S, D] \rightarrow [B, S, 1, D]$
2. **Explicit Engine Assignment (`EngineType`)**:
   - Every operation node declares its execution engine:
     - `NPU`: NVDLA hardware cores (Conv, GEMM lowered to Conv1x1, Add, Mul, LeakyReLU, Clip).
     - `DSP`: Cadence Vision Q7 DSP vector cores (RMSNorm, LayerNorm, Softmax, RoPE).
     - `CPU`: Host RISC-V CPU fallback for non-offloadable operators.
3. **Memory Residency Planning**:
   - Tensors specify residency in `SYSTEM` (DRAM) or `SRAM` (SPRAM, 4MB per die).
   - High-bandwidth intermediate feature maps can be scheduled for on-chip SPRAM tiling.
