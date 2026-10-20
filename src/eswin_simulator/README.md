# Open EsSimulator (`eswin_simulator`)

**eswin_simulator** is an open-source, 64-bit cross-platform functional simulator for the ESWIN EIC7700X / EIC7702 NPU and DSP loadable container (`.model` FlatBuffers).

It replaces the proprietary, 32-bit x86-only vendor `EsSimulator` binary with a clean, extensible, platform-independent execution environment.

---

## 1. Features & Implemented Milestones

### Milestone 1: Model Loader & Virtual Address Space Inspector
- **Universal `.model` Reader**: Full FlatBuffers (`NVDA` identifier) parsing of loadables across all model topologies (vision CNNs, transformer blocks, split dual-die LM heads).
- **Virtual Memory Architecture**: Simulates 64-bit DRAM address space alongside the on-chip 4 MiB SPRAM (`SRAM`), enforcing page alignments and buffer boundaries.
- **Static Blob Ingestion**: Automatically resolves and populates embedded weight blobs (`W_0`, tiled splits), hardware command packets (`kmd-task-op_list`, `kmd-task-surf_list`), and LUT tables into simulated physical memory.
- **Model Inspector CLI (`eswin-sim`)**:
  - Detailed terminal tables of input/output tensor shapes, data formats (`NCHW`/`NHWC`), and precisions (`HALF`/`INT8`/`FLOAT`).
  - Memory utilization breakdown (DRAM vs on-chip SPRAM footprint).
  - Submodel task topology (UMD vs KMD tasks) and relocation tables.
  - Machine-readable `--json` output format for toolchain automation and continuous integration.

### Milestone 2: Dynamic Relocations, BDMA Engine & Tensor File I/O
- **Dynamic Relocation Resolver**: Walks `RelocEntry` records and patches simulated 64-bit IOVA addresses (split into lower and upper 32-bit words) directly into hardware surface descriptors and command streams.
- **Bridge DMA (BDMA) Simulation Engine**:
  - Full functional simulation of multi-dimensional strided memory copies between DRAM and on-chip SPRAM.
  - High-throughput contiguous 1D linear copies, 2D pitch-strided sub-matrix extractions, and 3D planar volume transfers.
- **Tensor File I/O Integration**:
  - Ingestion of raw `.bin` tensor inputs (`--input=<ifmap.bin>`).
  - Output tensor generation and dumping to disk (`--output=<ofmap.bin>`).
  - Zero-copy tensor read/write buffers.

### Milestone 3: NPU Convolution & GEMM Functional Engine (CMAC + CACC)
- **CMAC Functional Engine (`engines/cmac.py`)**:
  - Direct 2D spatial convolution with arbitrary stride, padding, and dilation.
  - Grouped and depthwise separable convolutions ($G \ge 1$).
  - Pointwise $1\times 1$ convolution and linear GEMM ($y = x @ W^T$) across 2D, 3D (tokens), and 4D tensor inputs.
  - Dual precision support:
    - **FP16**: IEEE 754 half-precision multiplication with FP32 accumulator.
    - **INT8**: 8-bit signed integer multiplication with 32-bit signed integer accumulator (`int32_t`).
  - NVDLA Weight De-Tiling & Unpacking:
    - `unpack_nvdla_weights_conv2d`: De-tiles spatial convolution weights from NVDLA sub-channel blocks into canonical $(C_{out}, C_{in}, KH, KW)$ shape.
    - `unpack_nvdla_weights_1x1`: De-tiles 32-byte atomic $(C_{out\_sub}=8, C_{in\_sub}=2)$ memory burst tiles into canonical $(C_{out}, C_{in})$ GEMM matrices.
- **CACC Functional Engine (`engines/cacc.py`)**:
  - Multi-pass accumulation buffer management across tiled reduction steps.
  - FP16 bias addition and rounding.
  - INT8 channel scaling, bias addition, rounding, and strict hardware saturating clipping to $[-128, 127]$.
- **End-to-End CLI Simulation & Differential Verification**:
  - `--compare=<ref.bin>` mode computing Cosine Similarity, Max Relative Difference, and Max Absolute Error against reference golden data.
  - Bit-exact verification against vendor `EsSimulator`:
    - `test_conv_npu_b1.model`: Cosine Similarity $= \mathbf{1.000000}$, Relative Diff $= \mathbf{0.000000}$.
    - `test_linear_1x1conv_npu_b1.model`: Cosine Similarity $= \mathbf{1.000000}$, Relative Diff $= \mathbf{0.000518}$.

### Milestone 4: Single Data Point Engine (SDP) & Non-Linear LUTs
- **SDP Functional Engine (`engines/sdp.py`)**:
  - **Pointwise ALU Operations**: Channel/tensor bias addition with broadcasting, batch normalization scaling, and elementwise operations (`add`, `sub`, `mul`, `min`, `max`).
  - **Fixed-Function Activations**: `relu`, `leaky_relu`, `prelu` (with per-channel slope vectors), and `clip`.
  - **Piecewise Linear LUT Engine**:
    - Full parser and generator for 688-byte NVDLA LUT structures (`struct _lut_dev`).
    - Bit-accurate piecewise linear interpolation using 257-entry linear-only (`LO`) and 65-entry linear-exponential (`LE`) tables.
    - Automated boundary clamping and linear continuation for transcendental functions (SiLU / Swish, Sigmoid, Tanh, GELU).
  - **Precision Conversion**: FP16 $\leftrightarrow$ INT8 quantization with zero-points and saturating clipping.
- **Full SwiGLU MLP Verification**:
  - Multi-op pipeline simulation ($W_{\text{gate}} \to \text{SiLU(LUT)} \odot W_{\text{up}} \to W_{\text{down}}$) on `test_swiglu_mlp_npu_b1.model`.
  - Cosine Similarity $= \mathbf{1.000000}$ ($0.99999976$) against vendor `EsSimulator` golden output.

---

## 2. Mathematical Model of BDMA Strided Transfers

The Bridge DMA (BDMA) hardware engine transfers multi-dimensional tensor data between SYSTEM memory (DRAM) and the 4 MiB on-chip SPRAM (`SRAM`).

### 2.1 Coordinate Mapping & Address Derivation
For a tensor volume parameterized by:
- $\text{surf\_repeat} \in [1, S]$ (typically channels $C$ or batch $N$)
- $\text{line\_repeat} \in [1, H]$ (lines per surface, typically height $H$)
- $\text{line\_size} \in [1, W \times \text{element\_size}]$ (contiguous payload bytes per line)

The source and destination byte addresses for surface index $s$ and line index $l$ are given by:

$$\text{Addr}_{\text{src}}(s, l) = \text{Base}_{\text{src}} + s \times \text{stride}_{\text{src\_surf}} + l \times \text{stride}_{\text{src\_line}}$$

$$\text{Addr}_{\text{dst}}(s, l) = \text{Base}_{\text{dst}} + s \times \text{stride}_{\text{dst\_surf}} + l \times \text{stride}_{\text{dst\_line}}$$

Where:
- $\text{stride}_{\text{line}}$: Pitch distance in bytes between consecutive row starts (allows skipping padding bytes between rows).
- $\text{stride}_{\text{surf}}$: Distance in bytes between consecutive planar feature map surfaces.

### 2.2 Relocation Address Patching
When allocating virtual DMA memory regions, the base 64-bit address is split into two 32-bit little-endian fields matching hardware registers:

$$\text{Addr}_{\text{lo}} = \text{IOVA} \ \& \ \text{0xFFFFFFFF} \quad (\text{reloc\_type} = 1)$$

$$\text{Addr}_{\text{hi}} = (\text{IOVA} \gg 32) \ \& \ \text{0xFFFFFFFF} \quad (\text{reloc\_type} = 2)$$

These values are written directly at byte offset `offset` inside the command descriptor buffer `write_id`.

---

## 3. Mathematical Model of CMAC and CACC Engines

The Convolution MAC Array (CMAC) and Convolution Accumulator (CACC) form the computational core of the NVDLA/ESWIN NPU.

### 3.1 NVDLA Weight Memory Tiling Format

In `.model` FlatBuffers loadables, weight tensors are stored in hardware-aligned tiles optimized for 32-byte and 64-byte memory bursts:

#### Spatial 2D Convolutions ($KH, KW > 1$)
Weights are packed into sub-channel groups of $C_{\text{out\_sub}} = 8$ output channels, with input channels padded to an atomic alignment $C_{\text{in\_align}} = 4$:
- Number of output channel groups: $G_{\text{out}} = \lceil C_{\text{out}} / 8 \rceil$
- Aligned input channels: $C_{\text{in\_pad}} = \lceil C_{\text{in}} / 4 \rceil \times 4$
- 5D Stored Tensor Shape:
  $$(G_{\text{out}}, C_{\text{in\_pad}}, KH, KW, 8)$$
- Unpacking Mapping:
  $$W[g \cdot 8 + s, c, y, x] = \text{Buffer}[g, c, y, x, s] \quad \forall \ s \in [0, 7], \ c < C_{\text{in}}$$

#### Pointwise $1\times 1$ Convolutions & GEMM
Weights are packed into atomic blocks of 8 output channels and 2 input channel pairs:
- Stored Shape:
  $$(\lceil C_{\text{out}} / 8 \rceil, \lceil C_{\text{in}} / 2 \rceil, 8, 2)$$
- Unpacking Mapping:
  $$W[g \cdot 8 + s, k \cdot 2 + p] = \text{Buffer}[g, k, s, p] \quad \forall \ s \in [0, 7], \ p \in [0, 1]$$

### 3.2 Precision and Accumulation Mechanics

#### FP16 Execution Pipeline
Inputs and weights in half-precision IEEE 754 are multiplied in FP32 and accumulated across reduction channels:

$$y_{\text{acc}}[n, c_o, h_o, w_o] = \sum_{c_i=0}^{C_{\text{in}}-1} \sum_{ky=0}^{KH-1} \sum_{kx=0}^{KW-1} x[n, c_i, h_o \cdot s_h + ky \cdot d_h - p_t, w_o \cdot s_w + kx \cdot d_w - p_l] \times W[c_o, c_i, ky, kx]$$

CACC adds optional bias and converts the FP32 accumulator to IEEE 754 `float16`:

$$y_{\text{final}} = \text{float16}(y_{\text{acc}} + \text{bias})$$

#### INT8 Execution Pipeline & Hardware Saturation
Inputs and weights in 8-bit signed integer $[-128, 127]$ are multiplied and accumulated into a 32-bit signed accumulator ($\text{int32\_t}$). CACC adds 32-bit integer bias, applies channel quantization scale, and saturates into $[-128, 127]$:

$$y_{\text{scaled}} = \left\lfloor (y_{\text{acc}} + \text{bias}) \times \text{scale} + 0.5 \right\rfloor$$

$$y_{\text{final}} = \min(127, \max(-128, y_{\text{scaled}})) \in [-128, 127]$$

---

## 4. Quick Start & CLI Usage

### Installation
From the repository root (or within a virtual environment):
```bash
cd src/eswin_simulator
pip install -e .
```

### Inspecting a Model (`--info`)
```bash
# Basic inspection of input/output tensors, memory, and submodels
eswin-sim -m path/to/model.model --info

# Detailed inspection including memory regions and relocation tables
eswin-sim -m path/to/model.model --info -v

# Programmatic JSON inspection
eswin-sim -m path/to/model.model --json
```

### Example Output
```
================================================================================
  ESWIN EIC7700X Open NPU/DSP Functional Simulator (eswin-sim)
  Milestone 1: Model Loader & Virtual Address Space Inspector
================================================================================

[Model Information]
  • File Path         : test_conv_npu_b1.model
  • Container Magic   : NVDA [OK]
  • Loadable Version  : 0.7.0
  • File Size         : 37.67 KiB
  • Tasks / Submodels : 2
  • Blobs Embedded    : 12

[Input Tensors (1)]
  Name                     Shape              Type     Format   Size         Domain  
  ------------------------------------------------------------------------------
  input                    (1, 3, 32, 32)     HALF     NCHW     6.00 KiB     SYSTEM  

[Output Tensors (1)]
  Name                     Shape              Type     Format   Size         Domain  
  ------------------------------------------------------------------------------
  output                   (1, 16, 30, 30)    HALF     NCHW     28.12 KiB    SYSTEM  

[Simulated Memory Utilization]
  • DRAM Regions (SYSTEM) : 0.11 MiB (114096 bytes across 17 regions)
  • SPRAM (On-Chip SRAM)  : 4.00 MiB / 4.00 MiB (100.0%)
  • Total Address IDs     : 18
  • Relocation Entries    : 4
```

### Simulating Model Execution & Differential Parity (`--input`, `--output`, `--compare`)
```bash
# Execute 2D Convolution simulation and compare against golden reference
eswin-sim \
  --model test_conv_npu_b1.model \
  --input ifmap.bin \
  --output ofmap_sim.bin \
  --compare ofmap_golden.bin

# Execute 1x1 GEMM simulation with custom tolerance
eswin-sim \
  --model test_linear_1x1conv_npu_b1.model \
  --input in_vec.bin \
  --output out_sim.bin \
  --compare out_golden.bin \
  --tolerance 0.01
```

---

## 5. Python API

```python
from eswin_simulator.loader import LoadableModel
from eswin_simulator.memory import VirtualAddressSpace
from eswin_simulator.engines import (
    CMACEngine,
    CACCEngine,
    unpack_nvdla_weights_conv2d,
    unpack_nvdla_weights_1x1,
)
import numpy as np

# 1. Load .model container
model = LoadableModel.from_file("test_conv_npu_b1.model")
vas = VirtualAddressSpace(model)

# 2. Initialize hardware compute engines
cmac = CMACEngine(address_space=vas)
cacc = CACCEngine(address_space=vas)

# 3. Unpack NVDLA weight blobs
w_raw = model.blobs["W_0"].data
w = unpack_nvdla_weights_conv2d(w_raw, c_out=16, c_in=3, kh=3, kw=3, dtype=np.float16)

# 4. Bind input tensor and read ndarray
in_t = model.input_tensors[0]
x = np.ones(in_t.shape, dtype=in_t.dtype)
vas.bind_input(in_t, x)

# 5. Execute CMAC 2D convolution and CACC accumulator
acc = cmac.conv2d(x, w, stride=1, padding=0, precision="FP16")
y = cacc.finalize_fp16(acc, to_fp16=True)

# 6. Write to output tensor and inspect
out_t = model.output_tensors[0]
vas.write_tensor(out_t, y)
print("Computed output shape:", y.shape, "MACs:", cmac.total_macs)
```

---

## 6. Running the Test Suite

Run all unit and differential tests via `pytest`:
```bash
pytest -v
```

All 28 comprehensive tests cover:
- Binary magic validation (`NVDA`) and error detection on corrupted containers.
- 4D Tensor descriptor parsing (shape, strides, data format, data type, element sizes).
- Memory allocation across 64-bit DRAM and the 4 MiB on-chip SPRAM.
- Dynamic relocation resolution (splitting 64-bit IOVA into lower/upper words).
- BDMA strided transfers (1D contiguous, 2D pitch-strided, 3D volume, DRAM $\leftrightarrow$ SPRAM).
- CMAC 2D convolution across strides, paddings, dilations, and grouped/depthwise modes.
- CMAC 1x1 pointwise convolution and GEMM across 2D, 3D, and 4D tensor shapes.
- CACC 32-bit accumulation, bias addition, FP16 rounding, and INT8 saturating clipping ($[-128, 127]$).
- SDP pointwise ALU operations (`bias_add`, `batch_norm`, `eltwise_mul/add/sub/min/max`).
- SDP fixed activations (`relu`, `leaky_relu`, `prelu`, `clip`) and precision conversions.
- SDP 688-byte piecewise linear LUT interpolation (`SiLU`, `Sigmoid`, `Tanh`, `GELU`).
- Differential parity testing against vendor `EsSimulator` output across all reference models:
  * `test_conv_npu_b1.model`: Cosine Similarity $= \mathbf{1.000000}$, Relative Diff $= \mathbf{0.000000}$.
  * `test_linear_1x1conv_npu_b1.model`: Cosine Similarity $= \mathbf{1.000000}$, Relative Diff $= \mathbf{0.000518}$.
  * `test_swiglu_mlp_npu_b1.model`: Cosine Similarity $= \mathbf{1.000000}$ ($0.99999976$), Max Abs Diff $= \mathbf{0.000244}$.
