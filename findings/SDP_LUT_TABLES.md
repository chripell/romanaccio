# NVDLA / EIC7700X SDP Lookup Table (LUT) Specification & Theory

## 1. Overview of the SDP Engine

The **SDP (Single Data Point)** engine in the EIC7700X NPU is a versatile post-processing pipeline that operates on the output of the CMAC / CACC convolution engine (flying mode) or directly on DRAM/SRAM activation tensors (offline mode).

Each NPU core includes an SDP engine capable of executing:
1. **Bias Addition**: Constant or per-channel vector addition.
2. **Batch Normalization / Scaling**: Per-channel scale and bias multiplication.
3. **Pointwise ALU Operations**: Pointwise Addition, Subtraction, Multiplication with another tensor.
4. **Non-linear Activations**:
   - Fixed-function: ReLU, Leaky ReLU, Clip.
   - Arbitrary functions via Look-Up Tables: Sigmoid, SiLU (Swish), Tanh, GELU, Exponential.

---

## 2. Mathematical Theory of SDP Piecewise Interpolation

Non-linear functions are evaluated in hardware using **piecewise linear interpolation** with two lookup tables:
- **`LE` Table (Linear-Exponential)**: 65 sample points ($2^6 + 1$) for coarse or exponential regions.
- **`LO` Table (Linear-Only)**: 257 sample points ($2^8 + 1$) for fine-grained linear interpolation.

For an input activation $x \in [x_\text{min}, x_\text{max}]$:
1. The hardware normalizes $x$ to determine the interval index $i$:
   $$\text{idx} = \left\lfloor \frac{x - x_\text{min}}{\Delta x} \right\rfloor$$
2. The hardware extracts the base value $y_i = \text{table}[i]$ and the next value $y_{i+1} = \text{table}[i+1]$.
3. The fractional interpolation offset $\delta$ is scaled by the slope scale factor and added:
   $$y = y_i + \left( (x - x_i) \cdot \text{slope\_scale} \gg \text{slope\_shift} \right)$$

For activation functions such as Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$) and SiLU ($x \cdot \sigma(x)$), the input domain is bounded to $[-8.0, +8.0]$. Outside this domain:
- For $x < -8.0$: $\sigma(x) \approx 0.0$, $\text{silu}(x) \approx 0.0$.
- For $x > +8.0$: $\sigma(x) \approx 1.0$, $\text{silu}(x) \approx x$.

---

## 3. Binary Structure of `struct _lut_dev` (688 Bytes)

Extracted from DWARF symbols in `EsSimulator` and verified against vendor loadables (`test_swiglu_mlp_npu_b1.model`):

```c
struct _lut_dev {
    uint32_t precision;             // +0:   0x0002 for FP16, 0x0004 for INT8
    uint32_t lut_cfg;               // +4:   0x0031 (linear interpolation enabled)
    uint32_t lut_info;              // +8:   Routing and table selection
    float    le_start;              // +12:  LE domain minimum (-8.0f)
    float    le_end;                // +16:  LE domain maximum (+8.0f)
    float    lo_start;              // +20:  LO domain minimum (-8.0f)
    float    lo_end;                // +24:  LO domain maximum (+8.0f)
    uint32_t le_slope_scale;        // +28:  Fixed-point slope multiplier (0x3f80)
    uint32_t lo_slope_scale;        // +32:  Fixed-point slope multiplier (0x3f80)
    uint32_t le_slope_shift;        // +36:  Bit shift for LE slope interpolation (0)
    uint32_t lo_slope_shift;        // +40:  Bit shift for LO slope interpolation (0)
    int16_t  linear_exp_table[65];  // +44:  130 bytes (65 FP16 sample values)
    int16_t  linear_only_table[257];// +174: 514 bytes (257 FP16 sample values)
};
// Total Size: 44 + 130 + 514 = 688 bytes
```

### Table Layout & Offsets
| Offset (hex) | Offset (dec) | Size (bytes) | Field | Description |
|:---|:---|:---|:---|:---|
| `0x0000` | 0 | 4 | `precision` | Input/output precision (`2` = FP16) |
| `0x0004` | 4 | 4 | `lut_cfg` | Configuration flags (`0x0031`) |
| `0x0008` | 8 | 4 | `lut_info` | Route selector (`0x0000`) |
| `0x000c` | 12 | 4 | `le_start` | Domain minimum for LE table (`-8.0f`) |
| `0x0010` | 16 | 4 | `le_end` | Domain maximum for LE table (`+8.0f`) |
| `0x0014` | 20 | 4 | `lo_start` | Domain minimum for LO table (`-8.0f`) |
| `0x0018` | 24 | 4 | `lo_end` | Domain maximum for LO table (`+8.0f`) |
| `0x001c` | 28 | 4 | `le_slope_scale` | Slope scaling factor |
| `0x0020` | 32 | 4 | `lo_slope_scale` | Slope scaling factor |
| `0x0024` | 36 | 4 | `le_slope_shift` | Slope bit shift |
| `0x0028` | 40 | 4 | `lo_slope_shift` | Slope bit shift |
| `0x002c` | 44 | 130 | `linear_exp_table` | 65 piecewise FP16 sample points |
| `0x00ae` | 174 | 514 | `linear_only_table`| 257 piecewise FP16 sample points |
| **Total**| | **688** | | |

---

## 4. Compiler Integration & Lowering

1. **`passes/fusion_sdp.py` (`SDPFusionPass`)**:
   - Detects `CONV -> ADD (bias)` patterns and attaches the bias tensor directly to the convolution node.
   - Detects `CONV -> RELU / SIGMOID / SILU / CLIP` patterns and sets `conv_node.attributes["activation"]`.
2. **`backend/lut_generator.py` (`generate_sdp_lut`)**:
   - Synthesizes the exact 688-byte binary blob with FP16 values sampled over $[-8.0, +8.0]$.
   - Emits the blob as `kmd-task-lut_list` in the FlatBuffers package.
3. **Execution in `libnn_runtime.so` & `EsSimulator`**:
   - `dla_sdp_op_desc.lut_index` is set to `0` to activate the LUT table.
   - During execution, the SDP unit interpolates values on the fly with zero overhead.
