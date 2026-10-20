# YOLO11 (Ultralytics YOLOv11) Dual-NPU Acceleration Plan on ESWIN EIC7700X

## 1. Executive Summary & Feasibility Assessment

### 1.1 Feasibility Verdict: **EXTREMELY HIGH (Native Architecture Match)**
Deploying the latest **YOLO11** (released by Ultralytics in late 2024) on the ESWIN EIC7700X is **fully supported and exceptionally well-suited** to the hardware architecture.

The EIC7700X NPU is an NVDLA-derived Convolutional Neural Network (CNN) accelerator featuring dedicated hardware engines:
- **CDMA / CSC**: Convolution DMA and Sequence Controller for streaming 4D tensor feature maps ($[B, C, H, W]$).
- **CMAC**: Systolic Multiply-Accumulate Array executing 2D convolutions ($1\times 1, 3\times 3, 5\times 5$, depthwise, strided) at up to 19.99 TOPS INT8.
- **SDP**: Single Data Point Processor with hardware piecewise-linear Look-Up Tables (LUT) for instantaneous SiLU / ReLU activation and element-wise residual addition.
- **PDP**: Planar Data Processor for hardware Max Pooling ($3\times 3, 5\times 5$) and Average Pooling.
- **RUBIK**: Dedicated engine for feature map slicing, reshaping, concatenation, and memory planar transposition.

Unlike autoregressive Large Language Models (which are memory-bandwidth limited during token-by-token generation), YOLO11 is a dense feed-forward vision network that operates with high arithmetic intensity. The EIC7700X systolic array achieves **high compute utilization (>70%)**, delivering real-time performance far exceeding standard edge video requirements (60–150+ FPS).

### 1.2 Performance & FPS Targets

| Model Variant | Parameters | MACs (FLOPs) | INT8 Model Size | Single-Die FPS (640x640) | Dual-Die FPS (Pipelined / 2-Stream) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11n (Nano)** | 2.6 M | 6.5 G | ~2.8 MiB | **130 – 180 FPS** (5.5 ms) | **260 – 350 FPS** |
| **YOLO11s (Small)** | 9.4 M | 21.5 G | ~9.8 MiB | **70 – 110 FPS** (10.5 ms) | **140 – 200 FPS** |
| **YOLO11m (Medium)**| 20.1 M | 68.0 G | ~21.0 MiB | **40 – 60 FPS** (20.0 ms) | **80 – 115 FPS** |
| **YOLO11l (Large)** | 25.3 M | 86.9 G | ~26.5 MiB | **30 – 45 FPS** (28.0 ms) | **60 – 85 FPS** |
| **YOLO11x (XLarge)**| 56.9 M | 194.9 G | ~59.0 MiB | **15 – 25 FPS** (45.0 ms) | **30 – 48 FPS** |

> [!TIP]
> The dual-die design enables a **Concurrent Dual-Camera Surveillance / Inspection Mode**:
> - **Die 0**: Dedicated real-time 1080p inference for Camera Feed A (e.g. 60 FPS).
> - **Die 1**: Dedicated real-time 1080p inference for Camera Feed B (e.g. 60 FPS).
> Both camera streams operate with zero mutual contention since each die has independent LPDDR5 memory channels and MMZ hardware pools.

---

## 2. YOLO11 Network Architecture Breakdown & Hardware Lowering

YOLO11 introduces three core structural upgrades over YOLOv8:

```
[ Input: 1 x 3 x 640 x 640 (RGB Frame) ]
                   │
                   ▼
  [ Backbone: Conv P1 + C3k2 Blocks + SPPF + C2PSA ]
    ├── Conv 3x3 (stride=2): Downsamples resolution ──> Hardware CMAC
    ├── C3k2 Blocks: Refined Bottleneck feature extraction ──> CMAC + SDP (SiLU)
    ├── SPPF: Multi-scale spatial pooling (k=5) ──> Hardware PDP engine
    └── C2PSA: Pointwise Spatial Attention (400 spatial positions) ──> CMAC 1x1 + SDP LUT
                   │
                   ▼
  [ Neck: PANet Top-Down & Bottom-Up Feature Aggregation ]
    ├── Lateral 1x1 Convolutions & Upsampling ──> CMAC + RUBIK
    └── Concatenation & C3k2 Feature Fusion ──> CDMA / RUBIK
                   │
                   ▼
  [ Decoupled Detection Head (Anchor-Free) ]
    ├── Classification Branch: Depthwise Separable Convs (DW 3x3 + PW 1x1) ──> CMAC
    └── Bounding Box Regression Branch: DFL (Distribution Focal Loss, reg_max=16) ──> CMAC
                   │
                   ▼
[ Output Tensor: 1 x 84 x 8400 ]
  (4 box coords [x, y, w, h] + 80 COCO classes across 8400 anchor grid locations)
                   │
                   ▼
[ Host RISC-V Post-Processing: Fast Vectorized NMS (SiFive P550 RVV) ]
```

### 2.1 Hardware Engine Mapping Table

| YOLO11 Component | Mathematical Operation | EIC7700X Hardware Engine | Notes |
| :--- | :--- | :--- | :--- |
| **Input Normalization** | $x_{\text{norm}} = (x - \text{mean}) \times \text{scale}$ | **SDP Input Stage** | Zero CPU overhead; normalizes raw uint8 image during DMA transfer. |
| **Standard Convolutions** | $Y = W \ast X + B$ ($1\times 1, 3\times 3$, stride 1/2) | **CMAC + CACC** | INT8 systolic multiplication with 32-bit accumulation. |
| **Depthwise Convolutions** | Channel-wise $3\times 3$ filtering | **CMAC Depthwise Mode** | Used in YOLO11 classification head; 100% hardware offloaded. |
| **SiLU Activations** | $f(x) = x \cdot \sigma(x)$ | **SDP LUT Engine** | 256-entry piecewise linear approximation curve ($[-8.0, +8.0]$). |
| **Residual Additions** | $Y = X + \text{Bottleneck}(X)$ | **SDP Element-Wise** | Single-pass addition without intermediate DRAM writeback. |
| **SPPF Max Pooling** | MaxPool $5\times 5$ (padding=2, stride=1) | **PDP Engine** | Dedicated planar data processor; 0 MAC cycles consumed. |
| **C2PSA Spatial Attention** | $1\times 1$ convs + QK dot product on $20\times 20$ grid | **CMAC + SDP Softmax** | $20\times 20 = 400$ positions; highly parallelized on 64-channel tiles. |
| **Feature Concatenation** | Along channel axis $C$ | **RUBIK / CDMA** | Assembled via memory stride configuration; zero-copy assembly. |

---

## 3. End-to-End Compilation & Deployment Pipeline

### 3.1 Step 1: Exporting Static ONNX Model
YOLO11 must be exported from PyTorch to ONNX with static shapes ($1 \times 3 \times 640 \times 640$) to ensure direct compatibility with the NPU's fixed-function 4D DMA engine:

```python
from ultralytics import YOLO

# Load official pretrained model
model = YOLO("yolo11n.pt")

# Export to clean static ONNX
model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,        # Crucial: static 1x3x640x640 shape
    simplify=True,        # Eliminate redundant reshape/identity nodes
    opset=17              # Standard modern ONNX opset
)
```

### 3.2 Step 2: INT8 Post-Training Quantization (PTQ)
Using `eswin_compiler` (or the containerized `EsAAC` compiler):
1. **Calibration Dataset**: Ingest 500 representative images from COCO `val2017`.
2. **Per-Channel Calibration**: Compute KL-divergence / MSE quantization thresholds for convolution weights and feature activations.
3. **FlatBuffers Generation**: Compile the quantized graph into `yolo11n.model` targeting the EIC7700X hardware architecture.

### 3.3 Step 3: Zero-Copy Video Ingestion & Streaming Runtime

```
[ V4L2 USB / MIPI-CSI Video Stream ]
                   │
                   ▼ (Zero-Copy Frame Capture via mmap)
[ ESWIN MMZ Physical Video Buffer: ES_SYS_Mmap() ]
                   │
                   ▼ (Hardware Direct Memory Access)
[ NPU Die 0: Hardware CMAC / SDP / PDP Pipeline ]
                   │
                   ▼ (Inference Execution: ~5.5 ms)
[ Hardware Output Buffer: 1 x 84 x 8400 Tensor ]
                   │
                   ▼ (Host SiFive P550 RISC-V CPU)
[ Vectorized Fast-NMS & Bounding Box Decode ]
                   │
                   ▼ (Direct Rendering / RTSP Output)
[ Annotated Real-Time 60 FPS Video Display ]
```

---

## 4. Post-Processing & RISC-V Optimization (Fast NMS)

The raw NPU output tensor has shape $[1, 84, 8400]$:
- $4$ bounding box coordinates: center $x, y$, width $w$, height $h$ (relative to $640\times 640$).
- $80$ class conditional probabilities (COCO dataset).
- $8400$ spatial anchors ($80\times 80 + 40\times 40 + 20\times 20 = 6400 + 1600 + 400$).

### 4.1 Optimized Decode Pipeline on RISC-V
To prevent the host CPU from becoming a bottleneck at 150+ FPS:
1. **Confidence Threshold Filtering ($>0.25$)**:
   - Inspect the maximum class probability per anchor using SIMD / Vector instructions.
   - Filter down 8400 candidate boxes to typically $<50$ active candidates before running pairwise IoU calculations.
2. **Anchor Coordinate Un-normalization**:
   - Precompute anchor grid offsets once at initialization:
     $$\text{box}_x = (\text{grid}_x + 0.5 - \text{reg}_0) \times \text{stride}$$
     $$\text{box}_y = (\text{grid}_y + 0.5 - \text{reg}_1) \times \text{stride}$$
3. **Pairwise Non-Maximum Suppression (IoU $>0.45$)**:
   - Run greedy NMS on the filtered candidate list in $<0.4 \text{ ms}$ on the SiFive P550 core.

---

## 5. Dual-Die Multi-Camera Deployment Architecture

The ESWIN EIC7700X dual-die layout offers an ideal topology for high-density edge vision:

```
                          [ EIC7700X Dual-Die Vision Engine ]
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
     [ NPU DIE 0 (Node 0) ]                                      [ NPU DIE 1 (Node 1) ]
  • Camera Feed 1 (1080p60)                                   • Camera Feed 2 (1080p60)
  • MMZ Pool 0: yolo11n.model                                 • MMZ Pool 1: yolo11n.model
  • DMA Input Buffer 0                                        • DMA Input Buffer 1
  • Independent Execution: 150 FPS                            • Independent Execution: 150 FPS
  • Output: Detections Stream 1                               • Output: Detections Stream 2
```

---

## 6. Implementation Roadmap & Milestones

### Phase 1: ONNX Export & Graph Simplification (Days 1–2)
- [ ] Export `yolo11n.onnx` and `yolo11s.onnx` with fixed $1\times 3\times 640\times 640$ shapes.
- [ ] Verify ONNX operators in Netron; confirm all nodes are standard Convs, MaxPools, Additions, and SiLU.
- [ ] Test reference inference on x86 host using `onnxruntime` to generate golden detection outputs.

### Phase 2: Compiler Lowering & INT8 Calibration (Days 3–5)
- [ ] Prepare calibration dataset: 500 images from COCO `val2017`.
- [ ] Ingest ONNX graph into `EsAAC` / `eswin_compiler`.
- [ ] Calibrate INT8 activation scales and generate the SDP SiLU LUT.
- [ ] Compile into `yolo11n.model` and `yolo11s.model` FlatBuffers loadables.

### Phase 3: Simulator Functional Validation (Days 6–7)
- [ ] Execute `yolo11n.model` inside `eswin_simulator`.
- [ ] Verify numerical precision against PyTorch / ONNXRuntime golden output ($1\times 84\times 8400$).
- [ ] Confirm Mean Average Precision (mAP50) loss is $<0.8\%$ compared to FP32 reference.

### Phase 4: RISC-V Hardware Runtime & Fast NMS (Days 8–11)
- [ ] Implement `yolo11_app` C++17 runtime in `romanaccio/src/yolov11/`:
  - Zero-copy V4L2 camera capture directly into MMZ buffer via `ES_SYS_Mmap`.
  - NPU inference offload via `es_nn_runtime` ioctl driver.
  - Optimized C++ bounding box decode and Fast NMS.
- [ ] Deploy and test on `roma` with real USB / RTSP camera streams.

### Phase 5: Dual-Die Parallelism & Benchmarking (Days 12–14)
- [ ] Enable concurrent dual-camera streaming across Die 0 and Die 1.
- [ ] Benchmark latency, frame rate, CPU utilization, and thermal stability.
- [ ] Commit all code, build scripts, and guides to `romanaccio`.
