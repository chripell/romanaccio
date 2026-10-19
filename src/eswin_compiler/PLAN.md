# Open EsAAC Compiler (`eswin_compiler`) Implementation Plan

## 1. Project Overview & Objectives

The goal of `eswin_compiler` is to provide a fully transparent, modular, open-source compiler that converts neural network models (vision models and modern transformer LLMs) into the offline `.model` FlatBuffers loadable format executable on the ESWIN EIC7700X / EIC7702 NPU hardware and DSP cores.

### 1.1 Goals
- Replace the proprietary, closed-source, 162 MB `EsAAC` binary.
- Support both **General Neural Networks** (via ONNX) and **Modern Transformer LLMs** (via Hugging Face `safetensors`).
- Generate bit-compatible, production-ready `.model` FlatBuffers loadables matching the reverse-engineered `NVDA` schema.
- Implement intelligent hardware graph partitioning between NPU fixed-function hardware engines (CDMA, CSC, CMAC, CACC, SDP, PDP, CDP, RUBIK) and the 372 Cadence Vision Q7 DSP kernels (`dsp_kernels/libes_*.so`).
- Support Dual-Die parallel compilation (Die 0 + Die 1) and memory tiling across the 4 MiB on-chip SPRAM.

### 1.2 Non-Goals
- Real-time JIT compilation (this is strictly an Ahead-of-Time / AOT compiler).
- General C/C++ compiler for DSP kernels (we target the existing, tested DSP `.so` kernels or precompiled host-side stubs).

---

## 2. Model Input Format Evaluation: ONNX vs. Safetensors vs. GGUF

Choosing the right input representation is critical to avoid the severe bugs and friction observed in vendor tools (such as `getBatchFromInputShape failed` on dynamic 2D ONNX inputs):

| Evaluation Criteria | ONNX (`.onnx`) | Safetensors (`.safetensors` + `config.json`) | GGUF (`.gguf`) |
| :--- | :--- | :--- | :--- |
| **Data Included** | Computational graph topology + weights. | Pure raw weight tensors + JSON metadata (no computational graph). | Graph metadata + quantized tensor data in a single binary. |
| **LLM Graph Handling** | **Poor/Brittle**: Exporting dynamic causal attention, dynamic RoPE, and KV caching to ONNX frequently creates non-standard subgraphs, unsupported operators (`GatherND`, `Range`), or collapses 4D tensor constraints. | **Outstanding**: Since Transformer architectures (Llama, Qwen, SmolLM, Mistral) follow standard mathematical blocks, a compiler can construct the **exact, optimal 4D NPU execution graph** directly from `config.json`. | **Good**: Well-structured for LLMs, but primarily designed for CPU inference and custom quantized formats (Q4_K, Q5_K) not natively supported by the NPU's INT8 systolic array. |
| **Vision & General Models** | **Outstanding**: Native standard for YOLO, ResNet, MobileNet, EfficientNet, CLIP. | **Poor**: No standard graph format for arbitrary vision architectures. | **Unsupported**: Vision support in GGUF is nascent and limited. |
| **Quantization Metadata** | Embedded via ONNX QuantizeLinear or auxiliary JSON. | Raw tensors; scales stored in auxiliary `quant_stats.json` or per-tensor safetensors metadata. | Quantization metadata baked into tensor types. |
| **Tooling & Ecosystem** | `onnx`, `onnxruntime`, PyTorch exporter. | Native Hugging Face format; zero-copy memory mapping (`mmap`). | `gguf-py`, `llama.cpp`. |

### Architectural Decision: Hybrid Dual-Frontend
1. **Safetensors Frontend (Specialized for LLMs)**:
   - Input: Hugging Face directory containing `model.safetensors`, `config.json`, and optional `quant_stats.json`.
   - The compiler uses a parameterized Transformer IR generator that programmatically constructs the exact 4D tensor shapes ($[1, C, 1, 1]$), splits Q/K/V projections for CMAC $1\times 1$ convs, configures dynamic KV-cache strides, and binds the Dual-Die LM Head.
   - **Benefit**: Eliminates 100% of ONNX export failures and dynamic shape crashes.
2. **ONNX Frontend (General Purpose Vision & CNNs)**:
   - Input: Static-shape ONNX model (opset 13–18).
   - Ingests graph topology, constants, and attributes for arbitrary CNNs (YOLOv3–v11, ResNet, etc.).

---

## 3. Programming Language Evaluation & Selection

### Options Considered
- **C/C++**: High execution speed, strong binary layout control, but high implementation complexity for graph matching, algebraic rewriting, and shape inference.
- **Python (Selected)**:
  - Exceptional graph manipulation ecosystem (`networkx`, `numpy`, `onnx`, `safetensors`).
  - Google FlatBuffers has an official, high-performance pure-Python code generator and runtime (`flatbuffers` PyPI package).
  - Rapid prototyping, interactive inspection, and clean unit testing with `pytest`.
  - Compile-time speed for offline models (a few seconds) is dominated by weight I/O, where Python's `memoryview` and `mmap` achieve wire-speed throughput.

### Choice: Modern Python (3.10+) with Strict Typing & Heavy Commenting
- **Strict Typing**: Full `mypy` static typing (`typing.Annotated`, `TypeAlias`, dataclasses).
- **Docstrings & Comments**: Every pass, class, and register emitter must include extensive docstrings explaining the mathematical semantics, memory layout, and NVDLA hardware constraints.

---

## 4. Module & Directory Architecture

```
src/eswin_compiler/
├── PLAN.md                               # This detailed implementation roadmap
├── pyproject.toml                        # Build configuration (setuptools/poetry, pytest, black, mypy)
├── README.md                             # User guide, CLI examples, architecture documentation
├── eswin_compiler/
│   ├── __init__.py                       # Package exports and version
│   ├── cli.py                            # Command-line interface (`eswin-compile`)
│   ├── schema/                           # FlatBuffers schema definitions & generated code
│   │   ├── loadable.fbs                  # FlatBuffers schema for .model (NVDA)
│   │   └── loadable/                     # Compiled Python classes from flatc
│   ├── frontend/                         # Model importers
│   │   ├── base.py                       # Abstract ModelImporter interface
│   │   ├── onnx_importer.py              # Ingests ONNX graphs and converts to EIR
│   │   ├── safetensors_importer.py       # Direct Hugging Face LLM graph synthesizer
│   │   └── quant_loader.py               # Ingests quantization tables (INT8 scales & zero-points)
│   ├── ir/                               # ESWIN Intermediate Representation (EIR)
│   │   ├── graph.py                      # Computational DAG (Nodes, Edges, Subgraphs)
│   │   ├── node.py                       # IR Node (operator type, attributes, engine target)
│   │   ├── tensor.py                     # IR Tensor (dims [N,C,H,W], dtype, memory domain, layout)
│   │   └── op_types.py                   # Canonical operator enumerations (Conv, GEMM, SDP, etc.)
│   ├── passes/                           # Graph transformation & optimization passes
│   │   ├── base.py                       # Pass interface (run_on_graph)
│   │   ├── shape_validator.py            # Ensures all shapes are strictly 4D and statically known
│   │   ├── constant_folding.py           # Evaluates compile-time constant subgraphs
│   │   ├── gemm_to_conv1x1.py            # Lowers MatMul/GEMM into 1x1 Convolution for CMAC
│   │   ├── fusion_sdp.py                 # Fuses BiasAdd + ReLU/Sigmoid/Clip into SDP descriptor
│   │   ├── dsp_partitioner.py            # Detects non-NPU ops (RMSNorm, Softmax) & assigns DSP kernels
│   │   ├── sram_tiler.py                 # Memory planning: SPRAM (4MB) vs DRAM tiling
│   │   └── dual_die_splitter.py          # Partitions models across Die 0 and Die 1 (e.g. LM Head)
│   └── backend/                          # Code generation and .model emission
│       ├── nvdla_builder.py              # Generates hardware register programming for KMD submodels
│       ├── dsp_task_builder.py           # Packages DSP task descriptors and parameter blobs
│       ├── memory_planner.py             # Assigns memory IDs, offsets, and alignment (4KB/2MB)
│       ├── relocator.py                  # Generates RelocEntry list for dynamic IOVA patching
│       └── flatbuffer_emitter.py         # Serializes final FlatBuffers container with NVDA magic
└── tests/                                # Comprehensive test suite
    ├── test_schema_roundtrip.py          # FlatBuffers parsing and serialization tests
    ├── test_onnx_importer.py             # Verification of ONNX parsing
    ├── test_safetensors_importer.py      # LLM block synthesis tests
    ├── test_passes.py                    # Unit tests for each optimization pass
    ├── test_golden_comparison.py         # Differential tests comparing against EsAAC output
    └── test_simulator_execution.py       # Verifies compiled .model files run in EsSimulator
```

---

## 5. Incremental Step-by-Step Implementation Roadmap

Development is broken down into **6 small, independently reviewable and testable steps**. No step proceeds to the next without 100% passing test coverage.

### Milestone 1: FlatBuffers `.model` Specification, Parser & Round-Trip Serializer
- **Scope**:
  - Implement `loadable.fbs` matching the reverse-engineered schema in `findings/MODEL_FORMAT.md`.
  - Compile schema using `flatc` into Python bindings.
  - Implement `ModelReader` and `ModelWriter` in `eswin_compiler/backend/flatbuffer_emitter.py`.
- **Deliverables**:
  - `eswin_compiler/schema/loadable.fbs`
  - `eswin_compiler/backend/flatbuffer_emitter.py`
  - `tests/test_schema_roundtrip.py`
- **Testing Requirements**:
  - Load existing valid vendor models (`artifacts/test_conv_npu_b1.model`, `artifacts/test_swiglu_mlp_npu_b1.model`).
  - Verify every field: memory list, tensor descriptions (names, dims $[1, C, 1, 1]$, strides), submodels, relocations, blobs.
  - Re-serialize to binary and assert structural and semantic parity.
- **Documentation**:
  - Detailed docstrings in `flatbuffer_emitter.py` explaining each table's role.

### Milestone 2: ESWIN Intermediate Representation (EIR) & Frontend Importers
- **Scope**:
  - Implement `ir/graph.py`, `ir/node.py`, and `ir/tensor.py`.
  - Implement `frontend/onnx_importer.py` converting ONNX graphs into EIR.
  - Implement `frontend/safetensors_importer.py` reading Hugging Face weights and synthesizing EIR for SmolLM / Qwen MLP and Attention blocks.
- **Deliverables**:
  - Complete `ir/` and `frontend/` packages.
  - Unit tests ingesting `goal/SmolLM-135M.onnx` and `artifacts/test_conv.onnx`.
- **Testing Requirements**:
  - Assert that all extracted tensor shapes are correctly transformed to 4D $[N, C, H, W]$.
  - Assert that weight tensors extracted from `.safetensors` match PyTorch golden tensors bit-for-bit.
- **Documentation**:
  - `ir/README.md` defining EIR node types and invariants.

### Milestone 3: NPU Fixed-Function Hardware Lowering (Conv2D & 1x1 Pointwise GEMM)
- **Scope**:
  - Implement `passes/gemm_to_conv1x1.py`: transforms `MatMul`/`Gemm` nodes into 2D Convolution with $1\times 1$ kernel.
  - Implement `backend/nvdla_builder.py`: generates KMD submodel command streams for CDMA, CSC, CMAC, and CACC.
  - Implement memory allocation assigning input, output, and weight offsets.
  - Emit valid `.model` for a standalone Conv/Linear layer.
- **Deliverables**:
  - `passes/gemm_to_conv1x1.py`
  - `backend/nvdla_builder.py`
  - Compilation CLI test producing `output.model`.
- **Testing Requirements**:
  - Differential testing: Run generated `output.model` through `EsSimulator` against the golden tensor produced by PyTorch / ONNXRuntime. Assert cosine similarity $\ge 0.9999$.
  - Hardware integration: Run generated `output.model` on board `roma` via `/opt/eswin/bin/es_run_model -d 0 -m output.model -r 10` with exit code 0.
- **Documentation**:
  - Comments detailing NVDLA register offsets (e.g. `CACC_D_MISC_CFG_0`, `CMAC_A_CFG_0`).

### Milestone 4: Activation & Elementwise Fusion (SDP Engine Lowering)
- **Scope**:
  - Implement `passes/fusion_sdp.py`: detects and fuses Bias Addition, ReLU, Sigmoid, and Mul operations directly into the SDP (Single Data Point) engine.
  - Build piecewise LUT table generator for Sigmoid and SiLU activations.
  - Implement `passes/sram_tiler.py` for weight caching in 4MB SPRAM.
- **Deliverables**:
  - Complete SwiGLU MLP compilation capability (Gate Conv + Sigmoid + Mul + Up Conv + Down Conv).
- **Testing Requirements**:
  - Compile SmolLM SwiGLU block.
  - Verify against vendor-compiled `artifacts/test_swiglu_mlp_npu_b1.model` in `EsSimulator` with numerical tolerance $< 10^{-4}$.
  - Benchmark on physical `roma` board: assert execution time $\le 0.35\text{ ms}$.
- **Documentation**:
  - Theory and layout of SDP LUT piecewise tables.

### Milestone 5: DSP Submodel Partitioning & Offloading
- **Scope**:
  - Implement `passes/dsp_partitioner.py`: identifies operations outside NPU CMAC/SDP capabilities:
    - `RMSNorm` $\to$ `libes_onnx_rms_normalization_f16f16.so`
    - `Softmax` $\to$ `libes_onnx_softmax.so`
    - `RoPE` $\to$ `libes_qk_rotary_position_embedding_f16f16.so`
  - Implement `backend/dsp_task_builder.py`: formats UMD DSP submodels and parameter binary structures.
- **Deliverables**:
  - Full Attention Block compilation (QKV Linear + RoPE DSP + Score + Softmax DSP + Context Linear).
- **Testing Requirements**:
  - Verify complete attention block in `EsSimulator` against PyTorch reference.
  - Validate on physical `roma` board.
- **Documentation**:
  - Parameter struct layouts for each offloaded DSP kernel.

### Milestone 6: Dual-Die Partitioning & Full Transformer Pipeline
- **Scope**:
  - Implement `passes/dual_die_splitter.py`: automatically splits wide layers (e.g., LM Head $576 \to 49152$ or $896 \to 151936$) into parallel halves for Die 0 and Die 1.
  - Generate paired loadables (`*_d0.model` and `*_d1.model`).
  - End-to-end CLI: `eswin-compile --model smollm-135m/ --output build/ --target dual-die`.
- **Deliverables**:
  - Production-ready compilation pipeline for SmolLM2, Qwen2.5, and Llama-3.2.
  - Full end-to-end integration test suite.
- **Testing Requirements**:
  - End-to-end compilation of all 30 layers of SmolLM-135M.
  - Integration with `src/smollm/` chat engine on `roma`.
- **Documentation**:
  - Comprehensive `README.md`, CLI manual, and Developer Architecture Guide.

---

## 6. Coding Standards & Review Guidelines

1. **Comments & Rationales**: Every non-trivial algorithmic decision, bit shift, or register setting must have an accompanying explanation comment referencing the NPU hardware architecture or DSP ABI.
2. **Deterministic Output**: Compilation must be 100% bit-reproducible given identical inputs and flags.
3. **No External Network Dependencies**: The compiler must operate strictly in offline environments without fetching weights or schemas over the network during compilation.
