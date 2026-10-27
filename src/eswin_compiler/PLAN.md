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

### Milestone 7: Runtime-Bound Weight Surfaces (dynamic weight binding for MoE) [PLANNED]
- **Motivation**:
  - DeepSeek-V2-Lite needs 6 of 64 routed experts per layer, chosen at run time. A compiled
    `.model` bakes weights into blobs, so the naive mapping is one loadable per
    (layer, expert) = **1716 files**, 1716 `ES_NPU_LoadModelFromFile` calls, and an unknown
    concurrent-`modelId` limit. See `src/DeepSeek-V2-Lite/PLAN.md` §6.4.
  - `EsAAC` cannot produce the alternative: it **SIGSEGVs (exit 139)** on any ONNX graph with
    a second dynamic input, verified with a minimal `MatMul` and a control `Add`
    (§6.11 of that plan). So this capability has to come from *this* compiler, which emits
    loadables directly.
- **Evidence gathered (2026-10-24, `deep/artifacts/scratch/probe_weight_cube.py`)**:
  - `backend/gemm_builder.py` already reproduces the vendor's binding scheme exactly. Its
    `memory_list` has **Class A** entries (`contents`/`offsets` populated = blob-backed) and
    **Class B** entries (`contents=[]`, `flags=5` input / `flags=9` output, `tensor_desc_id`
    set) — matching `FINDINGS.md` §25 byte for byte.
  - The weight region is Class A: `mem_id=1 flags=3 size=1,769,472 contents=[WsplitM_0_0,
    WsplitM_1_1, WsplitM_2_2] offsets=[0, 589824, 1179648]` — three 589,824-byte M-slices.
  - `relocs` target **only** the Class B tensors: `address_id=4` (input) at `kmd-task-surf_list`
    offsets 3286/3290, 5334/5338, 7382/7386 and `address_id=5` (output) at 8470/8474, all
    `write_id=15`, `interface=2`, `sub_interface=4`, as `reloc_type` 1/2 pairs (the low and
    high halves of a 64-bit IOVA).
  - **The weight `dla_data_cube`s were located**: `kmd-task-surf_list` +0, +1024, +2048
    (1024-byte stride, one per M-slice), each `type=0` (DRAM), `address=1` (the address_list
    index of the weight region), `offset` = 0 / 589824 / 1179648 (**the cube already carries
    a slice-select offset**), `size=589824`, dims `[64,1,1,576]`. Companion `type=1`
    (SRAM) cubes at +64 and +1088 with `address=3` are the `.sram.edma` staging destinations.
- **Design**: emit the weight as a **Class B** entry — `contents=[]`, `flags=5`, a
  `bind_id`, and a matching `TensorDescEntry` — so the whole expert blob is supplied per
  submit and expert selection becomes `DevBuf.offset`. `MoERouter::expert_blob_offset()`
  (Milestone 12 of `eswin_runtime`) already computes exactly that offset, per die.
- **Open questions, ordered by cost to answer**:
  1. **RESOLVED (2026-10-24, on `roma`): the driver *does* honour a non-zero
     `DevBuf.offset`.** `tools/probe_devbuf_offset.cpp` (in `eswin_runtime`) submitted the
     same `test_linear_1x1conv_npu_b1.model` three times: pattern P0 at offset 0 of a shared
     buffer (A), pattern P1 at offset 4096 of **that same buffer** (B), and P1 at offset 0 of
     a separate buffer as the control (C). Result: `A != B`, **`B == C` byte-for-byte**. So B
     genuinely read from offset 4096 — offset is applied to the resolved IOVA, not ignored.
     **Approach (b) is viable**; full write-up in `FINDINGS.md` §26. This also demotes the
     load-time / concurrent-`modelId` measurements from blocking to merely interesting.
  2. **RESOLVED (2026-10-24, on `roma`): no reloc pair is needed.** Two variants were built
     by patching a compiled loadable (`deep/artifacts/scratch/make_dynamic_weight_model.py`):
     one with the weight converted to Class B and relocs left alone, one with six extra
     relocs added at weight-cube +22/+26 mirroring the activation I/O. They produced
     **byte-identical outputs** (22656.000 / 23680.000), so the added relocs changed nothing.
     A Class B weight is resolved purely through `address_list → mem_id → bind_id → DevBuf`.
     **Emit the simpler no-reloc form.** `FINDINGS.md` §27.
  3. **RESOLVED (2026-10-24, on `roma`): the SPRAM `.sram.edma` staging path does resolve a
     runtime-bound weight source.** The graph computed correctly with the weights coming from
     our own buffer, so MoE weights are **not** confined to `domain=0`. `FINDINGS.md` §27.
- **Scope**:
  - `backend/gemm_builder.py` / `backend/nvdla_builder.py`: add a runtime-bound weight mode
    (Class B memory entry, `bind_id` allocation, weight `TensorDescEntry`, and relocs if
    question 2 says they are needed).
  - `backend/flatbuffer_emitter.py`: `ModelWriter` must round-trip a Class B weight without
    dropping the (now empty) `contents`/`offsets` vectors — remember the FlatBuffers invariant
    that all 8 root vectors are always serialized (`FINDINGS.md` §14).
  - A `--dynamic-weights` compiler flag plus a per-tensor opt-in, so existing static paths
    are untouched and bit-reproducible.
- **Deliverables**: dynamic-weight emission in the GEMM/Conv path; a `bind_id` allocator
  shared with the activation I/O; `tests/test_dynamic_weight_binding.py`.
- **Testing Requirements**:
  - Emit a minimal 1×1 Conv with a Class B weight at `bind_id=1`; run it on `roma` **twice
    with two different weight buffers** and assert the outputs differ accordingly. This is
    the acceptance test — everything else is inference until it passes.
  - Then the real target: one loadable serving N experts by varying `DevBuf.offset`,
    bit-matching N separately-compiled static loadables (cosine ≥ 0.9999 via `eswin-sim`).
  - Verify `eswin-sim` executes the Class B weight path, so this stays testable off-board.
- **Documentation**: weight-binding specification (Class A vs Class B, `bind_id` numbering,
  cube/reloc offsets) appended to `findings/MODEL_FORMAT.md`.
- **Status**: **mechanism proven on hardware; production emitter not yet written.** All three
  open questions are resolved in favour of approach (b) — `FINDINGS.md` §26 (offset honoured)
  and §27 (Class B weight read at submit time, no reloc needed, SPRAM staging works). The
  proof used a loadable *patcher* (`deep/artifacts/scratch/make_dynamic_weight_model.py`)
  rather than an emitter change, which isolated the variable and kept the production path
  untouched: converting the weight memory entry to `contents=[]`, `flags=5`, `bind_id=1`,
  `tensor_desc_id=2` plus a `data_category=1` WEIGHT descriptor, and dropping the three
  `WsplitM_*` blobs, shrank the file from 1,838,632 to 68,872 bytes and it ran on `roma`
  with `inputFdNum=2`.
  **Bit-match validation: DONE (2026-10-24, on `roma`).** `probe_weight_bitmatch.cpp` fed the
  Class B loadable the original three `WsplitM_*` blobs concatenated in the layout the Class A
  entry declared, and ran it against the unmodified static loadable with the same varied
  deterministic activation. Output is **byte-identical over all 3072 bytes**
  (`mean|static| = mean|dynamic| = 35379.1686`). So the binding is not merely read, it is
  *faithful* — the earlier "outputs differ" evidence is now upgraded to bit-exact equality,
  and the emitter change is safe to build on. `FINDINGS.md` §27.

  **Emitter implemented (2026-10-24).** `NVDLAGEMMBuilder.build_linear_3pass()` and
  `build_from_node()` take `dynamic_weight: bool = False`. When set, the three `WsplitM_*`
  blobs are omitted, the weight memory entry becomes Class B (`contents=[]`, `flags=5`,
  `bind_id=1`, `tensor_desc_id=2`), and a `data_category=1` WEIGHT `TensorDescEntry` is
  appended — with **no** relocs, per question 2. The static path is untouched.
  Verified: emitter output is structurally correct (`flags=5 bind_id=1`, 3 descriptors,
  11 blobs, 8 relocs vs the static model's `flags=3`, 3 contents, 2 descriptors, 14 blobs),
  68,872 bytes vs 1,838,520, and **an emitter-produced dynamic model is byte-identical to
  its emitter-produced static counterpart on `roma`** (mean 31698.7018 both, 3072/3072 bytes).
  The existing compiler tests still pass (5 ok, 3 skipped for absent test-data fixtures).

  **CLI flag and test suite: DONE (2026-10-24).** `compiler.py` takes `--dynamic-weights`,
  carried on `CompilerConfig.dynamic_weights` (default `False`, so existing output is
  unchanged) and consumed in `_compile_single_layer()`. It routes to `NVDLAGEMMBuilder` with
  `dynamic_weight=True`, and **rejects a non-1×1 convolution with a clear error** rather than
  silently emitting a static loadable the caller would then mis-submit with `inputFdNum=2` —
  the 2D conv emitter still bakes weights into blobs.
  `tests/test_dynamic_weight_binding.py` (8 tests, all passing) pins the emitted contract:
  static stays Class A (`flags=3`, 3 contents, `bind_id=0`, no WEIGHT descriptor); dynamic is
  Class B (`contents=[]`, `flags=5`, `bind_id=1`, `tensor_desc_id=2`, one `data_category=1`
  descriptor sized `c_out × c_in`); no weight blobs and **no weight relocs** in either;
  activation relocs for address ids 4 and 5 survive; the dynamic model is >10× smaller;
  `ModelWriter` round-trips the empty `contents` and `bind_id`; both paths share task/memory/
  address geometry with the weight address still referenced by the task; and the flag appears
  in `--help`. The board bit-match remains `probe_weight_bitmatch.cpp`.

  **SwiGLU expert-FFN generalisation: DONE (2026-10-24), and it corrected an assumption in
  this plan.** `backend/dynamic_weight_binding.py` converts the weight regions of *any*
  compiled loadable — vendor or ours — from Class A to Class B, with a `bind_id` allocator,
  plus `extract_weight_blob()` to rebuild the submit-time buffer for bit-match testing.
  `tests/test_dynamic_weight_swiglu.py` (14 tests) runs against the real vendor
  `test_swiglu_mlp_npu_b1.model`.

  **The plan assumed "three runtime-bound weight tensors per expert". That is wrong.**
  EsAAC packs a whole FFN's gate/up/down into **one** Class A region:
  `mem_id=1, flags=3, size=5,308,416` = 3 × 1,769,472, holding nine blobs
  (`w_gatesplitM_0_0..2_2`, `w_upsplitM_0_3..2_5`, `w_downsplitM_0_6,1_7,2_8`). So an expert
  FFN needs **one** runtime-bound tensor and **one** `bind_id`, not three — which is better
  for MoE than planned: a single per-die blob can hold every expert's gate/up/down, and
  selecting an expert is one `DevBuf.offset`. The allocator still handles multiple regions,
  since nothing guarantees future templates pack the same way.

  Verified on the vendor SwiGLU block: auto-detects exactly one weight region; assigns
  `bind_id=1` (activation keeps 0); drops 9 weight blobs (20 → 11) while **all 11 KMD/UMD
  command streams survive**; tensor descriptors become `[WEIGHT, FEATURE, FEATURE]`; no
  weight relocs added and the activation relocs for address ids 4/5 preserved; weight bytes
  extractable and each blob present at its declared offset; `ModelWriter` round-trips the
  empty `contents` and `bind_id`; **5,482,752 → 173,448 bytes (31.6× smaller)**.
  The blob classifier is tested to reject every task stream and intermediate
  (`cvtPrecision`, `.sram.edma`, `tiling_`, `Concat`, `mul_`, `biasadd`, `_BIAS`) — dropping
  a command stream would brick the loadable, so that invariant is asserted explicitly.
  Also covered: idempotence (a second call is a no-op), explicit `mem_ids` override, and the
  converter working on our own `gemm_builder` output as well as vendor models.
  Full compiler suite: 25 tests pass.

  **Not yet done for this archetype: the board bit-match.** The single-weight GEMM case is
  proven byte-identical on `roma` (§27), but the SwiGLU conversion has only been verified
  structurally. `probe_weight_bitmatch.cpp` is hardcoded for one weight input at 1536×576;
  validating the FFN needs `inputFdNum=2` with a 5,308,416-byte weight buffer and the
  SwiGLU model's own activation geometry (1152 B in / 1152 B out). Until that passes, treat
  multi-blob-region conversion as structurally sound but numerically unconfirmed.

  Remaining work, in order:
  1. ~~Board bit-match for the SwiGLU conversion~~ **DONE (2026-10-27, on `roma`).**
     `probe_weight_bitmatch.cpp` now takes geometry as optional argv (and rejects a
     non-4096-aligned weight size); the converted vendor FFN is byte-identical to the static
     one over all 1152 output bytes, so the converter is hardware-proven for a multi-blob
     weight region and the 1716-loadable path is closed off. `FINDINGS.md` §28.
  1b. Measured alongside it with `tools/probe_submit_overhead.cpp`: **0.233–0.305 ms/submit**
     and **104–214 ms/model load** on a small FFN. At 318 submits/token that is a
     10.3–13.5 tok/s ceiling — not the binding constraint (§24's 5.0–7.2 tok/s bandwidth
     dominates), but only ~1.4–1.9× margin, so batching experts into fewer larger graphs is
     worth real design effort. Load cost also rules out approach (a) by measurement
     (~180–370 s startup), and ≥40 concurrent loaded models showed no saturation.
  2. Expose the converter through the CLI for pre-compiled loadables (`--convert-weights`),
     since it operates on vendor `.model` files that never pass through our emitter.
  3. Extend `--dynamic-weights` to the 2D conv emitter, or lower spatial convs through the
     GEMM path, so the flag is not 1×1-only.
  4. Wire the runtime side: a session path that keeps one expert blob resident per die and
     submits with `inputFdNum = 1 + n_weight_regions`, taking each offset from
     `MoERouter::expert_blob_offset()`.
  5. A native SwiGLU emitter remains unbuilt: `hw_templates.py` has only `kmd_*`/`umd_*` and
     `linear3p_*` command-stream templates, so emitting an FFN from scratch needs new
     `dep_graph`/`op_list`/`surf_list` templates. Note this means FINDINGS §15's "complete
     SwiGLU MLP compilation capability" describes the *passes* (SDP fusion, LUT generation,
     SPRAM tiling), not a from-scratch emitter — corrected there on 2026-10-27.
  6. **Silent single-op truncation: FIXED (2026-10-27).** `_compile_single_layer()` took
     `primary_conv = conv_nodes[0]` and discarded the rest, while `compile_safetensors_llm()`
     fed it real multi-node graphs and logged success — so a SwiGLU or attention "loadable"
     contained exactly one convolution. Measured node counts after lowering: **MLP = 3 CONV +
     1 MUL** (3 of 4 dropped) and **attention = 5 CONV + RMSNorm + RoPE + Softmax** (7 of 8
     dropped).
     Now guarded: any graph whose nodes would not all be emitted raises a `ValueError` naming
     the dropped count and op types, before anything is written. `allow_partial_graph`
     (CLI `--allow-partial-graph`) keeps the old behaviour reachable but logs a `PARTIAL`
     warning, so nothing is silent either way. Covered by
     `tests/test_partial_graph_guard.py` (9 tests): measured node counts asserted as
     regression evidence, rejection for MLP and attention, **no artifact written on
     rejection**, opt-in emits and warns, a genuine single-CONV graph still compiles (no
     false positive), and `--dynamic-weights` cannot smuggle a multi-op graph past the guard.
     Three existing tests that had been encoding the buggy behaviour now opt in explicitly
     with a comment stating they verify emission/simulator plumbing, not block correctness —
     notably `test_compiled_swiglu_mlp_execution` previously passed against a truncated
     loadable, which is precisely how this survived.
     Multi-op emission itself is still unbuilt (item 5); the guard only stops it from being
     mistaken for something that works.

---

## 6. Coding Standards & Review Guidelines

1. **Comments & Rationales**: Every non-trivial algorithmic decision, bit shift, or register setting must have an accompanying explanation comment referencing the NPU hardware architecture or DSP ABI.
2. **Deterministic Output**: Compilation must be 100% bit-reproducible given identical inputs and flags.
3. **No External Network Dependencies**: The compiler must operate strictly in offline environments without fetching weights or schemas over the network during compilation.
