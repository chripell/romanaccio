# Qwen2-VL-2B Multimodal Vision-Language Acceleration Plan on ESWIN EIC7700X

## 1. Executive Summary & Feasibility Assessment

### 1.1 Feasibility Verdict: **EXTREMELY HIGH (Leverages Existing Qwen2 Dual-Die Engine)**
Deploying **Qwen2-VL-2B-Instruct** (Vision-Language Model) on the ESWIN EIC7700X is **highly feasible and strategic**. It transforms the Roma board from a text-only system into a **fully multimodal edge AI workstation** capable of:
- Understanding camera frames and live video feeds.
- Parsing scanned documents, PDF forms, invoices, and charts (Visual Document OCR).
- Analyzing desktop UI screenshots for local automation.

Qwen2-VL-2B consists of two tightly coupled components:
1. **Vision Encoder (ViT / SigLIP)**:
   - Dynamic resolution patch projection using 2D Convolutions ($14\times 14$ patch size).
   - Passes image patches through a 24-layer Vision Transformer to generate visual tokens.
   - **Hardware Match**: The 2D patch convolutions and ViT matrix multiplications map to the NPU's CMAC array with high systolic efficiency.
2. **Language Decoder (Qwen2 1.5B/2B Core)**:
   - 28 Transformer layers, hidden dimension $d_{model} = 1536$, vocabulary $152,064$.
   - **Hardware Match**: We already have the production-tested Qwen2 dual-die engine operational in `romanaccio/src/qwen2_fix/` generating at ~10.5 tokens/second on 7B. On the 2B model, the same pipeline will achieve **~18 to 25 tokens/second**!

### 1.2 Resource Footprint & Performance Projections

| Component | Architecture | Weight Footprint (INT8) | Target Hardware | Latency / Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Tower** | ViT ($14\times 14$ patches, 24 layers) | ~450 MiB | NPU Die 0 (CMAC systolic array) | **~100 – 140 ms** per image ($448\times 448$) |
| **Multimodal Projector**| 2-Layer MLP ($1280 \to 1536$) | ~8 MiB | NPU Die 0 / Host CPU | **<1 ms** |
| **Language Decoder** | Qwen2 (28 layers, $d_{model}=1536$) | ~1.85 GiB | Dual-Die Pipeline (Die 0 + Die 1) | **~18 – 25 tokens/second** |
| **Total System VRAM** | Comprehensive Multimodal Package | **~2.31 GiB** | Fits easily within 22 GiB MMZ pool | **>18 GiB Headroom remaining** |

---

## 2. Multimodal Architecture & Hardware Pipeline

```
  [ Input Image: 448 x 448 RGB ]                      [ User Text Prompt: "What is in this picture?" ]
                 │                                                          │
                 ▼                                                          ▼
  [ NPU Die 0: Vision Patch Stem ]                            [ Host CPU: TikToken Tokenizer ]
  • Conv2D (3 -> 1280, k=14, stride=14) ──> CMAC Array                      │
  • Produces (32 x 32) = 1024 visual tokens                                 │
                 │                                                          │
                 ▼                                                          │
  [ NPU Die 0: Vision Transformer (ViT) ]                                   │
  • 24 layers of multi-head self-attention on visual tokens                 │
  • 2D RoPE (Rotary Position Embedding) for spatial awareness               │
                 │                                                          │
                 ▼                                                          │
  [ Spatial Merge & 2-Layer MLP Projector ]                                 │
  • Merges 2x2 neighboring patches -> 256 compact visual tokens             │
  • Projects from dim 1280 to dim 1536                                      │
                 │                                                          │
                 └───────────────────────────────┬──────────────────────────┘
                                                 │
                                                 ▼
                          [ Concatenated Token Sequence: <|vision_start|> ... <|vision_end|> User text ]
                                                 │
                                                 ▼
                          [ Proven Qwen2 Dual-Die Inference Engine ]
                            ├── Die 0: Layers 0 to 13 + Split LM Head
                            └── Die 1: Layers 14 to 27 + Split LM Head
                                                 │
                                                 ▼
                          [ Streaming Response: "The image depicts a circuit board..." ]
```

---

## 3. Implementation Roadmap & Milestones

### Phase 1: Model Ingestion & Vision Tower Lowering (Days 1–3)
- [ ] Ingest `Qwen/Qwen2-VL-2B-Instruct` from Hugging Face.
- [ ] Export Vision Encoder to ONNX with static image resolution ($448\times 448$ or $336\times 336$).
- [ ] Map $14\times 14$ patch convolution to NPU CMAC $2D$ convolution operator.
- [ ] Compile Vision Transformer layers to `.model` format using `eswin_compiler` / `EsAAC`.

### Phase 2: Dual-Die Language Model Integration (Days 4–6)
- [ ] Re-target our operational `qwen2_fix` dual-die pipeline for the 2B configuration ($d_{model}=1536$, 28 layers).
- [ ] Extract INT8 quantized layer weights and partition 14 layers to Die 0, 14 layers to Die 1.
- [ ] Compile the split LM Head ($152,064$ vocab $\to 76,032$ per die).

### Phase 3: Simulator Functional Validation (Days 7–8)
- [ ] Validate end-to-end vision token projection and cross-modal attention in `eswin_simulator`.
- [ ] Verify that visual token embeddings produce identical cosine similarity against PyTorch golden outputs.

### Phase 4: RISC-V Multimodal Runtime (`qwen2_vl_app`) (Days 9–12)
- [ ] Implement `qwen2_vl_app` C++17 runtime in `romanaccio/src/qwen2_vl/`:
  - Image loading and preprocessing (stb_image / libjpeg).
  - Vision token execution on Die 0.
  - Interleaved text/vision KV-cache injection.
  - Streaming conversational CLI with image argument (`--image photo.jpg --prompt "Describe this image"`).

### Phase 5: Hardware Benchmark & Tuning (Days 13–14)
- [ ] Deploy to `roma` and benchmark total latency from image ingestion to first token.
- [ ] Verify OCR accuracy on test document images and camera snapshots.
- [ ] Commit all code, build scripts, and guides to `romanaccio`.
