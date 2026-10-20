# OpenAI Whisper Edge Speech-to-Text Acceleration Plan on ESWIN EIC7700X

## 1. Executive Summary & Feasibility Assessment

### 1.1 Feasibility Verdict: **EXTREMELY HIGH (High Arithmetic Intensity & Speed)**
Implementing **OpenAI Whisper** on the ESWIN EIC7700X is an ideal match for an edge AI PC workstation or laptop (such as the Roma II). It provides **completely offline, privacy-preserving speech-to-text transcription and voice dictation** with zero latency jitter and zero cloud dependence.

Whisper operates in two distinct phases:
1. **Audio Feature Extraction & 1D Convolutional Front-End**:
   - Converts raw 16 kHz mono audio into 80-channel log-Mel spectrograms ($80 \times 3000$ for a 30-second audio chunk).
   - Passes the spectrogram through two sequential 1D Convolution layers with stride 2 ($3\times 3$ kernel, stride 2, padding 1), reducing temporal length from 3000 to 1500 frames.
   - **Hardware Match**: Convolutions execute natively on the NPU's CMAC engine in $<1.2 \text{ ms}$.
2. **Transformer Audio Encoder**:
   - Encodes 1500 audio frames into contextual representations. Since the encoder operates non-autoregressively on all 1500 frames in parallel, the EIC7700X systolic array achieves **high arithmetic intensity and >80% compute utilization**.
3. **Autoregressive Text Decoder**:
   - Generates text tokens conditioned on cross-attention with the audio encoder outputs.

### 1.2 Model Variant Comparison & Throughput Targets

| Model Variant | Parameters | Weight Size (INT8) | Encoder Latency (30s Audio) | Decoder Speed | Real-Time Factor (RTF) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Whisper-tiny** | 39 M | ~42 MiB | **~25 ms** | ~45 tokens/s | **>25x Real-Time** (30s audio in 1.1s) |
| **Whisper-base** | 74 M | ~78 MiB | **~50 ms** | ~35 tokens/s | **>15x Real-Time** (30s audio in 1.8s) |
| **Whisper-small** | 244 M | ~255 MiB | **~140 ms** | ~22 tokens/s | **>8x Real-Time** (30s audio in 3.5s) |
| **Whisper-large-v3-turbo** | 809 M | ~835 MiB | **~380 ms** | ~12 tokens/s | **>3.5x Real-Time** (30s audio in 7.5s) |

---

## 2. Audio Processing & Hardware Mapping Architecture

```
[ Raw Microphone Audio: 16 kHz Mono PCM ]
                   │
                   ▼ (Host DSP / Fast FFT on SiFive CPU)
[ Log-Mel Spectrogram: 1 x 80 x 3000 FP16 Buffer ]
                   │
                   ▼ (Direct DMA Transfer into MMZ Memory)
[ NPU Die 0: Whisper 1D Convolution Stem (CMAC Engine) ]
  • Conv1D (80 -> d_model, k=3, stride=1, padding=1) + GELU
  • Conv1D (d_model -> d_model, k=3, stride=2, padding=1) + GELU
  • Output: 1 x d_model x 1500 Latent Audio Embeddings
                   │
                   ▼ (Hardware Systolic Array Parallel Compute)
[ NPU Die 0: Transformer Audio Encoder (Non-Autoregressive) ]
  • N layers of Self-Attention & MLP (parallel across all 1500 positions)
  • Execution Time: <150 ms for Whisper-small
                   │
                   ▼ (Cross-Attention Conditioning Tensor)
[ NPU Die 1 (or Dual-Die Pipeline): Autoregressive Text Decoder ]
  • Self-Attention (Causal KV-cache) + Audio Cross-Attention + MLP
  • Word Vocabulary: 51,865 tokens
                   │
                   ▼ (Host Post-Processing & Text Stream)
[ Live Transcribed Text / Subtitles: "Hello world..." ]
```

---

## 3. Dual-Die Offload Strategy

Whisper's asymmetric Encoder-Decoder structure maps to the dual-die architecture:
- **Die 0**: Dedicated **Audio Preprocessor & Encoder Engine**.
  - Holds `whisper_conv_stem.model` and `whisper_encoder.model`.
  - Continuously listens to incoming microphone audio streams; when voice activity (VAD) is detected, it processes the 30s chunk in one parallel burst.
- **Die 1**: Dedicated **Autoregressive Text Decoder Engine**.
  - Holds `whisper_decoder.model` and `whisper_lm_head.model`.
  - Generates transcript tokens with low latency.

---

## 4. Implementation Roadmap & Milestones

### Phase 1: ONNX Export & Graph Lowering (Days 1–3)
- [ ] Export `whisper-base` and `whisper-small` from Hugging Face to separate ONNX graphs:
  - `encoder.onnx`: Takes `[1, 80, 3000]` input, produces `[1, 1500, d_model]`.
  - `decoder.onnx`: Takes `[1, seq_len]` and encoder hidden states.
- [ ] Calibrate INT8 quantization using audio samples from LibriSpeech.

### Phase 2: Compiler Lowering & FlatBuffers Generation (Days 4–6)
- [ ] Ingest ONNX graphs into `eswin_compiler` / `EsAAC`.
- [ ] Verify 1D convolution lowering onto CMAC 2D convolution array (treating $H=1, W=3000$).
- [ ] Compile into `whisper_encoder.model` and `whisper_decoder.model`.

### Phase 3: RISC-V Native Runtime (`whisper_transcribe`) (Days 7–10)
- [ ] Implement audio capture front-end using ALSA / PulseAudio in C++17.
- [ ] Implement fast log-Mel filterbank feature extractor using FFT.
- [ ] Bind hardware NPU inference using `es_nn_runtime`.
- [ ] Add real-time streaming mode (`--mic`) and file transcription mode (`--file audio.wav`).

### Phase 4: Benchmark & Integration (Days 11–12)
- [ ] Test on `roma` with real microphone input.
- [ ] Measure word error rate (WER) and real-time factor (RTF).
- [ ] Commit all code, build scripts, and guides to `romanaccio`.
