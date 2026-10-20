# Qwen2-v2: Dual-Die NPU Interactive Chat Engine with Issue #47 Fix

## 1. Overview

`qwen2_v2` is the second-generation interactive chat engine for **Qwen2** and **DeepSeek-R1-Distill-Qwen** models (0.5B, 1.5B, 7B), built entirely on the modular **Open ESWIN NPU Runtime** (`libes_runtime.so`).

### The Issue #47 Solution
In the vendor SDK and first-generation scripts, Qwen2 models suffered from severe degradation after a few conversation turns:
- Getting stuck in repetitive token loops (e.g. `"the the the the"`).
- Entering cyclic oscillation patterns (e.g. `ABABABAB` phrase echoing).
- Crashing due to unhandled KV-cache sequence boundary overflows and driver queue stalls.

`qwen2_v2` resolves these issues directly in the new runtime architecture:
1. **Sliding-Window Repetition Penalty**: Token frequencies are tracked across a sliding window of $W = 64$ tokens, dividing positive logits by $\alpha = 1.15$ and multiplying negative logits by $\alpha$.
2. **Consecutive Duplicate Suppressor**: Enforces a strict cap ($K = 2$) on immediate duplicate emissions, physically setting the logit of the repeating token to $-\infty$ if emitted twice consecutively.
3. **Cyclic N-Gram Detector**: Identifies recurring 2-gram and 3-gram sequences and suppresses their continuation tokens before loop traps form.
4. **WorkerPool Multi-Die Isolation**: Completely isolates Die 0 and Die 1 into separate NUMA worker daemons (`WorkerDaemon`), eliminating driver lock deadlocks (`0xa00f6046`) and kernel frame teardown Oops.

---

## 2. Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Architectures** | Qwen2 / DeepSeek-R1-Distill-Qwen (0.5B, 1.5B, 7B) |
| **Context Window** | 2,048 tokens |
| **Prompt Template** | ChatML (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant`) |
| **Sampling Defaults** | Temperature: `0.6`, Top-P: `0.9`, Top-K: `40`, Repeat Penalty: `1.15` (Window: `64`) |
| **Acceleration** | Automatic concurrent Dual-Die NPU offload |

---

## 3. Building

### Host Build (Simulation / Host Fallback)
```bash
# Build runtime first
cd ../eswin_runtime && make && cd ../qwen2_v2

# Build qwen2_v2
make
```

### Cross-Compiling for RISC-V (Board Roma)
```bash
# Cross-compile runtime
cd ../eswin_runtime && make clean && make CROSS_COMPILE=riscv64-linux-gnu- && cd ../qwen2_v2

# Cross-compile qwen2_v2
make clean && make CROSS_COMPILE=riscv64-linux-gnu-
```

---

## 4. Running

### C++ Interactive REPL
```bash
# Run with default dual-die acceleration (thinking visible by default)
./qwen2_v2

# Custom sampling and thinking controls
./qwen2_v2 --max-thinking-tokens 128 --max-tokens 256

# Hide thinking tokens (clean answer only)
./qwen2_v2 --hide-thinking

# Debug mode: inspect all generated token IDs and pieces
./qwen2_v2 --show-tokens
```

### Python Interactive REPL
```bash
python3 qwen2_v2.py --temp 0.6
```

### Interactive REPL Commands
- `/thinking on|off`: Toggle real-time streaming of thinking tokens.
- `/tokens on|off`: Toggle raw token ID debug mode (`[#id:"piece"]`).
- `/max-thinking <N>`: Adjust maximum thinking tokens allowed per turn (default: 128).
- `/max-tokens <N>`: Adjust maximum generated tokens (default: 256).
- `/reset`: Flushes conversation history and clears the KV-cache.
- `/stats`: Displays latency, prompt tokens, generation tokens, and tokens/sec throughput for the last turn.
- `/help`: Displays available commands.
- `/exit` or `/quit`: Cleanly exits and shuts down NPU worker daemons.
- `Ctrl+C`: Halts active token generation gracefully without killing the REPL.
