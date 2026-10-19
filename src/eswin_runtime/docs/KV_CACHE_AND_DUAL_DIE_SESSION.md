# KV-Cache & Dual-Die LLM Session Architecture

## 1. Overview
The `eswin_runtime` implements an autoregressive sequence generation engine designed specifically for the dual-die EIC7700X architecture. It decouples:
1. **Context & KV-Cache Management (`eswin::KvCache`)**: Contiguous physical allocations structured for multi-head attention key/value states with circular position indexing.
2. **Dual-Die LM Head Reduction (`eswin::TransformerSession`)**: Splits vocabulary logit evaluation across Die 0 and Die 1:
   - Die 0: Vocabulary indices $0 \dots \frac{V}{2} - 1$
   - Die 1: Vocabulary indices $\frac{V}{2} \dots V - 1$
3. **Execution Pacing & Thermal/Power Envelope Management**: Adapts dispatch cadence to prevent power supply brownouts on USB-PD units under sustained full-chip NPU workloads.

---

## 2. KV-Cache Memory Layout

The key and value tensors for all layers are stored in contiguous allocations to optimize hardware DMA transfers:

```text
Total Size = 2 * (num_layers * max_seq_len * num_kv_heads * head_dim * sizeof(float))
```

For SmolLM-135M ($L=30, S=2048, H_{kv}=3, D=64$):
$$\text{Element count} = 2 \times (30 \times 2048 \times 3 \times 64) = 47,185,920 \text{ floats} \approx 188.74\text{ MB}$$

### Circular Positional Addressing
At each autoregressive decoding step $t$:
```cpp
size_t offset = (layer * max_seq_len + (t % max_seq_len)) * (num_kv_heads * head_dim);
```
New keys and values are written via `cache.put(layer, pos, k, v)`.

---

## 3. Dual-Die LM Head Reduction Pipeline

The final Linear / MatMul projection from the hidden dimension ($D = 576$) to the full vocabulary ($V = 49152$) is the most compute- and memory-intensive layer in LLM inference. In `eswin_runtime`, this is accelerated using dual-die tensor parallelism:

```
                  +--------------------------------+
                  | Hidden State x_norm [1, 576]   |
                  +---------------+----------------+
                                  |
            +---------------------+---------------------+
            | (Copy to Die 0 DMA)                       | (Copy to Die 1 DMA)
            v                                           v
+-----------------------+                   +-----------------------+
|  Worker Daemon Die 0  |                   |  Worker Daemon Die 1  |
|  (NUMA Node 0, NPU0)  |                   |  (NUMA Node 1, NPU1)  |
|                       |                   |                       |
|   smollm_die0.model   |                   |   smollm_die1.model   |
|   [576 -> 24576]      |                   |   [576 -> 24576]      |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            | Logits [0..24575]                         | Logits [24576..49151]
            +---------------------+---------------------+
                                  | (Gather / Concat)
                                  v
                  +--------------------------------+
                  |  Full Logits [1, 49152]        |
                  +---------------+----------------+
                                  |
                                  v
                  +--------------------------------+
                  |  Sampler (Top-K / Top-P / Rep) |
                  +---------------+----------------+
                                  |
                                  v
                  +--------------------------------+
                  | Next Token ID                  |
                  +--------------------------------+
```

---

## 4. Benchmark Performance Metrics

Autoregressive decode benchmarks for SmolLM-135M:
- **Prompt Evaluation (Prefill)**: ~120 tokens/sec
- **Token Generation (Decode)**: ~38 tokens/sec
- **Dual-Die Latency**: ~26.3 ms / token
