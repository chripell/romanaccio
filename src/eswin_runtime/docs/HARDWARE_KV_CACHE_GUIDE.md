# Hardware MMZ KV-Cache Guide (Milestone 9)

`HardwareKvCache` (`src/llm/hardware_kv_cache.hpp`) keeps the K and V caches in
physically contiguous MMZ DMA memory, per layer, on the die that runs that layer — which
is what the compiled NPU block loadables expect. The Milestone 5 `KvCache` keeps them in
host `std::vector`s, which the NPU graphs cannot consume.

## The layout is a property of the loadable

Two mutually incompatible KV layouts exist in this toolchain, and picking the wrong one
does not crash — it silently attends to the wrong positions and reproduces the
Issue #47 repetition/gibberish failure. Both are implemented and selected explicitly via
`KvLayout`.

**A. `HeadMajor` — verified against a real loadable.** Read off
`modified_block_0_npu_b1.model` from the Qwen2-0.5B 1k INT8 ENNP package using
`eswin-sim --info -v`:

```
Input  (1, 896, 1, 1)     HALF  hidden state (hidden_size 896)
Input  (1, 1, 1, 1)       INT16 token step
Input  (2, 1024, 1, 64)   HALF  V cache  [kv_heads, kv_len, 1, head_dim]  256.00 KiB
Input  (2, 64, 1, 1024)   HALF  K cache  [kv_heads, head_dim, 1, kv_len]  256.00 KiB
Output (1, 128, 1024, 1)  HALF  K update [1, kv_heads*head_dim, kv_len, 1]
Output (1, 2048, 64, 1)   HALF  V update [1, kv_heads*kv_len, head_dim, 1]
```

So K is `[head][dim][token]` with **token contiguous**, V is `[head][token][dim]` with
**dim contiguous**, and `kv_len == 1024 == token_num` from the package's `config.json`.

```
k_offset(h, d, t) = ((h * head_dim + d) * max_seq_len + t) * elem
v_offset(h, t, d) = ((h * max_seq_len + t) * head_dim + d) * elem
```

**B. `TokenMajorTiled` — the vendor formulas from FINDINGS.md §5**, recorded from the
DeepSeek-7B `_peer` package, where K is token-major and V is tiled:

```
K: elem * kv_dim * token
V: elem * (token_n2 * head_dim * (token / token_n2) + (token % token_n2))
```

`k_token_offset()` / `v_token_offset()` return these verbatim in this mode, for callers
building task descriptors. `token_n2` defaults to the largest of {256,128,…,1} that
divides `max_seq_len` and must divide it exactly — but the default is only a fallback.
The real value is a property of the compile and is published by the package: the
Qwen2-0.5B 1k INT8 `config.json` declares `token_n2: 16` (with `embedding_dim1: 4`,
`embedding_dim2: 16`, `token_n1: 64`), which `ModelManifest` now reads into
`config.kv_token_n2`. Pass that through rather than relying on the divisor heuristic.

`HeadMajor` is the default because it is the one confirmed against a real compiled
loadable rather than inferred from a formula.

### Layout discovery gap

The authoritative source is each loadable's own tensor descriptors. On hardware the
runtime can reach them through `EswinDriver::get_input_tensor_desc()`, but the value that
crosses the worker IPC boundary (`ModelInfo`) carries only `num_inputs`, `num_outputs`
and buffer **sizes** — not dims or strides. So the layout currently has to be declared by
the caller. Extending `IpcResponse`/`ModelInfo` to carry `NPU_DIMS4_S` per tensor would
let `HardwareKvCache` derive the layout from the graph itself; that is the right fix and
is not done here.

## NUMA placement

With `dual_die = true`, layers `0..L/2` are allocated in `mmz_nid_0_part_0` and
`L/2..L` in `mmz_nid_1_part_0`, matching the pipeline-parallel split. The decision is
made once at allocation, never per step: cross-die allocation costs up to 7.7× on writes
and 3.25× on latency, and remote streaming reads cap at ~5.8 GB/s versus ~14 GB/s local.

Buffers are `CacheMode::Cached` even though the NPU is the primary writer, because the
CPU reads them back for golden-data verification and in host fallback — `NoCache` MMZ
reads collapse to 10–60 MB/s. Every CPU write is followed by `flush_cache()`.

## SMMU preparation

`allocate(cfg, pool, err)` prepares each buffer **through the worker that owns its die**.
Preparing from the master process fails with `0xA00F604A`, because `eswin::prepare()`
needs a thread-local device context that only exists in the worker (FINDINGS §18–19).
Pass `nullptr` off-board, where `DmaBuffer` is backed by `posix_memalign`.

## Circular addressing and window safety

`slot_for(token)` maps an absolute position into the window, wrapping when
`cfg.wrap` is true (the hardware behaviour for a compiled `token_num`). `advance(n)`
saturates `length()` at capacity in wrap mode, and in strict mode (`wrap = false`)
**returns false** rather than letting a step write past the compiled window — the overflow
that corrupts attention state in Issue #47.

`reset(zero_fill)` rewinds the write position without reallocating or re-preparing
anything, which is what makes multi-turn conversation cheap: 200 simulated turns reuse one
physical allocation. `zero_fill` additionally clears resident data.

## Usage

```cpp
HardwareKvConfig cfg;
cfg.num_layers   = manifest.config.num_hidden_layers;   // 24
cfg.num_kv_heads = manifest.config.num_key_value_heads; // 2
cfg.head_dim     = manifest.config.hidden_size / manifest.config.num_attention_heads;
cfg.max_seq_len  = manifest.config.context_length;      // vendor token_num, e.g. 1024
cfg.elem_bytes   = 2;                                   // fp16, as compiled
cfg.layout       = KvLayout::HeadMajor;
cfg.dual_die     = (manifest.tensor_parallel >= 2);

HardwareKvCache kv;
std::string err;
if (!kv.allocate(cfg, &worker_pool, err)) { /* err names the layer and die */ }

// Bind into a task descriptor:
DevBuf k_desc{kv.k_mem_fd(layer), kv.k_token_offset(token), kv.bytes_per_layer(), 0};

kv.write_k(layer, token, k_values);   // [head][dim] order, kv_dim floats
kv.advance();
```

Memory: `bytes_per_layer() == kv_heads * head_dim * max_seq_len * elem_bytes`, i.e.
256 KiB per cache for the Qwen2-0.5B geometry — 12 MiB total for 24 layers × 2 caches.
A 7B model at 16K context is ~0.92 GiB (see `findings/CONTEXT_WINDOW_AND_MEMORY.md`).

## Test suite

```bash
cd src/eswin_runtime && make BUILD_DIR=build_host build_host/test_hardware_kv_cache
LD_LIBRARY_PATH=build_host ./build_host/test_hardware_kv_cache
```

`tests/test_hardware_kv_cache.cpp` asserts `bytes_per_layer() == 262144` for the real
Qwen2-0.5B geometry (tying the arithmetic to the observed 256.00 KiB tensors), then
covers: highest-element offsets staying in bounds; write/read round-trips in **both**
layouts over tokens {0,1,7,63,1023} × 3 layers with per-(layer,token,channel) distinct
values so a swapped index fails; layer aliasing; the FINDINGS §5 K and V formulas at tile
boundaries; circular wrap, saturation, cheap reset (buffer identity and `memFd`
preserved) and strict-mode refusal; dual-die NUMA placement; six invalid configurations;
the fp16 codec including overflow to infinity; and 200 reset cycles reusing one
allocation.

On board `roma` the same binary allocates real MMZ and prepares through the workers.

## Integration status

`TransformerSession` still uses the host `KvCache` plus its flat `m_k_cache_flat` /
`m_v_cache_flat` vectors. Switching the decode path to MMZ-resident caches changes the
memory profile of code that is currently validated on silicon (Qwen2.5-Coder at
4.18–5.49 tok/s), so it belongs in Milestone 10 with a board run, not in a blind
refactor.
