# Multi-Layer Block Chaining & Dual-Die Pipeline Guide (Milestone 7)

`PipelineExecutor` (`src/backend/pipeline_executor.hpp`) chains the N compiled block
loadables of a large model into one activation path that stays resident in MMZ for the
whole chain, instead of bouncing each layer's activation back through the CPU.

## Why it exists

A 28-layer model compiles to 28+ `.model` loadables per die. The naive integration
submits layer *i*, copies its activation to a host buffer, then re-uploads it as layer
*i+1*'s input — two DMA round-trips per layer. On this SoC that is expensive in a
specific, measured way: CPU reads from `NoCache` MMZ collapse to 10–60 MB/s, while the
D2D interconnect caps at 21.85 GB/s. Per-layer round-trips turn a bandwidth-bound decode
into a latency-bound one.

## Design

**Ping-pong activation buffers.** Two buffers per die. Stage *i* reads slot `i & 1` and
writes slot `(i & 1) ^ 1`, so stage *i*'s output buffer *is* stage *i+1*'s input buffer.
Nothing is copied between stages on the same die.

**Buffers are `CacheMode::Cached`.** The cross-die handoff is performed by the CPU
through the mapped virtual addresses, and cached MMZ reads run at 1.74–2.89 GB/s versus
10–60 MB/s uncached. Every CPU write is followed by `flush_cache()` so the receiving
die's NPU sees it.

**Per-die MMZ zones and SMMU mapping.** Buffers are allocated in the zone of the die
that executes the stage (`mmz_nid_0_part_0` / `mmz_nid_1_part_0`) and prepared on that
die only — mapping a Die 0 buffer into Die 1's SMMU wastes a mapping and can leave stale
IOVA entries after a fault.

**One handoff per die boundary.** With pipeline parallelism (layers `0..N/2` on Die 0,
`N/2..N` on Die 1) the activation crosses the D2D link exactly once per token, which
matches the measured behaviour of the Qwen2.5-Coder port (a single ~3 KB hidden-state
transfer per token).

## Declaring a chain in the package manifest

Open format — explicit die assignment and chain position:

```json
"blocks": [
  { "name": "blk0.model", "die_id": 0, "stage": 0 },
  { "name": "blk1.model", "die_id": 0, "stage": 1 },
  { "name": "blk2.model", "die_id": 1, "stage": 2 }
]
```

Vendor ENNP `config.json` is accepted as-is; `model_list[].device_id` maps to `die_id`
and `block_models[]` order becomes the stage number:

```json
"model_list": [
  { "device_id": 0, "block_models": ["modified_block_0_npu_b1.model", "..."] },
  { "device_id": 1, "block_models": ["modified_block_14_npu_b1.model", "..."] }
]
```

`ModelManifest::blocks_for_die(die)` returns one die's entries ordered by `stage`
(a stable sort, so entries without an explicit stage keep declaration order).

## Usage

```cpp
WorkerPool pool;
pool.start(2);

std::vector<PipelineStage> chain;
for (const auto &b : manifest.blocks) {
    PipelineStage st;
    st.die_id    = b.die_id;
    st.model_id  = loaded_id_for(b.name);          // from WorkerPool::load_model
    st.in_bytes  = hidden_size * sizeof(float);
    st.out_bytes = hidden_size * sizeof(float);
    st.label     = b.name;
    chain.push_back(st);
}

PipelineExecutor exec(pool);
std::string err;
if (!exec.init(chain, err)) { /* err explains the failure */ }

PipelineStats stats;
exec.run(hidden_in, in_bytes, hidden_out, out_bytes, &stats);
// stats.avg_stage_us, stats.cross_die_handoffs, stats.handoff_ns, stats.stage_ns[]
```

`release()` unmaps each buffer from the die that prepared it and frees the DMA
allocations; the destructor calls it, and `init()` may be called again afterwards.

## Off-board behaviour

`EswinDriver::is_hardware_available()` is false on a development host, where
`WorkerPool::submit_task` reports success without computing anything and `DmaBuffer`
backs onto `posix_memalign`. In that mode the executor applies the caller-supplied
`set_host_stage_fn()` transform instead of submitting, so chaining order, ping-pong
alternation, handoff accounting and teardown remain verifiable in CI. The host function
is never consulted when real dies are present.

## Test suite

```bash
cd src/eswin_runtime && make BUILD_DIR=build_host build_host/test_layer_chaining
LD_LIBRARY_PATH=build_host ./build_host/test_layer_chaining
```

`tests/test_layer_chaining.cpp` covers: rejection of an empty chain and of an invalid
`die_id`; 2/8/28-stage single-die chains verified by an additive invariant
(activation == seed + stage count, so a skipped or duplicated stage fails loudly);
the same at 2/8/28 stages split across both dies with exactly one handoff asserted;
manifest parsing in both spellings plus serialization round-trip and a manifest-driven
run; per-stage latency (the PLAN target of <40 µs/layer is asserted only on hardware);
and RSS growth across 40 init/run/release cycles as a leak check.

## Not yet wired into the decode path

`TransformerSession::step()` still submits its per-layer GEMMs directly, because CPU work
(RoPE, attention softmax, KV update) is interleaved between them — the chain is not a
pure NPU-only sequence. Folding the executor into that hot path is Milestone 10 work and
needs board verification against the working Qwen2.5-Coder package before it replaces a
silicon-validated path.
