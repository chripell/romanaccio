# LFM3-1.7B on the ESWIN EIC7700X (dual-die)

Native C++17 engine (`lfm3_v2`) for LiquidAI's LFM3-1.7B tri-hybrid backbone: 13 gated short
convolutions, 5 sliding-window attentions (256) and 6 global attentions across 24 layers, run
layer-pipelined over both NPU dies (`PP=2`, layers 0-11 on die 0, 12-23 on die 1) with a
tensor-parallel LM head split 8 x 16,000.

Design and milestones live in `PLAN.md`. Measurements live in `../../FINDINGS.md` (§36-§43).
This file is only "how to run what is already built".

## 1. Which build to run today

| package | accuracy gate | decode | use it for |
|---|---|---|---|
| **FP16** (`LFM3-1.7B_FP16`) | **PASSES** -- logits cosine 0.99999267 vs PyTorch FP32, min-layer 0.99998623 (§43.1) | ~6.6 tok/s | everything real, right now |
| INT8 (`LFM3-1.7B_ENNP`) | FAILS -- min-layer 0.729, logits 0.887 (§43.2) | ~5.5 tok/s | investigation only; do not trust output |
| mixed, FP16 `proj2048` (`LFM3-1.7B_MIXED`) | FAILS -- min-layer 0.857, logits 0.816 | ~5.1 tok/s | the ablation series |

FP16 is the only configuration that reproduces its reference. The INT8 path's failure is a
quantization-boundary problem, not an engine problem -- the engine is shared by all three.

## 2. Run it

Everything is on `roma` (`ssh roma`), and the NPU nodes need root (`sudo`).

```bash
ssh roma
cd /home/chri/models/LFM3-1.7B_FP16

# one-shot prompt
sudo ./lfm3_v2 --prompt "Explain what a depthwise convolution does in one sentence." -n 64

# the full acceptance gate: 6-token teacher-forced prefill, cosine per layer,
# 4-token greedy match against the golden, 16-token sustained throughput
sudo ./lfm3_v2 --verify-golden golden_ref.bin
```

**`--model-dir` matters, and silently.** The engine now resolves its package from the **current
working directory** (falling back to the well-known board paths, and failing loudly if it finds
no `cpu_weights.bin`). It prints the resolved absolute path and, per die, the weight blob's path
with an FNV checksum of its first and last 64 KiB. Until §43 this defaulted to a hardcoded path,
which made a package-swap experiment "reproduce" the previous package's numbers bit for bit -- so
**check those two lines before believing any result**.

Useful flags and environment variables:

| | meaning |
|---|---|
| `--int8` | force INT8/INT16 boundary scaling (the INT8 and mixed packages need it; **FP16 must not have it** -- FP16 weights under INT8 scaling give cosine ≈ -0.06, worse than chance) |
| `--per-subgraph-scales` | use per-subgraph thresholds instead of the shared per-archetype ones |
| `--verify-golden FILE` | run the acceptance gate; exits non-zero if it fails |
| `--max-tokens N`, `--max-ctx N` | generation and context limits |
| `--weights-layer0` | timing experiment only: bind every layer to one weight region per archetype, to separate NPU streaming speed from per-submit overhead. **Generated text is garbage.** |
| `LFM3_DUMP_SURFACES=1` | print what `ModelInfo` reports for every input/output slot at load time |
| `LFM3_MEASURE_EXTENT=1` | write-extent probe: sentinel-fill each output buffer, submit, report the high-water mark per archetype (§45). Slow; diagnostic only |
| `LFM3_SMALL_SURFACES=1` | geometry-sized output buffers (~32% faster) -- **not deterministic on run 1, do not ship** (§45.2) |

### Gate-integrity checks (built in, not optional)

`--verify-golden` first proves the gate can fail before letting it pass anything:

1. **DETERMINISM** -- the same prefill re-run in the same process must be bit-identical.
2. **SENSITIVITY** -- `0xFF` XORed over 1 MiB at offset 0 of the resident weight blob must move
   the metrics (any difference counts; a non-finite metric counts loudest).
3. **RESTORE** -- XOR it back and the result must reproduce run 2.

If any fails, the run aborts before the decode test. This is what caught the staging-buffer race
(§42.5), and it is the reason a `FAIL` from these tests is more informative than a `PASS`.

## 3. Rebuilding a package

Host side, in this order. `deep/` holds the checkpoints and all generated artifacts.

```bash
V=/home/chri/aixigua/venv/bin/python
cd /home/chri/aixigua/romanaccio/src/LFM3

$V inspect_and_golden.py                      # config audit + golden oracle  (M1)
$V export_lfm3.py                             # 74 static 4D opset-13 ONNX + CPU weight sidecars (M2)
$V batch_compile_lfm3.py                      # EsAAC -> .model loadables + manifest (M3, container)
$V export_lfm3_prefill.py                     # T=64 prefill graphs               (M3.5)
$V calibrate_lfm3_int8.py                     # real-activation INT8 tables      (M5)
$V build_lfm3_int8_pkg.py                     # INT8 package via the NVDLA tiler (M5)
$V build_mixed_proj2048_fp16.py               # mixed FP16-proj2048 package
```

Then deploy to `roma:/home/chri/models/<name>/` and `cd` there to run. `embedding.bin` is 500 MB
and byte-identical across packages -- `md5sum` before re-sending it, the link to the board is
~5 MB/s and the blobs are ~1-3 GB.

The EsAAC container (`ssh simulator`) has two traps that cost real time: `-o` is ignored (output
lands in `<cwd>/model_files/`, or `/tmp/model_files/` if the cwd is not writable), and **it exits 0
even when it compiled nothing** -- a truncated upload reports `Model <x>.onnx has no graph` and
still returns success. Grep the logs and count output files. Uploads run at ~0.56 MB/s per stream
but scale ~8x in parallel; pulls run at ~134 MB/s.

## 4. Known limits and open bugs

- **INT8 accuracy.** min-layer 0.729 against FP16 0.99998 is too low for quantization noise, and
  FP16 `proj2048` moved min-layer *up* while moving logits *down* -- non-monotonic in precision,
  which points at a boundary representation mismatch (§43.2). Per-archetype substitution is the
  discriminator.
- **Calibration is thin and in-sample.** 32 sentences / **512 tokens** of real text, captured with
  PyTorch forward hooks (not synthetic noise), supplies every threshold -- and the same 512 tokens
  are then used to score perplexity. Needs a larger, held-out set.
- **Output-buffer sizing costs throughput.** `ES_SYS_MemFlushCache(fd)` takes an fd and no range
  (`nm` on `libes_memory.so` confirms: `ES_SYS_MemFlushCache`, `...All`, `...AllByNid`,
  `ES_VB_PoolFlushCache` -- nothing address/length scoped), so maintenance is over the whole
  allocation. Entry-sized (2 MiB) buffers are required for INT8, which is deterministic with them
  and *nondeterministic* with the old 32 KiB ones (§40.2, re-confirmed after the race fix); FP16
  stays deterministic with small buffers and runs ~32% faster (6.61 vs 5.03 tok/s). Sizing each
  buffer to what its graph actually writes -- rather than the driver's 2 MiB bucket or a fixed
  constant -- is the open optimization.
- **Prefill.** T=64 batching is built (M3.5) and reaches ~83 prompt tok/s; the non-MLP projections
  are still T=1, which floors prefill (§1.1 of `PLAN.md`).
- **Long context.** The global-attention KV lives on the host; measured cost crosses the NPU's
  weight-stream cost somewhere between 32K and 128K tokens (§1.1 item 3). Nothing above ~4K is
  validated.
- **Attribution experiment not yet run to completion.** `attribute_int8_error.py` measures the
  fp32 baseline, weights-only INT8, activations-only, and both, to say whether the INT8 gap is
  weight precision or boundary handling. It currently dies inside `modeling_lfm3.py`'s conv path
  (`Expected 2D or 3D input to conv1d, but got [1, 1, 2048, 6]`) once the boundary hooks are
  installed, because `Lfm3ShortConv` transposes around its projections -- wrapping the Linears is
  not shape-transparent there. Fixing that is the next step before drawing any conclusion from it.
- **Nothing exercises the >256-token paths end to end yet**: the INT8 SWAR attention kernel, the
  sliding-window ring wrap and flash-decoding splits all engage past position 256, and the golden
  test is 6 tokens. Each is unit-benchmarked; none is integration-tested.
