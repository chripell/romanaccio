# DeepSeek & Qwen2 Open-Runtime Deployment Guide (Milestone 10)

How to run the vendor DeepSeek-7B / Qwen2 / Qwen2.5-Coder packages through
`libes_runtime.so` instead of the vendor `Qwen2Client`, and how to gate performance and
quality regressions.

## Loading a vendor package directly

`es_model_load()` resolves, in order:

1. `<model_path>/manifest.json` — the open package format;
2. `<model_path>/config.json` — the **vendor ENNP format**, loaded through the same
   `ModelManifest` parser;
3. `~/.cache/eswin/models/<repo_id>/manifest.json`.

Vendor `config.json` is understood without conversion. The root-level field mapping was
confirmed against the Qwen2-0.5B 1k INT8 package and its compiled loadable:

| vendor key | `ModelConfig` field | Qwen2-0.5B value |
|---|---|---|
| `model_name` | `architecture` | `qwen2` |
| `vocab_size` | `vocab_size` | 151936 |
| `embedding_size` | `hidden_size` | 896 |
| `embedding_dim` | `head_dim` | 64 |
| `num_head` | `num_key_value_heads` | 2 |
| `token_num` | `context_length` | 1024 |
| `token_n2` | `kv_token_n2` | 16 |
| `precision` | `elem_bytes` | 2 (fp16) |
| `model_list[].device_id` | `blocks[].die_id` | 0 / 1 |
| `model_list[].block_models[]` | `blocks[].name` + `stage` | 24 entries |

`num_attention_heads` is derived as `hidden_size / head_dim` (896/64 = 14), because the
vendor config states KV heads but not attention heads. A nested `config` object always
wins over root-level keys, so an open manifest is never overridden.

Note `num_head` is **KV** heads, not attention heads — reading it as attention heads
silently breaks RoPE and grouped-query attention.

## Context window

The window comes from the package (`token_num` → `context_length`), never from a
constant. `es_model_get_info()` reports it as `info.max_seq_len`, and
`TransformerSession::generate()` clamps each turn so `prompt + generated` stays inside it;
a prompt that alone does not fit makes `generate()` return false. Declaring more than the
graphs were compiled with does not crash, it disagrees with them — raising the window
means recompiling the loadables with a matching `--kv-cache-len`.

## Generating

```c
es_model_config_t mcfg = {0};
mcfg.model_path  = "/home/chri/models/qwen2_5_coder";
mcfg.device_mode = ES_DEVICE_DUAL_DIE_AUTO;

es_model_t *model = NULL;
es_model_load(ctx, &mcfg, &model);

es_model_info_t info = {0};
es_model_get_info(model, &info);

es_session_config_t scfg = {0};
scfg.max_seq_len    = info.max_seq_len;
scfg.max_new_tokens = info.max_seq_len;   // per-turn cap is enforced in the callback
scfg.enable_dual_die = 1;
scfg.sampler_params.temperature = 0.6f;   // Issue #47 guards:
scfg.sampler_params.top_k = 40;
scfg.sampler_params.top_p = 0.9f;
scfg.sampler_params.repeat_penalty = 1.15f;
scfg.sampler_params.repeat_window = 64;
scfg.sampler_params.max_consecutive_duplicates = 2;

es_session_t *session = NULL;
es_session_create(model, &scfg, &session);
es_session_generate(session, prompt, my_callback, my_state, &stats);
```

`es_session_config_t` is snapshotted at creation and has no setter, so per-turn control
belongs in the token callback: return `false` to stop. `src/qwen2_v2/qwen2_v2.cpp` is the
reference implementation of that pattern (thinking-token streaming, `/max-tokens` in the
REPL, startup component card).

## Perplexity as a quality regression signal

```c
double ppl = 0.0, mean_logprob = 0.0;
es_session_perplexity(session, text, &ppl, &mean_logprob);
```

Teacher-forced: it runs the prefill path only, no sampling, so it is deterministic across
runs and builds. Returns `ES_ERR_INVALID_ARG` when the text encodes to fewer than two
tokens or nothing could be scored.

**Sanity check:** with host-fallback dummy weights the distribution is uniform, so
perplexity equals the vocabulary size exactly — 1024 for the synthetic test package,
151936 for Qwen2.5-Coder. A number equal to the vocab size off-board is *correct*, not a
bug; it is also the tell that you are not running on real weights.

## Regression harness

`tools/e2e_regression.sh` builds the benchmark binary, runs it against a package, parses
its `PERF`/`PPL` lines, and gates on both throughput and perplexity:

```bash
# On the board (real numbers):
tools/e2e_regression.sh --host roma \
    --model-dir /home/chri/models/qwen2_5_coder --record   # once, to set the baseline
tools/e2e_regression.sh --host roma \
    --model-dir /home/chri/models/qwen2_5_coder             # CI: compare

# Off-board (host fallback; perplexity is meaningless but plumbing is exercised):
BUILD_DIR=build_host tools/e2e_regression.sh --build-dir build_host \
    --model-dir ../qwen2_v2 --record
```

`--host` cross-compiles with `CROSS_COMPILE=riscv64-linux-gnu-`, copies the binary and
`libes_runtime.so` to `/tmp/eswin_e2e`, and runs over SSH. Exit status is non-zero when
throughput drops or perplexity rises by more than `--tolerance` percent (default 5). The
gate is one-directional per metric: improvement never fails.

Reference points from `findings/`: Qwen2.5-Coder-1.5B sustains 4.18–5.49 tok/s on board
with all 28 layers split 14/14 across the dies; a measured `qwen2_v2` run gave 3.88 tok/s
at 619 generated tokens. The PLAN's 10.5 tok/s target applies to the Qwen2-0.5B class,
not to 1.5B/7B dense models — decode is bandwidth-bound (~2 FLOPs/byte), so compare
against the same model, not across sizes.

## Test suite

```bash
make BUILD_DIR=build_host build_host/test_deepseek_e2e
LD_LIBRARY_PATH=build_host ./build_host/test_deepseek_e2e            # host mode
ES_E2E_MODEL_DIR=/home/chri/models/qwen2_5_coder ./test_deepseek_e2e # benchmark mode
```

`tests/test_deepseek_e2e.cpp` synthesises a vendor-shaped package (config.json only, no
manifest.json) and drives the whole public C ABI over it: field mapping including the
dual-die block chain, `es_model_load` fallback to `config.json`, metadata propagation,
streaming generation, telemetry consistency, window clamping when the session asks for
4096 against a 256-token package, multi-turn reset, and the perplexity API.

## What is not done

Milestone 10's stated goal of executing the full chained pipeline *through the new
Milestone 7–9 components* is **not** complete:

- `TransformerSession` still runs its own per-layer GEMM submission path with a single
  `m_d{0,1}_gemm_in/out` buffer pair, not `PipelineExecutor`.
- KV state is still the host `m_k_cache_flat` / `m_v_cache_flat` vectors, not
  `HardwareKvCache` in MMZ.
- Embeddings are still read into `std::vector` by `load_embeddings_fp16()`, not
  memory-mapped by `EmbeddingTable`.

All three switch-overs touch the decode path that currently produces correct output on
silicon (Qwen2.5-Coder at 3.88–5.49 tok/s on `roma`). Replacing it without a board
comparison is how a working engine gets broken, so each needs its own board-verified
change: run `tools/e2e_regression.sh --host roma --record` before the switch, then
compare after. The components are built, unit-tested and ready to be wired.
