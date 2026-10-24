# Weights & Embedding Memory-Mapping Specification (Milestone 8)

Two loaders, both read-only and zero-copy: `EmbeddingTable` for raw `embedding.bin`
tables, and `SafetensorsLoader` for Hugging Face `.safetensors` containers.

## Why mmap instead of reading into a vector

| Package | Table | Size |
|---|---|---|
| Qwen2.5-Coder-1.5B | 151936 × 1536 fp16 | 466 MB |
| DeepSeek-7B | — | ~1.08 GB |

`std::vector` loading pays a full read plus the heap copy, and doubles peak RSS during
the load. Mapping with `MAP_SHARED` + `PROT_READ` costs page-table setup only; pages
fault in during prefill when the rows are actually gathered, and the page cache is
shared with any other process using the same package.

Measured: mapping the full 466 MB Qwen2.5-Coder table takes **0.003 ms** (PLAN target was
<5 ms for 1.08 GB), i.e. open cost is independent of file size.

`MADV_WILLNEED` is issued at open to start readahead, but the mapping is deliberately
never touched there — that is what keeps open O(1).

## EmbeddingTable

```cpp
EmbeddingTable t;
std::string err;
t.open(dir + "/embedding.bin", vocab_size, hidden_size, EmbeddingPrecision::Fp16, err);

std::vector<float> row(hidden_size);
t.get_row(token_id, row.data(), row.size());     // bounds-checked, converts to fp32

std::vector<float> batch(n * hidden_size);
t.gather(token_ids, n, batch.data(), batch.size());  // out-of-range ids zero-fill
```

**Precision must be stated, not guessed.** fp16 and bf16 are both 2 bytes per element, so
file size cannot distinguish them. `open()` takes an explicit `EmbeddingPrecision`
(`Fp32` / `Fp16` / `Bf16`; ESWIN packages export fp16). `open_auto()` infers only the
unambiguous cases — `rows*cols*4` → fp32, `rows*cols*2` → fp16 — and rejects anything
else with a message naming both candidate sizes rather than reading garbage.

`open()` also validates that the file size is *exactly* `rows × cols × elem_bytes`, so a
truncated download is rejected at load instead of faulting mid-prefill.

Lookups are bounds-checked on both sides: an out-of-range token returns `false` from
`get_row()` / `nullptr` from `raw_row()`, and `gather()` zero-fills that row while still
processing the rest of the batch. An undersized destination is rejected rather than
overrun.

Conversions are portable bit manipulation (`es_f16_to_f32`, `es_bf16_to_f32`), handling
signed zero, subnormals (renormalized, not flushed), and inf/NaN. They live in `model/`
rather than reusing the copies in `llm/transformer_session.hpp` so the dependency stays
one-directional.

## SafetensorsLoader

Container layout:

```
[0..8)      little-endian u64 header length N
[8..8+N)    JSON header: name -> {dtype, shape, data_offsets:[begin,end]}
[8+N..EOF)  densely packed tensor payload
```

```cpp
SafetensorsLoader st;
st.open("model.safetensors", err);
for (const auto &name : st.tensor_names()) {          // alphabetical, not file order
    const SafetensorInfo *t = st.find(name);          // dtype, shape, offset, length
    std::vector<float> v;
    st.to_f32(name, v, err);                          // any supported dtype -> fp32
}
const void *raw = st.data("w", &nbytes);              // zero-copy pointer into mapping
```

Only the JSON header is copied out for parsing (hundreds of KB); tensor payloads are
never copied. The header length is read byte-by-byte so no alignment is assumed.

Supported dtypes for `to_f32`: `F64 F32 F16 BF16 I64 I32 I16 I8 U8 BOOL`. An unsupported
dtype (e.g. `F8_E4M3`) is reported by name rather than reinterpreted.

### Validation

Every tensor is checked before it is published, so a truncated download or hand-edited
header fails at `open()` with a message naming the offending tensor:

- header length must fit inside the file;
- `data_offsets` must be a sane `[begin, end)` pair;
- `end` must lie within the data section (catches truncation);
- `end - begin` must equal `prod(shape) × dtype_size` (catches shape/dtype mismatch).

`err` is cleared at entry to `open()` — `JsonValue::parse` only writes it on failure, so a
caller reusing one string across loads would otherwise see a previous file's message
attributed to the current one. (This was a real bug found by the test suite.)

## Test suite

```bash
cd src/eswin_runtime && make BUILD_DIR=build_host build_host/test_embedding_loader
LD_LIBRARY_PATH=build_host ./build_host/test_embedding_loader
```

`tests/test_embedding_loader.cpp` builds synthetic files on disk (no mocks) and covers:
fp16/bf16 conversion primitives including subnormals and infinity; fp32 row lookup,
gather with a bogus token in the middle, and rejection of out-of-range rows and
undersized destinations; fp16/bf16 decode and move semantics; precision inference,
truncation, missing-file and zero-dimension rejection; O(1) open against a 466 MB sparse
table (the exact Qwen2.5-Coder shape); `.safetensors` header/shape/offset/metadata
parsing with F32 and F16 decode; and rejection of truncated data, shape/dtype mismatch,
unknown dtype, oversized header length and missing files.

## Integration status

These loaders are standalone and fully tested, but `TransformerSession` still uses its
existing `load_embeddings_fp16()` / `load_weights()` paths, which read into
`std::vector`. Switching the session over is Milestone 10 work: it changes the memory
profile of a path that is currently validated on silicon, so it should be done with a
board run rather than blind.
