# Embedded Tokenizer & Advanced Sampler Guide

## 1. Overview
The `eswin_runtime` includes an embedded, high-performance, zero-external-dependency natural language processing module:
- **Tokenizer (`eswin::Tokenizer`)**: High-speed Byte-Level BPE, TikToken, and SentencePiece tokenizer supporting Hugging Face `tokenizer.json` formats (SmolLM, LLaMA, DeepSeek, Qwen2) without needing external Python runtimes or protobuf dependencies.
- **Sampler (`eswin::Sampler`)**: Production-grade sampling engine porting algorithms from `llama.cpp` alongside custom fixes for Issue #47 (degenerate repetitive text loops on Edge NPUs).

---

## 2. Sampler Features & Issue #47 Anti-Looping Engine

### Sampling Pipeline
1. **Sliding-Window Repetition Penalty**:
   - Given a context of the last $N$ generated tokens (default: 64 tokens), penalties are applied to logits before temperature scaling:
     $$\text{logit}_i = \begin{cases} \text{logit}_i / \text{penalty} & \text{if } \text{logit}_i > 0 \\ \text{logit}_i \times \text{penalty} & \text{otherwise} \end{cases}$$
2. **Consecutive Duplicate Suppressor**:
   - Immediately sets the logit of any token that has repeated $M$ consecutive times (default: 2) to $-\infty$. This eliminates stubborn stuttering loops (such as `"the the the"` or `"and and and"`).
3. **Cyclic $N$-Gram Loop Detector**:
   - Tracks alternating patterns ($ABABAB$) and suppresses the cyclical repeating token.
4. **Temperature Scaling**:
   - Scales remaining logits by $1 / T$. If $T \le 0.0$, falls back to deterministic greedy argmax.
5. **Top-$K$ Pruning**:
   - Keeps only the top $K$ highest-probability candidate tokens.
6. **Softmax Normalization**:
   - Converts pruned logits to a normalized probability distribution using numerically stable exponential shifts.
7. **Min-$P$ Cutoff**:
   - Removes any candidate whose probability is less than $\text{prob}_{\text{max}} \times \text{min\_p}$.
8. **Top-$P$ (Nucleus) Cutoff**:
   - Truncates candidates whose cumulative sum exceeds threshold $P$.
9. **Mersenne Twister Sampling**:
   - Draws a pseudo-random sample from the renormalized probability distribution.

---

## 3. Tokenizer Format Support

| Format | File Extension | Source Frameworks | Status |
| :--- | :--- | :--- | :--- |
| **Hugging Face Tokenizer** | `tokenizer.json` | SmolLM, Llama-3, Mistral, HuggingFace `tokenizers` | Supported |
| **OpenAI / Qwen TikToken** | `.tiktoken` | Qwen2.5, DeepSeek-V2/V3, GPT-4 | Supported |
| **Binary Compact Vocab** | `tokenizer.bin` | `romanaccio` standalone native engines | Supported |

### ChatML Templating
The runtime provides built-in ChatML formatting:
```cpp
std::string formatted = eswin::Tokenizer::format_chatml(
    "Explain quantum computing simply.",
    "You are a helpful and concise physics professor."
);
```
Output:
```text
<|im_start|>system
You are a helpful and concise physics professor.<|im_end|>
<|im_start|>user
Explain quantum computing simply.<|im_end|>
<|im_start|>assistant
```

---

## 4. C11 Public API Usage Example

```c
#include <es_runtime.h>
#include <stdio.h>

int main() {
    // 1. Initialize tokenizer
    es_tokenizer_t *tok = NULL;
    es_tokenizer_load("models/smollm/tokenizer.json", &tok);

    // 2. Encode text
    int32_t tokens[256];
    int32_t count = es_tokenizer_encode(tok, "Hello world!", tokens, 256);

    // 3. Initialize sampler
    es_sampler_params_t params = {
        .temperature = 0.7f,
        .top_k = 40,
        .top_p = 0.9f,
        .min_p = 0.05f,
        .repeat_penalty = 1.15f,
        .repeat_window = 64,
        .max_consecutive_duplicates = 2,
        .seed = 42
    };
    es_sampler_t *sampler = NULL;
    es_sampler_create(&params, &sampler);

    // 4. Sample token from logits
    float logits[49152] = { ... };
    int32_t next_token = es_sampler_sample(sampler, logits, 49152);
    es_sampler_accept(sampler, next_token);

    // 5. Decode token
    char piece[64];
    es_tokenizer_decode(tok, &next_token, 1, piece, sizeof(piece));
    printf("%s", piece);

    // 6. Cleanup
    es_sampler_free(sampler);
    es_tokenizer_free(tok);
    return 0;
}
```
