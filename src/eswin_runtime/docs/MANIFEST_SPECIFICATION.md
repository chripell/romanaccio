# Model Package Manifest Specification (`manifest.json`)

## 1. Overview
The `eswin_runtime` defines an open, standard package manifest format for packaging, distributing, verifying, and running NPU-accelerated neural networks (including LLMs like SmolLM and Qwen2, as well as vision models like YOLOv3).

A model package consists of a directory containing:
1. `manifest.json`: Structured JSON metadata, hardware execution topology, model hyperparameters, and file checksums.
2. Binary loadables: Compiled NPU models (`*.model`), one per die for dual-die pipelines.
3. Auxiliary assets: Tokenizer (`tokenizer.json`), vocabulary mappings, generation configs.

---

## 2. JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EswinModelManifest",
  "type": "object",
  "required": ["name", "version", "tensor_parallel", "config", "files"],
  "properties": {
    "name": {
      "type": "string",
      "description": "Unique model identifier, e.g. SmolLM-135M-int8"
    },
    "version": {
      "type": "string",
      "description": "Semantic version of the package, e.g. 1.0.0"
    },
    "description": {
      "type": "string",
      "description": "Human-readable description of the package and quantization scheme"
    },
    "tensor_parallel": {
      "type": "integer",
      "enum": [1, 2],
      "description": "1 = Single Die, 2 = Dual Die (NPU Core 0 + Core 1)"
    },
    "config": {
      "type": "object",
      "description": "Model architecture hyperparameters",
      "required": ["architecture", "vocab_size", "hidden_size"],
      "properties": {
        "architecture": { "type": "string" },
        "vocab_size": { "type": "integer" },
        "hidden_size": { "type": "integer" },
        "intermediate_size": { "type": "integer" },
        "num_hidden_layers": { "type": "integer" },
        "num_attention_heads": { "type": "integer" },
        "num_key_value_heads": { "type": "integer" },
        "rope_theta": { "type": "number" },
        "bos_token_id": { "type": "integer" },
        "eos_token_id": { "type": "integer" }
      }
    },
    "files": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "role"],
        "properties": {
          "name": { "type": "string" },
          "role": {
            "type": "string",
            "enum": ["weights_die0", "weights_die1", "tokenizer", "config", "weights_monolithic"]
          },
          "sha256": {
            "type": "string",
            "pattern": "^[a-f0-9]{64}$"
          },
          "size_bytes": { "type": "integer" },
          "url": { "type": "string" }
        }
      }
    }
  }
}
```

---

## 3. Example: SmolLM-135M Dual-Die Package

```json
{
  "name": "SmolLM-135M-int8",
  "version": "1.0.0",
  "description": "SmolLM 135M Int8 Dual-Die NPU model (Die 0 + Die 1 tensor parallel)",
  "tensor_parallel": 2,
  "config": {
    "architecture": "smollm",
    "vocab_size": 49152,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "rope_theta": 10000.0,
    "bos_token_id": 0,
    "eos_token_id": 0
  },
  "files": [
    {
      "name": "smollm_die0.model",
      "role": "weights_die0",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size_bytes": 74582912,
      "url": "https://huggingface.co/HuggingFaceTB/SmolLM-135M/raw/main/config.json"
    },
    {
      "name": "smollm_die1.model",
      "role": "weights_die1",
      "sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
      "size_bytes": 74582912,
      "url": "https://huggingface.co/HuggingFaceTB/SmolLM-135M/raw/main/config.json"
    },
    {
      "name": "tokenizer.json",
      "role": "tokenizer",
      "sha256": "a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef",
      "size_bytes": 2154321,
      "url": "https://huggingface.co/HuggingFaceTB/SmolLM-135M/raw/main/tokenizer.json"
    }
  ]
}
```

---

## 4. CLI Tool: `eswin-fetch`

The runtime provides the `eswin-fetch` tool to manage downloads and verification:

```bash
# Fetch and verify a built-in model
eswin-fetch smollm-135m

# Fetch from a custom manifest JSON
eswin-fetch /path/to/my_model_manifest.json --cache-dir /opt/models

# Verify existing model cache integrity
eswin-fetch smollm-135m --verify-only
```
