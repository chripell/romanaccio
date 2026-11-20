#!/usr/bin/env python3
"""
Unified OpenAI-Compatible HTTP API Gateway for LFM3-1.7B & LFM2.5-8B-A1B on ESWIN EIC7700X.

Provides:
- GET  /v1/models
- POST /v1/chat/completions (streaming SSE & non-streaming, multi-turn, multimodal, prefix cache telemetry)
- POST /v1/completions      (standard text completions with prefix cache telemetry)
- POST /v1/decisions        (single-pass calibrated decision engine for d1-2B / LFM3)
- GET  /v1/cache/stats      (RadixCache + Tier-0 RAM telemetry)
- POST /v1/cache/clear      (clear unpinned or all prefix cache entries)
- GET  /healthz

Routes dense LFM3 models (lfm3_v2 --daemon 8088) and Sparse MoE models
(lfm2_moe --daemon 8089) concurrently via --backend-map or --backend-port/--moe-port.
"""

import os
import re
import sys
import io
import time
import json
import uuid
import socket
import base64
import asyncio
import logging
import argparse
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
import numpy as np
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# Ensure local LFM3 directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lfm3_openai_server")

DEFAULT_MODELS: List[str] = [
    "lfm3-1.7b",
    "lfm3-1.7b-int8",
    "lfm3-1.7b-heretic",
    "lfm3-vl-2b",
    "d1-2b",
    "lfm2.5-8b-a1b",
]

app = FastAPI(
    title="Romanaccio EIC7700X Unified OpenAI API Server",
    description="Unified OpenAI-compatible gateway for LFM3-1.7B and LFM2.5-8B-A1B on dual-die ESWIN EIC7700X NPU",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Configuration & Multi-Backend Routing
# -----------------------------------------------------------------------------
class ServerConfig:
    def __init__(self) -> None:
        self.backend_host: str = "roma"
        self.backend_port: int = 8088
        self.moe_port: int = 8089
        self.backend_mode: str = "tcp"
        self.default_model: str = "lfm3-1.7b"
        self.backend_map: Dict[str, Tuple[str, int]] = {}
        self.available_models: List[str] = list(DEFAULT_MODELS)


config = ServerConfig()


def parse_backend_map(spec: str) -> Dict[str, Tuple[str, int]]:
    """Parses comma-separated 'model_or_prefix=host:port' routing overrides."""
    mapping: Dict[str, Tuple[str, int]] = {}
    if not spec:
        return mapping
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid --backend-map entry '{entry}': expected 'model=host:port'")
        model_key, target = entry.split("=", 1)
        model_key = model_key.strip().lower()
        target = target.strip()
        if not model_key or ":" not in target:
            raise ValueError(f"Invalid --backend-map target '{entry}': expected 'host:port'")
        host, port_str = target.rsplit(":", 1)
        host = host.strip()
        try:
            port = int(port_str.strip())
        except ValueError as e:
            raise ValueError(f"Invalid port in --backend-map entry '{entry}'") from e
        if not host or port <= 0 or port > 65535:
            raise ValueError(f"Invalid host/port in --backend-map entry '{entry}'")
        mapping[model_key] = (host, port)
    return mapping


def is_moe_model(model_name: Optional[str]) -> bool:
    """Returns True when model targets LFM2.5-8B-A1B Sparse MoE."""
    m = (model_name or "").strip().lower()
    return m.startswith("lfm2") or "moe" in m


def resolve_backend(model_name: Optional[str]) -> Tuple[str, int]:
    """Resolves (host, port) for the requested model via backend_map or family default."""
    m = (model_name or config.default_model or "lfm3-1.7b").strip().lower()
    if m in config.backend_map:
        return config.backend_map[m]

    wildcards = [k for k in config.backend_map if k.endswith("*")]
    wildcards.sort(key=len, reverse=True)
    for pat in wildcards:
        if m.startswith(pat[:-1]):
            return config.backend_map[pat]

    if is_moe_model(m):
        return config.backend_host, config.moe_port
    return config.backend_host, config.backend_port


def get_unique_backends() -> Dict[str, Tuple[str, int]]:
    """Returns deduplicated 'host:port' -> (host, port) map across all configured routes."""
    backends: Dict[str, Tuple[str, int]] = {}
    default_dense = (config.backend_host, config.backend_port)
    default_moe = (config.backend_host, config.moe_port)
    backends[f"{default_dense[0]}:{default_dense[1]}"] = default_dense
    backends[f"{default_moe[0]}:{default_moe[1]}"] = default_moe
    for m in config.available_models:
        hp = resolve_backend(m)
        backends[f"{hp[0]}:{hp[1]}"] = hp
    for hp in config.backend_map.values():
        backends[f"{hp[0]}:{hp[1]}"] = hp
    return backends


# -----------------------------------------------------------------------------
# Request Schemas
# -----------------------------------------------------------------------------
class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    cache_control: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, List[ContentPart], List[Dict[str, Any]], None] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = "lfm3-1.7b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: Optional[bool] = False
    prompt_cache_key: Optional[str] = None
    cache: Optional[bool] = True


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = "lfm3-1.7b"
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.1
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    prompt_cache_key: Optional[str] = None
    cache: Optional[bool] = True
    pin_blocks_up_to: Optional[int] = 0


class CacheClearRequest(BaseModel):
    model: Optional[str] = None
    include_pinned: bool = False


class DecisionRequest(BaseModel):
    model: Optional[str] = "d1-2b"
    state: Optional[str] = ""
    question: Union[str, Dict[str, Any]]
    type: Optional[str] = "noul"
    criteria: Optional[Any] = None
    image_url: Optional[str] = None


def _resolve_max_tokens(request: ChatCompletionRequest) -> int:
    if request.max_completion_tokens is not None:
        return int(request.max_completion_tokens)
    if request.max_tokens is not None:
        return int(request.max_tokens)
    return 256


# -----------------------------------------------------------------------------
# Vision, Tokenizer Helper & Multimodal Processing
# -----------------------------------------------------------------------------
_vision_encoder = None
_optional_tokenizer = None
_optional_tokenizer_checked = False


def get_vision_encoder():
    """Lazily loads the hybrid 164M vision encoder."""
    global _vision_encoder
    if _vision_encoder is None:
        try:
            from vision_encoder import Lfm3VisionEncoder
            _vision_encoder = Lfm3VisionEncoder()
            logger.info("Lfm3VisionEncoder loaded successfully for multimodal requests.")
        except Exception as e:
            logger.warning(f"Could not load Lfm3VisionEncoder: {e}")
    return _vision_encoder


def get_optional_tokenizer():
    """Optionally loads LFM tokenizer for exact pin_blocks_up_to estimation if configured."""
    global _optional_tokenizer, _optional_tokenizer_checked
    if _optional_tokenizer_checked:
        return _optional_tokenizer
    _optional_tokenizer_checked = True
    tok_path = os.environ.get("LFM3_TOKENIZER_PATH", "").strip()
    if tok_path and Path(tok_path).is_file():
        try:
            from tokenizers import Tokenizer as HfTokenizer
            _optional_tokenizer = HfTokenizer.from_file(tok_path)
        except Exception as e:
            logger.warning(f"Failed to load LFM3_TOKENIZER_PATH '{tok_path}': {e}")
    return _optional_tokenizer


def load_image_from_source(source: str):
    """Loads a PIL image from base64 data URI, HTTP/HTTPS URL, or local filepath."""
    from PIL import Image

    if source.startswith("data:image/"):
        _, encoded = source.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(data)).convert("RGB")
    elif source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "Lfm3OpenAIServer/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        return Image.open(p).convert("RGB")


def _encode_image_part(url: str, image_embeds: List[np.ndarray]) -> str:
    if not url:
        return ""
    venc = get_vision_encoder()
    if not venc:
        return ""
    try:
        img = load_image_from_source(url)
        feats = venc.encode_image(img)
        image_embeds.append(feats)
        num_toks = len(feats)
        return "<|vision_start|>" + ("<|image|>" * num_toks) + "<|vision_end|>\n"
    except Exception as e:
        logger.error(f"Failed to encode image from {url[:40]}...: {e}")
        return ""


# -----------------------------------------------------------------------------
# Deterministic Prefix-Stable Prompt Formatting
# -----------------------------------------------------------------------------
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_historic_think_blocks(text: str) -> str:
    """Strips historic <think>...</think> blocks from prior assistant messages."""
    if "<think>" not in text:
        return text
    return _THINK_BLOCK_RE.sub("", text)


def _is_cache_pinned(cache_control: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(cache_control, dict):
        return False
    ctype = str(cache_control.get("type", "")).strip().lower()
    return ctype in ("ephemeral", "pinned")


def _estimate_pin_blocks(prefix_text: str) -> int:
    if not prefix_text:
        return 0
    tok = get_optional_tokenizer()
    if tok is not None:
        try:
            est_tokens = len(tok.encode(prefix_text).ids)
        except Exception:
            est_tokens = len(prefix_text.encode("utf-8")) // 4
    else:
        est_tokens = len(prefix_text.encode("utf-8")) // 4
    return max(1, est_tokens // 64)


def format_messages_to_lfm3_prompt(
    messages: List[ChatMessage],
    enable_thinking: bool = False,
    model_name: str = "lfm3-1.7b",
) -> Tuple[str, Optional[bytes], int]:
    """Formats messages into deterministic append-only ChatML matching LFM3 / LFM2.5 templates."""
    has_system = any((m.role or "").strip().lower() == "system" for m in messages)
    moe = is_moe_model(model_name)

    prompt_parts: List[str] = ["<|startoftext|>"]
    cum_chars = len(prompt_parts[0])
    if not has_system and not moe:
        sys_block = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        prompt_parts.append(sys_block)
        cum_chars += len(sys_block)

    image_embeds: List[np.ndarray] = []
    pin_char_end = 0

    for msg in messages:
        role = (msg.role or "user").strip().lower()
        msg_pinned = _is_cache_pinned(msg.cache_control)
        content_pieces: List[str] = []

        if isinstance(msg.content, str):
            content_pieces.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict):
                    ptype = part.get("type", "text")
                    if _is_cache_pinned(part.get("cache_control")):
                        msg_pinned = True
                    if ptype == "text":
                        content_pieces.append(part.get("text") or "")
                    elif ptype == "image_url":
                        img_obj = part.get("image_url") or {}
                        content_pieces.append(_encode_image_part(img_obj.get("url", ""), image_embeds))
                elif hasattr(part, "type"):
                    if _is_cache_pinned(getattr(part, "cache_control", None)):
                        msg_pinned = True
                    if part.type == "text" and part.text:
                        content_pieces.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        content_pieces.append(_encode_image_part(part.image_url.get("url", ""), image_embeds))

        content_text = "".join(content_pieces)
        if role == "assistant":
            content_text = _strip_historic_think_blocks(content_text)

        segment = f"<|im_start|>{role}\n{content_text}<|im_end|>\n"
        prompt_parts.append(segment)
        cum_chars += len(segment)
        if msg_pinned:
            pin_char_end = cum_chars

    prompt_parts.append("<|im_start|>assistant\n")
    if moe:
        prompt_parts.append("<think>\n" if enable_thinking else "<think>\n</think>\n\n")
    else:
        prompt_parts.append("<think>" if enable_thinking else "<think></think>")

    full_prompt = "".join(prompt_parts)
    pin_blocks_up_to = _estimate_pin_blocks(full_prompt[:pin_char_end]) if pin_char_end > 0 else 0

    image_bytes = None
    if image_embeds:
        all_feats = np.concatenate(image_embeds, axis=0).astype(np.float32)
        image_bytes = all_feats.tobytes()

    return full_prompt, image_bytes, pin_blocks_up_to


# -----------------------------------------------------------------------------
# OpenAI Telemetry & HTTP Cache Headers
# -----------------------------------------------------------------------------
def compute_cache_status_header(event: Dict[str, Any]) -> str:
    raw_status = str(event.get("cache_status", "MISS")).upper()
    if raw_status == "BYPASS":
        return "BYPASS"
    prompt_tokens = int(event.get("prompt_tokens", 0))
    cached_tokens = int(event.get("cached_tokens", 0))
    if cached_tokens <= 0:
        return "MISS"
    if prompt_tokens > 0 and cached_tokens >= prompt_tokens:
        return "HIT"
    return "PARTIAL"


def build_cache_headers(event: Dict[str, Any]) -> Dict[str, str]:
    cached_tokens = int(event.get("cached_tokens", 0))
    cache_tier = str(event.get("cache_tier", "none"))
    restore_ms = float(event.get("cache_restore_ms", 0.0))
    return {
        "X-Cache-Status": compute_cache_status_header(event),
        "X-Cached-Tokens": str(cached_tokens),
        "X-Cache-Tier": cache_tier,
        "X-Cache-Restore-Ms": f"{restore_ms:.2f}",
    }


def build_usage_and_timings(event: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt_tokens = int(event.get("prompt_tokens", 0))
    completion_tokens = int(event.get("completion_tokens", 0))
    cached_tokens = int(event.get("cached_tokens", 0))
    computed_prompt = int(
        event.get("computed_prompt_tokens", max(0, prompt_tokens - cached_tokens))
    )
    reasoning_tokens = int(event.get("reasoning_tokens", 0))
    prefill_ms = float(event.get("prefill_ms", 0.0))
    decode_ms = float(event.get("decode_ms", 0.0))
    restore_ms = float(event.get("cache_restore_ms", 0.0))

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": cached_tokens,
            "audio_tokens": 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
        },
    }
    timings = {
        "cache_n": cached_tokens,
        "prompt_n": computed_prompt,
        "prompt_ms": round(prefill_ms, 3),
        "restore_ms": round(restore_ms, 3),
        "prompt_per_second": round(1000.0 * computed_prompt / prefill_ms, 2) if prefill_ms > 0.0 else 0.0,
        "predicted_n": completion_tokens,
        "predicted_ms": round(decode_ms, 3),
        "predicted_per_second": round(1000.0 * completion_tokens / decode_ms, 2) if decode_ms > 0.0 else 0.0,
    }
    return usage, timings


# -----------------------------------------------------------------------------
# Resident Backend Communication (TCP Socket)
# -----------------------------------------------------------------------------
async def _open_backend_connection(
    payload: Dict[str, Any],
    host: str,
    port: int,
    image_bytes: Optional[bytes] = None,
    timeout: Optional[float] = None,
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    try:
        conn = asyncio.open_connection(host, port)
        reader, writer = await (asyncio.wait_for(conn, timeout=timeout) if timeout else conn)
    except Exception as e:
        logger.error(f"Failed to connect to backend at {host}:{port}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Backend engine not reachable at {host}:{port}. Is the daemon running?",
        )

    try:
        send_payload = dict(payload)
        if image_bytes:
            send_payload["image_bytes"] = len(image_bytes)
        req_line = json.dumps(send_payload) + "\n"
        writer.write(req_line.encode("utf-8"))
        if image_bytes:
            writer.write(image_bytes)
        await writer.drain()
        return reader, writer
    except Exception as e:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"Failed writing to {host}:{port}: {e}") from e


async def _read_next_event(reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            return None
        line = line_bytes.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


async def query_backend_command(
    host: str, port: int, cmd_payload: Dict[str, Any], timeout: float = 5.0
) -> Dict[str, Any]:
    reader, writer = await _open_backend_connection(cmd_payload, host, port, timeout=timeout)
    try:
        event = await asyncio.wait_for(_read_next_event(reader), timeout=timeout)
        if event is None:
            raise HTTPException(status_code=502, detail=f"Empty response from {host}:{port}")
        if event.get("type") == "error":
            raise HTTPException(status_code=500, detail=event.get("message", "Engine error"))
        return event
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def query_backend_stream(
    payload: Dict[str, Any],
    image_bytes: Optional[bytes] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Connects to the resolved resident C++ daemon and yields JSON events."""
    h = host or config.backend_host
    p = port or config.backend_port
    reader, writer = await _open_backend_connection(payload, h, p, image_bytes=image_bytes)
    try:
        while True:
            event = await _read_next_event(reader)
            if event is None:
                break
            yield event
            if event.get("type") in ("done", "error"):
                break
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def _start_sse_stream(
    payload: Dict[str, Any],
    host: str,
    port: int,
    image_bytes: Optional[bytes] = None,
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter, Dict[str, str], List[Dict[str, Any]]]:
    """Opens stream and reads the prefill_done header event before returning SSE headers."""
    reader, writer = await _open_backend_connection(payload, host, port, image_bytes=image_bytes)
    try:
        first_event = await _read_next_event(reader)
        if first_event is None:
            raise HTTPException(status_code=502, detail=f"Backend {host}:{port} closed connection early")
        etype = first_event.get("type")
        if etype == "error":
            raise HTTPException(status_code=500, detail=first_event.get("message", "Engine error"))
        if etype == "prefill_done":
            return reader, writer, build_cache_headers(first_event), []
        if etype == "done":
            return reader, writer, build_cache_headers(first_event), [first_event]
        fallback_headers = {
            "X-Cache-Status": "STREAMING",
            "X-Cached-Tokens": "0",
            "X-Cache-Tier": "unknown",
            "X-Cache-Restore-Ms": "0.00",
        }
        return reader, writer, fallback_headers, [first_event]
    except Exception:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        raise


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz")
async def health_check():
    return {
        "status": "ok",
        "backend": f"{config.backend_host}:{config.backend_port}",
        "backends": {k: f"{v[0]}:{v[1]}" for k, v in get_unique_backends().items()},
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    model_entries = [
        {
            "id": m,
            "object": "model",
            "created": now,
            "owned_by": "liquidai" if "heretic" not in m else "mihaipopa",
            "permission": [],
            "root": m,
            "parent": None,
        }
        for m in config.available_models
    ]
    return {"object": "list", "data": model_entries}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = request.model or config.default_model
    host, port = resolve_backend(model_name)

    formatted_prompt, image_bytes, pin_blocks_up_to = format_messages_to_lfm3_prompt(
        request.messages,
        enable_thinking=bool(request.enable_thinking),
        model_name=model_name,
    )

    payload: Dict[str, Any] = {
        "id": req_id,
        "prompt": formatted_prompt,
        "raw": True,
        "max_tokens": _resolve_max_tokens(request),
        "temperature": request.temperature if request.temperature is not None else 0.1,
        "top_k": request.top_k if request.top_k is not None else 50,
        "no_think": not bool(request.enable_thinking),
        "cache": True if request.cache is None else bool(request.cache),
        "pin_blocks_up_to": pin_blocks_up_to,
    }
    if request.prompt_cache_key:
        payload["prompt_cache_key"] = request.prompt_cache_key

    if request.stream:
        include_usage = bool((request.stream_options or {}).get("include_usage", False))
        reader, writer, headers, pending = await _start_sse_stream(
            payload, host, port, image_bytes=image_bytes
        )

        async def event_generator():
            try:
                idx = 0
                while True:
                    if idx < len(pending):
                        event = pending[idx]
                        idx += 1
                    else:
                        event = await _read_next_event(reader)
                        if event is None:
                            break
                    etype = event.get("type")
                    if etype == "prefill_done":
                        continue
                    elif etype == "token":
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": event.get("text", "")},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif etype == "done":
                        usage, timings = build_usage_and_timings(event)
                        stop_chunk: Dict[str, Any] = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        if include_usage:
                            stop_chunk["usage"] = None
                        yield f"data: {json.dumps(stop_chunk)}\n\n"
                        if include_usage:
                            usage_chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": model_name,
                                "choices": [],
                                "usage": usage,
                                "timings": timings,
                            }
                            yield f"data: {json.dumps(usage_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif etype == "error":
                        yield f"data: {json.dumps({'error': event.get('message', 'Internal engine error')})}\n\n"
                        yield "data: [DONE]\n\n"
                        break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

    completion_text = ""
    done_event: Dict[str, Any] = {}
    async for event in query_backend_stream(payload, image_bytes=image_bytes, host=host, port=port):
        etype = event.get("type")
        if etype == "token":
            completion_text += event.get("text", "")
        elif etype == "done":
            done_event = event
        elif etype == "error":
            raise HTTPException(status_code=500, detail=event.get("message", "Engine error"))

    usage, timings = build_usage_and_timings(done_event)
    headers = build_cache_headers(done_event)
    return JSONResponse(
        content={
            "id": req_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "timings": timings,
        },
        headers=headers,
    )


@app.post("/v1/completions")
async def raw_completions(request: CompletionRequest):
    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = request.model or config.default_model
    host, port = resolve_backend(model_name)

    raw_prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
    payload: Dict[str, Any] = {
        "id": req_id,
        "prompt": raw_prompt,
        "max_tokens": request.max_tokens if request.max_tokens is not None else 256,
        "temperature": request.temperature if request.temperature is not None else 0.1,
        "top_k": request.top_k if request.top_k is not None else 50,
        "no_think": True,
        "raw": True,
        "cache": True if request.cache is None else bool(request.cache),
        "pin_blocks_up_to": int(request.pin_blocks_up_to or 0),
    }
    if request.prompt_cache_key:
        payload["prompt_cache_key"] = request.prompt_cache_key

    if request.stream:
        include_usage = bool((request.stream_options or {}).get("include_usage", False))
        reader, writer, headers, pending = await _start_sse_stream(payload, host, port)

        async def event_generator():
            try:
                idx = 0
                while True:
                    if idx < len(pending):
                        event = pending[idx]
                        idx += 1
                    else:
                        event = await _read_next_event(reader)
                        if event is None:
                            break
                    etype = event.get("type")
                    if etype == "prefill_done":
                        continue
                    elif etype == "token":
                        chunk = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "text": event.get("text", ""), "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif etype == "done":
                        usage, timings = build_usage_and_timings(event)
                        stop_chunk: Dict[str, Any] = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model_name,
                            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                        }
                        if include_usage:
                            stop_chunk["usage"] = None
                        yield f"data: {json.dumps(stop_chunk)}\n\n"
                        if include_usage:
                            usage_chunk = {
                                "id": req_id,
                                "object": "text_completion",
                                "created": created_time,
                                "model": model_name,
                                "choices": [],
                                "usage": usage,
                                "timings": timings,
                            }
                            yield f"data: {json.dumps(usage_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif etype == "error":
                        yield f"data: {json.dumps({'error': event.get('message', 'Internal engine error')})}\n\n"
                        yield "data: [DONE]\n\n"
                        break
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

    completion_text = ""
    done_event: Dict[str, Any] = {}
    async for event in query_backend_stream(payload, host=host, port=port):
        etype = event.get("type")
        if etype == "token":
            completion_text += event.get("text", "")
        elif etype == "done":
            done_event = event
        elif etype == "error":
            raise HTTPException(status_code=500, detail=event.get("message", "Engine error"))

    usage, timings = build_usage_and_timings(done_event)
    headers = build_cache_headers(done_event)
    return JSONResponse(
        content={
            "id": req_id,
            "object": "text_completion",
            "created": created_time,
            "model": model_name,
            "choices": [{"index": 0, "text": completion_text, "finish_reason": "stop"}],
            "usage": usage,
            "timings": timings,
        },
        headers=headers,
    )


# -----------------------------------------------------------------------------
# Prefix Cache Management Endpoints (Milestone 17C)
# -----------------------------------------------------------------------------
def _enrich_cache_stats(raw: Dict[str, Any], backend_label: str) -> Dict[str, Any]:
    lookups = int(raw.get("total_lookups", 0))
    hits = int(raw.get("total_block_hits", 0))
    disk_saved = int(raw.get("total_tokens_saved", 0))
    tier0_saved = int(raw.get("tier0_tokens_saved", 0))
    out = dict(raw)
    out["backend"] = backend_label
    out["block_hit_rate"] = round(hits / lookups, 4) if lookups > 0 else 0.0
    out["total_saved_tokens"] = disk_saved + tier0_saved
    return out


@app.get("/v1/cache/stats")
async def get_cache_stats(model: Optional[str] = Query(default=None)):
    if model:
        host, port = resolve_backend(model)
        st = await query_backend_command(host, port, {"cmd": "cache_stats"})
        return _enrich_cache_stats(st, f"{host}:{port}")

    targets = get_unique_backends()
    coros = [
        query_backend_command(host, port, {"cmd": "cache_stats"}, timeout=2.0)
        for host, port in targets.values()
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    backends_out: Dict[str, Any] = {}
    sum_keys = (
        "resident_blocks",
        "pinned_blocks",
        "disk_bytes_used",
        "total_lookups",
        "total_block_hits",
        "total_tokens_saved",
        "total_evictions",
        "pending_writes",
        "tier0_hits",
        "tier0_tokens_saved",
        "tier0_live_tokens",
    )
    agg: Dict[str, Any] = {k: 0 for k in sum_keys}
    agg["disk_bytes_quota"] = 0

    for label, res in zip(targets.keys(), results):
        if isinstance(res, Exception):
            continue
        enriched = _enrich_cache_stats(res, label)
        backends_out[label] = enriched
        for k in sum_keys:
            agg[k] += int(res.get(k, 0))
        agg["disk_bytes_quota"] = max(agg["disk_bytes_quota"], int(res.get("disk_bytes_quota", 0)))

    if not backends_out:
        raise HTTPException(status_code=503, detail="No cache-capable backend daemons reachable")

    lookups = agg["total_lookups"]
    agg["type"] = "cache_stats"
    agg["block_hit_rate"] = round(agg["total_block_hits"] / lookups, 4) if lookups > 0 else 0.0
    agg["total_saved_tokens"] = agg["total_tokens_saved"] + agg["tier0_tokens_saved"]
    agg["backends"] = backends_out
    return agg


@app.post("/v1/cache/clear")
async def post_cache_clear(body: Optional[CacheClearRequest] = Body(default=None)):
    req = body or CacheClearRequest()
    include_pinned = bool(req.include_pinned)
    cmd = {"cmd": "cache_clear", "include_pinned": include_pinned}

    if req.model:
        host, port = resolve_backend(req.model)
        res = await query_backend_command(host, port, cmd)
        removed = int(res.get("removed_blocks", 0))
        return {
            "status": "ok",
            "removed_blocks": removed,
            "include_pinned": include_pinned,
            "backends": {f"{host}:{port}": res},
        }

    targets = get_unique_backends()
    coros = [
        query_backend_command(host, port, cmd, timeout=5.0)
        for host, port in targets.values()
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    per_backend: Dict[str, Any] = {}
    total_removed = 0
    for label, res in zip(targets.keys(), results):
        if isinstance(res, Exception):
            continue
        per_backend[label] = res
        total_removed += int(res.get("removed_blocks", 0))

    if not per_backend:
        raise HTTPException(status_code=503, detail="No cache-capable backend daemons reachable")

    return {
        "status": "ok",
        "removed_blocks": total_removed,
        "include_pinned": include_pinned,
        "backends": per_backend,
    }


# -----------------------------------------------------------------------------
# Decision API Endpoint (Milestone 12)
# -----------------------------------------------------------------------------
@app.post("/v1/decisions")
async def make_decision(request: DecisionRequest):
    """Executes a typed single-pass calibrated decision (noul, choice, score, estimate) with zero decoding."""
    try:
        from d1_runner import D1Runner
        host, port = resolve_backend(request.model or "d1-2b")
        runner = D1Runner(backend_host=host, backend_port=port)
        image_bytes = None
        if request.image_url:
            venc = get_vision_encoder()
            if venc:
                img = load_image_from_source(request.image_url)
                feats = venc.encode_image(img)
                image_bytes = feats.tobytes()

        res = runner.decide(
            state=request.state,
            question=request.question,
            question_type=request.type or "noul",
            criteria=request.criteria,
            image_bytes=image_bytes,
        )
        return res
    except Exception as e:
        logger.error(f"Decision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Main CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified LFM3 / LFM2.5-MoE OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP server bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server bind port (default: 8000)")
    parser.add_argument("--backend-host", type=str, default="roma", help="Default C++ daemon host (default: roma)")
    parser.add_argument("--backend-port", type=int, default=8088, help="LFM3 daemon port (default: 8088)")
    parser.add_argument("--moe-port", type=int, default=8089, help="LFM2.5-8B-A1B MoE daemon port (default: 8089)")
    parser.add_argument(
        "--backend-map",
        type=str,
        default="",
        help="Optional comma-separated routing map: 'lfm3-1.7b=roma:8088,lfm2.5-8b-a1b=roma:8089'",
    )
    args = parser.parse_args()

    config.backend_host = args.backend_host
    config.backend_port = args.backend_port
    config.moe_port = args.moe_port
    config.backend_map = parse_backend_map(args.backend_map)
    for key in config.backend_map:
        if not key.endswith("*") and key not in config.available_models:
            config.available_models.append(key)

    import uvicorn
    logger.info(f"Starting Unified OpenAI API Server on http://{args.host}:{args.port}")
    logger.info(f"LFM3 backend: {config.backend_host}:{config.backend_port} | LFM2.5-MoE backend: {config.backend_host}:{config.moe_port}")
    if config.backend_map:
        logger.info(f"Custom backend overrides: {config.backend_map}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
