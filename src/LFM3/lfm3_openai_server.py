#!/usr/bin/env python3
"""
OpenAI-Compatible Standard HTTP API Endpoint for LFM3-1.7B on ESWIN EIC7700X.

Provides:
- GET  /v1/models
- POST /v1/chat/completions (streaming SSE & non-streaming, multi-turn, multimodal)
- POST /v1/completions (standard text completions)
- GET  /healthz

Bridges to the resident C++ engine (lfm3_v2) via TCP socket or persistent stdio IPC.
"""

import os
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure local LFM3 directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lfm3_openai_server")

app = FastAPI(
    title="LFM3 EIC7700X OpenAI API Server",
    description="OpenAI-compatible inference server for LFM3-1.7B running on dual-die ESWIN EIC7700X NPU",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Configuration & State
# -----------------------------------------------------------------------------
class ServerConfig:
    backend_host: str = "roma"
    backend_port: int = 8088
    backend_mode: str = "tcp"  # "tcp" or "stdio_ssh"
    default_model: str = "lfm3-1.7b"
    available_models: List[str] = [
        "lfm3-1.7b",
        "lfm3-1.7b-int8",
        "lfm3-1.7b-heretic",
        "lfm3-vl-2b",
        "d1-2b"
    ]

config = ServerConfig()

# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class ContentPart(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart], List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "lfm3-1.7b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = Field(default=256, alias="max_completion_tokens")
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: Optional[str] = "lfm3-1.7b"
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

# -----------------------------------------------------------------------------
# Vision & Multimodal Processing
# -----------------------------------------------------------------------------
_vision_encoder = None

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

def load_image_from_source(source: str):
    """Loads a PIL image from base64 data URI, HTTP/HTTPS URL, or local filepath."""
    from PIL import Image

    if source.startswith("data:image/"):
        header, encoded = source.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(data)).convert("RGB")
    elif source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "Lfm3OpenAIServer/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        return Image.open(p).convert("RGB")

# -----------------------------------------------------------------------------
# Prompt Formatting
# -----------------------------------------------------------------------------
def format_messages_to_lfm3_prompt(messages: List[ChatMessage], enable_thinking: bool = False) -> Tuple[str, Optional[bytes]]:
    """Formats chat messages into LFM3 chat template format, encoding any attached images."""
    prompt_parts = []
    has_system = False
    image_embeds = []

    for msg in messages:
        role = msg.role.strip().lower()
        if role == "system":
            has_system = True

        content_text = ""
        if isinstance(msg.content, str):
            content_text = msg.content
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict):
                    ptype = part.get("type", "text")
                    if ptype == "text":
                        content_text += part.get("text", "")
                    elif ptype == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            venc = get_vision_encoder()
                            if venc:
                                try:
                                    img = load_image_from_source(url)
                                    feats = venc.encode_image(img)
                                    image_embeds.append(feats)
                                    num_toks = len(feats)
                                    content_text += "<|vision_start|>" + ("<|image|>" * num_toks) + "<|vision_end|>\n"
                                except Exception as e:
                                    logger.error(f"Failed to encode image from {url[:40]}...: {e}")
                elif hasattr(part, "type"):
                    if part.type == "text" and part.text:
                        content_text += part.text
                    elif part.type == "image_url" and part.image_url:
                        url = part.image_url.get("url", "")
                        if url:
                            venc = get_vision_encoder()
                            if venc:
                                try:
                                    img = load_image_from_source(url)
                                    feats = venc.encode_image(img)
                                    image_embeds.append(feats)
                                    num_toks = len(feats)
                                    content_text += "<|vision_start|>" + ("<|image|>" * num_toks) + "<|vision_end|>\n"
                                except Exception as e:
                                    logger.error(f"Failed to encode image from {url[:40]}...: {e}")

        prompt_parts.append(f"<|im_start|>{role}\n{content_text}<|im_end|>\n")

    if not has_system:
        prompt_parts.insert(0, "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n")

    # Start assistant turn
    prompt_parts.append("<|im_start|>assistant\n")
    if not enable_thinking:
        prompt_parts.append("<think>\n</think>\n")

    image_bytes = None
    if image_embeds:
        all_feats = np.concatenate(image_embeds, axis=0).astype(np.float32)
        image_bytes = all_feats.tobytes()

    return "".join(prompt_parts), image_bytes

# -----------------------------------------------------------------------------
# Resident Backend Communication (TCP / IPC Socket)
# -----------------------------------------------------------------------------
async def query_backend_stream(payload: Dict[str, Any], image_bytes: Optional[bytes] = None) -> AsyncGenerator[Dict[str, Any], None]:
    """Connects to the resident lfm3_v2 engine and yields token events."""
    host = config.backend_host
    port = config.backend_port
    
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except Exception as e:
        logger.error(f"Failed to connect to lfm3_v2 backend at {host}:{port}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LFM3 backend engine not reachable at {host}:{port}. Is lfm3_v2 running in daemon mode?"
        )

    try:
        if image_bytes:
            payload["image_bytes"] = len(image_bytes)
        req_line = json.dumps(payload) + "\n"
        writer.write(req_line.encode("utf-8"))
        if image_bytes:
            writer.write(image_bytes)
        await writer.drain()

        while True:
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            yield event
            if event.get("type") in ("done", "error"):
                break
    finally:
        writer.close()
        await writer.wait_closed()

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz")
async def health_check():
    return {"status": "ok", "backend": f"{config.backend_host}:{config.backend_port}"}

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
            "parent": None
        }
        for m in config.available_models
    ]
    return {"object": "list", "data": model_entries}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = request.model or config.default_model

    formatted_prompt, image_bytes = format_messages_to_lfm3_prompt(
        request.messages,
        enable_thinking=bool(request.enable_thinking)
    )

    payload = {
        "id": req_id,
        "prompt": formatted_prompt,
        "max_tokens": request.max_tokens or 256,
        "temperature": request.temperature if request.temperature is not None else 0.1,
        "top_k": request.top_k or 50,
        "no_think": not bool(request.enable_thinking)
    }

    if request.stream:
        async def event_generator():
            try:
                async for event in query_backend_stream(payload, image_bytes):
                    etype = event.get("type")
                    if etype == "token":
                        text_piece = event.get("text", "")
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text_piece},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif etype == "done":
                        final_chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": event.get("prompt_tokens", 0),
                                "completion_tokens": event.get("completion_tokens", 0),
                                "total_tokens": event.get("prompt_tokens", 0) + event.get("completion_tokens", 0)
                            }
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif etype == "error":
                        err_chunk = {"error": event.get("message", "Internal engine error")}
                        yield f"data: {json.dumps(err_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                err_chunk = {"error": str(e)}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming response
    completion_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for event in query_backend_stream(payload, image_bytes):
        etype = event.get("type")
        if etype == "token":
            completion_text += event.get("text", "")
        elif etype == "done":
            prompt_tokens = event.get("prompt_tokens", 0)
            completion_tokens = event.get("completion_tokens", 0)
        elif etype == "error":
            raise HTTPException(status_code=500, detail=event.get("message", "Engine error"))

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

@app.post("/v1/completions")
async def raw_completions(request: CompletionRequest):
    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = request.model or config.default_model

    raw_prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
    payload = {
        "id": req_id,
        "prompt": raw_prompt,
        "max_tokens": request.max_tokens or 256,
        "temperature": request.temperature if request.temperature is not None else 0.1,
        "top_k": 50,
        "no_think": True,
        "raw": True
    }

    if request.stream:
        async def event_generator():
            try:
                async for event in query_backend_stream(payload):
                    etype = event.get("type")
                    if etype == "token":
                        text_piece = event.get("text", "")
                        chunk = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": text_piece,
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif etype == "done":
                        yield "data: [DONE]\n\n"
                        break
            except Exception as e:
                err_chunk = {"error": str(e)}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    completion_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for event in query_backend_stream(payload):
        etype = event.get("type")
        if etype == "token":
            completion_text += event.get("text", "")
        elif etype == "done":
            prompt_tokens = event.get("prompt_tokens", 0)
            completion_tokens = event.get("completion_tokens", 0)
        elif etype == "error":
            raise HTTPException(status_code=500, detail=event.get("message", "Engine error"))

    return {
        "id": req_id,
        "object": "text_completion",
        "created": created_time,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": completion_text,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

# -----------------------------------------------------------------------------
# Decision API Endpoint (Milestone 12)
# -----------------------------------------------------------------------------
class DecisionRequest(BaseModel):
    state: Optional[str] = ""
    question: Union[str, Dict[str, Any]]
    type: Optional[str] = "noul"
    criteria: Optional[Any] = None
    image_url: Optional[str] = None

@app.post("/v1/decisions")
async def make_decision(request: DecisionRequest):
    """Executes a typed single-pass calibrated decision (noul, choice, score, estimate) with zero decoding."""
    try:
        from d1_runner import D1Runner
        runner = D1Runner(backend_host=config.backend_host, backend_port=config.backend_port)
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
            image_bytes=image_bytes
        )
        return res
    except Exception as e:
        logger.error(f"Decision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Main CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LFM3-1.7B OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP server bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server bind port (default: 8000)")
    parser.add_argument("--backend-host", type=str, default="roma", help="lfm3_v2 backend host (default: roma)")
    parser.add_argument("--backend-port", type=int, default=8088, help="lfm3_v2 backend port (default: 8088)")
    args = parser.parse_args()

    config.backend_host = args.backend_host
    config.backend_port = args.backend_port

    import uvicorn
    logger.info(f"Starting OpenAI API Server on http://{args.host}:{args.port}")
    logger.info(f"Targeting backend engine at {config.backend_host}:{config.backend_port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
