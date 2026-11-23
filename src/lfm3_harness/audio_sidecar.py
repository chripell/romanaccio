"""Host x86_64 FastAPI / ONNX Runtime audio sidecar for Romanaccio Harness (Milestone 19F)."""

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LFM3_DIR = _REPO_ROOT / "src" / "LFM3"
if str(_LFM3_DIR) not in sys.path:
    sys.path.insert(0, str(_LFM3_DIR))

from audio_encoder import (  # type: ignore  # noqa: E402
    AUDIO_EMBED_DIM,
    MAX_AUDIO_SECONDS,
    Lfm3AudioEncoder,
    Lfm3AudioSynthesizer,
)

MAX_AUDIO_UPLOAD_BYTES = 16 * 1024 * 1024


def create_audio_sidecar_app(
    *,
    encoder: Optional[Any] = None,
    synthesizer: Optional[Any] = None,
    onnx_dir: Optional[str] = None,
    codebook_bin_path: Optional[str] = None,
    default_daemon_host: str = "127.0.0.1",
    default_daemon_port: int = 8091,
    enable_tts: bool = True,
) -> FastAPI:
    """Create host ONNX audio encoder + MaskGIT TTS synthesizer sidecar app."""
    enc = encoder if encoder is not None else Lfm3AudioEncoder(onnx_dir=onnx_dir)
    synth = synthesizer
    if synth is None and enable_tts:
        try:
            synth = Lfm3AudioSynthesizer(onnx_dir=onnx_dir, codebook_bin_path=codebook_bin_path)
        except FileNotFoundError:
            synth = None

    encode_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    model_name = Path(getattr(enc, "onnx_path", "audio_encoder.onnx")).name

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield

    app = FastAPI(title="Romanaccio Audio Sidecar", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": model_name,
            "embed_dim": AUDIO_EMBED_DIM,
            "asr_encoder": True,
            "tts_synthesizer": synth is not None,
            "daemon_host": default_daemon_host,
            "daemon_port": int(default_daemon_port),
        }

    @app.post("/encode")
    async def encode(
        request: Request,
        format: str = "raw",
        filename: str = "clip.wav",
    ) -> Response:
        ctype = (request.headers.get("content-type") or "").lower()
        chosen_name = (request.headers.get("x-filename") or filename or "clip.wav").strip()
        raw_bytes = b""

        if "multipart/form-data" in ctype:
            form = await request.form()
            up = form.get("file") or form.get("audio")
            if up is None:
                raise HTTPException(status_code=400, detail="Missing 'file' form field")
            chosen_name = getattr(up, "filename", None) or chosen_name
            raw_bytes = await up.read()
        elif "application/json" in ctype:
            body_raw = await request.body()
            try:
                obj = json.loads(body_raw.decode("utf-8"))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
            b64_val = str(obj.get("audio_b64") or "")
            if not b64_val:
                raise HTTPException(status_code=400, detail="Missing 'audio_b64' field")
            raw_bytes = base64.b64decode(b64_val)
            chosen_name = str(obj.get("filename") or chosen_name)
        else:
            raw_bytes = await request.body()

        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Audio payload must not be empty")
        if len(raw_bytes) > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"Audio exceeds {MAX_AUDIO_UPLOAD_BYTES} B limit")

        try:
            async with encode_lock:
                res = await asyncio.to_thread(
                    enc.encode_wav_bytes,
                    raw_bytes,
                    chosen_name,
                    max_duration_s=MAX_AUDIO_SECONDS,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audio encode failed: {exc}") from exc

        feats_f32 = np.ascontiguousarray(res["embeddings"], dtype=np.float32)
        if feats_f32.ndim != 2 or feats_f32.shape[1] != AUDIO_EMBED_DIM:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected audio embedding shape {tuple(feats_f32.shape)}",
            )
        num_tokens = int(feats_f32.shape[0])
        num_frames = int(res.get("num_frames") or 0)
        duration_s = float(res.get("duration_s") or 0.0)
        encode_ms = float(res.get("encode_ms") or 0.0)
        embed_bytes = feats_f32.tobytes()

        if format.lower() == "json":
            return JSONResponse(
                {
                    "ok": True,
                    "num_tokens": num_tokens,
                    "embed_dim": AUDIO_EMBED_DIM,
                    "num_frames": num_frames,
                    "duration_s": duration_s,
                    "encode_ms": encode_ms,
                    "model": model_name,
                    "embeddings_b64": base64.b64encode(embed_bytes).decode("ascii"),
                }
            )

        return Response(
            content=embed_bytes,
            media_type="application/octet-stream",
            headers={
                "X-Audio-Tokens": str(num_tokens),
                "X-Audio-Duration-S": f"{duration_s:.3f}",
                "X-Audio-Frames": str(num_frames),
                "X-Audio-Encode-Ms": f"{encode_ms:.2f}",
                "X-Audio-Model": model_name,
            },
        )

    @app.post("/tts/synthesize")
    async def tts_synthesize(request: Request) -> Response:
        if synth is None:
            raise HTTPException(status_code=503, detail="LFM3-Audio TTS synthesizer is not configured")
        body_raw = await request.body()
        try:
            obj = json.loads(body_raw.decode("utf-8")) if body_raw else {}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(obj, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")

        text = str(obj.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="TTS 'text' must not be empty")
        if len(text) > 4000:
            text = text[:4000]

        language = str(obj.get("language") or "American English").strip() or "American English"
        gender = str(obj.get("gender") or "female").strip() or "female"
        max_frames = max(1, min(600, int(obj.get("max_frames") or 250)))
        temperature = max(0.0, min(2.0, float(obj.get("temperature", 0.8))))
        top_k = max(0, min(2017, int(obj.get("top_k", 128))))
        rep_pen = max(0.5, min(2.5, float(obj.get("repetition_penalty", 1.1))))
        daemon_host = str(obj.get("daemon_host") or default_daemon_host).strip() or default_daemon_host
        daemon_port = int(obj.get("daemon_port") or default_daemon_port)
        seed = int(obj["seed"]) if "seed" in obj and obj["seed"] is not None else None

        t0 = time.monotonic()
        try:
            async with tts_lock:
                res = await asyncio.to_thread(
                    synth.synthesize_via_daemon,
                    text,
                    daemon_host,
                    daemon_port,
                    language=language,
                    gender=gender,
                    max_frames=max_frames,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=rep_pen,
                    seed=seed,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, EOFError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"TTS daemon synthesis failed: {exc}") from exc

        wav_bytes = bytes(res["wav_bytes"])
        total_ms = float(res.get("total_ms") or round((time.monotonic() - t0) * 1000.0, 2))
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-TTS-Engine": "lfm3-audio",
                "X-TTS-Frames": str(res.get("num_frames", 0)),
                "X-TTS-Duration-S": f"{float(res.get('duration_s', 0.0)):.3f}",
                "X-TTS-Total-Ms": f"{total_ms:.2f}",
                "X-TTS-RTF": f"{float(res.get('rtf', 0.0)):.3f}",
                "X-TTS-Prefill-Ms": f"{float(res.get('prefill_ms', 0.0)):.2f}",
                "X-TTS-Depth-Ms": f"{float(res.get('depth_ms', 0.0)):.2f}",
                "X-TTS-Codec-Ms": f"{float(res.get('codec_ms', 0.0)):.2f}",
                "X-TTS-EOS": "true" if res.get("eos") else "false",
            },
        )

    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Romanaccio Host ONNX Audio Sidecar (Milestone 19F)")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8096, help="HTTP listen port (default 8096)")
    parser.add_argument("--onnx-dir", default="", help="Path to LFM3-Audio-2B-ONNX directory")
    parser.add_argument("--codebook-bin", default="", help="Path to audio_codebook_embed.bin")
    parser.add_argument("--audio-daemon-host", default="127.0.0.1", help="Target LFM3-Audio daemon host")
    parser.add_argument("--audio-daemon-port", type=int, default=8091, help="Target LFM3-Audio daemon port")
    parser.add_argument("--no-tts", action="store_true", help="Disable depth_decoder/codec_decoder TTS endpoint")
    ns = parser.parse_args(argv)

    app = create_audio_sidecar_app(
        onnx_dir=ns.onnx_dir or None,
        codebook_bin_path=ns.codebook_bin or None,
        default_daemon_host=ns.audio_daemon_host,
        default_daemon_port=ns.audio_daemon_port,
        enable_tts=not ns.no_tts,
    )
    print(
        f"[romanaccio-audio-sidecar] http://{ns.bind}:{ns.port} | "
        f"daemon={ns.audio_daemon_host}:{ns.audio_daemon_port}"
    )

    import uvicorn

    uvicorn.run(app, host=ns.bind, port=ns.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
