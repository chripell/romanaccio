"""Host x86_64 FastAPI / ONNX Runtime vision encoder sidecar for Romanaccio Harness (Milestone 19E)."""

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

from vision_encoder import Lfm3VisionEncoder  # type: ignore  # noqa: E402
from .vision import MAX_IMAGE_BYTES, VISION_EMBED_DIM, inspect_image_bytes  # noqa: E402

DEFAULT_Q8_ONNX = Path("/home/chri/aixigua/deep/models/LFM3-VL-2B-ONNX/onnx/vision_encoder_q8.onnx")


def resolve_default_vision_onnx(explicit_path: str = "") -> Optional[Path]:
    """Resolve explicit ONNX path or prefer LFM3-VL-2B vision_encoder_q8.onnx when present."""
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Vision ONNX file not found: {p}")
        return p
    if DEFAULT_Q8_ONNX.is_file():
        return DEFAULT_Q8_ONNX
    return None


def create_vision_sidecar_app(
    *,
    encoder: Optional[Any] = None,
    onnx_path: Optional[Path] = None,
    default_max_dim: int = 512,
) -> FastAPI:
    """Create the FastAPI vision encoding sidecar exposing /healthz and /encode."""
    if encoder is None:
        resolved = onnx_path or resolve_default_vision_onnx()
        enc = Lfm3VisionEncoder(str(resolved) if resolved else None)
    else:
        enc = encoder

    model_name = Path(getattr(enc, "onnx_path", onnx_path or "vision_encoder.onnx")).name
    encode_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield

    app = FastAPI(title="Romanaccio Vision Sidecar", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": model_name,
            "embed_dim": VISION_EMBED_DIM,
            "default_max_dim": int(default_max_dim),
        }

    @app.post("/encode")
    async def encode(
        request: Request,
        format: str = "json",
        max_dim: int = default_max_dim,
        filename: str = "image.png",
    ) -> Response:
        ctype = (request.headers.get("content-type") or "").lower()
        chosen_name = filename
        raw_bytes = b""

        if "multipart/form-data" in ctype:
            form = await request.form()
            up = form.get("file") or form.get("image")
            if up is None:
                raise HTTPException(status_code=400, detail="Missing 'file' form field")
            chosen_name = getattr(up, "filename", None) or filename
            raw_bytes = await up.read()
        elif "application/json" in ctype:
            body_raw = await request.body()
            try:
                obj = json.loads(body_raw.decode("utf-8"))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
            b64_val = str(obj.get("image_b64") or "")
            if "," in b64_val and b64_val.startswith("data:"):
                b64_val = b64_val.split(",", 1)[1]
            if not b64_val:
                raise HTTPException(status_code=400, detail="Missing 'image_b64' field")
            raw_bytes = base64.b64decode(b64_val)
            chosen_name = str(obj.get("filename") or filename)
            if "max_dim" in obj:
                max_dim = int(obj["max_dim"])
        else:
            raw_bytes = await request.body()

        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"Image exceeds {MAX_IMAGE_BYTES} byte limit")

        try:
            fmt, width, height = inspect_image_bytes(raw_bytes, chosen_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        bounded_dim = max(128, min(1024, int(max_dim)))
        t0 = time.monotonic()
        async with encode_lock:
            feats = await asyncio.to_thread(enc.encode_image, raw_bytes, max_dim=bounded_dim)
        feats_f32 = np.ascontiguousarray(feats, dtype=np.float32)
        if feats_f32.ndim != 2 or feats_f32.shape[1] != VISION_EMBED_DIM:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected embedding shape {tuple(feats_f32.shape)} (expected [N, {VISION_EMBED_DIM}])",
            )
        num_tokens = int(feats_f32.shape[0])
        encode_ms = round((time.monotonic() - t0) * 1000.0, 2)
        embed_bytes = feats_f32.tobytes()

        if format.lower() == "raw":
            return Response(
                content=embed_bytes,
                media_type="application/octet-stream",
                headers={
                    "X-Image-Tokens": str(num_tokens),
                    "X-Image-Width": str(width),
                    "X-Image-Height": str(height),
                    "X-Image-Format": fmt,
                    "X-Encode-Ms": f"{encode_ms:.2f}",
                    "X-Vision-Model": model_name,
                },
            )

        return JSONResponse(
            {
                "ok": True,
                "num_tokens": num_tokens,
                "embed_dim": VISION_EMBED_DIM,
                "width": width,
                "height": height,
                "format": fmt,
                "encode_ms": encode_ms,
                "model": model_name,
                "embeddings_b64": base64.b64encode(embed_bytes).decode("ascii"),
            }
        )

    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Romanaccio Host ONNX Vision Encoder Sidecar (Milestone 19E)")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8095, help="HTTP listen port (default 8095)")
    parser.add_argument("--onnx-path", default="", help="Path to vision_encoder*.onnx (defaults to LFM3-VL-2B vision_encoder_q8.onnx)")
    parser.add_argument("--max-dim", type=int, default=512, help="Default max image edge pixels (default 512)")
    ns = parser.parse_args(argv)

    onnx_p = resolve_default_vision_onnx(ns.onnx_path)
    app = create_vision_sidecar_app(onnx_path=onnx_p, default_max_dim=ns.max_dim)
    print(f"[romanaccio-vision-sidecar] http://{ns.bind}:{ns.port} | onnx={onnx_p or 'auto'}")

    import uvicorn

    uvicorn.run(app, host=ns.bind, port=ns.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
