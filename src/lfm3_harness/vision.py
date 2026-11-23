"""Pure-stdlib image container inspection, placeholder formatting, and host Vision Sidecar client."""

import asyncio
import hashlib
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

ALLOWED_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
VISION_EMBED_DIM = 2048
MAX_IMAGE_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class ImageMetadata:
    """Verified image header metadata and vision-token geometry."""

    filename: str
    bytes: int
    sha256: str
    format: str
    width: int
    height: int
    num_tokens: int = 0
    saved_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_vision_placeholder(num_tokens: int) -> str:
    """Render LFM3 vision placeholder tags matching exact encoded visual feature count."""
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    return "<|vision_start|>" + ("<|image|>" * num_tokens) + "<|vision_end|>"


def _parse_webp_dimensions(raw: bytes) -> Tuple[int, int]:
    if len(raw) < 30:
        raise ValueError("Truncated WebP header")
    chunk = raw[12:16]
    if chunk == b"VP8X":
        w = int.from_bytes(raw[24:27], "little") + 1
        h = int.from_bytes(raw[27:30], "little") + 1
        return w, h
    if chunk == b"VP8L" and raw[20] == 0x2F:
        b0 = struct.unpack("<I", raw[21:25])[0]
        w = (b0 & 0x3FFF) + 1
        h = ((b0 >> 14) & 0x3FFF) + 1
        return w, h
    if chunk == b"VP8 " and raw[23:26] == b"\x9d\x01\x2a":
        w, h = struct.unpack("<HH", raw[26:30])
        return w & 0x3FFF, h & 0x3FFF
    raise ValueError("Unsupported or corrupt WebP bitstream")


def _parse_jpeg_dimensions(raw: bytes) -> Tuple[int, int]:
    pos = 2
    n = len(raw)
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while pos < n:
        while pos < n and raw[pos] != 0xFF:
            pos += 1
        while pos < n and raw[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = raw[pos]
        pos += 1
        if marker == 0x00 or marker == 0x01 or (0xD0 <= marker <= 0xD9):
            if marker == 0xD9:
                break
            continue
        if marker == 0xDA:
            break
        if pos + 2 > n:
            break
        seg_len = struct.unpack(">H", raw[pos : pos + 2])[0]
        if seg_len < 2 or pos + seg_len > n:
            break
        if marker in sof_markers and seg_len >= 7:
            h, w = struct.unpack(">HH", raw[pos + 3 : pos + 7])
            return w, h
        pos += seg_len
    raise ValueError("Could not locate JPEG SOF frame dimensions")


def inspect_image_bytes(raw: bytes, filename: str = "") -> Tuple[str, int, int]:
    """Parse (format, width, height) strictly from binary headers using pure stdlib."""
    if not raw:
        raise ValueError("Empty image payload")

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension '{ext}'; allowed: {sorted(ALLOWED_IMAGE_EXTS)}")

    fmt = ""
    width = 0
    height = 0
    if len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
        fmt = "png"
        width, height = struct.unpack(">II", raw[16:24])
    elif len(raw) >= 10 and raw[:6] in (b"GIF87a", b"GIF89a"):
        fmt = "gif"
        width, height = struct.unpack("<HH", raw[6:10])
    elif len(raw) >= 26 and raw[:2] == b"BM":
        fmt = "bmp"
        dib_sz = struct.unpack("<I", raw[14:18])[0]
        if dib_sz == 12:
            width, height = struct.unpack("<HH", raw[18:22])
        else:
            width, raw_h = struct.unpack("<ii", raw[18:26])
            height = abs(raw_h)
    elif len(raw) >= 30 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        fmt = "webp"
        width, height = _parse_webp_dimensions(raw)
    elif len(raw) >= 4 and raw[:2] == b"\xff\xd8":
        fmt = "jpeg"
        width, height = _parse_jpeg_dimensions(raw)
    else:
        raise ValueError("Unrecognized or unsupported image header (expected PNG/JPEG/WebP/GIF/BMP)")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions {width}x{height}")
    return fmt, int(width), int(height)


def build_image_metadata(raw: bytes, filename: str, *, num_tokens: int = 0, saved_path: str = "") -> ImageMetadata:
    """Inspect image bytes and build verified ImageMetadata."""
    clean_name = Path((filename or "image.png").strip() or "image.png").name
    fmt, width, height = inspect_image_bytes(raw, clean_name)
    return ImageMetadata(
        filename=clean_name,
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        format=fmt,
        width=width,
        height=height,
        num_tokens=num_tokens,
        saved_path=saved_path,
    )


class VisionClient:
    """Async HTTP client talking to a host x86_64 or LAN vision_sidecar instance."""

    def __init__(self, base_url: str, *, timeout_s: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _health_sync(self) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=min(5.0, self.timeout_s)) as resp:
            import json
            return dict(json.loads(resp.read().decode("utf-8")))

    async def health(self) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(self._health_sync)
        except Exception as exc:
            raise RuntimeError(f"Vision sidecar health check failed ({self.base_url}): {exc}") from exc

    def _encode_sync(self, raw_bytes: bytes, filename: str, max_dim: int) -> Tuple[bytes, int, Dict[str, Any]]:
        qs = urllib.parse.urlencode({"format": "raw", "max_dim": int(max_dim), "filename": filename})
        url = f"{self.base_url}/encode?{qs}"
        req = urllib.request.Request(
            url,
            data=raw_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                embed_bytes = resp.read()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vision sidecar HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Vision sidecar encode failed ({self.base_url}): {exc}") from exc

        try:
            num_tokens = int(hdrs["x-image-tokens"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Vision sidecar response missing valid X-Image-Tokens header") from exc

        expected_bytes = num_tokens * VISION_EMBED_DIM * 4
        if num_tokens <= 0 or len(embed_bytes) != expected_bytes:
            raise RuntimeError(
                f"Vision embedding byte mismatch: got {len(embed_bytes)} B for {num_tokens} tokens "
                f"(expected {expected_bytes} B)"
            )

        info = {
            "num_tokens": num_tokens,
            "width": int(hdrs.get("x-image-width", "0") or 0),
            "height": int(hdrs.get("x-image-height", "0") or 0),
            "format": hdrs.get("x-image-format", ""),
            "encode_ms": float(hdrs.get("x-encode-ms", "0") or 0.0),
            "model": hdrs.get("x-vision-model", ""),
        }
        return embed_bytes, num_tokens, info

    async def encode(
        self,
        raw_bytes: bytes,
        *,
        filename: str = "image.png",
        max_dim: int = 512,
    ) -> Tuple[bytes, int, Dict[str, Any]]:
        """Encode an image into raw float32 [num_tokens, 2048] embeddings via the vision sidecar."""
        if not raw_bytes:
            raise ValueError("Image payload must not be empty")
        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image payload exceeds {MAX_IMAGE_BYTES} B limit ({len(raw_bytes)} B)")
        return await asyncio.to_thread(self._encode_sync, raw_bytes, filename, max_dim)
