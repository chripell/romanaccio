"""Tier-1 offline web voice and audio processing service (ffmpeg + espeak-ng / optional piper-tts)."""

import asyncio
import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

AUDIO_EMBED_DIM = 2048
MAX_AUDIO_UPLOAD_BYTES = 16 * 1024 * 1024
_ALLOWED_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"})
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LFM3_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_LFM2_CALL_RE = re.compile(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


@dataclass(frozen=True)
class AudioMetadata:
    """Verified container telemetry returned for uploaded or normalized audio clips."""

    filename: str
    bytes: int
    sha256: str
    duration_s: float
    sample_rate: int
    channels: int
    codec: str
    normalized_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_text_for_tts(text: str, *, max_chars: int = 4000) -> str:
    """Strip chain-of-thought, tool markup, code blocks, and Markdown punctuation for speech."""
    cleaned = text or ""
    cleaned = _THINK_RE.sub(" ", cleaned)
    cleaned = _LFM3_CALL_RE.sub(" ", cleaned)
    cleaned = _LFM2_CALL_RE.sub(" ", cleaned)
    cleaned = _CODE_FENCE_RE.sub(" code block omitted ", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"[`*_>#~|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "."
    return cleaned


def _parse_wav_bytes(data: bytes) -> Optional[Tuple[int, int, int, float]]:
    """Return (sample_rate, channels, sampwidth, duration_s) for a standard RIFF/WAVE PCM clip."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    import io

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            ch = int(wf.getnchannels())
            sr = int(wf.getframerate())
            sw = int(wf.getsampwidth())
            nf = int(wf.getnframes())
            dur = (nf / float(sr)) if sr > 0 else 0.0
            return sr, ch, sw, round(dur, 4)
    except Exception:
        return None


def _is_real_piper_binary(bin_path: Path) -> bool:
    """Verify binary is Rhasspy Piper TTS and not Ubuntu's GTK gaming-mouse 'piper' GUI."""
    if not bin_path.exists() or not bin_path.is_file():
        return False
    # Check ELF vs python script mentioning ratbagd/libratbag.
    try:
        head = bin_path.read_bytes()[:4096]
        if b"ratbag" in head.lower() or b"gi.repository" in head:
            return False
    except OSError:
        return False
    return True


def inspect_wav_bytes(data: bytes) -> Optional[Tuple[int, int, int, float]]:
    """Return (sample_rate, channels, sample_width_bytes, duration_s) for a PCM WAV buffer."""
    return _parse_wav_bytes(data)


def format_audio_placeholder(num_tokens: int) -> str:
    """Render exact LFM3-Audio <|audio|> placeholder tokens matching encoded audio features."""
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    return "<|audio|>" * int(num_tokens)


class AudioClient:
    """Pure-stdlib async HTTP client talking to host x86_64 audio_sidecar."""

    def __init__(self, base_url: str, *, timeout_s: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _health_sync(self) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=min(5.0, self.timeout_s)) as resp:
            return dict(json.loads(resp.read().decode("utf-8")))

    async def health(self) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(self._health_sync)
        except Exception as exc:
            raise RuntimeError(f"Audio sidecar health check failed ({self.base_url}): {exc}") from exc

    def _encode_sync(self, raw_bytes: bytes, filename: str) -> Tuple[bytes, int, Dict[str, Any]]:
        qs = urllib.parse.urlencode({"format": "raw", "filename": filename})
        url = f"{self.base_url}/encode?{qs}"
        req = urllib.request.Request(
            url,
            data=raw_bytes,
            headers={"Content-Type": "application/octet-stream", "X-Filename": filename},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                embed_bytes = resp.read()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Audio sidecar HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Audio sidecar encode failed ({self.base_url}): {exc}") from exc

        try:
            num_tokens = int(hdrs["x-audio-tokens"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Audio sidecar response missing valid X-Audio-Tokens header") from exc

        expected_bytes = num_tokens * AUDIO_EMBED_DIM * 4
        if num_tokens <= 0 or len(embed_bytes) != expected_bytes:
            raise RuntimeError(
                f"Audio embedding byte mismatch: got {len(embed_bytes)} B for {num_tokens} tokens "
                f"(expected {expected_bytes} B)"
            )
        info = {
            "num_tokens": num_tokens,
            "num_frames": int(hdrs.get("x-audio-frames", "0") or 0),
            "duration_s": float(hdrs.get("x-audio-duration-s", "0") or 0.0),
            "encode_ms": float(hdrs.get("x-audio-encode-ms", "0") or 0.0),
            "model": hdrs.get("x-audio-model", ""),
        }
        return embed_bytes, num_tokens, info

    async def encode(self, raw_bytes: bytes, *, filename: str = "clip.wav") -> Tuple[bytes, int, Dict[str, Any]]:
        if not raw_bytes:
            raise ValueError("Audio payload must not be empty")
        if len(raw_bytes) > MAX_AUDIO_UPLOAD_BYTES:
            raise ValueError(f"Audio payload exceeds {MAX_AUDIO_UPLOAD_BYTES} B limit ({len(raw_bytes)} B)")
        return await asyncio.to_thread(self._encode_sync, raw_bytes, filename)

    def _tts_sync(
        self,
        text: str,
        *,
        language: str,
        gender: str,
        max_frames: int,
        daemon_host: Optional[str],
        daemon_port: Optional[int],
    ) -> Tuple[bytes, Dict[str, Any]]:
        body: Dict[str, Any] = {
            "text": text,
            "language": language,
            "gender": gender,
            "max_frames": int(max_frames),
        }
        if daemon_host:
            body["daemon_host"] = daemon_host
        if daemon_port:
            body["daemon_port"] = int(daemon_port)
        raw_req = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/tts/synthesize",
            data=raw_req,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                wav_bytes = resp.read()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Audio sidecar TTS HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Audio sidecar TTS failed ({self.base_url}): {exc}") from exc

        if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
            raise RuntimeError("Audio sidecar TTS returned invalid WAV payload")
        meta = {
            "frames": hdrs.get("x-tts-frames", ""),
            "duration_s": hdrs.get("x-tts-duration-s", ""),
            "total_ms": hdrs.get("x-tts-total-ms", ""),
            "rtf": hdrs.get("x-tts-rtf", ""),
            "prefill_ms": hdrs.get("x-tts-prefill-ms", ""),
            "depth_ms": hdrs.get("x-tts-depth-ms", ""),
            "codec_ms": hdrs.get("x-tts-codec-ms", ""),
            "eos": hdrs.get("x-tts-eos", ""),
        }
        return wav_bytes, meta

    async def synthesize_tts(
        self,
        text: str,
        *,
        language: str = "American English",
        gender: str = "female",
        max_frames: int = 250,
        daemon_host: Optional[str] = None,
        daemon_port: Optional[int] = None,
    ) -> Tuple[bytes, Dict[str, Any]]:
        return await asyncio.to_thread(
            self._tts_sync,
            text,
            language=language,
            gender=gender,
            max_frames=max_frames,
            daemon_host=daemon_host,
            daemon_port=daemon_port,
        )


class AudioService:
    """Offline Tier-1 + native LFM3-Audio ASR/TTS audio processing service."""

    def __init__(
        self,
        *,
        ffmpeg_bin: Optional[str] = None,
        espeak_bin: Optional[str] = None,
        piper_bin: Optional[str] = None,
        piper_model: Optional[Path | str] = None,
        asr_hook: Optional[Callable[[AudioMetadata], Any]] = None,
        tts_hook: Optional[Callable[[str, float], Any]] = None,
        asr_engine_name: Optional[str] = None,
        native_tts_hook: Optional[Callable[..., Any]] = None,
        native_tts_engine_name: str = "lfm3-audio",
        native_tts_default: bool = False,
    ):
        self.ffmpeg_bin = ffmpeg_bin if ffmpeg_bin is not None else shutil.which("ffmpeg")
        self.espeak_bin = espeak_bin if espeak_bin is not None else shutil.which("espeak-ng")

        resolved_piper_bin: Optional[str] = None
        if piper_bin and _is_real_piper_binary(Path(piper_bin)):
            resolved_piper_bin = str(Path(piper_bin))
        elif piper_model:
            cand = shutil.which("piper-tts") or shutil.which("piper")
            if cand and _is_real_piper_binary(Path(cand)):
                resolved_piper_bin = cand

        model_path = Path(piper_model) if piper_model else None
        if resolved_piper_bin and model_path is not None and model_path.is_file():
            self.piper_bin: Optional[str] = resolved_piper_bin
            self.piper_model: Optional[Path] = model_path
        else:
            self.piper_bin = None
            self.piper_model = None

        self._asr_hook = asr_hook
        self._tts_hook = tts_hook
        self._asr_engine_name = asr_engine_name
        self._native_tts_hook = native_tts_hook
        self._native_tts_engine_name = native_tts_engine_name or "lfm3-audio"
        self.native_tts_default = bool(native_tts_default)

    def configure_native_audio(
        self,
        *,
        asr_hook: Optional[Callable[[AudioMetadata], Any]] = None,
        asr_engine_name: Optional[str] = None,
        native_tts_hook: Optional[Callable[..., Any]] = None,
        native_tts_engine_name: Optional[str] = None,
        native_tts_default: Optional[bool] = None,
    ) -> None:
        """Attach native ASR/TTS callbacks without overwriting explicit test hooks."""
        if asr_hook is not None and self._asr_hook is None:
            self._asr_hook = asr_hook
            if asr_engine_name is not None:
                self._asr_engine_name = asr_engine_name
        if native_tts_hook is not None and self._native_tts_hook is None:
            self._native_tts_hook = native_tts_hook
            if native_tts_engine_name is not None:
                self._native_tts_engine_name = native_tts_engine_name
        if native_tts_default is not None:
            self.native_tts_default = bool(native_tts_default)

    @property
    def asr_available(self) -> bool:
        return self._asr_hook is not None

    @property
    def native_tts_available(self) -> bool:
        return self._native_tts_hook is not None

    @property
    def tts_engines(self) -> List[str]:
        engines: List[str] = []
        if self._tts_hook is not None:
            engines.append("custom-tts")
        if self._native_tts_hook is not None:
            engines.append(self._native_tts_engine_name)
        if self.piper_bin and self.piper_model:
            engines.append("piper-tts")
        if self.espeak_bin:
            engines.append("espeak-ng")
        return engines

    @property
    def tts_engine(self) -> Optional[str]:
        if self._tts_hook is not None:
            return "custom-tts"
        if self._native_tts_hook is not None and (
            self.native_tts_default or (not (self.piper_bin and self.piper_model) and not self.espeak_bin)
        ):
            return self._native_tts_engine_name
        if self.piper_bin and self.piper_model:
            return "piper-tts"
        if self.espeak_bin:
            return "espeak-ng"
        return None

    def capabilities(self) -> Dict[str, Any]:
        return {
            "voice_asr": bool(self._asr_hook is not None),
            "voice_tts": bool(self.tts_engine is not None),
            "tts_engine": self.tts_engine,
            "tts_engines": self.tts_engines,
            "asr_engine": (self._asr_engine_name or "custom-asr") if self._asr_hook is not None else None,
            "native_tts": self.native_tts_available,
            "native_tts_default": self.native_tts_default,
            "ffmpeg_available": bool(self.ffmpeg_bin),
        }

    async def inspect_and_normalize(
        self,
        raw_bytes: bytes,
        filename: str,
        dest_dir: Path,
        *,
        max_bytes: int = 25 * 1024 * 1024,
        timeout_s: float = 30.0,
    ) -> AudioMetadata:
        """Validate uploaded audio and normalize to 16 kHz mono 16-bit PCM WAV."""
        if not raw_bytes:
            raise ValueError("Uploaded audio payload is empty")
        if len(raw_bytes) > max_bytes:
            raise ValueError(f"Audio upload exceeds {max_bytes} byte limit ({len(raw_bytes)} B)")

        clean_name = Path((filename or "clip.wav").strip() or "clip.wav").name
        ext = Path(clean_name).suffix.lower() or ".wav"
        if ext not in _ALLOWED_AUDIO_EXTS:
            raise ValueError(f"Unsupported audio format '{ext}'; allowed: {sorted(_ALLOWED_AUDIO_EXTS)}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(raw_bytes).hexdigest()
        norm_path = (dest_dir / f"{digest[:16]}.wav").resolve()

        wav_info = _parse_wav_bytes(raw_bytes) if ext == ".wav" else None
        if wav_info is not None and wav_info[0] == 16000 and wav_info[1] == 1 and wav_info[2] == 2:
            norm_path.write_bytes(raw_bytes)
            sr, ch, _sw, dur = wav_info
            return AudioMetadata(
                filename=clean_name,
                bytes=len(raw_bytes),
                sha256=digest,
                duration_s=dur,
                sample_rate=sr,
                channels=ch,
                codec="pcm_s16le",
                normalized_path=str(norm_path),
            )

        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg is required to decode/resample non-16kHz-mono WAV audio")

        src_path = dest_dir / f"{digest[:16]}_raw{ext}"
        src_path.write_bytes(raw_bytes)
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(src_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                "-f",
                "wav",
                str(norm_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"ffmpeg timed out after {timeout_s}s") from exc

            if proc.returncode != 0 or not norm_path.exists():
                msg = (err or b"").decode("utf-8", errors="replace").strip()
                raise ValueError(f"ffmpeg rejected audio '{clean_name}': {msg or proc.returncode}")

            norm_bytes = norm_path.read_bytes()
            parsed = _parse_wav_bytes(norm_bytes)
            if parsed is None:
                raise ValueError("ffmpeg output was not a valid RIFF PCM WAV")
            sr, ch, _sw, dur = parsed
            return AudioMetadata(
                filename=clean_name,
                bytes=len(norm_bytes),
                sha256=hashlib.sha256(norm_bytes).hexdigest(),
                duration_s=dur,
                sample_rate=sr,
                channels=ch,
                codec="pcm_s16le",
                normalized_path=str(norm_path),
            )
        finally:
            try:
                src_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def transcribe(self, meta: AudioMetadata) -> Tuple[Optional[str], Optional[str]]:
        """Transcribe normalized audio if an ASR engine is attached (never fabricates text)."""
        if self._asr_hook is None:
            return None, None
        result = self._asr_hook(meta)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def synthesize_tts_detailed(
        self,
        text: str,
        *,
        speed: float = 1.0,
        engine: str = "auto",
        language: str = "American English",
        gender: str = "female",
        max_frames: int = 250,
        timeout_s: float = 90.0,
    ) -> Tuple[bytes, str, float, Dict[str, Any]]:
        """Synthesize WAV speech bytes with explicit engine selection and telemetry metadata."""
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            raise ValueError("No speakable text after stripping markup")
        sp = max(0.5, min(2.0, float(speed)))
        t0 = time.monotonic()

        chosen = (engine or "auto").strip().lower()
        if chosen in ("auto", ""):
            chosen = (self.tts_engine or "").lower()

        if chosen == "custom-tts" and self._tts_hook is not None:
            out = self._tts_hook(cleaned, sp)
            if asyncio.iscoroutine(out):
                out = await out
            wav_bytes, eng = out
            if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
                raise RuntimeError("Custom TTS hook returned invalid WAV bytes")
            return wav_bytes, eng, round((time.monotonic() - t0) * 1000.0, 2), {}

        if chosen in ("lfm3-audio", "lfm3", "native", self._native_tts_engine_name.lower()):
            if self._native_tts_hook is None:
                raise RuntimeError("Native lfm3-audio TTS is not configured (--audio-url / --audio-port)")
            out = self._native_tts_hook(
                cleaned,
                speed=sp,
                language=language,
                gender=gender,
                max_frames=max_frames,
            )
            if asyncio.iscoroutine(out):
                out = await out
            if len(out) == 3:
                wav_bytes, eng, meta = out
            else:
                wav_bytes, eng = out
                meta = {}
            if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
                raise RuntimeError("Native TTS hook returned invalid WAV bytes")
            return wav_bytes, str(eng), round((time.monotonic() - t0) * 1000.0, 2), dict(meta or {})

        if chosen in ("piper-tts", "piper"):
            if not (self.piper_bin and self.piper_model):
                raise RuntimeError("piper-tts is not installed or voice model is missing")
            cmd = [self.piper_bin, "--model", str(self.piper_model), "--output_file", "-"]
            if sp != 1.0:
                cmd.extend(["--length_scale", f"{1.0 / sp:.3f}"])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate((cleaned + "\n").encode("utf-8")),
                    timeout=min(timeout_s, 30.0),
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"piper-tts timed out after {timeout_s}s") from exc
            if proc.returncode != 0 or len(stdout) < 44 or stdout[:4] != b"RIFF":
                msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"piper-tts failed ({proc.returncode}): {msg}")
            return stdout, "piper-tts", round((time.monotonic() - t0) * 1000.0, 2), {}

        if chosen in ("espeak-ng", "espeak"):
            if not self.espeak_bin:
                raise RuntimeError("espeak-ng is not installed")
            wpm = max(80, min(360, int(round(170 * sp))))
            proc = await asyncio.create_subprocess_exec(
                self.espeak_bin,
                "--stdout",
                "-s",
                str(wpm),
                "--",
                cleaned,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=min(timeout_s, 20.0))
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"espeak-ng timed out after {timeout_s}s") from exc
            if proc.returncode != 0 or len(stdout) < 44 or stdout[:4] != b"RIFF":
                msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"espeak-ng failed ({proc.returncode}): {msg}")
            return stdout, "espeak-ng", round((time.monotonic() - t0) * 1000.0, 2), {}

        if chosen:
            raise ValueError(f"Unsupported or unavailable TTS engine '{engine}'")
        raise RuntimeError("No Tier-1 TTS engine available (install espeak-ng, piper-tts, or configure lfm3-audio)")

    async def synthesize_tts(
        self,
        text: str,
        *,
        speed: float = 1.0,
        engine: str = "auto",
        language: str = "American English",
        gender: str = "female",
        max_frames: int = 250,
        timeout_s: float = 90.0,
    ) -> Tuple[bytes, str, float]:
        """Synthesize WAV speech bytes using lfm3-audio, piper-tts, or espeak-ng."""
        wav_bytes, chosen_eng, elapsed_ms, _ = await self.synthesize_tts_detailed(
            text,
            speed=speed,
            engine=engine,
            language=language,
            gender=gender,
            max_frames=max_frames,
            timeout_s=timeout_s,
        )
        return wav_bytes, chosen_eng, elapsed_ms
