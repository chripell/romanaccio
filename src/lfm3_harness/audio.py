"""Tier-1 offline web voice and audio processing service (ffmpeg + espeak-ng / optional piper-tts)."""

import asyncio
import hashlib
import re
import shutil
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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


class AudioService:
    """Offline Tier-1 audio normalizer, optional ASR hook, and espeak-ng / piper-tts synthesizer."""

    def __init__(
        self,
        *,
        ffmpeg_bin: Optional[str] = None,
        espeak_bin: Optional[str] = None,
        piper_bin: Optional[str] = None,
        piper_model: Optional[Path | str] = None,
        asr_hook: Optional[Callable[[AudioMetadata], Tuple[Optional[str], Optional[str]]]] = None,
        tts_hook: Optional[Callable[[str, float], Tuple[bytes, str]]] = None,
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

    @property
    def tts_engine(self) -> Optional[str]:
        if self._tts_hook is not None:
            return "custom-tts"
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
            "asr_engine": "custom-asr" if self._asr_hook is not None else None,
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

    async def synthesize_tts(
        self,
        text: str,
        *,
        speed: float = 1.0,
        timeout_s: float = 15.0,
    ) -> Tuple[bytes, str, float]:
        """Synthesize WAV speech bytes using Rhasspy piper-tts (if configured) or espeak-ng."""
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            raise ValueError("No speakable text after stripping markup")
        sp = max(0.5, min(2.0, float(speed)))
        t0 = time.monotonic()

        if self._tts_hook is not None:
            out = self._tts_hook(cleaned, sp)
            if asyncio.iscoroutine(out):
                out = await out
            wav_bytes, eng = out
            if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
                raise RuntimeError("Custom TTS hook returned invalid WAV bytes")
            return wav_bytes, eng, round((time.monotonic() - t0) * 1000.0, 2)

        if self.piper_bin and self.piper_model:
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
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"piper-tts timed out after {timeout_s}s") from exc
            if proc.returncode != 0 or len(stdout) < 44 or stdout[:4] != b"RIFF":
                msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"piper-tts failed ({proc.returncode}): {msg}")
            return stdout, "piper-tts", round((time.monotonic() - t0) * 1000.0, 2)

        if self.espeak_bin:
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
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"espeak-ng timed out after {timeout_s}s") from exc
            if proc.returncode != 0 or len(stdout) < 44 or stdout[:4] != b"RIFF":
                msg = (stderr or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"espeak-ng failed ({proc.returncode}): {msg}")
            return stdout, "espeak-ng", round((time.monotonic() - t0) * 1000.0, 2)

        raise RuntimeError("No Tier-1 TTS engine available (install espeak-ng or configure piper-tts + voice .onnx)")
