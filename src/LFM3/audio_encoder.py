#!/usr/bin/env python3
"""CPU-side ONNX audio encoder and MaskGIT/LiquidCodec synthesizer for LiquidAI LFM3-Audio-2B.

Uses only Python stdlib + NumPy + ONNX Runtime (no PyTorch dependency at runtime).
"""

import io
import json
import os
import socket
import struct
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - tested with optional mocks when ort unavailable
    ort = None  # type: ignore[assignment]

AUDIO_EMBED_DIM = 2048
AUDIO_SAMPLE_RATE = 16000
CODEC_SAMPLE_RATE = 24000
CODEC_SAMPLES_PER_FRAME = 1920  # 12.5 Hz -> 24000 / 12.5 = 1920
NUM_CODEBOOKS = 12
CODEBOOK_SIZE = 2017
AUDIO_EOS_CODE = 2016
MASKGIT_SCHEDULE = (4, 4, 4)
MAX_AUDIO_SECONDS = 30.0

DEFAULT_AUDIO_ONNX_DIRS = (
    "/home/chri/aixigua/deep/models/LFM3-Audio-2B-ONNX",
)
DEFAULT_CODEBOOK_PATHS = (
    "/home/chri/aixigua/deep/models/LFM3-Audio-2B/audio_codebook_embed.bin",
    "/home/chri/aixigua/deep/models/LFM3-Audio-2B/eswin_pkg_tp2/audio_codebook_embed.bin",
    "/home/chri/aixigua/deep/models/LFM3-Audio-2B/eswin_pkg_int8/audio_codebook_embed.bin",
)


def _hertz_to_mel_slaney(freq: Any) -> Any:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    if isinstance(freq, np.ndarray):
        mels = 3.0 * freq / 200.0
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
        return mels
    if freq >= min_log_hertz:
        return min_log_mel + float(np.log(freq / min_log_hertz) * logstep)
    return 3.0 * float(freq) / 200.0


def _mel_to_hertz_slaney(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _build_slaney_mel_filter_bank(
    num_frequency_bins: int = 201,
    num_mel_filters: int = 128,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
    sampling_rate: int = AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    """Build triangular Slaney-normalized mel filterbank [201, 128] FP32."""
    fft_freqs = np.linspace(0.0, sampling_rate // 2, num_frequency_bins)
    mel_freqs = np.linspace(
        _hertz_to_mel_slaney(min_frequency),
        _hertz_to_mel_slaney(max_frequency),
        num_mel_filters + 2,
    )
    filter_freqs = _mel_to_hertz_slaney(mel_freqs)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters.astype(np.float32)


_MEL_FILTER_BANK_F32 = _build_slaney_mel_filter_bank()
_MEL_HANN_WIN_F32 = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(400, dtype=np.float64) / 400.0)).astype(np.float32)
_ISTFT_HANN_WIN_F64 = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1280, dtype=np.float64) / 1280.0)
_ISTFT_WIN_SQ_F64 = _ISTFT_HANN_WIN_F64 * _ISTFT_HANN_WIN_F64


def compute_log_mel(wav_16k: np.ndarray) -> np.ndarray:
    """Compute [1, 128, N_frames] FP32 Slaney log-mel spectrogram matching Lfm3AudioFeatureExtractor."""
    x = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot compute log-mel spectrogram of empty audio")
    n_fft = 400
    hop = 160
    pad_len = n_fft if x.size < n_fft else x.size + (-(x.size - n_fft) % hop)
    if pad_len > x.size:
        x = np.pad(x, (0, pad_len - x.size))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    spec = np.fft.rfft((frames * _MEL_HANN_WIN_F32[None, :]).astype(np.float32), n=n_fft, axis=-1)
    power = (spec.real.astype(np.float32) ** 2) + (spec.imag.astype(np.float32) ** 2)
    mel = np.maximum(power @ _MEL_FILTER_BANK_F32, np.float32(1e-30))
    log_mel = np.maximum(np.log10(mel), np.float32(-12.0))
    norm_mel = (log_mel + np.float32(4.0)) / np.float32(4.0)
    return np.ascontiguousarray(norm_mel.T[None, :, :], dtype=np.float32)


def decode_audio_to_16k_mono(
    raw_bytes: bytes,
    filename: str = "",
    *,
    max_duration_s: float = MAX_AUDIO_SECONDS,
) -> Tuple[np.ndarray, float]:
    """Decode WAV or compressed audio bytes to mono 16 kHz FP32 waveform [-1, 1] and duration_s."""
    if not raw_bytes:
        raise ValueError("Empty audio payload")

    # Fast pure-stdlib path when already 16 kHz mono 16-bit PCM WAV.
    if len(raw_bytes) >= 44 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == AUDIO_SAMPLE_RATE:
                    pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
                    dur = pcm.size / float(AUDIO_SAMPLE_RATE)
                    if dur <= 0.0:
                        raise ValueError("Decoded WAV contains zero samples")
                    if dur > max_duration_s:
                        raise ValueError(f"Audio duration {dur:.2f}s exceeds {max_duration_s:.1f}s limit")
                    return pcm, dur
        except wave.Error:
            pass

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=raw_bytes,
            capture_output=True,
            timeout=20.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg is required to decode non-16kHz-mono audio ({filename or 'blob'})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg audio decoding timed out") from exc

    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise ValueError(f"Audio decode failed for '{filename or 'clip'}': {msg or proc.returncode}")

    if len(proc.stdout) % 2 != 0:
        proc.stdout = proc.stdout[:-1]
    pcm = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    dur = pcm.size / float(AUDIO_SAMPLE_RATE)
    if dur <= 0.0:
        raise ValueError("Decoded audio contains zero samples")
    if dur > max_duration_s:
        raise ValueError(f"Audio duration {dur:.2f}s exceeds {max_duration_s:.1f}s limit")
    return pcm, dur


def liquid_codec_istft(stft_features: np.ndarray) -> np.ndarray:
    """Pure-NumPy overlap-add iSTFT matching LiquidCodecISTFT + LiquidCodecModel trimming."""
    x = np.asarray(stft_features)
    if x.ndim == 3:
        x = x[0]
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 1282:
        raise ValueError(f"Expected stft_features shape [S, 1282], got {tuple(x.shape)}")

    n_fft = 1280
    hop = 320
    win_len = 1280
    log_mag = np.clip(x[:, :641], -20.0, 20.0)
    angle = x[:, 641:]
    spec = np.exp(log_mag) * np.exp(1j * angle)
    frames = np.fft.irfft(spec, n=n_fft, axis=-1)
    n_stft = frames.shape[0]
    out_len = (n_stft - 1) * hop + win_len
    y = np.zeros(out_len, dtype=np.float64)
    env = np.zeros(out_len, dtype=np.float64)
    w_frames = frames * _ISTFT_HANN_WIN_F64[None, :]
    for i in range(n_stft):
        s0 = i * hop
        y[s0 : s0 + win_len] += w_frames[i]
        env[s0 : s0 + win_len] += _ISTFT_WIN_SQ_F64
    trim = (win_len - hop) // 2  # 480 samples
    y_trim = y[trim : out_len - trim]
    env_trim = np.maximum(env[trim : out_len - trim], 1e-12)
    return np.ascontiguousarray((y_trim / env_trim).astype(np.float32))


def pcm_f32_to_wav_bytes(samples: np.ndarray, sample_rate: int = CODEC_SAMPLE_RATE) -> bytes:
    """Encode mono FP32 waveform [-1, 1] into a 16-bit PCM WAV byte buffer."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = np.rint(clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _resolve_onnx_dir(explicit_dir: Optional[str] = None) -> Path:
    if explicit_dir:
        p = Path(explicit_dir).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Audio ONNX directory/file not found: {p}")
    for cand in DEFAULT_AUDIO_ONNX_DIRS:
        p = Path(cand)
        if p.exists():
            return p
    raise FileNotFoundError(f"LFM3-Audio ONNX directory not found in {DEFAULT_AUDIO_ONNX_DIRS}")


def _resolve_onnx_file(base: Path, filename: str) -> Path:
    if base.is_file() and base.name == filename:
        return base
    for cand in (base / "onnx" / filename, base / filename):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"{filename} not found under {base}")


def _resolve_codebook_bin(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Audio codebook embedding file not found: {p}")
    for cand in DEFAULT_CODEBOOK_PATHS:
        p = Path(cand)
        if p.is_file():
            return p
    raise FileNotFoundError(f"audio_codebook_embed.bin not found in {DEFAULT_CODEBOOK_PATHS}")


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: List[bytes] = []
    got = 0
    while got < nbytes:
        chunk = sock.recv(nbytes - got)
        if not chunk:
            raise EOFError(f"Socket closed after {got}/{nbytes} bytes")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _recv_json_line(sock: socket.socket, rx_buf: bytearray) -> Dict[str, Any]:
    while True:
        nl = rx_buf.find(b"\n")
        if nl >= 0:
            line = bytes(rx_buf[:nl]).decode("utf-8", errors="replace").strip()
            del rx_buf[: nl + 1]
            if not line:
                continue
            return dict(json.loads(line))
        chunk = sock.recv(4096)
        if not chunk:
            raise EOFError("Daemon closed connection before sending JSON line")
        rx_buf.extend(chunk)


def _recv_exact_buffered(sock: socket.socket, rx_buf: bytearray, nbytes: int) -> bytes:
    out = bytearray()
    if rx_buf:
        take = min(len(rx_buf), nbytes)
        out.extend(rx_buf[:take])
        del rx_buf[:take]
    if len(out) < nbytes:
        out.extend(_recv_exact(sock, nbytes - len(out)))
    return bytes(out)


class Lfm3AudioEncoder:
    """CPU ONNX Runtime audio encoder producing [N_tok, 2048] FP32 embeddings for LFM3-Audio-2B."""

    def __init__(self, onnx_dir: Optional[str] = None, *, num_threads: int = 4):
        if ort is None:
            raise RuntimeError("onnxruntime is required for Lfm3AudioEncoder")
        base = _resolve_onnx_dir(onnx_dir)
        self.onnx_path = _resolve_onnx_file(base, "audio_encoder.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(num_threads)
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def encode_pcm_16k(self, wav_16k: np.ndarray) -> Tuple[np.ndarray, int]:
        mel = compute_log_mel(wav_16k)
        num_frames = int(mel.shape[-1])
        feats = self.session.run(None, {"input_features": mel})[0]
        feats_f32 = np.ascontiguousarray(feats, dtype=np.float32)
        if feats_f32.ndim != 2 or feats_f32.shape[1] != AUDIO_EMBED_DIM:
            raise RuntimeError(f"Unexpected audio_encoder output shape {tuple(feats_f32.shape)}")
        return feats_f32, num_frames

    def encode_wav_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "",
        *,
        max_duration_s: float = MAX_AUDIO_SECONDS,
    ) -> Dict[str, Any]:
        t0 = time.monotonic()
        wav_16k, duration_s = decode_audio_to_16k_mono(audio_bytes, filename, max_duration_s=max_duration_s)
        feats_f32, num_frames = self.encode_pcm_16k(wav_16k)
        encode_ms = round((time.monotonic() - t0) * 1000.0, 2)
        return {
            "embeddings": feats_f32,
            "num_tokens": int(feats_f32.shape[0]),
            "num_frames": num_frames,
            "duration_s": round(float(duration_s), 3),
            "encode_ms": encode_ms,
            "model": self.onnx_path.name,
        }


class Lfm3AudioSynthesizer:
    """Host CPU MaskGIT depth_decoder + LiquidCodec ONNX TTS synthesizer paired with NPU daemon."""

    def __init__(
        self,
        onnx_dir: Optional[str] = None,
        codebook_bin_path: Optional[str] = None,
        *,
        num_threads: int = 4,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required for Lfm3AudioSynthesizer")
        base = _resolve_onnx_dir(onnx_dir)
        self.depth_onnx_path = _resolve_onnx_file(base, "depth_decoder.onnx")
        self.codec_onnx_path = _resolve_onnx_file(base, "codec_decoder.onnx")
        self.codebook_bin_path = _resolve_codebook_bin(codebook_bin_path)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(num_threads)
        self.depth_session = ort.InferenceSession(
            str(self.depth_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.codec_session = ort.InferenceSession(
            str(self.codec_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        raw_cb = np.fromfile(str(self.codebook_bin_path), dtype=np.float32)
        expected = NUM_CODEBOOKS * CODEBOOK_SIZE * AUDIO_EMBED_DIM
        if raw_cb.size != expected:
            raise ValueError(
                f"Expected {expected} float32 entries in {self.codebook_bin_path}, got {raw_cb.size}"
            )
        self.codebook_table = np.ascontiguousarray(
            raw_cb.reshape(NUM_CODEBOOKS * CODEBOOK_SIZE, AUDIO_EMBED_DIM),
            dtype=np.float32,
        )
        self.offsets = (np.arange(NUM_CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE)

    @staticmethod
    def build_tts_prompt(
        text: str,
        *,
        language: str = "American English",
        gender: str = "female",
    ) -> str:
        clean_text = text.strip()
        lang = language.strip() or "American English"
        gen = gender.strip() or "female"
        return (
            f"<|startoftext|><|im_start|>system\n"
            f"Perform TTS in {lang} using a {gen} voice.<|im_end|>\n"
            f"<|im_start|>user\n{clean_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<|audio_start|>"
        )

    def embed_frame(self, codes_12: np.ndarray) -> np.ndarray:
        """Sum the 12 codebook embeddings for one speech frame into [2048] FP32."""
        idx = np.asarray(codes_12, dtype=np.int64) + self.offsets
        return np.ascontiguousarray(self.codebook_table[idx].sum(axis=0), dtype=np.float32)

    def sample_audio_frame(
        self,
        hidden_vec: np.ndarray,
        rng: np.random.Generator,
        history: Sequence[np.ndarray],
        *,
        temperature: float = 0.8,
        top_k: int = 128,
        repetition_penalty: float = 1.1,
        repetition_window: int = 4096,
    ) -> Tuple[np.ndarray, bool]:
        """Run 3-step MaskGIT [4, 4, 4] sampling on depth_decoder.onnx for one 80 ms speech frame."""
        codes = np.zeros((1, NUM_CODEBOOKS), dtype=np.int64)
        masked = np.ones((1, NUM_CODEBOOKS), dtype=bool)
        h_in = np.ascontiguousarray(hidden_vec, dtype=np.float32).reshape(1, AUDIO_EMBED_DIM)
        order = rng.permutation(NUM_CODEBOOKS)
        recent = history[-repetition_window:] if (repetition_penalty != 1.0 and history) else ()

        cursor = 0
        for count in MASKGIT_SCHEDULE:
            logits = self.depth_session.run(
                None,
                {"hidden_states": h_in, "codes": codes, "masked": masked},
            )[0][0]
            pick = order[cursor : cursor + count]
            cursor += count
            for cb in pick:
                scores = logits[cb].astype(np.float64, copy=True)
                if recent:
                    prev = np.unique([int(frame[cb]) for frame in recent])
                    neg = scores[prev] < 0.0
                    scores[prev] = np.where(neg, scores[prev] * repetition_penalty, scores[prev] / repetition_penalty)
                if temperature <= 0.0:
                    chosen = int(np.argmax(scores))
                else:
                    scores /= float(temperature)
                    k = int(top_k)
                    if 0 < k < scores.size:
                        kth = np.partition(scores, -k)[-k]
                        scores[scores < kth] = -np.inf
                    scores -= np.max(scores)
                    probs = np.exp(scores)
                    probs_sum = float(probs.sum())
                    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
                        chosen = int(np.argmax(scores))
                    else:
                        probs /= probs_sum
                        chosen = int(rng.choice(scores.size, p=probs))
                codes[0, cb] = chosen
                masked[0, cb] = False

        is_eos = bool(np.any(codes[0] == AUDIO_EOS_CODE))
        return codes[0].copy(), is_eos

    def decode_codes_to_wav(self, frames: Sequence[np.ndarray]) -> Tuple[bytes, float]:
        """Decode [F, 12] non-EOS speech frames via codec_decoder.onnx + iSTFT into 24 kHz WAV bytes."""
        t0 = time.monotonic()
        if not frames:
            silence = np.zeros((2400,), dtype=np.float32)
            return pcm_f32_to_wav_bytes(silence, CODEC_SAMPLE_RATE), round((time.monotonic() - t0) * 1000.0, 2)
        codes_arr = np.stack(frames, axis=0).T[None, :, :].astype(np.int64)
        stft_feat = self.codec_session.run(None, {"audio_codes": codes_arr})[0]
        wav_f32 = liquid_codec_istft(stft_feat)
        wav_bytes = pcm_f32_to_wav_bytes(wav_f32, CODEC_SAMPLE_RATE)
        codec_ms = round((time.monotonic() - t0) * 1000.0, 2)
        return wav_bytes, codec_ms

    def synthesize_via_daemon(
        self,
        text: str,
        daemon_host: str,
        daemon_port: int,
        *,
        language: str = "American English",
        gender: str = "female",
        max_frames: int = 250,
        temperature: float = 0.8,
        top_k: int = 128,
        repetition_penalty: float = 1.1,
        audio_out_token_id: int = 125032,
        seed: Optional[int] = None,
        timeout_s: float = 90.0,
    ) -> Dict[str, Any]:
        """Stream hidden states over TCP `tts_session` with LFM3-Audio-2B daemon and synthesize 24 kHz WAV."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("TTS text must not be empty")

        t_start = time.monotonic()
        rng = np.random.default_rng(seed)
        prompt = self.build_tts_prompt(clean_text, language=language, gender=gender)
        req_obj = {
            "cmd": "tts_session",
            "prompt": prompt,
            "raw": True,
            "no_think": True,
            "max_frames": int(max(1, max_frames)),
            "audio_out_token_id": int(audio_out_token_id),
            "cache": True,
        }
        payload = (json.dumps(req_obj) + "\n").encode("utf-8")

        frames: List[np.ndarray] = []
        eos_hit = False
        depth_ms = 0.0
        prefill_ms = 0.0
        daemon_steps = 0
        hidden_nbytes = AUDIO_EMBED_DIM * 4

        with socket.create_connection((daemon_host, int(daemon_port)), timeout=float(timeout_s)) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(float(timeout_s))
            sock.sendall(payload)

            rx_buf = bytearray()
            ready = _recv_json_line(sock, rx_buf)
            if ready.get("type") == "error":
                raise RuntimeError(str(ready.get("message") or "Daemon tts_session rejected request"))
            if ready.get("type") != "tts_ready":
                raise RuntimeError(f"Unexpected tts_session handshake: {ready}")

            prefill_ms = float(ready.get("prefill_ms") or 0.0)
            eff_max = min(int(max(1, max_frames)), int(ready.get("effective_max_frames") or max_frames))
            h_raw = _recv_exact_buffered(sock, rx_buf, hidden_nbytes)
            h_vec = np.frombuffer(h_raw, dtype="<f4").copy()

            while len(frames) < eff_max:
                td0 = time.monotonic()
                codes, is_eos = self.sample_audio_frame(
                    h_vec,
                    rng,
                    frames,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
                depth_ms += (time.monotonic() - td0) * 1000.0
                if is_eos:
                    eos_hit = True
                    sock.sendall(b"\x00")
                    break
                frames.append(codes)
                if len(frames) >= eff_max:
                    sock.sendall(b"\x00")
                    break
                next_emb = self.embed_frame(codes)
                sock.sendall(b"\x01" + next_emb.tobytes())
                h_raw = _recv_exact_buffered(sock, rx_buf, hidden_nbytes)
                h_vec = np.frombuffer(h_raw, dtype="<f4").copy()

            try:
                done = _recv_json_line(sock, rx_buf)
                daemon_steps = int(done.get("steps") or 0)
            except (EOFError, json.JSONDecodeError):
                daemon_steps = max(0, len(frames) - (0 if eos_hit else 1))

        wav_bytes, codec_ms = self.decode_codes_to_wav(frames)
        total_ms = round((time.monotonic() - t_start) * 1000.0, 2)
        duration_s = round(len(frames) / 12.5, 3) if frames else 0.1
        rtf = round((duration_s / (total_ms / 1000.0)) if total_ms > 0 else 0.0, 3)
        return {
            "wav_bytes": wav_bytes,
            "num_frames": len(frames),
            "daemon_steps": daemon_steps,
            "duration_s": duration_s,
            "total_ms": total_ms,
            "prefill_ms": round(prefill_ms, 2),
            "depth_ms": round(depth_ms, 2),
            "codec_ms": round(codec_ms, 2),
            "rtf": rtf,
            "eos": eos_hit,
            "sample_rate": CODEC_SAMPLE_RATE,
        }


def main() -> None:
    enc = Lfm3AudioEncoder()
    sr = AUDIO_SAMPLE_RATE
    t = np.linspace(0.0, 0.5, sr // 2, dtype=np.float32)
    sine = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    emb, n_frames = enc.encode_pcm_16k(sine)
    print(f"[+] Lfm3AudioEncoder self-test: {n_frames} mel frames -> {tuple(emb.shape)} {emb.dtype}")


if __name__ == "__main__":
    main()
