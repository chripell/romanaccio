"""Host end-to-end test suite for Milestone 19F: LFM3-Audio-2B ASR, Native TTS & Full Voice Harness."""

import asyncio
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LFM3_DIR = _REPO_ROOT / "src" / "LFM3"
if str(_LFM3_DIR) not in sys.path:
    sys.path.insert(0, str(_LFM3_DIR))

from src.lfm3_harness import (  # noqa: E402
    AudioService,
    BackendEndpoint,
    Capabilities,
    GenerateRequest,
    HarnessConfig,
    SafetyLimits,
    get_tool_schemas,
)
from src.lfm3_harness.audio import AUDIO_EMBED_DIM, AudioClient, AudioMetadata, inspect_wav_bytes  # noqa: E402
from src.lfm3_harness.backend import DaemonClient, TokenizerLookup  # noqa: E402
from src.lfm3_harness.paths import WorkspaceSandbox  # noqa: E402
from src.lfm3_harness.server import create_app  # noqa: E402
from src.lfm3_harness.tools import ToolExecutor  # noqa: E402


def _make_sine_wav_16k(duration_s: float = 0.5, freq_hz: float = 440.0) -> bytes:
    import io
    sr = 16000
    n = int(round(sr * duration_s))
    frames = b"".join(
        struct.pack("<h", int(round(0.25 * 32767.0 * math.sin(2.0 * math.pi * freq_hz * i / sr))))
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(frames)
    return buf.getvalue()


def _make_pcm24_silence_wav(num_frames: int = 2) -> bytes:
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * (1920 * num_frames))
    return buf.getvalue()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _MultimodalFakeDaemon:
    """In-process TCP fake daemon recording image/audio payloads, embed_token_id, suppress_tokens, and tts_session."""

    def __init__(self, handler: Callable[[Dict[str, Any], bytes], Tuple[List[str], float]]):
        self.handler = handler
        self.port = _free_port()
        self.requests: List[Dict[str, Any]] = []
        self.received_media: List[bytes] = []
        self.cleared_flags: List[bool] = []
        self.tts_requests: List[Dict[str, Any]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop_ev: Optional[asyncio.Event] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        if self._loop and self._stop_ev:
            self._loop.call_soon_threadsafe(self._stop_ev.set)
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_ev = asyncio.Event()
        srv = await asyncio.start_server(self._handle_conn, "127.0.0.1", self.port)
        self._ready.set()
        await self._stop_ev.wait()
        srv.close()
        await srv.wait_closed()

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            cmd = req.get("cmd")
            if cmd == "cache_stats":
                resp = {
                    "type": "cache_stats",
                    "enabled": True,
                    "resident_blocks": 1,
                    "pinned_blocks": 0,
                    "disk_bytes_used": 1024,
                    "disk_bytes_quota": 1048576,
                    "total_lookups": 1,
                    "total_block_hits": 1,
                    "total_tokens_saved": 64,
                    "total_evictions": 0,
                    "pending_writes": 0,
                    "tier0_hits": 1,
                    "tier0_tokens_saved": 64,
                    "tier0_live_tokens": 64,
                }
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
                return
            if cmd == "cache_clear":
                self.cleared_flags.append(bool(req.get("include_pinned", False)))
                writer.write((json.dumps({"type": "cache_cleared", "removed_blocks": 1}) + "\n").encode("utf-8"))
                await writer.drain()
                return
            if cmd == "tts_session":
                import numpy as np

                self.tts_requests.append(req)
                eff_max = min(int(req.get("max_frames") or 250), 32)
                writer.write(
                    (
                        json.dumps(
                            {
                                "type": "tts_ready",
                                "prompt_tokens": 20,
                                "cached_tokens": 0,
                                "cache_status": "MISS",
                                "prefill_ms": 4.0,
                                "effective_max_frames": eff_max,
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                writer.write(np.full((AUDIO_EMBED_DIM,), 0.25, dtype="<f4").tobytes())
                await writer.drain()
                steps = 0
                while True:
                    op_b = await reader.readexactly(1)
                    if op_b[0] == 0:
                        break
                    emb_b = await reader.readexactly(AUDIO_EMBED_DIM * 4)
                    assert len(emb_b) == AUDIO_EMBED_DIM * 4
                    steps += 1
                    writer.write(np.full((AUDIO_EMBED_DIM,), 0.125 * (steps + 1), dtype="<f4").tobytes())
                    await writer.drain()
                writer.write(
                    (
                        json.dumps(
                            {
                                "type": "tts_done",
                                "steps": steps,
                                "prefill_ms": 4.0,
                                "decode_ms": 8.0,
                                "fps": 125.0,
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
                return

            nbytes = int(req.get("audio_bytes") or req.get("image_bytes") or 0)
            media = await reader.readexactly(nbytes) if nbytes > 0 else b""
            self.requests.append(req)
            self.received_media.append(media)

            if req.get("return_logits"):
                yes_id = TokenizerLookup().single_token_id("yes") or 11683
                no_id = TokenizerLookup().single_token_id("no") or 2243
                logits = {str(tid): -5.0 for tid in (req.get("requested_tokens") or [])}
                logits[str(yes_id)] = 3.0
                logits[str(no_id)] = -3.0
                writer.write(
                    (json.dumps({"type": "logits", "prefill_ms": 2.0, "cache_status": "HIT_TIER0", "logits": logits}) + "\n").encode("utf-8")
                )
                await writer.drain()
                return

            writer.write((json.dumps({"type": "prefill_done", "prompt_tokens": 16, "cached_tokens": 0}) + "\n").encode("utf-8"))
            await writer.drain()
            pieces, _ = self.handler(req, media)
            for idx, piece in enumerate(pieces, start=1):
                writer.write(
                    (json.dumps({"type": "token", "token_id": 100 + idx, "text": piece, "thinking": False}) + "\n").encode("utf-8")
                )
                await writer.drain()
            writer.write(
                (json.dumps({"type": "done", "prompt_tokens": 16, "completion_tokens": len(pieces), "finish_reason": "stop"}) + "\n").encode("utf-8")
            )
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


class _FakeEncoder:
    onnx_path = Path("fake_audio_encoder.onnx")

    def encode_wav_bytes(self, raw_bytes: bytes, filename: str = "", *, max_duration_s: float = 30.0) -> Dict[str, Any]:
        import numpy as np

        _ = max_duration_s
        info = inspect_wav_bytes(raw_bytes)
        if info is None:
            raise ValueError(f"Bad wav: {filename}")
        return {
            "embeddings": np.full((6, AUDIO_EMBED_DIM), 0.5, dtype=np.float32),
            "num_tokens": 6,
            "num_frames": 48,
            "duration_s": round(info[3], 3),
            "encode_ms": 3.5,
            "model": "fake_audio_encoder.onnx",
        }


class _FakeSynthesizer:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

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
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "text": text,
                "daemon_host": daemon_host,
                "daemon_port": daemon_port,
                "language": language,
                "gender": gender,
                "max_frames": max_frames,
            }
        )
        wav_bytes = _make_pcm24_silence_wav(2)
        return {
            "wav_bytes": wav_bytes,
            "num_frames": 2,
            "duration_s": 0.16,
            "total_ms": 40.0,
            "prefill_ms": 5.0,
            "depth_ms": 15.0,
            "codec_ms": 10.0,
            "rtf": 4.0,
            "eos": True,
        }


def test_mel_and_istft_purity_and_shapes() -> None:
    import numpy as np
    from audio_encoder import compute_log_mel, liquid_codec_istft, pcm_f32_to_wav_bytes
    short_pcm = np.zeros((320,), dtype=np.float32)
    mel_short = compute_log_mel(short_pcm)
    assert mel_short.dtype == np.float32
    assert mel_short.shape == (1, 128, 1)
    assert np.all(np.isfinite(mel_short))

    pcm_1s = 0.2 * np.sin(2.0 * np.pi * 440.0 * np.arange(16000, dtype=np.float32) / 16000.0)
    mel_1s = compute_log_mel(pcm_1s)
    assert mel_1s.shape == (1, 128, 99)
    assert mel_1s.min() >= -2.0

    # iSTFT shape & WAV serialization check (6 * 4 = 24 STFT frames -> 4 * 1920 = 7680 samples)
    stft_zeros = np.zeros((1, 24, 1282), dtype=np.float32)
    wav_f32 = liquid_codec_istft(stft_zeros)
    assert wav_f32.dtype == np.float32
    assert wav_f32.shape == (7680,)
    assert liquid_codec_istft(np.zeros((1, 0, 1282), dtype=np.float32)).size == 0

    wav_bytes = pcm_f32_to_wav_bytes(wav_f32, sample_rate=24000)
    parsed = inspect_wav_bytes(wav_bytes)
    assert parsed == (24000, 1, 2, 0.32)


def test_audio_sidecar_encode_and_tts_endpoints() -> None:
    from src.lfm3_harness.audio_sidecar import create_audio_sidecar_app

    fake_enc = _FakeEncoder()
    fake_synth = _FakeSynthesizer()
    sidecar_app = create_audio_sidecar_app(encoder=fake_enc, synthesizer=fake_synth, default_daemon_port=8091)
    tc = TestClient(sidecar_app)

    h = tc.get("/health")
    assert h.status_code == 200
    assert h.json()["asr_encoder"] is True
    assert h.json()["tts_synthesizer"] is True

    wav_in = _make_sine_wav_16k(0.4)
    enc_raw = tc.post("/encode?format=raw&filename=probe.wav", content=wav_in)
    assert enc_raw.status_code == 200
    assert int(enc_raw.headers["x-audio-tokens"]) == 6
    assert len(enc_raw.content) == 6 * AUDIO_EMBED_DIM * 4

    enc_json = tc.post("/encode?format=json&filename=probe.wav", content=wav_in)
    assert enc_json.status_code == 200
    assert enc_json.json()["num_tokens"] == 6

    bad = tc.post("/encode", content=b"")
    assert bad.status_code == 400

    tts = tc.post("/tts/synthesize", json={"text": "System nominal.", "language": "British English", "gender": "male"})
    assert tts.status_code == 200
    assert tts.headers["content-type"].startswith("audio/wav")
    assert tts.headers["x-tts-engine"] == "lfm3-audio"
    assert tts.headers["x-tts-frames"] == "2"
    assert tts.headers["x-tts-rtf"] == "4.000"
    assert tts.headers["x-tts-eos"] == "true"
    assert tts.content[:4] == b"RIFF"
    assert fake_synth.calls[-1]["language"] == "British English"
    assert fake_synth.calls[-1]["gender"] == "male"


def test_backend_generate_request_audio_and_vision_serialization() -> None:
    import numpy as np

    daemon = _MultimodalFakeDaemon(lambda _req, _m: (["ok"], 1.0))
    daemon.start()
    try:
        cli = DaemonClient("127.0.0.1", daemon.port)

        async def _run() -> None:
            aud_vec = np.ones((2, AUDIO_EMBED_DIM), dtype=np.float32).tobytes()
            aud_req = GenerateRequest(
                prompt="<|audio|><|audio|>",
                image_bytes=aud_vec,
                embed_token_id=125031,
                suppress_tokens=(125012, 125015, 125032),
            )
            evs = [ev async for ev in cli.stream_generate(aud_req)]
            assert any(ev.type == "done" for ev in evs)

            vis_vec = np.full((3, AUDIO_EMBED_DIM), 0.5, dtype=np.float32).tobytes()
            vis_req = GenerateRequest(prompt="<|image|><|image|><|image|>", image_bytes=vis_vec)
            evs2 = [ev async for ev in cli.stream_generate(vis_req)]
            assert any(ev.type == "done" for ev in evs2)

        asyncio.run(_run())
        assert len(daemon.requests) == 2
        r0, r1 = daemon.requests
        assert r0["audio_bytes"] == 2 * AUDIO_EMBED_DIM * 4
        assert "image_bytes" not in r0
        assert r0["embed_token_id"] == 125031
        assert r0["suppress_tokens"] == [125012, 125015, 125032]
        assert len(daemon.received_media[0]) == 2 * AUDIO_EMBED_DIM * 4

        assert r1["image_bytes"] == 3 * AUDIO_EMBED_DIM * 4
        assert "audio_bytes" not in r1
        assert "suppress_tokens" not in r1
    finally:
        daemon.stop()


def test_audio_service_asr_and_multi_engine_tts(tmp_path: Path) -> None:
    fake_espeak = tmp_path / "espeak-ng"
    wav_espeak = _make_sine_wav_16k(0.25, 330.0)
    fake_espeak.write_bytes(
        b"#!/usr/bin/env python3\n"
        b"import sys\n"
        + f"sys.stdout.buffer.write({wav_espeak!r})\n".encode("ascii")
    )
    fake_espeak.chmod(0o755)

    wav_native = _make_pcm24_silence_wav(1)

    def _native_hook(text: str, *, speed: float = 1.0, language: str = "American English", gender: str = "female", max_frames: int = 250):
        return wav_native, "lfm3-audio", {"frames": 1, "language": language, "gender": gender}

    svc = AudioService(
        espeak_bin=str(fake_espeak),
        asr_hook=lambda meta: (f"heard:{meta.filename}", "lfm3-audio"),
        asr_engine_name="lfm3-audio",
        native_tts_hook=_native_hook,
        native_tts_default=False,
    )
    assert svc.asr_available is True
    assert svc.native_tts_available is True
    assert svc.tts_engine == "espeak-ng"
    assert svc.tts_engines == ["lfm3-audio", "espeak-ng"]

    async def _run() -> None:
        w1, e1, _, m1 = await svc.synthesize_tts_detailed("hello")
        assert e1 == "espeak-ng" and m1 == {}
        assert w1[:4] == b"RIFF"

        w2, e2, _, m2 = await svc.synthesize_tts_detailed("hello", engine="lfm3-audio", language="French", gender="male")
        assert e2 == "lfm3-audio"
        assert m2["language"] == "French" and m2["gender"] == "male"
        assert w2[:4] == b"RIFF"

        svc.native_tts_default = True
        assert svc.tts_engine == "lfm3-audio"
        _, e3, _, _ = await svc.synthesize_tts_detailed("hello", engine="auto")
        assert e3 == "lfm3-audio"
        _, e4, _, _ = await svc.synthesize_tts_detailed("hello", engine="espeak-ng")
        assert e4 == "espeak-ng"

    asyncio.run(_run())


def test_transcribe_tool_gating_sandbox_and_end_to_end_turn(tmp_path: Path) -> None:
    names_off = [s["function"]["name"] for s in get_tool_schemas(Capabilities(voice_asr=False))]
    names_on = [s["function"]["name"] for s in get_tool_schemas(Capabilities(voice_asr=True))]
    assert "transcribe" not in names_off
    assert "transcribe" in names_on

    ws = tmp_path / "ws"
    ws.mkdir()
    wav_bytes = _make_sine_wav_16k(0.4)
    (ws / "sample.wav").write_bytes(wav_bytes)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(wav_bytes)

    ex = ToolExecutor(WorkspaceSandbox(ws), Capabilities(voice_asr=True), SafetyLimits())
    ok_res = ex.execute("transcribe", {"path": "sample.wav"})
    assert ok_res.ok
    assert json.loads(ok_res.content)["sample_rate"] == 16000
    bad_res = ex.execute("transcribe", {"path": "../outside.wav"})
    assert not bad_res.ok
    assert "Sandbox violation" in (bad_res.error or "")

    # End-to-end tool call inside server session with real AudioClient talking to TestClient sidecar URL via local HTTP server
    import uvicorn
    from src.lfm3_harness.audio_sidecar import create_audio_sidecar_app

    sidecar_port = _free_port()
    sidecar_app = create_audio_sidecar_app(encoder=_FakeEncoder(), synthesizer=_FakeSynthesizer(), default_daemon_port=8091)
    srv_cfg = uvicorn.Config(sidecar_app, host="127.0.0.1", port=sidecar_port, log_level="warning")
    uv_srv = uvicorn.Server(srv_cfg)
    t_side = threading.Thread(target=uv_srv.run, daemon=True)
    t_side.start()
    import time
    for _ in range(50):
        if uv_srv.started:
            break
        time.sleep(0.05)
    assert uv_srv.started

    def _actor_handler(req: Dict[str, Any], _media: bytes) -> Tuple[List[str], float]:
        if "<tool_response>" in req["prompt"]:
            assert "port eight zero nine one" in req["prompt"]
            return ["Transcript verified cleanly."], 1.0
        return ['<tool_call>{"name": "transcribe", "arguments": {"path": "sample.wav"}}</tool_call>'], 1.0

    def _audio_handler(req: Dict[str, Any], _media: bytes) -> Tuple[List[str], float]:
        assert req["embed_token_id"] == 125031
        assert req["audio_bytes"] == 6 * AUDIO_EMBED_DIM * 4
        assert req["suppress_tokens"] == [125012, 125015, 125032]
        return ["The NPU server is ready on port eight zero nine one."], 1.0

    d_actor = _MultimodalFakeDaemon(_actor_handler)
    d_audio = _MultimodalFakeDaemon(_audio_handler)
    d_actor.start()
    d_audio.start()
    try:
        cfg = HarnessConfig(
            workspace_root=ws,
            db_path=tmp_path / "harness.db",
            actor=BackendEndpoint(name="lfm3-1.7b", port=d_actor.port, family="lfm3"),
            audio_daemon=BackendEndpoint(name="lfm3-audio-2b", port=d_audio.port, family="lfm3", supports_audio=True),
            audio_url=f"http://127.0.0.1:{sidecar_port}",
            native_tts_default=False,
        )
        app = create_app(cfg, allow_no_auth=True)
        client = TestClient(app)

        # Check /api/models, /api/health, /api/audio/status, /api/cache/stats & clear
        m = client.get("/api/models").json()
        assert m["audio"]["reachable"] is True
        assert m["capabilities"]["voice_asr"] is True
        assert m["capabilities"]["asr_engine"] == "lfm3-audio"
        assert "lfm3-audio" in m["capabilities"]["tts_engines"]

        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/audio/status").json()["sidecar_reachable"] is True

        cs = client.get("/api/cache/stats").json()
        assert "audio" in cs["models"] and cs["models"]["audio"]["reachable"] is True
        cc = client.post("/api/cache/clear", json={"model": "audio"}).json()
        assert "audio" in cc["results"]
        assert d_audio.cleared_flags == [False]

        # /api/audio/upload -> ASR transcript
        up = client.post("/api/audio/upload?filename=sample.wav", content=wav_bytes)
        assert up.status_code == 200
        uj = up.json()
        assert uj["asr_available"] is True
        assert uj["asr_engine"] == "lfm3-audio"
        assert "eight zero nine one" in uj["transcript"]

        # /api/tts -> native lfm3-audio selectable
        tts_resp = client.post("/api/tts", json={"text": "Ready.", "engine": "lfm3-audio", "language": "American English", "gender": "female"})
        assert tts_resp.status_code == 200
        assert tts_resp.headers["x-tts-engine"] == "lfm3-audio"
        assert tts_resp.headers["x-tts-frames"] == "2"

        # Chat turn invoking workspace transcribe("sample.wav") tool
        sid = client.post("/api/sessions", json={}).json()["session"]["id"]
        chat = client.post(f"/api/sessions/{sid}/messages", json={"content": "Please transcribe sample.wav."})
        assert chat.status_code == 200
        assert "event: tool_result" in chat.text
        assert "Transcript verified cleanly." in chat.text
    finally:
        d_actor.stop()
        d_audio.stop()
        uv_srv.should_exit = True
        t_side.join(timeout=3.0)


def test_audio_synthesizer_tts_session_wire_protocol(tmp_path: Path) -> None:
    """Exercise Lfm3AudioSynthesizer.synthesize_via_daemon end-to-end against _MultimodalFakeDaemon."""
    import numpy as np
    from audio_encoder import Lfm3AudioSynthesizer

    cb_path = tmp_path / "codebook.bin"
    np.full((12 * 2017, AUDIO_EMBED_DIM), 0.01, dtype=np.float32).tofile(cb_path)

    class _FakeSess:
        def __init__(self, mode: str):
            self.mode = mode
            self.calls = 0

        def run(self, _out, feeds: Dict[str, Any]):
            if self.mode == "depth":
                self.calls += 1
                logits = np.full((1, 12, 2017), -10.0, dtype=np.float32)
                # First 2 frames emit normal tokens (<2016); frame 3 emits EOS=2016.
                target_tok = 2016 if self.calls > 6 else 5
                logits[:, :, target_tok] = 10.0
                return [logits]
            frames = int(feeds["audio_codes"].shape[-1])
            return [np.zeros((1, frames * 6, 1282), dtype=np.float32)]

    synth = object.__new__(Lfm3AudioSynthesizer)
    synth.depth_session = _FakeSess("depth")
    synth.codec_session = _FakeSess("codec")
    synth.codebook_table = np.memmap(cb_path, dtype=np.float32, mode="r", shape=(12 * 2017, AUDIO_EMBED_DIM))
    synth.offsets = np.arange(12, dtype=np.int64) * 2017

    daemon = _MultimodalFakeDaemon(lambda _req, _m: ([], 1.0))
    daemon.start()
    try:
        res = synth.synthesize_via_daemon(
            "Test phrase.",
            "127.0.0.1",
            daemon.port,
            max_frames=10,
            temperature=0.0,
            seed=0,
        )
        assert res["eos"] is True
        assert res["num_frames"] == 2
        assert res["daemon_steps"] == 2
        assert res["wav_bytes"][:4] == b"RIFF"
        assert inspect_wav_bytes(res["wav_bytes"]) == (24000, 1, 2, 0.16)
        assert len(daemon.tts_requests) == 1
        assert daemon.tts_requests[0]["audio_out_token_id"] == 125032
    finally:
        daemon.stop()


def test_server_module_is_pure_stdlib_no_heavy_ml_imports() -> None:
    code = (
        "import sys; "
        "import src.lfm3_harness.server, src.lfm3_harness.audio, src.lfm3_harness.backend; "
        "banned = [m for m in ('numpy', 'torch', 'onnxruntime') if m in sys.modules]; "
        "assert not banned, f'Unexpected heavy imports in server process: {banned}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
