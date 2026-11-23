"""Host end-to-end test suite for Milestone 19E: Triple-daemon d1 isolation, dual-actor switching & Vision sidecar."""

import asyncio
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import uvicorn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness import (  # noqa: E402
    BackendEndpoint,
    Capabilities,
    HarnessConfig,
    SafetyLimits,
    format_vision_placeholder,
    get_tool_schemas,
    inspect_image_bytes,
)
from src.lfm3_harness.server import create_app  # noqa: E402


def _make_png(width: int = 32, height: int = 24, rgb: Tuple[int, int, int] = (220, 30, 30)) -> bytes:
    raw_scanlines = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw_scanlines)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _make_gif(width: int = 16, height: int = 12) -> bytes:
    return b"GIF89a" + struct.pack("<HHBBB", width, height, 0x80, 0, 0) + b"\x00\x00\x00\xff\xff\xff;"


def _make_bmp(width: int = 20, height: int = 10) -> bytes:
    return b"BM" + struct.pack("<IHHI", 54, 0, 0, 54) + struct.pack("<IiiHHIIIIII", 40, width, height, 1, 24, 0, 0, 0, 0, 0, 0)


def _make_minimal_jpeg(width: int = 40, height: int = 30) -> bytes:
    sof0_payload = struct.pack(">BHHBBBB", 8, height, width, 1, 1, 0x11, 0)
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload + b"\xff\xd9"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _VisionAwareFakeDaemon:
    """In-process TCP fake of lfm3_v2 / lfm2_moe supporting binary image_bytes trailing payloads."""

    def __init__(self, handler: Callable[[Dict[str, Any], bytes], Tuple[List[str], float]]):
        self.handler = handler
        self.port = _free_port()
        self.requests: List[Dict[str, Any]] = []
        self.received_images: List[bytes] = []
        self.cleared_flags: List[bool] = []
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
        try:
            await self._stop_ev.wait()
        finally:
            srv.close()
            await srv.wait_closed()

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            img_len = int(req.get("image_bytes") or 0)
            img_blob = b""
            if img_len > 0:
                img_blob = await reader.readexactly(img_len)
                self.received_images.append(img_blob)
            self.requests.append(req)

            cmd = req.get("cmd")
            if cmd == "cache_stats":
                writer.write((json.dumps({
                    "type": "cache_stats",
                    "resident_blocks": 8,
                    "pinned_blocks": 2,
                    "disk_bytes_used": 16 * 1024 * 1024,
                    "disk_bytes_quota": 256 * 1024**3,
                    "total_lookups": 4,
                    "total_block_hits": 16,
                    "total_tokens_saved": 1024,
                    "tier0_hits": 3,
                    "tier0_live_tokens": 512,
                }) + "\n").encode())
                await writer.drain()
                return
            if cmd == "cache_clear":
                inc = bool(req.get("include_pinned", False))
                self.cleared_flags.append(inc)
                writer.write((json.dumps({"type": "cache_cleared", "removed_blocks": 8 if inc else 6, "include_pinned": inc}) + "\n").encode())
                await writer.drain()
                return

            if req.get("return_logits"):
                req_toks = req.get("requested_tokens") or []
                logits = {str(tid): 0.0 for tid in req_toks}
                if req_toks:
                    logits[str(req_toks[0])] = 5.0
                writer.write((json.dumps({
                    "type": "logits",
                    "prefill_ms": 9.5,
                    "cache_status": "HIT_TIER0",
                    "cached_tokens": 64,
                    "logits": logits,
                }) + "\n").encode())
                await writer.drain()
                return

            chunks, delay_s = self.handler(req, img_blob)
            writer.write((json.dumps({
                "type": "prefill_done",
                "prompt_tokens": 128,
                "cached_tokens": 64 if req.get("prompt_cache_key") else 0,
                "cache_status": "HIT_TIER0",
                "cache_tier": "tier0_ram",
                "cache_restore_ms": 0.0,
                "prefill_ms": 8.0,
            }) + "\n").encode())
            await writer.drain()
            for ch in chunks:
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                writer.write((json.dumps({"type": "token", "text": ch, "thinking": False}) + "\n").encode())
                await writer.drain()
            writer.write((json.dumps({
                "type": "done",
                "prompt_tokens": 128,
                "cached_tokens": 64,
                "cache_status": "HIT_TIER0",
                "cache_tier": "tier0_ram",
                "cache_restore_ms": 0.0,
                "prefill_ms": 8.0,
                "decode_ms": 12.0,
                "completion_tokens": len(chunks),
            }) + "\n").encode())
            await writer.drain()
        finally:
            try:
                writer.close()
            except Exception:
                pass


class _UvicornRunner:
    def __init__(self, app, port: int):
        self.config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, port: int) -> None:
        self.thread.start()
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(f"uvicorn server did not start on {port}")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


def _start_sidecar_subprocess(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.lfm3_harness.vision_sidecar", "--bind", "127.0.0.1", "--port", str(port)],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vision_sidecar exited early rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"vision_sidecar failed to bind port {port}")


def _http_json(url: str, method: str = "GET", body: Optional[dict] = None) -> Tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _read_sse(url: str, body: dict) -> List[Tuple[str, dict]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: List[Tuple[str, dict]] = []
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        cur_ev = "message"
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("event: "):
                cur_ev = line[7:].strip()
            elif line.startswith("data: "):
                events.append((cur_ev, json.loads(line[6:])))
    return events


def test_inspect_image_bytes_and_metadata():
    assert inspect_image_bytes(_make_png(32, 24), "a.png") == ("png", 32, 24)
    assert inspect_image_bytes(_make_gif(16, 12), "a.gif") == ("gif", 16, 12)
    assert inspect_image_bytes(_make_bmp(20, 10), "a.bmp") == ("bmp", 20, 10)
    assert inspect_image_bytes(_make_minimal_jpeg(40, 30), "a.jpg") == ("jpeg", 40, 30)
    assert format_vision_placeholder(3) == "<|vision_start|><|image|><|image|><|image|><|vision_end|>"

    with pytest.raises(ValueError):
        inspect_image_bytes(b"", "empty.png")
    with pytest.raises(ValueError):
        inspect_image_bytes(b"not an image", "bad.png")
    with pytest.raises(ValueError):
        inspect_image_bytes(_make_png(4, 4), "bad.txt")


def test_vision_capability_gating_off(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    assert "look" not in {s["function"]["name"] for s in get_tool_schemas(Capabilities(vision=False))}
    assert "look" in {s["function"]["name"] for s in get_tool_schemas(Capabilities(vision=True))}

    daemon = _VisionAwareFakeDaemon(lambda _req, _img: (["ok"], 0.0))
    daemon.start()
    srv = None
    try:
        cfg = HarnessConfig(
            workspace_root=ws,
            db_path=tmp_path / "h.db",
            actor=BackendEndpoint(name="lfm3-1.7b", port=daemon.port, family="lfm3"),
        )
        port = _free_port()
        app = create_app(cfg, allow_no_auth=True)
        srv = _UvicornRunner(app, port)
        srv.start(port)

        code, mod = _http_json(f"http://127.0.0.1:{port}/api/models")
        assert code == 200
        assert mod["capabilities"]["vision"] is False
        assert mod["d1"]["separate_daemon"] is False

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/images/upload?filename=shot.png",
            data=_make_png(16, 16),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5.0)
        assert exc_info.value.code == 503
    finally:
        if srv:
            srv.stop()
        daemon.stop()


def test_triple_daemon_d1_isolation_actor_switching_and_vision_e2e(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    shot_path = ws / "shot.png"
    shot_path.write_bytes(_make_png(32, 24))

    def lfm3_handler(req: Dict[str, Any], img: bytes) -> Tuple[List[str], float]:
        prompt = req.get("prompt") or ""
        last_user = prompt.rsplit("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
        if "<|image|>" in last_user and len(img) > 0:
            if "What shape is in" in last_user:
                return ["A bright red rectangle filling the 32x24 frame."], 0.0
            return ["Vision upload seen cleanly."], 0.0
        return ["LFM3 text reply ok."], 0.0

    def moe_handler(req: Dict[str, Any], _img: bytes) -> Tuple[List[str], float]:
        prompt = req.get("prompt") or ""
        after_last_user = prompt.rsplit("<|im_start|>user\n", 1)[-1]
        if "<|im_start|>tool\n" in after_last_user or "<tool_response>" in after_last_user:
            if "Sandbox violation" in after_last_user:
                return ["Look sandbox blocked cleanly."], 0.0
            return ["MoE saw shot.png: bright red rectangle."], 0.0
        if "escape_look_test" in after_last_user:
            return ["<|tool_call_start|>[look(path='../escape.png', question='Read it')]<|tool_call_end|>"], 0.0
        if "trigger_look_test" in after_last_user:
            return ["<|tool_call_start|>[look(path='shot.png', question='What shape is in shot.png?')]<|tool_call_end|>"], 0.0
        return ["MoE text reply ok."], 0.0

    d1_daemon = _VisionAwareFakeDaemon(lambda _r, _i: (["unused"], 0.0))
    lfm3_daemon = _VisionAwareFakeDaemon(lfm3_handler)
    moe_daemon = _VisionAwareFakeDaemon(moe_handler)
    d1_daemon.start()
    lfm3_daemon.start()
    moe_daemon.start()

    sidecar_port = _free_port()
    sidecar_proc = _start_sidecar_subprocess(sidecar_port)

    harness_srv = None
    try:
        cfg = HarnessConfig(
            workspace_root=ws,
            db_path=tmp_path / "harness.db",
            actor=BackendEndpoint(name="lfm3-1.7b", port=lfm3_daemon.port, family="lfm3", supports_thinking=True, supports_vision=True),
            secondary_actor=BackendEndpoint(name="lfm2.5-8b-a1b", port=moe_daemon.port, family="lfm2", supports_thinking=False),
            d1=BackendEndpoint(name="d1-2b", port=d1_daemon.port, family="lfm3", supports_thinking=True),
            vision_url=f"http://127.0.0.1:{sidecar_port}",
            capabilities=Capabilities(vision=True),
            limits=SafetyLimits(),
        )
        port = _free_port()
        harness_srv = _UvicornRunner(create_app(cfg, allow_no_auth=True), port)
        harness_srv.start(port)
        base = f"http://127.0.0.1:{port}"

        # 1. Verify /api/models reports dual actors, separate d1, and live vision_encoder_q8.onnx sidecar.
        code, models_res = _http_json(f"{base}/api/models")
        assert code == 200
        assert [m["name"] for m in models_res["models"]] == ["lfm3-1.7b", "lfm2.5-8b-a1b"]
        assert models_res["d1"]["separate_daemon"] is True
        assert models_res["d1"]["reachable"] is True
        assert models_res["d1"]["port"] == d1_daemon.port
        assert models_res["capabilities"]["vision"] is True
        assert models_res["capabilities"]["vision_reachable"] is True
        assert "vision_encoder" in models_res["capabilities"]["vision_model"]

        # 2. Create session and switch seamlessly between lfm3-1.7b and lfm2.5-8b-a1b.
        _, created = _http_json(f"{base}/api/sessions", method="POST", body={"title": "Dual Actor + Vision"})
        sid = created["session"]["id"]

        evs1 = _read_sse(f"{base}/api/sessions/{sid}/messages", {"content": "Please analyze how this module is organized cleanly."})
        assert any(e == "done" and d["status"] == "completed" for e, d in evs1)
        assert lfm3_daemon.requests[-1]["cache"] is True

        # Switch session actor to lfm2.5-8b-a1b
        code, patched = _http_json(f"{base}/api/sessions/{sid}", method="PATCH", body={"model_name": "lfm2.5-8b-a1b"})
        assert code == 200 and patched["session"]["model_name"] == "lfm2.5-8b-a1b"

        evs2 = _read_sse(f"{base}/api/sessions/{sid}/messages", {"content": "Please summarize the key design tradeoffs cleanly."})
        assert any(e == "token" and "MoE text reply ok." in d["text"] for e, d in evs2)

        # Zero return_logits ever hit either actor daemon (100% isolated on d1_daemon).
        assert sum(1 for r in lfm3_daemon.requests if r.get("return_logits")) == 0
        assert sum(1 for r in moe_daemon.requests if r.get("return_logits")) == 0
        assert sum(1 for r in d1_daemon.requests if r.get("return_logits")) >= 2

        # Cache clear supports target="d1"
        c_code, c_d1 = _http_json(f"{base}/api/cache/clear", method="POST", body={"model": "d1", "include_pinned": False})
        assert c_code == 200 and "d1" in c_d1["results"] and d1_daemon.cleared_flags == [False]

        # 3. Upload PNG to real ONNX sidecar -> send with lfm3-1.7b, reject on lfm2.5-8b-a1b.
        up_req = urllib.request.Request(
            f"{base}/api/images/upload?filename=shot.png",
            data=_make_png(32, 24),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(up_req, timeout=15.0) as up_resp:
            up_data = json.loads(up_resp.read().decode("utf-8"))
        assert up_data["ok"] is True
        img_id = up_data["image_id"]
        n_tok = up_data["metadata"]["num_tokens"]
        assert up_data["metadata"]["width"] == 32 and up_data["metadata"]["height"] == 24
        assert n_tok == 64  # 32x24 upscaled to min_dim=256 => 8x8 merged grid tokens

        bad_code, bad_err = _http_json(
            f"{base}/api/sessions/{sid}/messages",
            method="POST",
            body={"content": "What is here?", "model_name": "lfm2.5-8b-a1b", "image_id": img_id},
        )
        assert bad_code == 400 and "switch to" in bad_err["detail"]

        vis_evs = _read_sse(
            f"{base}/api/sessions/{sid}/messages",
            {"content": "Describe attached image please.", "model_name": "lfm3-1.7b", "image_id": img_id},
        )
        assert any(e == "token" and "Vision upload seen cleanly." in d["text"] for e, d in vis_evs), f"vis_evs={vis_evs}"
        assert len(lfm3_daemon.received_images) == 1
        assert len(lfm3_daemon.received_images[0]) == n_tok * 2048 * 4
        v0 = struct.unpack("<f", lfm3_daemon.received_images[0][:4])[0]
        assert math.isfinite(v0)
        assert format_vision_placeholder(n_tok) in lfm3_daemon.requests[-1]["prompt"]

        # Next turn does not carry orphan <|image|> tags when image_bytes=0
        _read_sse(f"{base}/api/sessions/{sid}/messages", {"content": "Followup question?", "model_name": "lfm3-1.7b"})
        assert "<|image|>" not in lfm3_daemon.requests[-1]["prompt"]

        # 4. Workspace look() tool from lfm2.5-8b-a1b delegates visual pass to lfm3-1.7b + rejects path escape.
        look_evs = _read_sse(
            f"{base}/api/sessions/{sid}/messages",
            {"content": "Run trigger_look_test on workspace file shot.png", "model_name": "lfm2.5-8b-a1b"},
        )
        tool_res_ev = [d for e, d in look_evs if e == "tool_result"]
        assert len(tool_res_ev) == 1 and tool_res_ev[0]["ok"] is True
        parsed_res = json.loads(tool_res_ev[0]["result"])
        assert parsed_res["width"] == 32 and parsed_res["height"] == 24
        assert parsed_res["visual_tokens"] == n_tok
        assert "bright red rectangle" in parsed_res["description"]
        assert len(lfm3_daemon.received_images) == 2

        esc_evs = _read_sse(
            f"{base}/api/sessions/{sid}/messages",
            {"content": "Run escape_look_test outside workspace", "model_name": "lfm2.5-8b-a1b"},
        )
        esc_tool_res = [d for e, d in esc_evs if e == "tool_result"]
        assert len(esc_tool_res) == 1 and esc_tool_res[0]["ok"] is False
        assert "Sandbox violation" in esc_tool_res[0]["result"]
    finally:
        if harness_srv:
            harness_srv.stop()
        sidecar_proc.terminate()
        try:
            sidecar_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            sidecar_proc.kill()
        d1_daemon.stop()
        lfm3_daemon.stop()
        moe_daemon.stop()


def test_append_only_harness_state_preserves_tier0_prefix_and_single_tokenizer_load(tmp_path: Path, monkeypatch):
    from src.lfm3_harness import server as srv_mod

    ws = tmp_path / "ws"
    ws.mkdir()
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps({"model": {"vocab": {"yes": 11683, "no": 2243}}, "added_tokens": []}))

    orig_from_file = srv_mod.TokenizerLookup.from_file
    load_calls: List[str] = []

    def _counting_from_file(cls, p):
        load_calls.append(str(p))
        return orig_from_file(p)

    monkeypatch.setattr(srv_mod.TokenizerLookup, "from_file", classmethod(_counting_from_file))

    d1_daemon = _VisionAwareFakeDaemon(lambda _r, _i: (["unused"], 0.0))
    lfm3_daemon = _VisionAwareFakeDaemon(lambda _r, _i: (["Acknowledged cleanly."], 0.0))
    moe_daemon = _VisionAwareFakeDaemon(lambda _r, _i: (["unused"], 0.0))
    d1_daemon.start()
    lfm3_daemon.start()
    moe_daemon.start()

    harness_srv = None
    try:
        cfg = HarnessConfig(
            workspace_root=ws,
            db_path=tmp_path / "h.db",
            actor=BackendEndpoint(name="lfm3-1.7b", port=lfm3_daemon.port, family="lfm3", tokenizer_path=str(tok_path)),
            secondary_actor=BackendEndpoint(name="lfm2.5-8b-a1b", port=moe_daemon.port, family="lfm2", supports_thinking=False, tokenizer_path=str(tok_path)),
            d1=BackendEndpoint(name="d1-2b", port=d1_daemon.port, family="lfm3", tokenizer_path=str(tok_path)),
        )
        app = create_app(cfg, allow_no_auth=True)
        # Separate d1 loads tokenizer once; primary and secondary actors skip duplicate tokenizer.json loads.
        assert len(load_calls) == 1

        port = _free_port()
        harness_srv = _UvicornRunner(app, port)
        harness_srv.start(port)
        base = f"http://127.0.0.1:{port}"

        _, created = _http_json(f"{base}/api/sessions", method="POST", body={"title": "Tier0 Append Prefix"})
        sid = created["session"]["id"]

        evs1 = _read_sse(f"{base}/api/sessions/{sid}/messages", {"content": "First chat turn please remember ALPHA-42."})
        assert any(e == "done" and d["status"] == "completed" for e, d in evs1)
        prompt_t1 = lfm3_daemon.requests[-1]["prompt"]

        evs2 = _read_sse(f"{base}/api/sessions/{sid}/messages", {"content": "Second chat turn what was the code?"})
        assert any(e == "done" and d["status"] == "completed" for e, d in evs2)
        prompt_t2 = lfm3_daemon.requests[-1]["prompt"]

        assert prompt_t2.startswith(prompt_t1), (
            f"Turn 2 prompt must extend Turn 1 prompt byte-for-byte for HIT_TIER0!\nT1_tail={prompt_t1[-120:]!r}\nT2_at_T1={prompt_t2[len(prompt_t1)-120:len(prompt_t1)+80]!r}"
        )

        # Persisted SQLite messages shown in the Web UI stay free of <harness_state> suffix notes.
        _, sess_detail = _http_json(f"{base}/api/sessions/{sid}")
        user_msgs = [m for m in sess_detail["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "First chat turn please remember ALPHA-42."
        assert "<harness_state>" not in user_msgs[0]["content"]
    finally:
        if harness_srv:
            harness_srv.stop()
        d1_daemon.stop()
        lfm3_daemon.stop()
        moe_daemon.stop()


def test_text_only_lfm3_delegates_vision_and_look_to_d1_daemon(tmp_path: Path):
    class _StubVisionClient:
        async def health(self) -> Dict[str, Any]:
            return {"status": "ok", "model": "stub_vision_q8.onnx"}

        async def encode(self, raw_bytes: bytes, *, filename: str = "image.png", max_dim: int = 512) -> Tuple[bytes, int, Dict[str, Any]]:
            # 4 tokens x 2048 FP32
            emb = struct.pack("<f", 0.25) * (4 * 2048)
            return emb, 4, {"num_tokens": 4, "width": 32, "height": 24, "format": "png", "encode_ms": 1.2, "model": "stub_vision_q8.onnx"}

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "shape.png").write_bytes(_make_png(32, 24))

    def d1_handler(req: Dict[str, Any], img: bytes) -> Tuple[List[str], float]:
        prompt = req.get("prompt") or ""
        if img:
            assert "<|image|>" in prompt
            return ["A bold red circle on a white background."], 0.0
        return ["unused"], 0.0

    def lfm3_handler(req: Dict[str, Any], img: bytes) -> Tuple[List[str], float]:
        assert not img, "Text-only lfm3 actor must never receive raw vision embeddings when delegated to d1-2b"
        prompt = req.get("prompt") or ""
        after_last_user = prompt.rsplit("<|im_start|>user\n", 1)[-1]
        if "<|im_start|>tool\n" in after_last_user or "<tool_response>" in after_last_user:
            return ["Workspace look saw a bold red circle."], 0.0
        if "trigger_look_file" in after_last_user:
            return ["<tool_call>\n<function=look>\n<parameter=path>\nshape.png\n</parameter>\n<parameter=question>\nWhat shape?\n</parameter>\n</function>\n</tool_call>"], 0.0
        assert "d1-2b vision: A bold red circle on a white background." in after_last_user
        return ["The attached image shows a red circle."], 0.0

    d1_daemon = _VisionAwareFakeDaemon(d1_handler)
    lfm3_daemon = _VisionAwareFakeDaemon(lfm3_handler)
    d1_daemon.start()
    lfm3_daemon.start()

    harness_srv = None
    try:
        cfg = HarnessConfig(
            workspace_root=ws,
            db_path=tmp_path / "h.db",
            actor=BackendEndpoint(name="lfm3-1.7b", port=lfm3_daemon.port, family="lfm3", supports_thinking=True, supports_vision=False),
            d1=BackendEndpoint(name="d1-2b", port=d1_daemon.port, family="lfm3", supports_thinking=True),
            vision_url="http://127.0.0.1:18095",
            capabilities=Capabilities(vision=True),
        )
        port = _free_port()
        harness_srv = _UvicornRunner(create_app(cfg, vision_client=_StubVisionClient(), allow_no_auth=True), port)
        harness_srv.start(port)
        base = f"http://127.0.0.1:{port}"

        _, created = _http_json(f"{base}/api/sessions", method="POST", body={"title": "Delegated Vision"})
        sid = created["session"]["id"]

        up_req = urllib.request.Request(
            f"{base}/api/images/upload?filename=shape.png",
            data=_make_png(32, 24),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(up_req, timeout=5.0) as resp:
            img_id = json.loads(resp.read().decode("utf-8"))["image_id"]

        # 1. Direct +Img delegates vision decoding to d1-2b, sends 0 image_bytes to lfm3-1.7b
        vis_evs = _read_sse(
            f"{base}/api/sessions/{sid}/messages",
            {"content": "Describe this attached image.", "model_name": "lfm3-1.7b", "image_id": img_id},
        )
        vis_dec = [d for e, d in vis_evs if e == "vision_decode"]
        assert len(vis_dec) == 1 and vis_dec[0]["vision_actor"] == "d1-2b"
        assert "red circle" in vis_dec[0]["observation"].lower()
        assert len(d1_daemon.received_images) == 1
        assert len(lfm3_daemon.received_images) == 0
        assert any(e == "token" and "red circle" in d["text"].lower() for e, d in vis_evs)

        # 2. Workspace look() delegates to d1-2b and returns vision_actor='d1-2b'
        look_evs = _read_sse(
            f"{base}/api/sessions/{sid}/messages",
            {"content": "Please run trigger_look_file on shape.png.", "model_name": "lfm3-1.7b"},
        )
        look_res_evs = [d for e, d in look_evs if e == "tool_result" and d.get("name") == "look"]
        assert len(look_res_evs) == 1 and look_res_evs[0]["ok"] is True
        look_payload = json.loads(look_res_evs[0]["result"])
        assert look_payload["vision_actor"] == "d1-2b"
        assert "red circle" in look_payload["description"].lower()
        assert len(d1_daemon.received_images) == 2
        assert len(lfm3_daemon.received_images) == 0
    finally:
        if harness_srv:
            harness_srv.stop()
        d1_daemon.stop()
        lfm3_daemon.stop()
