"""Host end-to-end test suite for Milestone 19D: FastAPI LAN server, Auth/CSRF, SSE/Stop, Diff Cards, Missions & Audio."""

import asyncio
import http.cookiejar
import io
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import uvicorn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness.audio import AudioService  # noqa: E402
from src.lfm3_harness.config import BackendEndpoint, Capabilities, HarnessConfig, SafetyLimits  # noqa: E402
from src.lfm3_harness.server import create_app  # noqa: E402


def _make_wav_bytes(duration_s: float = 0.25, sample_rate: int = 22050, channels: int = 2) -> bytes:
    buf = io.BytesIO()
    n_frames = max(1, int(round(duration_s * sample_rate)))
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x01\x00" * (n_frames * channels))
    return buf.getvalue()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeDaemonServer:
    """In-process line-JSON TCP fake of lfm3_v2 / lfm2_moe supporting slow streaming, decisions & cache cmds."""

    def __init__(self, handler: Callable[[Dict[str, Any]], Tuple[List[str], float]]):
        self.handler = handler
        self.port = _free_port()
        self.requests: List[Dict[str, Any]] = []
        self.aborted_count = 0
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
            self.requests.append(req)
            cmd = req.get("cmd")
            if cmd == "cache_stats":
                writer.write(
                    (
                        json.dumps(
                            {
                                "type": "cache_stats",
                                "resident_blocks": 12,
                                "pinned_blocks": 4,
                                "disk_bytes_used": 64 * 1024 * 1024,
                                "disk_bytes_quota": 256 * 1024**3,
                                "total_lookups": 8,
                                "total_block_hits": 40,
                                "total_tokens_saved": 2560,
                                "tier0_hits": 5,
                                "tier0_live_tokens": 3000,
                            }
                        )
                        + "\n"
                    ).encode()
                )
                await writer.drain()
                return
            if cmd == "cache_clear":
                inc = bool(req.get("include_pinned", False))
                self.cleared_flags.append(inc)
                writer.write((json.dumps({"type": "cache_cleared", "removed_blocks": 12 if inc else 8, "include_pinned": inc}) + "\n").encode())
                await writer.drain()
                return

            if req.get("return_logits"):
                req_toks = req.get("requested_tokens") or []
                logits = {str(tid): 0.0 for tid in req_toks}
                # Favour contradicts ('C' = token 43) on WriteGate DESTRUCTIVE_WIPE checks
                if "DESTRUCTIVE_WIPE" in (req.get("prompt") or "") and "PROPOSED ACTION" in (req.get("prompt") or "") and 43 in req_toks:
                    logits["43"] = 12.0
                elif len(req_toks) >= 2:
                    logits[str(req_toks[0])] = 4.0
                writer.write(
                    (
                        json.dumps(
                            {
                                "type": "logits",
                                "prefill_ms": 12.5,
                                "cache_status": "HIT_TIER0",
                                "cached_tokens": 64,
                                "logits": logits,
                            }
                        )
                        + "\n"
                    ).encode()
                )
                await writer.drain()
                return

            chunks, delay_s = self.handler(req)
            writer.write(
                (
                    json.dumps(
                        {
                            "type": "prefill_done",
                            "prompt_tokens": 128,
                            "cached_tokens": 64 if req.get("prompt_cache_key") else 0,
                            "cache_status": "HIT_TIER0",
                            "cache_tier": "tier0_ram",
                            "cache_restore_ms": 0.0,
                            "prefill_ms": 10.0,
                        }
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()
            for ch in chunks:
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                writer.write((json.dumps({"type": "token", "text": ch, "thinking": False}) + "\n").encode())
                await writer.drain()
            writer.write(
                (
                    json.dumps(
                        {
                            "type": "done",
                            "prompt_tokens": 128,
                            "cached_tokens": 64,
                            "cache_status": "HIT_TIER0",
                            "cache_tier": "tier0_ram",
                            "cache_restore_ms": 0.0,
                            "prefill_ms": 10.0,
                            "decode_ms": 15.0,
                            "completion_tokens": len(chunks),
                        }
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            self.aborted_count += 1
        finally:
            try:
                writer.close()
            except Exception:
                pass


class _UvicornRunner:
    def __init__(self, app, port: int):
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(cfg)
        self.port = port
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("Uvicorn server failed to start")

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5.0)


def _http_req(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    json_body: Any = None,
    raw_bytes: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    hdrs = dict(headers or {})
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    elif raw_bytes is not None:
        data = raw_bytes
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with opener.open(req, timeout=20) as resp:
            return int(resp.status), {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        return int(err.code), {k.lower(): v for k, v in err.headers.items()}, err.read()


def _read_sse_events(opener: urllib.request.OpenerDirector, url: str, json_body: Optional[Dict[str, Any]] = None) -> Tuple[int, List[Tuple[str, Dict[str, Any]]]]:
    data = None if json_body is None else json.dumps(json_body).encode("utf-8")
    method = "GET" if json_body is None else "POST"
    hdrs = {"Content-Type": "application/json"} if json_body is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    events: List[Tuple[str, Dict[str, Any]]] = []
    with opener.open(req, timeout=20) as resp:
        ev_name = "message"
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event: "):
                ev_name = line[7:].strip()
            elif line.startswith("data: "):
                events.append((ev_name, json.loads(line[6:])))
            elif line == "":
                ev_name = "message"
        return int(resp.status), events


def test_milestone19d_end_to_end_http_sse_auth_diffs_missions_and_audio(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.txt").write_text("Quarterly total is 420.\n", encoding="utf-8")

    # Deterministic scripted NPU fake supporting missions, chat read+write, and slow stream for Stop test.
    def fake_handler(req: Dict[str, Any]) -> Tuple[List[str], float]:
        p = str(req.get("prompt") or "")
        # Mission desk_task branch
        if "inbox/q1_sales.txt" in p and "Phase 1 (plan)" in p:
            if "<tool_response>" not in p:
                return ['<tool_call>\n<function=todo_write>\n<parameter=items>\n[{"id":"1","title":"Read","status":"in_progress"}]\n</parameter>\n</function>\n</tool_call>'], 0.0
            if "verified_deliverables=" in p:
                return ['<tool_call>\n<function=finish>\n<parameter=summary>\nNorth 120 South 85 Total 205 Revenue 6150.\n</parameter>\n<parameter=deliverables>\n["reports/desk_summary.md"]\n</parameter>\n</function>\n</tool_call>'], 0.0
            if "written_files=" in p:
                return ['<tool_call>\n<function=read_file>\n<parameter=path>\nreports/desk_summary.md\n</parameter>\n</function>\n</tool_call>'], 0.0
            if "'inbox/q1_sales.txt'" not in p:
                return ['<tool_call>\n<function=read_file>\n<parameter=path>\ninbox/q1_sales.txt\n</parameter>\n</function>\n</tool_call>'], 0.0
            if "'inbox/q2_sales.txt'" not in p:
                return ['<tool_call>\n<function=read_file>\n<parameter=path>\ninbox/q2_sales.txt\n</parameter>\n</function>\n</tool_call>'], 0.0
            if "'inbox/notes.txt'" not in p:
                return ['<tool_call>\n<function=read_file>\n<parameter=path>\ninbox/notes.txt\n</parameter>\n</function>\n</tool_call>'], 0.0
            return ['<tool_call>\n<function=write_file>\n<parameter=path>\nreports/desk_summary.md\n</parameter>\n<parameter=content>\nNorth: 120 units (3600 USD), South: 85 units (2550 USD), Total Units: 205, Total Revenue: 6150\n</parameter>\n</function>\n</tool_call>'], 0.0

        # Interactive chat branches (inspect latest user turn only, ignoring prior history)
        last_turn = p.rsplit("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
        if "SLOW_ABORT_TEST" in last_turn:
            return ["tok1 ", "tok2 ", "tok3 ", "tok4 ", "tok5 ", "tok6 "], 0.18
        if "DESTRUCTIVE_WIPE" in last_turn:
            if "held_by_write_gate" in p:
                return ["Write was blocked by safety policy."], 0.0
            return ['<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>\nWIPED\n</parameter>\n</function>\n</tool_call>'], 0.0
        if "APPROVE_WRITE_TEST" in last_turn:
            return ['<tool_call>\n<function=write_file>\n<parameter=path>\nreport.txt\n</parameter>\n<parameter=content>\nApproved Value=900\n</parameter>\n</function>\n</tool_call>'], 0.0
        if "DENY_WRITE_TEST" in last_turn:
            return ['<tool_call>\n<function=write_file>\n<parameter=path>\ndeny_me.txt\n</parameter>\n<parameter=content>\nShouldNotExist\n</parameter>\n</function>\n</tool_call>'], 0.0
        if "READ_CHAT_TEST" in last_turn:
            if "<tool_response>" not in p:
                return ['<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>'], 0.0
            return ["Observed quarterly total 420 in notes.txt."], 0.0
        return ["Hello cleanly from harness."], 0.0

    daemon = FakeDaemonServer(fake_handler)
    daemon.start()
    port = _free_port()
    token = "secret-test-token-19d"
    cfg = HarnessConfig(
        workspace_root=ws,
        db_path=tmp_path / "harness.db",
        actor=BackendEndpoint(name="lfm3-1.7b", host="127.0.0.1", port=daemon.port, family="lfm3"),
        auth_token=token,
    )
    stub_wav = _make_wav_bytes(0.1, sample_rate=16000, channels=1)
    audio_svc = AudioService(tts_hook=lambda _txt, _sp: (stub_wav, "espeak-ng"))
    app = create_app(cfg, audio_service=audio_svc)
    srv = _UvicornRunner(app, port)
    srv.start()

    try:
        base = f"http://127.0.0.1:{port}"
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

        # 1. Static SPA assets & healthz are reachable; protected endpoints enforce 401 before login
        for asset in ("/", "/static/app.js", "/static/audio.js", "/static/styles.css", "/healthz"):
            code, _, _ = _http_req(opener, base + asset)
            assert code == 200, f"GET {asset} returned {code}"
        code_unauth, _, _ = _http_req(opener, base + "/api/sessions")
        assert code_unauth == 401

        # 2. CSRF Origin enforcement + Auth Cookie login
        code_csrf, _, _ = _http_req(
            opener,
            base + "/api/auth/login",
            method="POST",
            json_body={"token": token},
            headers={"Origin": "https://evil.example.com"},
        )
        assert code_csrf == 403

        code_login, _, _ = _http_req(
            opener,
            base + "/api/auth/login",
            method="POST",
            json_body={"token": token},
            headers={"Origin": base},
        )
        assert code_login == 200

        # 3. Create session, read-only tool chat turn + EvidenceGate
        code, _, raw = _http_req(opener, base + "/api/sessions", method="POST", json_body={"title": "Test Session"})
        assert code == 200
        sid = json.loads(raw)["session"]["id"]

        code_sse, evs = _read_sse_events(opener, f"{base}/api/sessions/{sid}/messages", {"content": "Please inspect READ_CHAT_TEST"})
        assert code_sse == 200
        ev_types = [t for t, _ in evs]
        assert "route" in ev_types and "tool_call" in ev_types and "tool_result" in ev_types and "evidence_gate" in ev_types and ev_types[-1] == "done"

        # 4. Mid-stream Stop abort + Concurrent 409 check + clean recovery
        stop_evs: List[Tuple[str, Dict[str, Any]]] = []
        conflict_codes: List[int] = []

        def _stream_slow() -> None:
            _c, evs_s = _read_sse_events(opener, f"{base}/api/sessions/{sid}/messages", {"content": "SLOW_ABORT_TEST run"})
            stop_evs.extend(evs_s)

        t_slow = threading.Thread(target=_stream_slow)
        t_slow.start()
        time.sleep(0.25)
        c_busy, _, _ = _http_req(opener, f"{base}/api/sessions/{sid}/messages", method="POST", json_body={"content": "Concurrent"})
        conflict_codes.append(c_busy)
        c_stop, _, _ = _http_req(opener, f"{base}/api/sessions/{sid}/stop", method="POST", json_body={})
        assert c_stop == 200
        t_slow.join(timeout=5.0)
        assert conflict_codes == [409]
        assert any(t == "done" and d.get("status") == "cancelled" for t, d in stop_evs)

        # Follow-up recovery turn works immediately
        _, rec_evs = _read_sse_events(opener, f"{base}/api/sessions/{sid}/messages", {"content": "Ping"})
        assert rec_evs[-1][0] == "done" and rec_evs[-1][1].get("status") == "completed"

        # 5. Staged Diff Approval (Allow once -> disk written) & Deny (disk untouched) & WriteGate hold
        _, app_evs = _read_sse_events(opener, f"{base}/api/sessions/{sid}/messages", {"content": "APPROVE_WRITE_TEST"})
        cards = [d["action"] for t, d in app_evs if t == "approval_card"]
        assert len(cards) == 1
        act_id = cards[0]["id"]
        c_allow, _, raw_allow = _http_req(opener, f"{base}/api/actions/{act_id}/allow", method="POST", json_body={})
        assert c_allow == 200 and json.loads(raw_allow)["status"] == "applied"
        assert (ws / "report.txt").read_text(encoding="utf-8") == "Approved Value=900"

        _, deny_evs = _read_sse_events(opener, f"{base}/api/sessions/{sid}/messages", {"content": "DENY_WRITE_TEST"})
        deny_id = [d["action"]["id"] for t, d in deny_evs if t == "approval_card"][0]
        c_deny, _, _ = _http_req(opener, f"{base}/api/actions/{deny_id}/deny", method="POST", json_body={})
        assert c_deny == 200
        assert not (ws / "deny_me.txt").exists()

        _, hold_evs = _read_sse_events(opener, f"{base}/api/sessions/{sid}/messages", {"content": "DESTRUCTIVE_WIPE please"})
        assert any(t == "write_gate_hold" for t, _ in hold_evs)
        assert not any(t == "approval_card" for t, _ in hold_evs)

        # 6. Verified Mission (desk_task) over HTTP + SSE stream
        c_mis, _, raw_mis = _http_req(
            opener,
            base + "/api/missions/run",
            method="POST",
            json_body={"template": "desk_task", "wait": True},
        )
        assert c_mis == 200
        mis_obj = json.loads(raw_mis)
        assert mis_obj["mission"]["status"] == "passed", f"mission failed: {mis_obj}"
        _, m_evs = _read_sse_events(opener, f"{base}/api/missions/{mis_obj['run_id']}/stream")
        assert any(t == "mission_done" and d.get("status") == "passed" for t, d in m_evs)

        # 7. Cache HUD + System Health (RSS < 100 MiB)
        c_cs, _, raw_cs = _http_req(opener, base + "/api/cache/stats")
        assert c_cs == 200 and json.loads(raw_cs)["models"]["lfm3-1.7b"]["resident_blocks"] == 12
        c_cc, _, _ = _http_req(opener, base + "/api/cache/clear", method="POST", json_body={"include_pinned": False})
        assert c_cc == 200 and daemon.cleared_flags == [False]
        c_sys, _, raw_sys = _http_req(opener, base + "/api/system/health")
        sys_obj = json.loads(raw_sys)
        assert c_sys == 200 and 0 < sys_obj["rss_mib"] < 100.0

        # 8. Audio Upload (22.05kHz stereo -> ffmpeg normalized to 16kHz mono PCM) & TTS WAV
        stereo_wav = _make_wav_bytes(0.25, sample_rate=22050, channels=2)
        c_aud, _, raw_aud = _http_req(
            opener,
            base + "/api/audio/upload?filename=sample.wav",
            method="POST",
            raw_bytes=stereo_wav,
            headers={"Content-Type": "audio/wav"},
        )
        assert c_aud == 200
        aud_meta = json.loads(raw_aud)["metadata"]
        assert aud_meta["sample_rate"] == 16000 and aud_meta["channels"] == 1 and aud_meta["duration_s"] > 0.2

        c_bad_aud, _, _ = _http_req(
            opener,
            base + "/api/audio/upload?filename=bad.exe",
            method="POST",
            raw_bytes=b"MZ1234",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert c_bad_aud == 400

        c_tts, hdr_tts, raw_tts = _http_req(opener, base + "/api/tts", method="POST", json_body={"text": "<think>secret</think>Hello world!"})
        assert c_tts == 200 and raw_tts[:4] == b"RIFF" and hdr_tts.get("x-tts-engine") == "espeak-ng"
    finally:
        srv.stop()
        daemon.stop()
