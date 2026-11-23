#!/usr/bin/env python3
"""Live LAN verification suite for Milestone 19D against roma FastAPI harness + NPU daemon."""

import argparse
import http.cookiejar
import io
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _make_wav_bytes(duration_s: float = 0.25, sample_rate: int = 22050, channels: int = 2) -> bytes:
    buf = io.BytesIO()
    n_frames = max(1, int(round(duration_s * sample_rate)))
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x01\x00" * (n_frames * channels))
    return buf.getvalue()


def _http_req(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    json_body: Any = None,
    raw_bytes: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
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
        with opener.open(req, timeout=timeout_s) as resp:
            return int(resp.status), {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        return int(err.code), {k.lower(): v for k, v in err.headers.items()}, err.read()


def _read_sse_events(
    opener: urllib.request.OpenerDirector,
    url: str,
    json_body: Optional[Dict[str, Any]] = None,
    *,
    on_event=None,
    timeout_s: float = 180.0,
) -> Tuple[int, List[Tuple[str, Dict[str, Any]]]]:
    data = None if json_body is None else json.dumps(json_body).encode("utf-8")
    method = "GET" if json_body is None else "POST"
    hdrs = {"Content-Type": "application/json"} if json_body is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    events: List[Tuple[str, Dict[str, Any]]] = []
    with opener.open(req, timeout=timeout_s) as resp:
        ev_name = "message"
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event: "):
                ev_name = line[7:].strip()
            elif line.startswith("data: "):
                obj = json.loads(line[6:])
                events.append((ev_name, obj))
                if on_event is not None:
                    on_event(ev_name, obj)
            elif line == "":
                ev_name = "message"
        return int(resp.status), events


def run_suite(base_url: str, token: str, model_name: str, verify_tls: bool = False) -> Dict[str, Any]:
    handlers: List[Any] = [urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())]
    if base_url.startswith("https://") and not verify_tls:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    opener = urllib.request.build_opener(*handlers)

    print(f"=== Milestone 19D Live LAN Suite against {base_url} (model={model_name}) ===", flush=True)

    # 1. Static SPA assets & auth
    for asset in ("/", "/static/app.js", "/static/audio.js", "/static/styles.css", "/healthz"):
        c, _, _ = _http_req(opener, base_url + asset)
        assert c == 200, f"GET {asset} failed with {c}"
    c_unauth, _, _ = _http_req(opener, base_url + "/api/sessions")
    assert c_unauth == 401, f"Expected 401 unauth, got {c_unauth}"
    c_csrf, _, _ = _http_req(
        opener,
        base_url + "/api/auth/login",
        method="POST",
        json_body={"token": token},
        headers={"Origin": "https://evil.example.com"},
    )
    assert c_csrf == 403, f"Expected 403 CSRF, got {c_csrf}"
    c_login, _, _ = _http_req(opener, base_url + "/api/auth/login", method="POST", json_body={"token": token}, headers={"Origin": base_url})
    assert c_login == 200, f"Login failed: {c_login}"
    print("[1/7] SPA assets, 401 Auth, 403 CSRF, HttpOnly Cookie login PASS", flush=True)

    # 2. Create session & live streaming SSE chat
    c, _, raw_sess = _http_req(opener, base_url + "/api/sessions", method="POST", json_body={"title": "Live 19D Test", "model_name": model_name})
    assert c == 200
    sid = json.loads(raw_sess)["session"]["id"]

    _, evs_chat = _read_sse_events(opener, f"{base_url}/api/sessions/{sid}/messages", {"content": "Reply briefly: what is 2+2?", "model_name": model_name})
    ev_types = [t for t, _ in evs_chat]
    assert "route" in ev_types and "prefill_done" in ev_types and "done" in ev_types
    tok_txt = "".join(d.get("text", "") for t, d in evs_chat if t == "token")
    print(f"[2/7] Live SSE streaming PASS ({len(evs_chat)} events, text={tok_txt[:80]!r})", flush=True)

    # 3. Mid-stream Stop abort + clean recovery
    stop_evs: List[Tuple[str, Dict[str, Any]]] = []
    seen_tok = threading.Event()

    def _on_ev(name: str, _d: Dict[str, Any]) -> None:
        if name == "token":
            seen_tok.set()

    def _slow_stream() -> None:
        _, e = _read_sse_events(
            opener,
            f"{base_url}/api/sessions/{sid}/messages",
            {"content": "Count from 1 to 200 separated by newlines.", "model_name": model_name, "max_tokens": 256},
            on_event=_on_ev,
        )
        stop_evs.extend(e)

    th = threading.Thread(target=_slow_stream)
    th.start()
    assert seen_tok.wait(timeout=45.0), "Timed out waiting for first token before Stop"
    c_stop, _, _ = _http_req(opener, f"{base_url}/api/sessions/{sid}/stop", method="POST", json_body={})
    assert c_stop == 200
    th.join(timeout=20.0)
    assert any(t == "done" and d.get("status") == "cancelled" for t, d in stop_evs), f"Expected cancelled status, got {stop_evs[-2:]}"

    _, rec_evs = _read_sse_events(opener, f"{base_url}/api/sessions/{sid}/messages", {"content": "Reply OK only.", "model_name": model_name, "max_tokens": 24})
    assert rec_evs[-1][0] == "done" and rec_evs[-1][1].get("status") == "completed"
    print("[3/7] Live Stop abort + immediate clean recovery PASS", flush=True)

    # 4. Staged Write Diff Card: Allow once & Deny
    _, w1_evs = _read_sse_events(
        opener,
        f"{base_url}/api/sessions/{sid}/messages",
        {
            "content": "Use the write_file tool to create file allowed_note.txt with content: Alpha 42 Beta",
            "model_name": model_name,
            "max_tokens": 256,
        },
    )
    cards1 = [d["action"] for t, d in w1_evs if t == "approval_card"]
    assert len(cards1) == 1, f"Expected 1 staged approval card, got {w1_evs}"
    c_allow, _, raw_allow = _http_req(opener, f"{base_url}/api/actions/{cards1[0]['id']}/allow", method="POST", json_body={})
    assert c_allow == 200 and json.loads(raw_allow)["status"] == "applied"
    allow_rel = cards1[0]["rel_path"]

    _, w2_evs = _read_sse_events(
        opener,
        f"{base_url}/api/sessions/{sid}/messages",
        {
            "content": "Use the write_file tool to create file denied_note.txt with content: SecretBlocked",
            "model_name": model_name,
            "max_tokens": 256,
        },
    )
    cards2 = [d["action"] for t, d in w2_evs if t == "approval_card"]
    assert len(cards2) == 1, f"Expected 1 staged approval card for deny test, got {w2_evs}"
    c_deny, _, raw_deny = _http_req(opener, f"{base_url}/api/actions/{cards2[0]['id']}/deny", method="POST", json_body={})
    assert c_deny == 200 and json.loads(raw_deny)["status"] == "denied"
    deny_rel = cards2[0]["rel_path"]
    print(f"[4/7] Staged Diff Approval allow={allow_rel} + deny={deny_rel} PASS", flush=True)

    # 5. Live Verified Task (desk_task) over HTTP + SSE
    c_mrun, _, raw_mrun = _http_req(opener, base_url + "/api/missions/run", method="POST", json_body={"template": "desk_task", "model_name": model_name})
    assert c_mrun == 200
    run_id = json.loads(raw_mrun)["run_id"]
    _, m_evs = _read_sse_events(opener, f"{base_url}/api/missions/{run_id}/stream", timeout_s=300.0)
    done_ev = [d for t, d in m_evs if t == "mission_done"][-1]
    assert done_ev["status"] == "passed", f"desk_task failed: {done_ev}"
    rounds = done_ev.get("rounds") or []
    later_hits = [r for r in rounds if r["round_num"] >= 2 and r["cached_tokens"] > 0]
    assert later_hits and all(r["cache_restore_ms"] < 25.0 for r in later_hits)
    print(f"[5/7] Live desk_task mission PASS ({len(rounds)} rounds, cache hits={len(later_hits)}/{max(0, len(rounds)-1)})", flush=True)

    # 6. Cache HUD & System Health (RSS < 100 MiB)
    c_cs, _, raw_cs = _http_req(opener, base_url + "/api/cache/stats")
    assert c_cs == 200 and json.loads(raw_cs)["models"][model_name]["reachable"] is True
    c_cc, _, raw_cc = _http_req(opener, base_url + "/api/cache/clear", method="POST", json_body={"include_pinned": False, "model": model_name})
    assert c_cc == 200
    c_sys, _, raw_sys = _http_req(opener, base_url + "/api/system/health")
    sys_health = json.loads(raw_sys)
    assert c_sys == 200 and 0 < sys_health["rss_mib"] < 100.0, f"Unexpected RSS/health: {sys_health}"
    print(f"[6/7] Cache HUD & System Health PASS (RSS={sys_health['rss_mib']} MiB, freq={sys_health['npu_freq_hz']})", flush=True)

    # 7. Audio Upload (ffmpeg normalization) & TTS WAV (espeak-ng)
    wav_in = _make_wav_bytes(0.3, sample_rate=22050, channels=2)
    c_aud, _, raw_aud = _http_req(
        opener,
        base_url + "/api/audio/upload?filename=lan_probe.wav",
        method="POST",
        raw_bytes=wav_in,
        headers={"Content-Type": "audio/wav"},
    )
    assert c_aud == 200
    aud_meta = json.loads(raw_aud)["metadata"]
    assert aud_meta["sample_rate"] == 16000 and aud_meta["channels"] == 1 and aud_meta["duration_s"] > 0.25

    c_tts, hdr_tts, raw_tts = _http_req(opener, base_url + "/api/tts", method="POST", json_body={"text": "Romanaccio voice output operational."})
    assert c_tts == 200 and raw_tts[:4] == b"RIFF" and hdr_tts.get("x-tts-engine")
    print(f"[7/7] Audio Upload (16kHz mono) & TTS WAV ({hdr_tts.get('x-tts-engine')}, {len(raw_tts)} B) PASS", flush=True)

    return {
        "allow_rel": allow_rel,
        "deny_rel": deny_rel,
        "mission_run": done_ev,
        "system_health": sys_health,
        "tts_engine": hdr_tts.get("x-tts-engine"),
        "tts_bytes": len(raw_tts),
        "audio_meta": aud_meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live LAN verification suite for Milestone 19D")
    parser.add_argument("--url", default="http://roma:8000")
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", default="lfm3-1.7b")
    parser.add_argument("--out-json", default="")
    ns = parser.parse_args()
    res = run_suite(ns.url.rstrip("/"), ns.token, ns.model)
    if ns.out_json:
        Path(ns.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
