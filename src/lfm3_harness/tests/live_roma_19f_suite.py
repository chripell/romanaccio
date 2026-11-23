#!/usr/bin/env python3
"""Live LAN 4-daemon verification suite on roma for Milestone 19F (LFM3-Audio-2B ASR + Native TTS)."""

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _http_json(
    url: str,
    *,
    token: str,
    method: str = "GET",
    body: Optional[dict] = None,
    raw_bytes: Optional[bytes] = None,
    content_type: str = "application/json",
    timeout: float = 90.0,
) -> Tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data: Optional[bytes] = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = content_type
    elif raw_bytes is not None:
        data = raw_bytes
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw}


def _http_bytes(
    url: str,
    *,
    token: str,
    method: str = "POST",
    body: Optional[dict] = None,
    timeout: float = 120.0,
) -> Tuple[int, bytes, Dict[str, str]]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, resp.read(), hdrs
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


def _read_sse(
    url: str,
    *,
    token: str,
    method: str = "POST",
    body: Optional[dict] = None,
    timeout: float = 180.0,
) -> List[Tuple[str, dict]]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    events: List[Tuple[str, dict]] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cur_ev = "message"
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event: "):
                cur_ev = line[7:].strip()
            elif line.startswith("data: "):
                events.append((cur_ev, json.loads(line[6:])))
    return events


def _inspect_wav(raw: bytes) -> Dict[str, Any]:
    with wave.open(io.BytesIO(raw), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nf = wf.getnframes()
        return {"channels": ch, "sample_rate": sr, "sampwidth": sw, "frames": nf, "duration_s": round(nf / float(sr), 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Live LAN Milestone 19F 4-daemon verification suite on roma")
    parser.add_argument("--host", default="roma")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auth-token", required=True)
    parser.add_argument("--wav-path", required=True, help="16 kHz speech WAV for ASR check")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    tok = args.auth_token
    wav_bytes = Path(args.wav_path).read_bytes()
    report: Dict[str, Any] = {"host": args.host, "port": args.port}

    # [0/5] Discovery, health, audio status, 4-daemon cache stats, RSS < 95 MiB
    c_h, health = _http_json(f"{base}/api/health", token=tok)
    assert c_h == 200 and health.get("status") == "ok", f"/api/health failed: {c_h} {health}"
    assert float(health["rss_mib"]) < 95.0, f"server RSS too high: {health['rss_mib']} MiB >= 95 MiB"

    c_ast, ast = _http_json(f"{base}/api/audio/status", token=tok)
    assert c_ast == 200 and ast.get("sidecar_reachable") is True and ast.get("daemon", {}).get("reachable") is True, (
        f"/api/audio/status unhealthy: {ast}"
    )

    c_mod, mod = _http_json(f"{base}/api/models", token=tok)
    assert c_mod == 200, f"/api/models failed: {c_mod}"
    assert [m["name"] for m in mod["models"]] == ["lfm3-1.7b", "lfm2.5-8b-a1b"]
    assert all(m["reachable"] for m in mod["models"]), f"Actor offline: {mod}"
    assert mod["d1"]["separate_daemon"] is True and mod["d1"]["reachable"] is True
    assert mod["audio"]["configured"] is True and mod["audio"]["reachable"] is True, f"Audio role offline: {mod['audio']}"
    assert mod["capabilities"]["vision"] is True and mod["capabilities"]["vision_reachable"] is True
    assert mod["capabilities"]["voice_asr"] is True and mod["capabilities"]["asr_reachable"] is True
    assert "lfm3-audio" in mod["capabilities"]["tts_engines"], f"Missing native TTS in {mod['capabilities']}"

    c_cs, cs = _http_json(f"{base}/api/cache/stats", token=tok)
    assert c_cs == 200 and {"lfm3-1.7b", "lfm2.5-8b-a1b", "d1", "audio"}.issubset(cs["models"].keys()), cs
    report["discovery"] = {"health": health, "audio_status": ast, "models": mod, "cache_stats": cs}
    print(
        f"[0/5] Discovery PASS: roles={[m['name'] for m in mod['models']] + ['d1', 'audio']}, "
        f"sidecar_asr={ast.get('asr_engine')}, tts_engines={ast.get('tts_engines')}, RSS={health['rss_mib']} MiB",
        flush=True,
    )

    # [1/5] Tier-0 HIT_TIER0 non-eviction on :8088 across an interleaved :8091 ASR request
    _, s_t0 = _http_json(f"{base}/api/sessions", token=tok, method="POST", body={"title": "M19F Tier0", "model_name": "lfm3-1.7b"})
    sid_t0 = s_t0["session"]["id"]
    t1_evs = _read_sse(
        f"{base}/api/sessions/{sid_t0}/messages",
        token=tok,
        body={
            "content": "Remember this exact codeword: SIGNAL-ORBIT-904. Reply without calling any tools: NOTED_T1.",
            "model_name": "lfm3-1.7b",
            "max_tokens": 24,
        },
    )
    t1_done = [d for e, d in t1_evs if e == "done"][-1]
    assert t1_done["status"] == "completed", f"Tier-0 Turn 1 failed: {t1_done}"

    # Interleaved :8091 ASR request while :8088 holds Tier-0 state
    c_up, up_res = _http_json(
        f"{base}/api/audio/upload?filename=test_asr_check.wav",
        token=tok,
        method="POST",
        raw_bytes=wav_bytes,
        content_type="application/octet-stream",
    )
    assert c_up == 200 and up_res["ok"] is True and up_res["asr_available"] is True, f"Upload ASR failed: {up_res}"
    transcript_up = (up_res.get("transcript") or "").strip()
    assert transcript_up, f"Empty ASR transcript on test_asr_check.wav: {up_res}"

    t2_evs = _read_sse(
        f"{base}/api/sessions/{sid_t0}/messages",
        token=tok,
        body={
            "content": "Without calling any tools, what was the exact codeword I asked you to remember?",
            "model_name": "lfm3-1.7b",
            "max_tokens": 48,
        },
    )
    t2_done = [d for e, d in t2_evs if e == "done"][-1]
    t2_tel = t2_done["telemetry"]
    assert t2_tel["cache_status"] == "HIT_TIER0" and float(t2_tel["cache_restore_ms"]) == 0.0, (
        f"Expected HIT_TIER0 after interleaved :8091 ASR, got {t2_tel}"
    )
    t2_text = "".join(d.get("text", "") for e, d in t2_evs if e == "token").strip()
    assert "SIGNAL-ORBIT-904" in t2_text, f"Codeword recall failed: {t2_text!r}"
    report["tier0_non_eviction"] = t2_tel
    report["upload_asr"] = up_res
    print(
        f"[1/5] Tier-0 Non-Eviction PASS: :8088 kept {t2_tel['cache_status']} "
        f"({t2_tel['cached_tokens']}/{t2_tel['prompt_tokens']}, restore={t2_tel['cache_restore_ms']}ms) "
        f"across interleaved :8091 ASR; recalled='{t2_text}'",
        flush=True,
    )

    # [2/5] Verify ASR upload accuracy + workspace transcribe() tool through agent turn
    up_low = transcript_up.lower()
    assert ("ready" in up_low and ("port" in up_low or "eight" in up_low or "91" in up_low or "nine" in up_low)), (
        f"Unexpected ASR transcript on test_asr_check.wav: {transcript_up!r}"
    )

    _, s_tr = _http_json(f"{base}/api/sessions", token=tok, method="POST", body={"title": "M19F Transcribe Tool", "model_name": "lfm3-1.7b"})
    sid_tr = s_tr["session"]["id"]
    tr_evs = _read_sse(
        f"{base}/api/sessions/{sid_tr}/messages",
        token=tok,
        body={
            "content": "Use the transcribe tool on sample.wav and quote what the speaker says.",
            "model_name": "lfm3-1.7b",
            "max_tokens": 128,
        },
    )
    tr_tool_res = [d for e, d in tr_evs if e == "tool_result" and d.get("name") == "transcribe"]
    tr_text = "".join(d.get("text", "") for e, d in tr_evs if e == "token").strip()
    assert tr_tool_res and tr_tool_res[0].get("ok") is True, f"transcribe tool did not execute cleanly: {tr_evs}"
    tr_obj = json.loads(tr_tool_res[0]["result"])
    assert tr_obj.get("asr_engine") == "lfm3-audio" and tr_obj.get("transcript"), f"Bad transcribe payload: {tr_obj}"
    assert "ready" in tr_text.lower() or "port" in tr_text.lower() or "8091" in tr_text.lower() or "eight" in tr_text.lower(), (
        f"Agent answer missed transcript content: {tr_text!r} (tool={tr_obj})"
    )
    report["transcribe_tool"] = {"tool_result": tr_obj, "answer": tr_text}
    print(
        f"[2/5] ASR Upload + transcribe() Tool PASS: upload='{transcript_up}', tool='{tr_obj['transcript']}', agent='{tr_text}'",
        flush=True,
    )

    # [3/5] 19F.2 Native TTS benchmark + espeak-ng check + Closed-Loop TTS -> ASR verification
    c_esp, esp_wav, esp_hdr = _http_bytes(f"{base}/api/tts", token=tok, body={"text": "Fallback espeak check.", "engine": "espeak-ng"})
    assert c_esp == 200 and esp_wav[:4] == b"RIFF" and esp_hdr.get("x-tts-engine") == "espeak-ng"

    tts_benchmarks: List[Dict[str, Any]] = []
    for phrase, expected_words in (
        ("System status nominal on dual NPU dies.", ("system", "status", "nominal")),
        ("The NPU server is ready.", ("server", "ready")),
    ):
        t0 = time.monotonic()
        c_t, wav_t, hdr_t = _http_bytes(
            f"{base}/api/tts",
            token=tok,
            body={"text": phrase, "engine": "lfm3-audio", "language": "American English", "gender": "female", "max_frames": 180},
        )
        wall_ms = round((time.monotonic() - t0) * 1000.0, 2)
        assert c_t == 200 and wav_t[:4] == b"RIFF", f"Native TTS failed: {c_t} {wav_t[:120]!r}"
        assert hdr_t.get("x-tts-engine") == "lfm3-audio"
        wav_meta = _inspect_wav(wav_t)
        assert wav_meta["sample_rate"] == 24000 and wav_meta["channels"] == 1

        # Closed-loop TTS -> ASR
        c_rt, rt_res = _http_json(
            f"{base}/api/audio/upload?filename=tts_roundtrip.wav",
            token=tok,
            method="POST",
            raw_bytes=wav_t,
            content_type="application/octet-stream",
        )
        assert c_rt == 200 and rt_res.get("ok") is True
        rt_txt = (rt_res.get("transcript") or "").strip()
        rec = {
            "phrase": phrase,
            "wall_ms": wall_ms,
            "wav_meta": wav_meta,
            "frames": int(hdr_t.get("x-tts-frames") or 0),
            "duration_s": float(hdr_t.get("x-tts-duration-s") or wav_meta["duration_s"]),
            "total_ms": float(hdr_t.get("x-tts-total-ms") or wall_ms),
            "prefill_ms": float(hdr_t.get("x-tts-prefill-ms") or 0.0),
            "depth_ms": float(hdr_t.get("x-tts-depth-ms") or 0.0),
            "codec_ms": float(hdr_t.get("x-tts-codec-ms") or 0.0),
            "rtf": float(hdr_t.get("x-tts-rtf") or 0.0),
            "eos": hdr_t.get("x-tts-eos"),
            "closed_loop_asr": rt_txt,
            "expected_words": list(expected_words),
        }
        tts_benchmarks.append(rec)
        print(
            f"  TTS sample '{phrase}' -> {rec['frames']} frames ({rec['duration_s']}s), "
            f"total={rec['total_ms']}ms, RTF={rec['rtf']}x (prefill={rec['prefill_ms']}ms depth={rec['depth_ms']}ms codec={rec['codec_ms']}ms), "
            f"closed-loop ASR='{rt_txt}'",
            flush=True,
        )
        assert rec["frames"] >= 5 and any(w in rt_txt.lower() for w in expected_words), (
            f"Closed-loop TTS->ASR unintelligible for '{phrase}': got {rt_txt!r}"
        )

    avg_rtf = round(sum(r["rtf"] for r in tts_benchmarks) / len(tts_benchmarks), 3)
    report["native_tts"] = {
        "samples": tts_benchmarks,
        "avg_rtf": avg_rtf,
        "espeak_fallback_bytes": len(esp_wav),
        "promote_default_gate_passed": avg_rtf >= 1.5,
    }
    print(f"[3/5] Native TTS & Closed-Loop ASR PASS: avg_rtf={avg_rtf}x (>=1.5x gate: {avg_rtf >= 1.5})", flush=True)

    # Check final health RSS
    _, health_end = _http_json(f"{base}/api/health", token=tok)
    assert float(health_end["rss_mib"]) < 95.0, f"End server RSS exceeded 95 MiB: {health_end['rss_mib']}"
    report["health_end"] = health_end

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
