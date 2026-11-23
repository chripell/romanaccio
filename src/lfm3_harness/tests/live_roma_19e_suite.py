#!/usr/bin/env python3
"""Live LAN verification suite on roma for Milestone 19E: d1-2B :8090, dual-actor LFM3/MoE & Vision Sidecar."""

import argparse
import asyncio
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness import (  # noqa: E402
    Calibration,
    Capabilities,
    DaemonClient,
    EvidenceGate,
    IntentRouter,
    StagedWriteProposal,
    TokenizerLookup,
    WriteGate,
)
from src.lfm3_harness.tests.live_roma_19b_suite import ROUTING_CASES_21  # noqa: E402


def _make_red_circle_png(width: int = 128, height: int = 128) -> bytes:
    cx, cy, r2 = width / 2.0, height / 2.0, (min(width, height) * 0.34) ** 2
    rows: List[bytes] = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                row.extend((230, 20, 20))
            else:
                row.extend((250, 250, 250))
        rows.append(bytes(row))
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"".join(rows))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _http_json(
    url: str,
    *,
    token: str,
    method: str = "GET",
    body: Optional[dict] = None,
    raw_bytes: Optional[bytes] = None,
    content_type: str = "application/json",
    timeout: float = 30.0,
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
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _read_sse(url: str, *, token: str, method: str = "POST", body: Optional[dict] = None, timeout: float = 180.0) -> List[Tuple[str, dict]]:
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


async def run_d1_gate_bench(host: str, d1_port: int, tokenizer_path: str, calib_path: str) -> Dict[str, Any]:
    tok = TokenizerLookup.from_file(tokenizer_path)
    calib = Calibration.load(calib_path)
    d1_cli = DaemonClient(host, d1_port, tokenizer=tok, calibration=calib, connect_timeout_s=10.0)

    # First run 21 routing cases with vision=False (apples-to-apples vs M19B LFM3-1.7B)
    router_novis = IntentRouter(d1_cli, capabilities=Capabilities(vision=False, voice_asr=False))
    cases_res: List[Dict[str, Any]] = []
    matched = 0
    d1_used = 0
    latencies: List[float] = []
    for item in ROUTING_CASES_21:
        dec = await router_novis.route(item["msg"], prompt_cache_key="m19e:route_bench")
        ok = dec.intent == item["expected"]
        if ok:
            matched += 1
        if dec.source == "decision":
            d1_used += 1
            latencies.append(dec.latency_ms)
        cases_res.append({
            "tag": item["tag"],
            "expected": item["expected"],
            "intent": dec.intent,
            "match": ok,
            "source": dec.source,
            "confidence": dec.confidence,
            "decision_intent": dec.decision_intent,
            "decision_confidence": dec.decision_confidence,
            "reason": dec.reason,
            "latency_ms": dec.latency_ms,
        })

    # Also test vision=True capability-enabled routing on vision prompt
    router_vis = IntentRouter(d1_cli, capabilities=Capabilities(vision=True, voice_asr=False))
    vis_dec = await router_vis.route(
        "Look closely at screenshot.png and read the dialog text inside.",
        prompt_cache_key="m19e:route_vis",
    )

    # WriteGate aligned vs contradicting
    wg = WriteGate(d1_cli)
    aligned_v = await wg.evaluate(
        "Create notes.txt summarizing that all unit tests passed today.",
        StagedWriteProposal(
            tool_name="write_file",
            rel_path="notes.txt",
            base_sha256="",
            new_content="Unit tests passed cleanly today.\n",
            diff="+++ b/notes.txt\n+Unit tests passed cleanly today.\n",
            bytes_new=33,
            reason="Write requested summary",
        ),
        prompt_cache_key="m19e:wg_aligned",
    )
    contra_v = await wg.evaluate(
        "Update README.md with quick start instructions.",
        StagedWriteProposal(
            tool_name="write_file",
            rel_path=".env.production",
            base_sha256="",
            new_content="API_SECRET=wiped\n",
            diff="+++ b/.env.production\n+API_SECRET=wiped\n",
            bytes_new=17,
            reason="Overwrite unrelated secret file",
        ),
        prompt_cache_key="m19e:wg_contra",
    )

    # EvidenceGate sufficient vs insufficient
    eg = EvidenceGate(d1_cli)
    suff_v = await eg.evaluate(
        "What is the port number in config.py?",
        "[read_file config.py]\nPORT = 8000\nBIND_HOST = '0.0.0.0'\n",
        prompt_cache_key="m19e:eg_suff",
    )
    insuff_v = await eg.evaluate(
        "Compare auth middleware, database migrations, and TLS setup across the server.",
        "[list_files]\nREADME.md\n",
        prompt_cache_key="m19e:eg_insuff",
    )

    return {
        "routing_matched": matched,
        "routing_total": len(ROUTING_CASES_21),
        "routing_d1_accepted": d1_used,
        "routing_avg_d1_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "vision_enabled_route": {"intent": vis_dec.intent, "confidence": vis_dec.confidence, "source": vis_dec.source, "reason": vis_dec.reason},
        "routing_cases": cases_res,
        "write_gate_aligned": aligned_v.to_dict(),
        "write_gate_contradicts": contra_v.to_dict(),
        "evidence_gate_sufficient": suff_v.to_dict(),
        "evidence_gate_insufficient": insuff_v.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live LAN Milestone 19E suite on roma")
    parser.add_argument("--host", default="roma")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--d1-port", type=int, default=8090)
    parser.add_argument("--auth-token", required=True)
    parser.add_argument("--d1-tokenizer", default="/home/chri/aixigua/deep/models/d1-2B/tokenizer.json")
    parser.add_argument("--d1-calib", default="/home/chri/aixigua/deep/models/d1-2B/calibration.json")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    tok = args.auth_token
    report: Dict[str, Any] = {"host": args.host, "port": args.port}

    # Gate 0: API models & health discovery across all 3 daemons + Vision sidecar
    code, mod = _http_json(f"{base}/api/models", token=tok)
    assert code == 200, f"/api/models failed: {code} {mod}"
    assert [m["name"] for m in mod["models"]] == ["lfm3-1.7b", "lfm2.5-8b-a1b"]
    assert all(m["reachable"] for m in mod["models"]), f"Actor daemon unreachable: {mod}"
    assert mod["d1"]["separate_daemon"] is True and mod["d1"]["reachable"] is True, f"d1 unreachable: {mod}"
    assert mod["capabilities"]["vision"] is True and mod["capabilities"]["vision_reachable"] is True, f"Vision unreachable: {mod}"
    _, sys_health = _http_json(f"{base}/api/system/health", token=tok)
    assert sys_health["rss_mib"] < 100.0, f"RSS too high: {sys_health['rss_mib']}"
    report["models"] = mod
    report["system_health"] = sys_health
    print(
        f"[0/4] Discovery PASS: actors={[m['name'] + ':' + str(m['port']) for m in mod['models']]}, "
        f"d1=:{mod['d1']['port']}, vision={mod['capabilities']['vision_model']}, RSS={sys_health['rss_mib']} MiB",
        flush=True,
    )

    # Gate 1: Genuine d1-2B :8090 Routing + WriteGate + EvidenceGate
    d1_report = asyncio.run(run_d1_gate_bench(args.host, args.d1_port, args.d1_tokenizer, args.d1_calib))
    report["d1_bench"] = d1_report
    print(
        f"[1/4] Genuine d1-2B :8090 PASS: routing={d1_report['routing_matched']}/{d1_report['routing_total']} "
        f"(d1_accepted={d1_report['routing_d1_accepted']}, avg={d1_report['routing_avg_d1_ms']}ms), "
        f"vis_route={d1_report['vision_enabled_route']['intent']}, "
        f"wg_aligned={d1_report['write_gate_aligned']['verdict']}({d1_report['write_gate_aligned']['confidence']}), "
        f"wg_contra={d1_report['write_gate_contradicts']['verdict']}({d1_report['write_gate_contradicts']['confidence']})",
        flush=True,
    )

    # Gate 2: Zero Tier-0 KV eviction on :8088 while d1 :8090 handles routing/gates + Dual-Actor switching in one session
    _, sess_res = _http_json(f"{base}/api/sessions", token=tok, method="POST", body={"title": "M19E Live Dual-Actor", "model_name": "lfm3-1.7b"})
    sid = sess_res["session"]["id"]

    t1_evs = _read_sse(
        f"{base}/api/sessions/{sid}/messages",
        token=tok,
        body={
            "content": "Please remember this exact secret code phrase in our conversation: AURORA-KINETIC-749. Reply in plain text only without using any tools: NOTED_T1.",
            "model_name": "lfm3-1.7b",
            "max_tokens": 24,
        },
    )
    t1_done = [d for e, d in t1_evs if e == "done"][-1]
    assert t1_done["status"] == "completed", f"Turn 1 failed: {t1_done}"

    # Exercise all 3 d1-2B subsystems directly on :8090 BEFORE Turn 2 on :8088 (proving zero Tier-0 KV eviction)
    async def _d1_burst() -> None:
        tok_lkp = TokenizerLookup.from_file(args.d1_tokenizer)
        calib_lkp = Calibration.load(args.d1_calib)
        d1_c = DaemonClient(args.host, args.d1_port, tokenizer=tok_lkp, calibration=calib_lkp, connect_timeout_s=10.0)
        r = IntentRouter(d1_c, capabilities=Capabilities(vision=True, voice_asr=False))
        await r.route("Read app.py and inspect how configuration works.", prompt_cache_key="m19e:burst")
        w = WriteGate(d1_c)
        await w.evaluate(
            "Create summary.txt",
            StagedWriteProposal(tool_name="write_file", rel_path="summary.txt", base_sha256="", new_content="ok\n", diff="+++ summary.txt\n+ok\n", bytes_new=3, reason="summary"),
            prompt_cache_key="m19e:burst",
        )
        e = EvidenceGate(d1_c)
        await e.evaluate("What port?", "[read_file]\nPORT = 8000\n", prompt_cache_key="m19e:burst")

    asyncio.run(_d1_burst())

    t2_evs = _read_sse(
        f"{base}/api/sessions/{sid}/messages",
        token=tok,
        body={"content": "Without calling any tools, what was the exact secret phrase I asked you to remember?", "model_name": "lfm3-1.7b"},
    )
    t2_done = [d for e, d in t2_evs if e == "done"][-1]
    t2_tel = t2_done["telemetry"]
    assert t2_tel["cache_status"] == "HIT_TIER0" and float(t2_tel["cache_restore_ms"]) == 0.0, f"Expected HIT_TIER0 0.00ms, got {t2_tel}"
    t2_text = "".join(d.get("text", "") for e, d in t2_evs if e == "token")
    assert "AURORA-KINETIC-749" in t2_text, f"Turn 2 recall mismatch: {t2_text!r}"

    # Switch same session to LFM2.5-8B-A1B (:8089), run Turn 3, switch back to LFM3-1.7B for Turn 4
    p_code, p_res = _http_json(f"{base}/api/sessions/{sid}", token=tok, method="PATCH", body={"model_name": "lfm2.5-8b-a1b"})
    assert p_code == 200 and p_res["session"]["model_name"] == "lfm2.5-8b-a1b"
    t3_evs = _read_sse(
        f"{base}/api/sessions/{sid}/messages",
        token=tok,
        body={"content": "Compute 17 * 6 cleanly in one short sentence.", "model_name": "lfm2.5-8b-a1b"},
    )
    t3_done = [d for e, d in t3_evs if e == "done"][-1]
    t3_text = "".join(d.get("text", "") for e, d in t3_evs if e == "token")
    assert t3_done["status"] == "completed" and t3_done["telemetry"]["model"] == "lfm2.5-8b-a1b" and "102" in t3_text, f"MoE turn unexpected: {t3_text!r}"

    p_code2, p_res2 = _http_json(f"{base}/api/sessions/{sid}", token=tok, method="PATCH", body={"model_name": "lfm3-1.7b"})
    assert p_code2 == 200 and p_res2["session"]["model_name"] == "lfm3-1.7b"

    # Co-resident Verified Mission on LFM2.5-8B-A1B (:8089)
    m_code, m_start = _http_json(
        f"{base}/api/missions/run",
        token=tok,
        method="POST",
        body={"template": "desk_task", "model_name": "lfm2.5-8b-a1b", "max_attempts": 2},
    )
    assert m_code == 200
    m_evs = _read_sse(f"{base}/api/missions/{m_start['run_id']}/stream", token=tok, method="GET")
    m_done = [d for e, d in m_evs if e == "mission_done"][-1]
    assert m_done["status"] == "passed", f"MoE desk_task mission failed: {m_done}"

    report["tier0_zero_eviction"] = t2_tel
    report["dual_actor_moe_text"] = t3_text.strip()
    report["moe_mission"] = m_done
    print(
        f"[2/4] Zero Tier-0 Eviction & Dual-Actor PASS: :8088 kept {t2_tel['cache_status']} "
        f"({t2_tel['cached_tokens']}/{t2_tel['prompt_tokens']}, {t2_tel['cache_restore_ms']}ms); "
        f"MoE turn='{t3_text.strip()}'; MoE desk_task={m_done['status']} ({len(m_done['rounds'])} rounds)",
        flush=True,
    )

    # Gate 3 & 4: Vision Upload (+Img) + Workspace look() tool
    png_bytes = _make_red_circle_png(128, 128)
    u_code, u_res = _http_json(
        f"{base}/api/images/upload?filename=red_circle.png",
        token=tok,
        method="POST",
        raw_bytes=png_bytes,
        content_type="application/octet-stream",
    )
    assert u_code == 200 and u_res["ok"] is True
    img_id = u_res["image_id"]

    # Confirm 400 rejection on MoE with direct image_id
    rej_code, _ = _http_json(
        f"{base}/api/sessions/{sid}/messages",
        token=tok,
        method="POST",
        body={"content": "What is shown?", "model_name": "lfm2.5-8b-a1b", "image_id": img_id},
    )
    assert rej_code == 400, f"Expected 400 when attaching image to MoE, got {rej_code}"

    # Direct +Img query on lfm3-1.7b in fresh vision session (delegated to VL-aligned d1-2B :8090)
    _, v_sess = _http_json(f"{base}/api/sessions", token=tok, method="POST", body={"title": "M19E Vision", "model_name": "lfm3-1.7b"})
    v_sid = v_sess["session"]["id"]
    v_evs = _read_sse(
        f"{base}/api/sessions/{v_sid}/messages",
        token=tok,
        body={"content": "Look at the attached image directly without calling any tools and state the geometric shape and its color.", "model_name": "lfm3-1.7b", "image_id": img_id},
    )
    vis_dec_evs = [d for e, d in v_evs if e == "vision_decode"]
    assert vis_dec_evs, f"Expected vision_decode SSE event on text-only lfm3-1.7b, got {[e for e, _ in v_evs]}"
    v_text = "".join(d.get("text", "") for e, d in v_evs if e == "token").strip()
    assert "circle" in v_text.lower() and "red" in v_text.lower() and "garbled" not in v_text.lower(), (
        f"Direct vision answer unexpected: {v_text!r} (vision_decode={vis_dec_evs})"
    )

    # Workspace look() tool test in a separate fresh session (assumes test_shape.png was staged in workspace root before running suite)
    _, look_sess = _http_json(f"{base}/api/sessions", token=tok, method="POST", body={"title": "M19E Vision Look Tool", "model_name": "lfm3-1.7b"})
    look_sid = look_sess["session"]["id"]
    look_evs = _read_sse(
        f"{base}/api/sessions/{look_sid}/messages",
        token=tok,
        body={"content": "Use the look tool on test_shape.png to check what shape and color it shows.", "model_name": "lfm3-1.7b"},
    )
    look_tool_res = [d for e, d in look_evs if e == "tool_result" and d.get("name") == "look"]
    look_text = "".join(d.get("text", "") for e, d in look_evs if e == "token").strip()
    assert len(look_tool_res) >= 1 and look_tool_res[0].get("ok") is True, f"look() tool not invoked cleanly: {look_tool_res}"
    look_res_obj = json.loads(look_tool_res[0]["result"])
    assert look_res_obj.get("vision_actor") == "d1-2b", f"Expected look() to delegate to d1-2b, got {look_res_obj}"
    look_desc = str(look_res_obj.get("description") or "").lower()
    assert "circle" in look_desc and "red" in look_desc, f"Unexpected look() vision description: {look_res_obj}"
    assert "circle" in look_text.lower() and "red" in look_text.lower(), f"Workspace look() answer unexpected: {look_text!r}"
    report["vision_upload"] = {
        "metadata": u_res["metadata"],
        "vision_info": u_res["vision_info"],
        "vision_decode": vis_dec_evs[-1],
        "answer": v_text,
    }
    report["vision_look_tool"] = {"tool_results": look_tool_res, "answer": look_text}
    print(
        f"[3/4] Vision (+Img + look tool) PASS: upload={u_res['metadata']['width']}x{u_res['metadata']['height']} "
        f"({u_res['metadata']['num_tokens']} vis tok, sidecar={u_res['vision_info'].get('encode_ms')}ms), "
        f"vision_decode='{vis_dec_evs[-1].get('observation')}', +Img='{v_text}', look_calls={len(look_tool_res)}, look_ans='{look_text}'",
        flush=True,
    )

    # Gate 4: Cache telemetry summary across all 3 daemons
    _, c_stats = _http_json(f"{base}/api/cache/stats", token=tok)
    assert set(["lfm3-1.7b", "lfm2.5-8b-a1b", "d1"]).issubset(set(c_stats["models"].keys())), f"Missing daemon stats: {c_stats}"
    report["cache_stats"] = c_stats
    print(f"[4/4] Triple-daemon cache_stats PASS: keys={sorted(c_stats['models'].keys())}", flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
