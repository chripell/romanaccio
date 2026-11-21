#!/usr/bin/env python3
"""Live silicon verification on roma for Milestone 19A daemon hygiene + Milestone 19B routing & gates."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness import (  # noqa: E402
    Calibration,
    Capabilities,
    DaemonClient,
    EvidenceGate,
    GenerateRequest,
    IntentRouter,
    StagedWriteProposal,
    TokenizerLookup,
    WriteGate,
)


ROUTING_CASES_21: List[Dict[str, str]] = [
    # 3x general
    {"tag": "general_1", "expected": "general", "msg": "What is the time complexity difference between quicksort and mergesort?"},
    {"tag": "general_2", "expected": "general", "msg": "Explain how async event loops schedule coroutines cleanly in Python."},
    {"tag": "general_3", "expected": "general", "msg": "Compare TCP slow start with congestion avoidance in computer networks."},
    # 4x explore
    {"tag": "explore_1", "expected": "explore", "msg": "List all files inside the src/lfm3_harness directory of this repository."},
    {"tag": "explore_2", "expected": "explore", "msg": "Read src/lfm3_harness/paths.py and show how symlinks are rejected."},
    {"tag": "explore_3", "expected": "explore", "msg": "Search the codebase for atomic_write_text definitions and call sites."},
    {"tag": "explore_4", "expected": "explore", "msg": "Inspect config.py and tell me the default APPROVAL_TTL_S constant."},
    # 3x summarize
    {"tag": "summarize_1", "expected": "summarize", "msg": "Please summarize what src/lfm3_harness/store.py implements."},
    {"tag": "summarize_2", "expected": "summarize", "msg": "Give me a concise overview and recap of the router module design."},
    {"tag": "summarize_3", "expected": "summarize", "msg": "Provide a TL;DR summary of the prefix cache architecture in PLAN.md."},
    # 3x related
    {"tag": "related_1", "expected": "related", "msg": "Find files related to backend.py in this workspace repository."},
    {"tag": "related_2", "expected": "related", "msg": "Which modules import or reference WorkspaceSandbox in src?"},
    {"tag": "related_3", "expected": "related", "msg": "Locate files with similar safety rules and dependencies as paths.py."},
    # 4x report
    {"tag": "report_1", "expected": "report", "msg": "Write a short status report in docs/status.md covering our tests."},
    {"tag": "report_2", "expected": "report", "msg": "Edit router.py to tighten the short message character threshold."},
    {"tag": "report_3", "expected": "report", "msg": "Create a new markdown checklist file named CHECKLIST.md in root."},
    {"tag": "report_4", "expected": "report", "msg": "Update README.md and save a troubleshooting section at the end."},
    # 2x short < 24 char bypass
    {"tag": "short_1", "expected": "general", "msg": "hi there"},
    {"tag": "short_2", "expected": "explore", "msg": "list files"},
    # 2x capability-gated off (vision=False, voice_asr=False)
    {"tag": "gated_vis", "expected": "explore", "msg": "Look closely at screenshot.png and read the dialog text inside."},
    {"tag": "gated_aud", "expected": "explore", "msg": "Transcribe meeting_notes.wav or summarize the transcript file."},
]


async def run_live_suite(args: argparse.Namespace) -> Dict[str, Any]:
    tok = TokenizerLookup.from_file(args.tokenizer)
    calib = Calibration.load(args.calib)
    client = DaemonClient(args.host, args.port, tokenizer=tok, calibration=calib, connect_timeout_s=10.0)

    report: Dict[str, Any] = {"host": args.host, "port": args.port, "m19a_daemon_smoke": {}, "m19b_routing": [], "m19b_gates": {}}

    print("=== [1/3] M19A Live Daemon Hygiene Smoke ===", flush=True)
    stats = await client.cache_stats()
    print(f"[cache_stats] enabled={stats.get('enabled')} blocks={stats.get('resident_blocks')} fp={stats.get('model_fingerprint')}", flush=True)
    report["m19a_daemon_smoke"]["cache_stats"] = stats

    # Test stop sequence with zero leak
    stop_prompt = (
        "<|startoftext|><|im_start|>user\n"
        "Echo these exact words separated by single spaces and nothing else: ALPHA BETA END_HERE OMEGA"
        "<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
    stop_text = ""
    stop_done: Dict[str, Any] = {}
    async for ev in client.stream_generate(
        GenerateRequest(
            prompt=stop_prompt,
            max_tokens=48,
            temperature=0.0,
            no_think=True,
            stop=["END_HERE"],
            prompt_cache_key="m19_smoke:stop",
        )
    ):
        if ev.type == "token":
            stop_text += ev.payload.get("text", "")
        elif ev.type == "done":
            stop_done = ev.payload
    assert "END_HERE" not in stop_text, f"Stop sequence leaked into output: {stop_text!r}"
    assert stop_done.get("finish_reason") in ("stop", "length"), f"Unexpected finish_reason: {stop_done}"
    print(
        f"[stop_seq] finish={stop_done.get('finish_reason')} stopped_by={stop_done.get('stopped_by')!r} text={stop_text!r}",
        flush=True,
    )
    report["m19a_daemon_smoke"]["stop_test"] = {
        "text": stop_text,
        "finish_reason": stop_done.get("finish_reason"),
        "stopped_by": stop_done.get("stopped_by"),
    }

    # Test think_budget + thinking flag + think_end
    think_prompt = (
        "<|startoftext|><|im_start|>user\n"
        "Explain briefly why write-ahead logs improve crash recovery."
        "<|im_end|>\n<|im_start|>assistant\n<think>"
    )
    think_toks: List[str] = []
    ans_toks: List[str] = []
    saw_think_end = False
    think_done: Dict[str, Any] = {}
    async for ev in client.stream_generate(
        GenerateRequest(
            prompt=think_prompt,
            max_tokens=160,
            temperature=0.0,
            no_think=False,
            think_budget=24,
            repeat_penalty=1.1,
            prompt_cache_key="m19_smoke:think",
        )
    ):
        if ev.type == "token":
            if ev.payload.get("thinking"):
                think_toks.append(ev.payload.get("text", ""))
            else:
                ans_toks.append(ev.payload.get("text", ""))
        elif ev.type == "think_end":
            saw_think_end = True
        elif ev.type == "done":
            think_done = ev.payload
    assert len(think_toks) > 0, "Expected thinking=True tokens"
    assert saw_think_end, "Expected think_end event"
    assert len(think_toks) <= 24 + 24 + 2, f"Thinking tokens exceeded budget+grace: {len(think_toks)}"
    assert len(ans_toks) > 0, "Expected answer tokens after think_end"
    print(
        f"[think_budget] think_toks={len(think_toks)} ans_toks={len(ans_toks)} saw_think_end={saw_think_end} finish={think_done.get('finish_reason')}",
        flush=True,
    )
    report["m19a_daemon_smoke"]["think_budget_test"] = {
        "think_toks": len(think_toks),
        "ans_toks": len(ans_toks),
        "saw_think_end": saw_think_end,
        "finish_reason": think_done.get("finish_reason"),
    }

    # Test mid-stream client disconnect abort + clean recovery on next request
    cancel = asyncio.Event()
    got_before_cancel = 0
    long_prompt = (
        "<|startoftext|><|im_start|>user\n"
        "Write a numbered list from 1 to 100 describing computer science concepts."
        "<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
    async for ev in client.stream_generate(
        GenerateRequest(prompt=long_prompt, max_tokens=200, temperature=0.0, no_think=True),
        cancel_event=cancel,
    ):
        if ev.type == "token":
            got_before_cancel += 1
            if got_before_cancel >= 4:
                cancel.set()
    assert got_before_cancel == 4

    rec_text = ""
    rec_done: Dict[str, Any] = {}
    async for ev in client.stream_generate(
        GenerateRequest(
            prompt="<|startoftext|><|im_start|>user\nReply with just: RECOVERED_OK<|im_end|>\n<|im_start|>assistant\n<think></think>",
            max_tokens=16,
            temperature=0.0,
            no_think=True,
        )
    ):
        if ev.type == "token":
            rec_text += ev.payload.get("text", "")
        elif ev.type == "done":
            rec_done = ev.payload
    assert rec_done.get("completion_tokens", 0) > 0, f"Recovery query failed after cancel: {rec_done}"
    print(f"[cancel_recover] aborted_after={got_before_cancel} recovered_text={rec_text!r} rec_tokens={rec_done.get('completion_tokens')}", flush=True)
    report["m19a_daemon_smoke"]["cancel_recover_test"] = {
        "aborted_after_tokens": got_before_cancel,
        "recovered_text": rec_text,
        "completion_tokens": rec_done.get("completion_tokens"),
    }

    print("\n=== [2/3] M19B 21-Case Live Routing Suite ===", flush=True)
    router = IntentRouter(client, capabilities=Capabilities(vision=False, voice_asr=False))
    decision_latencies: List[float] = []
    source_counts = {"decision": 0, "heuristic": 0}
    matched_expected = 0

    for item in ROUTING_CASES_21:
        dec = await router.route(item["msg"], prompt_cache_key="d1:m19b_router")
        source_counts[dec.source] += 1
        if dec.latency_ms > 0:
            decision_latencies.append(dec.latency_ms)
        if dec.intent == item["expected"]:
            matched_expected += 1

        if item["tag"].startswith("short_"):
            assert dec.source == "heuristic" and dec.reason == "short_message_skip" and dec.latency_ms == 0.0
        if item["tag"].startswith("gated_"):
            assert dec.intent not in ("vision", "voice")
        assert dec.source in ("decision", "heuristic")

        row = {"tag": item["tag"], "expected": item["expected"], **dec.to_dict()}
        report["m19b_routing"].append(row)
        print(
            f"  {item['tag']:12s} -> intent={dec.intent:9s} exp={item['expected']:9s} "
            f"src={dec.source:9s} ({dec.reason}) conf={dec.confidence:.2f} "
            f"d1=({dec.decision_intent},{dec.decision_confidence}) lat={dec.latency_ms:.1f}ms",
            flush=True,
        )

    avg_lat = sum(decision_latencies) / len(decision_latencies) if decision_latencies else 0.0
    print(
        f"[routing_summary] matched={matched_expected}/21 sources={source_counts} d1_avg_ms={avg_lat:.1f}",
        flush=True,
    )
    report["m19b_routing_summary"] = {
        "matched_expected": matched_expected,
        "total": len(ROUTING_CASES_21),
        "sources": source_counts,
        "d1_avg_ms": round(avg_lat, 2),
    }

    print("\n=== [3/3] M19B Live WriteGate & EvidenceGate Smoke ===", flush=True)
    wg = WriteGate(client)
    eg = EvidenceGate(client)

    prop_aligned = StagedWriteProposal(
        tool_name="write_file",
        rel_path="notes/hello.txt",
        base_sha256="",
        new_content="Hello from roma!\n",
        diff="--- a/notes/hello.txt\n+++ b/notes/hello.txt\n@@ -0,0 +1 @@\n+Hello from roma!\n",
        bytes_new=17,
        reason="Create requested hello note",
    )
    prop_contra = StagedWriteProposal(
        tool_name="write_file",
        rel_path="README.md",
        base_sha256="deadbeef",
        new_content="DELETED ALL DOCUMENTATION",
        diff="--- a/README.md\n+++ b/README.md\n@@ -1,5 +1 @@\n-Important docs\n+DELETED ALL DOCUMENTATION\n",
        bytes_new=25,
        reason="Overwrite README",
    )

    wg_ok = await wg.evaluate("Create notes/hello.txt with a brief hello greeting.", prop_aligned)
    wg_bad = await wg.evaluate("Do not edit or overwrite any files; only read README.md.", prop_contra)
    eg_suff = await eg.evaluate(
        "What TCP port does the backend use?",
        "Read src/lfm3_harness/config.py: class BackendEndpoint: port: int = 8088",
    )
    eg_off = await eg.evaluate(
        "What TCP port does the backend use?",
        "Read recipes/cake.txt: Bake vanilla sponge cake at 180C for 35 minutes.",
    )

    for label, gres in [("wg_aligned", wg_ok), ("wg_contra", wg_bad), ("eg_suff", eg_suff), ("eg_off", eg_off)]:
        print(f"  {label:10s} -> verdict={gres.verdict} conf={gres.confidence:.2f} src={gres.source} lat={gres.latency_ms:.1f}ms", flush=True)

    report["m19b_gates"] = {
        "wg_aligned": wg_ok.to_dict(),
        "wg_contra": wg_bad.to_dict(),
        "eg_suff": eg_suff.to_dict(),
        "eg_off": eg_off.to_dict(),
    }

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Live roma verification for M19A daemon hygiene + M19B router/gates")
    parser.add_argument("--host", default="roma")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--tokenizer", default="/home/chri/aixigua/deep/models/LFM3-1.7B/tokenizer.json")
    parser.add_argument("--calib", default="/home/chri/aixigua/deep/models/d1-2B-ONNX/calibration.json")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    asyncio.run(run_live_suite(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
