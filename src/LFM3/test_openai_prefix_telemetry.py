#!/usr/bin/env python3
"""
Host Mock-Daemon Test Suite for Milestone 17C (lfm3_openai_server.py).

Uses asyncio mock TCP daemons + a background uvicorn server on 127.0.0.1 +
stdlib http.client so it runs locally without requiring external HTTP client
packages (httpx/requests) or a live FML13V03 board.
"""

import asyncio
import http.client
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfm3_openai_server as srv  # noqa: E402


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class MockDaemon:
    """Lightweight asyncio TCP mock speaking the lfm3_v2 / lfm2_moe JSON line protocol."""

    def __init__(self, name: str, emit_prefill_done: bool = True) -> None:
        self.name = name
        self.emit_prefill_done = emit_prefill_done
        self.requests: List[Dict[str, Any]] = []
        self.commands: List[Dict[str, Any]] = []
        self.resident_blocks = 60
        self.pinned_blocks = 8
        self.tier0_hits = 1
        self.clear_calls: List[bool] = []
        self._server: Optional[asyncio.AbstractServer] = None
        self.port: int = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        sock = self._server.sockets[0]
        self.port = int(sock.getsockname()[1])

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))

            if "cmd" in req:
                self.commands.append(req)
                cmd = req["cmd"]
                if cmd == "cache_stats":
                    resp = {
                        "type": "cache_stats",
                        "enabled": True,
                        "cache_dir": f"/tmp/mock_{self.name}",
                        "model_fingerprint": "00112233445566778899aabbccddeeff",
                        "resident_blocks": self.resident_blocks,
                        "pinned_blocks": self.pinned_blocks,
                        "disk_bytes_used": self.resident_blocks * 4177920,
                        "disk_bytes_quota": 256 * 1024 * 1024 * 1024,
                        "total_lookups": 10,
                        "total_block_hits": 8,
                        "total_tokens_saved": 3776,
                        "total_evictions": 0,
                        "pending_writes": 0,
                        "tier0_hits": self.tier0_hits,
                        "tier0_tokens_saved": 1280,
                        "tier0_live_tokens": 1344,
                    }
                elif cmd == "cache_clear":
                    inc = bool(req.get("include_pinned", False))
                    self.clear_calls.append(inc)
                    removed = self.resident_blocks if inc else (self.resident_blocks - self.pinned_blocks)
                    self.resident_blocks -= removed
                    if inc:
                        self.pinned_blocks = 0
                    resp = {
                        "type": "cache_cleared",
                        "removed_blocks": removed,
                        "include_pinned": inc,
                    }
                else:
                    resp = {"type": "error", "message": f"Unknown cmd {cmd}"}
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
                return

            self.requests.append(req)
            cache_enabled = bool(req.get("cache", True))
            key = req.get("prompt_cache_key", "")
            prompt = req.get("prompt", "")

            if not cache_enabled:
                cached_tokens = 0
                cache_status = "BYPASS"
                cache_tier = "none"
                restore_ms = 0.0
                prompt_tokens = 128
            elif key == "exact_hit":
                cached_tokens = 256
                cache_status = "HIT_TIER0"
                cache_tier = "tier0_ram"
                restore_ms = 0.0
                prompt_tokens = 256
            elif key == "miss":
                cached_tokens = 0
                cache_status = "MISS"
                cache_tier = "none"
                restore_ms = 0.0
                prompt_tokens = 3845
            else:
                cached_tokens = 3776
                cache_status = "HIT_DISK"
                cache_tier = "tier1_nvme"
                restore_ms = 22.73
                prompt_tokens = 3845

            computed_prompt = max(0, prompt_tokens - cached_tokens)
            prefill_ms = 42.5 if cached_tokens > 0 else 48000.0
            if self.emit_prefill_done:
                prefill_ev = {
                    "type": "prefill_done",
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "cached_tier0_tokens": cached_tokens if cache_tier == "tier0_ram" else 0,
                    "cached_disk_tokens": cached_tokens if cache_tier == "tier1_nvme" else 0,
                    "computed_prompt_tokens": computed_prompt,
                    "cache_lookup_ms": 0.45,
                    "cache_restore_ms": restore_ms,
                    "cache_status": cache_status,
                    "cache_tier": cache_tier,
                    "prefill_ms": prefill_ms,
                }
                writer.write((json.dumps(prefill_ev) + "\n").encode("utf-8"))

            pieces = ["Hello", " from ", self.name]
            for tid, txt in enumerate(pieces, start=100):
                writer.write((json.dumps({"type": "token", "token_id": tid, "text": txt}) + "\n").encode("utf-8"))

            done_ev = {
                "type": "done",
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "cached_tier0_tokens": cached_tokens if cache_tier == "tier0_ram" else 0,
                "cached_disk_tokens": cached_tokens if cache_tier == "tier1_nvme" else 0,
                "computed_prompt_tokens": computed_prompt,
                "reasoning_tokens": 19 if not req.get("no_think", True) else 0,
                "completion_tokens": len(pieces) + (19 if not req.get("no_think", True) else 0),
                "prefill_ms": prefill_ms,
                "decode_ms": 250.0,
                "cache_lookup_ms": 0.45,
                "cache_restore_ms": restore_ms,
                "cache_status": cache_status,
                "cache_tier": cache_tier,
                "pld_accepted": 1,
                "pld_drafted": 2,
                "pld_steps": 1,
                "prompt_len_chars": len(prompt),
            }
            writer.write((json.dumps(done_ev) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


class GatewayTestHarness:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.dense_mock = MockDaemon("dense", emit_prefill_done=True)
        self.moe_mock = MockDaemon("moe", emit_prefill_done=True)
        self.legacy_mock = MockDaemon("legacy", emit_prefill_done=False)
        self._loop_thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._uvicorn_thread: Optional[threading.Thread] = None
        self.http_port: int = 0

    def start(self) -> None:
        ready = threading.Event()

        def run_mocks() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(asyncio.gather(
                self.dense_mock.start(),
                self.moe_mock.start(),
                self.legacy_mock.start(),
            ))
            ready.set()
            self.loop.run_forever()

        self._loop_thread = threading.Thread(target=run_mocks, daemon=True)
        self._loop_thread.start()
        assert ready.wait(timeout=5.0)

        srv.config.backend_host = "127.0.0.1"
        srv.config.backend_port = self.dense_mock.port
        srv.config.moe_port = self.moe_mock.port
        srv.config.backend_map = {
            "lfm3-legacy": ("127.0.0.1", self.legacy_mock.port),
        }
        srv.config.available_models = list(srv.DEFAULT_MODELS) + ["lfm3-legacy"]

        self.http_port = _find_free_port()
        u_cfg = uvicorn.Config(srv.app, host="127.0.0.1", port=self.http_port, log_level="error")
        self._uvicorn_server = uvicorn.Server(u_cfg)
        self._uvicorn_thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
        self._uvicorn_thread.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.http_port, timeout=1.0)
                conn.request("GET", "/healthz")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    return
            except Exception:
                time.sleep(0.02)
        raise RuntimeError("uvicorn server failed to become healthy")

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._uvicorn_thread is not None:
            self._uvicorn_thread.join(timeout=5.0)
        if self.loop.is_running():
            async def _stop_all() -> None:
                await self.dense_mock.stop()
                await self.moe_mock.stop()
                await self.legacy_mock.stop()

            asyncio.run_coroutine_threadsafe(_stop_all(), self.loop).result(timeout=5.0)
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

    def request_json(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.http_port, timeout=5.0)
        raw = json.dumps(body).encode("utf-8") if body is not None else b""
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=raw, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        return resp.status, hdrs, json.loads(data)

    def request_sse(
        self, path: str, body: Dict[str, Any]
    ) -> Tuple[int, Dict[str, str], List[Dict[str, Any]], bool]:
        conn = http.client.HTTPConnection("127.0.0.1", self.http_port, timeout=5.0)
        raw = json.dumps(body).encode("utf-8")
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        payload = resp.read().decode("utf-8")
        conn.close()
        events: List[Dict[str, Any]] = []
        saw_done = False
        for block in payload.split("\n\n"):
            for line in block.splitlines():
                if not line.startswith("data: "):
                    continue
                val = line[6:].strip()
                if val == "[DONE]":
                    saw_done = True
                else:
                    events.append(json.loads(val))
        return resp.status, hdrs, events, saw_done


@pytest.fixture(scope="module")
def harness():
    h = GatewayTestHarness()
    h.start()
    yield h
    h.stop()


def test_models_and_backend_routing(harness: GatewayTestHarness) -> None:
    status, _, data = harness.request_json("GET", "/v1/models")
    assert status == 200
    model_ids = {m["id"] for m in data["data"]}
    assert {"lfm3-1.7b", "lfm3-1.7b-int8", "d1-2b", "lfm2.5-8b-a1b"}.issubset(model_ids)

    parsed = srv.parse_backend_map("lfm3-1.7b=roma:8088,lfm2.5-*=10.0.0.2:9001")
    assert parsed["lfm3-1.7b"] == ("roma", 8088)
    assert parsed["lfm2.5-*"] == ("10.0.0.2", 9001)

    n_dense_before = len(harness.dense_mock.requests)
    n_moe_before = len(harness.moe_mock.requests)

    st1, _, b1 = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": "Hi dense"}],
        "max_tokens": 32,
    })
    assert st1 == 200
    assert b1["choices"][0]["message"]["content"] == "Hello from dense"
    assert len(harness.dense_mock.requests) == n_dense_before + 1
    assert harness.dense_mock.requests[-1]["max_tokens"] == 32

    st2, _, b2 = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm2.5-8b-a1b",
        "messages": [{"role": "user", "content": "Hi moe"}],
        "max_completion_tokens": 48,
    })
    assert st2 == 200
    assert b2["choices"][0]["message"]["content"] == "Hello from moe"
    assert len(harness.moe_mock.requests) == n_moe_before + 1
    assert harness.moe_mock.requests[-1]["max_tokens"] == 48


def test_prefix_stable_chatml_thinking_and_pinning(harness: GatewayTestHarness) -> None:
    # Turn 1 + Turn 2 with historic <think> block must share identical prefix bytes up to Turn 2
    sys_msg = {"role": "system", "content": "A " * 200 + "system spec.", "cache_control": {"type": "pinned"}}
    u1 = {"role": "user", "content": [{"type": "text", "text": "Question 1", "cache_control": {"type": "ephemeral"}}]}

    st1, _, _ = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [sys_msg, u1],
        "enable_thinking": False,
        "prompt_cache_key": "miss",
        "cache": True,
    })
    assert st1 == 200
    req1 = harness.dense_mock.requests[-1]
    p1 = req1["prompt"]
    assert p1.startswith("<|startoftext|><|im_start|>system\n")
    assert p1.endswith("<|im_start|>assistant\n<think></think>")
    assert req1["raw"] is True
    assert req1["pin_blocks_up_to"] >= 1

    # Turn 2 appends assistant response containing <think>...</think> which MUST be stripped
    a1 = {"role": "assistant", "content": "<think>\nsecret CoT\n</think>\n\nAnswer 1"}
    u2 = {"role": "user", "content": "Question 2"}
    st2, _, _ = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [sys_msg, u1, a1, u2],
        "enable_thinking": True,
        "prompt_cache_key": "sess-alpha",
    })
    assert st2 == 200
    req2 = harness.dense_mock.requests[-1]
    p2 = req2["prompt"]
    turn1_user_prefix = p1.rsplit("<|im_start|>assistant\n", 1)[0]
    assert p2.startswith(turn1_user_prefix)
    assert "secret CoT" not in p2
    assert "<|im_start|>assistant\nAnswer 1<|im_end|>\n" in p2
    assert p2.endswith("<|im_start|>assistant\n<think>")
    assert req2["prompt_cache_key"] == "sess-alpha"

    # MoE omits default system prompt when absent and uses "<think>\n</think>\n\n"
    st3, _, _ = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm2.5-8b-a1b",
        "messages": [{"role": "user", "content": "Hi"}],
        "enable_thinking": False,
    })
    assert st3 == 200
    pmoe = harness.moe_mock.requests[-1]["prompt"]
    assert pmoe == "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n"


def test_non_streaming_telemetry_headers_and_bypass(harness: GatewayTestHarness) -> None:
    # MISS
    st_m, hdr_m, body_m = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": "cold"}],
        "prompt_cache_key": "miss",
    })
    assert st_m == 200
    assert hdr_m["x-cache-status"] == "MISS"
    assert hdr_m["x-cached-tokens"] == "0"
    assert hdr_m["x-cache-tier"] == "none"
    assert body_m["usage"]["prompt_tokens_details"] == {"cached_tokens": 0, "audio_tokens": 0}

    # PARTIAL (3776 / 3845)
    st_p, hdr_p, body_p = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": "warm"}],
        "enable_thinking": True,
    })
    assert st_p == 200
    assert hdr_p["x-cache-status"] == "PARTIAL"
    assert hdr_p["x-cached-tokens"] == "3776"
    assert hdr_p["x-cache-tier"] == "tier1_nvme"
    assert hdr_p["x-cache-restore-ms"] == "22.73"
    assert body_p["usage"]["prompt_tokens"] == 3845
    assert body_p["usage"]["prompt_tokens_details"] == {"cached_tokens": 3776, "audio_tokens": 0}
    assert body_p["usage"]["completion_tokens_details"] == {"reasoning_tokens": 19}
    assert body_p["timings"]["cache_n"] == 3776
    assert body_p["timings"]["prompt_n"] == 69
    assert body_p["timings"]["restore_ms"] == 22.73

    # Exact HIT
    st_h, hdr_h, body_h = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": "exact"}],
        "prompt_cache_key": "exact_hit",
    })
    assert st_h == 200
    assert hdr_h["x-cache-status"] == "HIT"
    assert hdr_h["x-cache-tier"] == "tier0_ram"
    assert body_h["usage"]["prompt_tokens_details"]["cached_tokens"] == 256

    # Explicit BYPASS (cache: false)
    st_b, hdr_b, body_b = harness.request_json("POST", "/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": "bypass"}],
        "cache": False,
    })
    assert st_b == 200
    assert hdr_b["x-cache-status"] == "BYPASS"
    assert harness.dense_mock.requests[-1]["cache"] is False
    assert body_b["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_sse_streaming_include_usage_and_legacy_fallback(harness: GatewayTestHarness) -> None:
    # 1. Standard OpenAI stream_options={"include_usage": True} contract
    st, hdrs, events, saw_done = harness.request_sse("/v1/chat/completions", {
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": "stream telemetry"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert st == 200
    assert saw_done is True
    assert hdrs["x-cache-status"] == "PARTIAL"
    assert hdrs["x-cached-tokens"] == "3776"
    assert hdrs["x-cache-tier"] == "tier1_nvme"
    assert hdrs["x-cache-restore-ms"] == "22.73"

    token_text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-2])
    assert token_text == "Hello from dense"

    stop_chunk = events[-2]
    assert stop_chunk["choices"][0]["finish_reason"] == "stop"
    assert stop_chunk.get("usage") is None

    usage_chunk = events[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 3845
    assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 3776
    assert usage_chunk["timings"]["cache_n"] == 3776
    assert usage_chunk["timings"]["restore_ms"] == 22.73

    # 2. Without stream_options.include_usage: no trailing choices=[] chunk
    st2, _, events2, saw_done2 = harness.request_sse("/v1/chat/completions", {
        "model": "lfm2.5-8b-a1b",
        "messages": [{"role": "user", "content": "stream plain"}],
        "stream": True,
    })
    assert st2 == 200 and saw_done2 is True
    assert events2[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(e["choices"] != [] for e in events2)

    # 3. Legacy daemon without prefill_done: no dropped first token, usage chunk intact
    st3, hdrs3, events3, saw_done3 = harness.request_sse("/v1/chat/completions", {
        "model": "lfm3-legacy",
        "messages": [{"role": "user", "content": "legacy"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert st3 == 200 and saw_done3 is True
    assert hdrs3["x-cache-status"] == "STREAMING"
    legacy_text = "".join(e["choices"][0]["delta"].get("content", "") for e in events3[:-2])
    assert legacy_text == "Hello from legacy"
    assert events3[-1]["choices"] == []
    assert events3[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 3776

    # 4. /v1/completions telemetry (non-stream + SSE)
    st4, hdrs4, b4 = harness.request_json("POST", "/v1/completions", {
        "model": "lfm2.5-8b-a1b",
        "prompt": "raw completion prefix",
        "prompt_cache_key": "exact_hit",
    })
    assert st4 == 200
    assert hdrs4["x-cache-status"] == "HIT"
    assert b4["usage"]["prompt_tokens_details"]["cached_tokens"] == 256

    st5, hdrs5, events5, saw_done5 = harness.request_sse("/v1/completions", {
        "model": "lfm2.5-8b-a1b",
        "prompt": "raw stream prefix",
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert st5 == 200 and saw_done5 is True
    assert hdrs5["x-cache-status"] == "PARTIAL"
    assert events5[-1]["choices"] == []
    assert events5[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 3776


def test_cache_stats_and_clear_endpoints(harness: GatewayTestHarness) -> None:
    st_s, _, single = harness.request_json("GET", "/v1/cache/stats?model=lfm2.5-8b-a1b")
    assert st_s == 200
    assert single["resident_blocks"] == 60
    assert single["pinned_blocks"] == 8
    assert single["block_hit_rate"] == 0.8
    assert single["total_saved_tokens"] == 5056

    st_a, _, agg = harness.request_json("GET", "/v1/cache/stats")
    assert st_a == 200
    assert agg["resident_blocks"] == 180  # dense + moe + legacy
    assert len(agg["backends"]) == 3

    st_c1, _, clr1 = harness.request_json("POST", "/v1/cache/clear", {
        "model": "lfm2.5-8b-a1b",
        "include_pinned": False,
    })
    assert st_c1 == 200
    assert clr1["removed_blocks"] == 52
    assert harness.moe_mock.clear_calls[-1] is False
    assert harness.moe_mock.resident_blocks == 8

    st_c2, _, clr2 = harness.request_json("POST", "/v1/cache/clear", {
        "model": "lfm2.5-8b-a1b",
        "include_pinned": True,
    })
    assert st_c2 == 200
    assert clr2["removed_blocks"] == 8
    assert harness.moe_mock.clear_calls[-1] is True
    assert harness.moe_mock.resident_blocks == 0
