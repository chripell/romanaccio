"""Fast host pytest suite for Milestone 19A: daemon hygiene, paths, tools, protocol, backend, and store."""

import asyncio
import json
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness import (
    BackendDisconnectError,
    BackendTimeoutError,
    BackendUnavailableError,
    Calibration,
    Capabilities,
    DaemonClient,
    GenerateRequest,
    HarnessStore,
    PathSandboxError,
    SessionBusyError,
    TokenizerLookup,
    ToolExecutor,
    WorkspaceSandbox,
    get_tool_schemas,
    parse_assistant_output,
    render_prompt,
    validate_workspace_root,
)
from src.lfm3_harness.tools import get_tool_specs


# ---------------------------------------------------------------------------
# 1. Path Sandbox & Atomic TOCTOU Writes
# ---------------------------------------------------------------------------
def test_validate_workspace_root_rejects_sensitive_paths(tmp_path: Path):
    for bad in [Path("/"), Path("/etc"), Path("/proc"), Path("/sys"), Path("/dev")]:
        if bad.exists():
            with pytest.raises(ValueError):
                validate_workspace_root(bad)
    ssh_dir = Path.home() / ".ssh"
    if ssh_dir.exists():
        with pytest.raises(ValueError):
            validate_workspace_root(ssh_dir)

    assert validate_workspace_root(tmp_path) == tmp_path.resolve()


def test_workspace_sandbox_rejects_escapes_symlinks_and_reserved_names(tmp_path: Path):
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "sub").mkdir()
    (ws / "sub" / "hello.txt").write_text("hi\n", encoding="utf-8")
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")

    # Symlinks pointing outside, inside, and dangling must all be rejected.
    (ws / "link_out.txt").symlink_to(outside / "secret.txt")
    (ws / "link_in.txt").symlink_to(ws / "sub" / "hello.txt")
    (ws / "link_dir").symlink_to(outside)
    (ws / "dangling").symlink_to(ws / "nonexistent")

    sandbox = WorkspaceSandbox(ws)
    assert sandbox.resolve_read("sub/hello.txt") == ws / "sub" / "hello.txt"

    bad_read_paths = [
        "",
        "   ",
        ".",
        "../outside/secret.txt",
        "sub/../../outside/secret.txt",
        "/etc/passwd",
        "C:\\secret.txt",
        "sub\\hello.txt",
        "bad\x00path.txt",
        "CON",
        "con.txt",
        "nul.py",
        "COM1.log",
        "LPT9",
        "link_out.txt",
        "link_in.txt",
        "link_dir/secret.txt",
        "dangling",
    ]
    for bad in bad_read_paths:
        with pytest.raises(PathSandboxError):
            sandbox.resolve_read(bad)
        with pytest.raises(PathSandboxError):
            sandbox.resolve_write(bad)

    with pytest.raises(PathSandboxError):
        sandbox.resolve_write("sub")  # Cannot overwrite existing directory


def test_atomic_write_toctou_and_size_limit(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sandbox = WorkspaceSandbox(ws)

    # New file write with expected_sha256="" succeeds and leaves no temp files.
    receipt = sandbox.atomic_write_text("notes/a.txt", "alpha\n", expected_sha256="")
    assert receipt.rel_path == "notes/a.txt"
    assert receipt.bytes_written == 6
    assert (ws / "notes" / "a.txt").read_text(encoding="utf-8") == "alpha\n"
    assert not any(p.name.startswith(".lfm3_tmp_") for p in (ws / "notes").iterdir())

    # Creating again with expected_sha256="" fails TOCTOU check.
    with pytest.raises(PathSandboxError) as exc_new:
        sandbox.atomic_write_text("notes/a.txt", "beta\n", expected_sha256="")
    assert exc_new.value.reason == "toctou_mismatch"

    # Correct digest succeeds; stale digest fails.
    r2 = sandbox.atomic_write_text("notes/a.txt", "beta\n", expected_sha256=receipt.sha256)
    with pytest.raises(PathSandboxError) as exc_stale:
        sandbox.atomic_write_text("notes/a.txt", "gamma\n", expected_sha256=receipt.sha256)
    assert exc_stale.value.reason == "toctou_mismatch"
    assert r2.sha256 != receipt.sha256

    # Enforce size limit cleanly.
    with pytest.raises(PathSandboxError) as exc_sz:
        sandbox.atomic_write_text("notes/big.txt", "x" * 64, max_bytes=32)
    assert exc_sz.value.reason == "size_limit"
    assert not (ws / "notes" / "big.txt").exists()


# ---------------------------------------------------------------------------
# 2. Confined Tool Registry & Staged Writes
# ---------------------------------------------------------------------------
def test_tools_capability_gating_no_shell_and_staged_writes(tmp_path: Path):
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "alpha.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 1\n", encoding="utf-8")
    (ws / "beta.py").write_text("from alpha import foo\nval = foo()\n", encoding="utf-8")
    (outside / "leak.py").write_text("password = 'nope'\n", encoding="utf-8")
    (ws / "sym_leak.py").symlink_to(outside / "leak.py")

    sandbox = WorkspaceSandbox(ws)

    # Default capabilities: look and transcribe must NOT be advertised or executable; no shell/fetch tools exist.
    schemas_off = get_tool_schemas(Capabilities())
    names_off = [s["function"]["name"] for s in schemas_off]
    assert names_off == sorted(names_off)
    assert "look" not in names_off
    assert "transcribe" not in names_off
    for forbidden in ("shell", "bash", "exec", "run_command", "fetch", "http_get"):
        assert forbidden not in names_off

    schemas_on = get_tool_schemas(Capabilities(vision=True, voice_asr=True))
    names_on = [s["function"]["name"] for s in schemas_on]
    assert "look" in names_on and "transcribe" in names_on
    assert all(spec.name in names_on for spec in get_tool_specs(Capabilities(vision=True, voice_asr=True)))

    exec_off = ToolExecutor(sandbox, Capabilities())
    assert not exec_off.execute("look", {"path": "alpha.py"}).ok
    assert not exec_off.execute("transcribe", {"path": "alpha.py"}).ok
    assert not exec_off.execute("shell", {"cmd": "id"}).ok

    # Read tools ignore symlinks and work deterministically.
    lst = json.loads(exec_off.execute("list_files", {"path": "."}).content)
    listed_paths = [e["path"] for e in lst["entries"]]
    assert listed_paths == ["alpha.py", "beta.py"]
    assert "sym_leak.py" not in listed_paths

    rd = json.loads(exec_off.execute("read_file", {"path": "alpha.py", "start_line": 1, "end_line": 2}).content)
    assert rd["content"] == "def foo():\n    return 1"
    assert rd["total_lines"] == 5

    srch = json.loads(exec_off.execute("search_files", {"query": "return 1"}).content)
    assert len(srch["matches"]) == 2

    rel_res = json.loads(exec_off.execute("related_files", {"path": "alpha.py"}).content)
    assert rel_res["related"][0]["path"] == "beta.py"

    # Phase gating blocks write_file during explore phase.
    gated = exec_off.execute(
        "write_file",
        {"path": "new.txt", "content": "hi"},
        allowed_tools=["list_files", "read_file"],
    )
    assert not gated.ok

    # Staged write_file NEVER writes to disk.
    staged_new = exec_off.execute("write_file", {"path": "new.txt", "content": "hello\n", "reason": "add note"})
    assert staged_new.ok and staged_new.staged_proposal is not None
    assert staged_new.staged_proposal.base_sha256 == ""
    assert not (ws / "new.txt").exists()

    # edit_file rejects ambiguous matches, succeeds when line range disambiguates, never writes disk.
    ambig = exec_off.execute("edit_file", {"path": "alpha.py", "target": "return 1", "replacement": "return 2"})
    assert not ambig.ok and "Ambiguous edit" in (ambig.error or "")

    staged_edit = exec_off.execute(
        "edit_file",
        {"path": "alpha.py", "target": "return 1", "replacement": "return 2", "start_line": 4, "end_line": 5},
    )
    assert staged_edit.ok and staged_edit.staged_proposal is not None
    assert "+    return 2" in staged_edit.staged_proposal.diff
    assert "return 2" not in (ws / "alpha.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. Protocol: Prefix Stability + LFM3/LFM2.5 Render & Parse Round-Trips
# ---------------------------------------------------------------------------
def test_protocol_prefix_invariance_and_append_only_rounds():
    schemas = get_tool_schemas(Capabilities())
    contract = "You are a helpful local workspace assistant. Never invent file contents."
    ws = "/home/chri/work/demo"

    # Varying dynamic route/phase/todo/advisories across two different turns MUST yield identical pinned_prefix_text.
    r1 = render_prompt(
        [{"role": "user", "content": "Inspect alpha.py"}],
        system_contract=contract,
        workspace_root=ws,
        tool_schemas=schemas,
        family="lfm3",
        enable_thinking=False,
        route_hint="explore (0.82)",
        phase="plan",
        todo_summary="1. Read alpha.py",
        advisories=["Read before editing"],
    )
    r2 = render_prompt(
        [{"role": "user", "content": "Write summary.md"}],
        system_contract=contract,
        workspace_root=ws,
        tool_schemas=schemas,
        family="lfm3",
        enable_thinking=True,
        route_hint="report (0.91)",
        phase="work",
        todo_summary="2. Stage diff",
        advisories=["Evidence sufficient"],
    )
    assert r1.pinned_prefix_text == r2.pinned_prefix_text
    assert r1.pin_blocks_up_to == r2.pin_blocks_up_to >= 3
    assert "route=" not in r1.pinned_prefix_text
    assert "phase=" not in r2.pinned_prefix_text
    assert "<|startoftext|><|im_start|>system\n" in r1.pinned_prefix_text
    assert "<tools>" in r1.pinned_prefix_text
    assert r1.stop_sequences == ["<tool_response>", "<|im_start|>"]

    # Append-only intra-turn tool loop check: Round 1 prompt up to assistant generation open matches prefix of Round 2.
    msgs_round1 = [{"role": "user", "content": "Read alpha.py"}]
    p_round1 = render_prompt(msgs_round1, system_contract=contract, workspace_root=ws, tool_schemas=schemas, family="lfm3")
    gen_marker = "<|im_start|>assistant\n<think></think>"
    assert p_round1.prompt.endswith(gen_marker)
    round1_prefix_before_asst = p_round1.prompt[: -len(gen_marker)]

    msgs_round2 = [
        {"role": "user", "content": "Read alpha.py"},
        {
            "role": "assistant",
            "thinking": "Need to open alpha.py",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "alpha.py"}}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"path":"alpha.py","content":"x=1"}'},
    ]
    p_round2 = render_prompt(msgs_round2, system_contract=contract, workspace_root=ws, tool_schemas=schemas, family="lfm3")
    assert p_round2.prompt.startswith(round1_prefix_before_asst)
    # Intra-turn thinking (> last_user_idx) is preserved cleanly for LFM3.
    assert "<think>Need to open alpha.py</think><tool_call>" in p_round2.prompt

    # Once a new user turn arrives, prior assistant <think> is stripped per chat_template.jinja!
    msgs_next_turn = msgs_round2 + [
        {"role": "assistant", "thinking": "Done", "content": "File contains x=1."},
        {"role": "user", "content": "Now explain it."},
    ]
    p_next = render_prompt(msgs_next_turn, system_contract=contract, workspace_root=ws, tool_schemas=schemas, family="lfm3")
    assert "Need to open alpha.py" not in p_next.prompt
    assert "<think></think>File contains x=1." in p_next.prompt


def test_protocol_lfm3_and_lfm2_tool_call_roundtrips():
    # LFM3 XML round-trip (multiline content + ints + nested dicts + multiple calls)
    lfm3_raw = (
        "<think>I will check and stage a file.</think>\n"
        "<tool_call>\n"
        "<function=write_file>\n"
        "<parameter=path>\nsrc/demo.py\n</parameter>\n"
        "<parameter=content>\ndef fn():\n    return {'k': [1, 2]}\n</parameter>\n"
        "<parameter=flag>\ntrue\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\nREADME.md\n</parameter>\n"
        "<parameter=start_line>\n10\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "Staged both actions."
    )
    parsed3 = parse_assistant_output(lfm3_raw, family="lfm3")
    assert parsed3.thinking == "I will check and stage a file."
    assert parsed3.content == "Staged both actions."
    assert parsed3.parse_errors == []
    assert len(parsed3.tool_calls) == 2
    assert parsed3.tool_calls[0].name == "write_file"
    assert parsed3.tool_calls[0].arguments["flag"] is True
    assert "def fn():" in parsed3.tool_calls[0].arguments["content"]
    assert parsed3.tool_calls[1].arguments == {"path": "README.md", "start_line": 10}

    # LFM2 bracket call AST round-trip (quotes, commas inside strings, true/false/null, nested dict)
    lfm2_raw = (
        '<|tool_call_start|>[edit_file(path=\'a,b(1).py\', target="x=1", replacement="x=2", '
        "start_line=4, dry=false, meta={'tags': ['a', 'b'], 'none_val': null})]"
        "<|tool_call_end|>\nProposed edit."
    )
    parsed2 = parse_assistant_output(lfm2_raw, family="lfm2")
    assert parsed2.parse_errors == []
    assert parsed2.content == "Proposed edit."
    assert len(parsed2.tool_calls) == 1
    tc2 = parsed2.tool_calls[0]
    assert tc2.name == "edit_file"
    assert tc2.arguments == {
        "path": "a,b(1).py",
        "target": "x=1",
        "replacement": "x=2",
        "start_line": 4,
        "dry": False,
        "meta": {"tags": ["a", "b"], "none_val": None},
    }

    # Malformed block records parse error safely without throwing.
    bad = parse_assistant_output("<tool_call><function=broken>", family="lfm3")
    assert len(bad.parse_errors) >= 1 and bad.tool_calls == []


# ---------------------------------------------------------------------------
# 4. Backend Client, Pure-Python TokenizerLookup & Disconnect/Cancel Handling
# ---------------------------------------------------------------------------
def test_tokenizer_lookup_parity():
    fallback = TokenizerLookup()
    assert fallback.single_token_id("yes") == 11683
    assert fallback.single_token_id(" yes") == 18171
    assert fallback.single_token_id("no") == 2243
    assert fallback.single_token_id("A") == 41
    assert fallback.single_token_id("0") == 24

    real_tok_path = Path("/home/chri/aixigua/deep/models/LFM3-1.7B/tokenizer.json")
    if real_tok_path.is_file():
        real_tok = TokenizerLookup.from_file(real_tok_path)
        for word in ("yes", "Yes", " yes", " Yes", " YES", "no", "No", "NO", " no", " No", " NO", "A", " A", "0"):
            exp = real_tok.single_token_id(word)
            assert exp is not None
            if word in ("yes", "Yes", " yes", " Yes", " YES", "no", "No", "NO", " no", " No", " NO", "A", "0"):
                assert fallback.single_token_id(word) == exp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_daemon_client_stream_cancel_stop_rst_and_d1():
    received_reqs: List[Dict[str, Any]] = []
    server_saw_abort = asyncio.Event()
    mode = {"behavior": "normal"}

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            line = await reader.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            received_reqs.append(req)
            beh = mode["behavior"]

            if beh == "rst":
                sock = writer.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                writer.close()
                return
            if beh == "eof":
                writer.close()
                await writer.wait_closed()
                return
            if beh == "hang":
                await asyncio.sleep(0.35)
                return

            if req.get("cmd") == "cache_stats":
                writer.write((json.dumps({"type": "cache_stats", "resident_blocks": 5}) + "\n").encode())
                await writer.drain()
                return
            if req.get("cmd") == "cache_clear":
                writer.write(
                    (json.dumps({"type": "cache_cleared", "removed_blocks": 2, "include_pinned": req["include_pinned"]}) + "\n").encode()
                )
                await writer.drain()
                return
            if req.get("return_logits"):
                req_ids = req.get("requested_tokens", [])
                neg_ids = {2243, 4547, 19598, 794, 2752, 14580, 42, 378, 24, 25, 27, 28}
                logits = {str(tid): (-5.0 if tid in neg_ids else 4.0) for tid in req_ids}
                writer.write((json.dumps({"type": "logits", "prefill_ms": 11.5, "logits": logits, "cache_status": "HIT"}) + "\n").encode())
                await writer.drain()
                return

            writer.write((json.dumps({"type": "prefill_done", "prompt_tokens": 120, "cached_tokens": 64, "cache_status": "PARTIAL"}) + "\n").encode())
            await writer.drain()

            if beh == "slow_stream":
                for idx in range(20):
                    await asyncio.sleep(0.03)
                    writer.write((json.dumps({"type": "token", "text": f"t{idx}", "thinking": False}) + "\n").encode())
                    await writer.drain()
                    # Detect peer close cleanly on drain/write.
                    if reader.at_eof():
                        server_saw_abort.set()
                        return
                return

            events = [
                {"type": "token", "text": "plan...", "thinking": True},
                {"type": "think_end"},
                {"type": "token", "text": "done!", "thinking": False},
                {"type": "done", "completion_tokens": 3, "reasoning_tokens": 1, "finish_reason": "stop", "stopped_by": "<tool_response>"},
            ]
            for ev in events:
                writer.write((json.dumps(ev) + "\n").encode())
                await writer.drain()
        except (OSError, ConnectionError):
            server_saw_abort.set()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        client = DaemonClient("127.0.0.1", port, calibration=Calibration(), connect_timeout_s=1.0)

        # cache_stats & cache_clear
        st = await client.cache_stats()
        assert st["resident_blocks"] == 5
        clr = await client.cache_clear(include_pinned=True)
        assert clr["removed_blocks"] == 2 and clr["include_pinned"] is True

        # Streaming request preserves repeat_penalty, think_budget, stop, thinking flag, think_end, and finish_reason.
        req = GenerateRequest(
            prompt="<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>",
            max_tokens=64,
            repeat_penalty=1.15,
            think_budget=32,
            stop=["<tool_response>", "<|im_start|>"],
            pin_blocks_up_to=3,
            prompt_cache_key="chat:s1",
            session_id="s1",
        )
        seen: List[Any] = []
        async for ev in client.stream_generate(req):
            seen.append((ev.type, ev.payload))
        assert [t for t, _ in seen] == ["prefill_done", "token", "think_end", "token", "done"]
        assert seen[1][1]["thinking"] is True
        assert seen[3][1]["thinking"] is False
        assert seen[4][1]["stopped_by"] == "<tool_response>"
        assert received_reqs[-1]["repeat_penalty"] == pytest.approx(1.15)
        assert received_reqs[-1]["think_budget"] == 32
        assert received_reqs[-1]["stop"] == ["<tool_response>", "<|im_start|>"]

        # Mid-stream cancellation closes socket and allows immediate next request.
        mode["behavior"] = "slow_stream"
        cancel = asyncio.Event()
        got_tokens = 0
        async for ev in client.stream_generate(GenerateRequest(prompt="slow", session_id="s1"), cancel_event=cancel):
            if ev.type == "token":
                got_tokens += 1
                if got_tokens == 2:
                    cancel.set()
        assert got_tokens == 2

        mode["behavior"] = "normal"
        next_types = [ev.type async for ev in client.stream_generate(GenerateRequest(prompt="recover", session_id="s1"))]
        assert next_types[-1] == "done"

        # Single-pass d1 decide() for noul, choice, and score.
        res_noul = await client.decide("STATE", "noul", "Should proceed?", prompt_cache_key="d1:write")
        assert res_noul.decision is True and res_noul.probability > 0.99
        assert received_reqs[-1]["prompt"].startswith("<|startoftext|><|im_start|>user\n")
        assert received_reqs[-1]["prompt"].endswith("<|im_end|>\n<|im_start|>assistant\n<think></think>")
        res_choice = await client.decide("STATE", "choice", "Pick intent", criteria=["explore", "report"])
        assert res_choice.choice == "explore"
        res_score = await client.decide("STATE", "score", "Quality", criteria=["low", "mid", "high"])
        assert res_score.level == "high" and res_score.level_index == 2

        # TCP RST and EOF cleanly raise BackendDisconnectError (502), then recover.
        mode["behavior"] = "rst"
        with pytest.raises(BackendDisconnectError) as exc_rst:
            await client.cache_stats()
        assert exc_rst.value.status_code == 502

        mode["behavior"] = "eof"
        with pytest.raises(BackendDisconnectError):
            await client.cache_stats()

        mode["behavior"] = "hang"
        with pytest.raises(BackendTimeoutError) as exc_to:
            await client.command({"cmd": "cache_stats"}, timeout_s=0.05)
        assert exc_to.value.status_code == 504

        mode["behavior"] = "normal"
        assert (await client.cache_stats())["resident_blocks"] == 5

        # SessionBusy check raises SessionBusyError (409).
        with pytest.raises(SessionBusyError) as exc_busy:
            async for _ in client.stream_generate(req, session_busy_check=lambda sid: sid == "s1"):
                pass
        assert exc_busy.value.status_code == 409

    # Unreachable closed port raises BackendUnavailableError (503).
    dead_client = DaemonClient("127.0.0.1", port, connect_timeout_s=0.5)
    with pytest.raises(BackendUnavailableError) as exc_unavail:
        await dead_client.cache_stats()
    assert exc_unavail.value.status_code == 503


# ---------------------------------------------------------------------------
# 5. SQLite WAL Store & Crash Reconciliation
# ---------------------------------------------------------------------------
def test_store_wal_and_crash_reconciliation(tmp_path: Path):
    db_path = tmp_path / "harness.db"
    store1 = HarnessStore(db_path)
    assert store1.journal_mode() == "wal"

    ws = store1.register_workspace("demo", tmp_path)
    sess_active = store1.create_session(ws["id"], "Active before crash", status="streaming")
    sess_done = store1.create_session(ws["id"], "Finished cleanly", status="idle")
    msg_active = store1.add_message(sess_active["id"], "assistant", "partial", status="streaming")
    msg_done = store1.add_message(sess_done["id"], "assistant", "complete", status="completed")

    # One already-expired pending action + one still-valid pending action.
    act_expired = store1.create_action(
        sess_active["id"],
        msg_active["id"],
        tool_name="write_file",
        rel_path="a.txt",
        base_sha256="",
        new_content="hi",
        diff_text="+hi",
        ttl_s=-5,
    )
    act_fresh = store1.create_action(
        sess_done["id"],
        msg_done["id"],
        tool_name="write_file",
        rel_path="b.txt",
        base_sha256="",
        new_content="ok",
        diff_text="+ok",
        ttl_s=600,
    )
    job_active = store1.create_job(ws["id"], "Crashed job", session_id=sess_active["id"], status="running")
    _, mis_active = store1.create_mission_run(ws["id"], '{"task":"desk"}', "sha_demo_1", status="running")
    store1.close()

    # Reopen store: startup reconciliation must cancel active entities and expire stale pending approvals.
    store2 = HarnessStore(db_path)
    assert store2.get_session(sess_active["id"])["status"] == "cancelled"
    assert store2.get_session(sess_done["id"])["status"] == "idle"
    msgs_a = store2.list_messages(sess_active["id"])
    msgs_d = store2.list_messages(sess_done["id"])
    assert msgs_a[0]["status"] == "interrupted"
    assert msgs_d[0]["status"] == "completed"
    assert store2.get_action(act_expired["id"])["status"] == "expired"
    assert store2.get_action(act_fresh["id"])["status"] == "pending"
    assert store2.conn.execute("SELECT status FROM jobs WHERE id = ?", (job_active["id"],)).fetchone()[0] == "cancelled"
    assert store2.conn.execute("SELECT status FROM mission_runs WHERE id = ?", (mis_active["id"],)).fetchone()[0] == "interrupted"

    ev_types = [e["event_type"] for e in store2.list_events()]
    assert "startup_reconciled" in ev_types
    assert "startup_expired" in ev_types

    # Verify two-stage approval transition ledger: pending -> approved -> applied.
    assert store2.transition_action(act_fresh["id"], expected_status="pending", new_status="approved")
    assert store2.transition_action(act_fresh["id"], expected_status="approved", new_status="applied")
    assert not store2.transition_action(act_fresh["id"], expected_status="pending", new_status="denied")
    store2.close()
