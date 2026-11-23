"""Fast deterministic host pytest suite for Milestone 19C: 3-phase AgentLoop & Verified Tasks."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness import (  # noqa: E402
    AcceptanceCheck,
    AcceptanceSpec,
    AgentLoop,
    Capabilities,
    DaemonEvent,
    HarnessStore,
    MissionRunner,
    WorkspaceSandbox,
    make_doc_extract_spec,
    make_screen_qa_spec,
    setup_desk_task_workspace,
    verify_acceptance_spec,
    verify_json_schema_stdlib,
)


class ScriptedDaemonClient:
    """In-process scripted fake DaemonClient that verifies append-only cache keys and streams canned turns."""

    def __init__(self, script_fn):
        self.script_fn = script_fn
        self.requests: List[Any] = []
        self.prompts: List[str] = []

    async def stream_generate(self, req, *, cancel_event=None, read_timeout_s=120.0, session_busy_check=None):
        step_idx = len(self.requests)
        self.requests.append(req)
        self.prompts.append(req.prompt)
        out_text = self.script_fn(step_idx, req)

        cached = 64 if step_idx >= 1 else 0
        yield DaemonEvent(
            "prefill_done",
            {
                "type": "prefill_done",
                "prompt_tokens": 128 + step_idx * 32,
                "cached_tokens": cached,
                "cache_status": "HIT_TIER0" if cached else "MISS",
                "cache_tier": "tier0_ram" if cached else "none",
                "cache_restore_ms": 0.0,
                "prefill_ms": 15.0 if cached else 90.0,
            },
        )
        yield DaemonEvent("token", {"type": "token", "text": out_text, "thinking": False})
        yield DaemonEvent(
            "done",
            {
                "type": "done",
                "prompt_tokens": 128 + step_idx * 32,
                "cached_tokens": cached,
                "cache_status": "HIT_TIER0" if cached else "MISS",
                "cache_tier": "tier0_ram" if cached else "none",
                "cache_restore_ms": 0.0,
                "prefill_ms": 15.0 if cached else 90.0,
                "decode_ms": 20.0,
                "completion_tokens": 12,
                "finish_reason": "stop",
            },
        )


def _xml_call(name: str, args: Dict[str, Any]) -> str:
    params = []
    for k, v in args.items():
        val = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        params.append(f"<parameter={k}>\n{val}\n</parameter>")
    return f"<tool_call>\n<function={name}>\n" + "\n".join(params) + "\n</function>\n</tool_call>"


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# 1. AgentLoop 3-Phase Gating, Todo Demotion, Unread-Finish Refusal & Append-Only Prefix
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_agent_loop_phase_gating_unread_refusal_and_append_only_prefix(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "in.txt").write_text("North=120 South=85 Total=205\n", encoding="utf-8")

    sandbox = WorkspaceSandbox(ws)
    store = HarnessStore(tmp_path / "harness.db")
    ws_row = store.register_workspace("ws", ws)

    def scripted(step_idx: int, req) -> str:
        # Step 0: attempt write_file in 'plan' before todo_write -> must be blocked!
        if step_idx == 0:
            return _xml_call("write_file", {"path": "out.txt", "content": "Total 205"})
        # Step 1: mark todo item 'done' prematurely while check fails -> must be demoted to in_progress!
        if step_idx == 1:
            return _xml_call(
                "todo_write",
                {
                    "items": [
                        {
                            "id": "write_out",
                            "title": "Write out.txt",
                            "status": "done",
                            "check": {"kind": "file_exists", "path": "out.txt"},
                        }
                    ]
                },
            )
        # Step 2: write out.txt
        if step_idx == 2:
            return _xml_call("read_file", {"path": "in.txt"}) + "\n" + _xml_call(
                "write_file", {"path": "out.txt", "content": "Total 205\n"}
            )
        # Step 3: call finish BEFORE calling read_file on out.txt -> MUST be refused!
        if step_idx == 3:
            return _xml_call("finish", {"summary": "Wrote Total 205", "deliverables": ["out.txt"]})
        # Step 4: read back out.txt and mark todo done
        if step_idx == 4:
            return _xml_call("read_file", {"path": "out.txt"}) + "\n" + _xml_call(
                "todo_write",
                {
                    "items": [
                        {
                            "id": "write_out",
                            "title": "Write out.txt",
                            "status": "done",
                            "check": {"kind": "file_exists", "path": "out.txt"},
                        }
                    ]
                },
            )
        # Step 5: now finish cleanly!
        return _xml_call("finish", {"summary": "Verified Total 205 in out.txt", "deliverables": ["out.txt"]})

    client = ScriptedDaemonClient(scripted)
    loop = AgentLoop(client, sandbox, store, family="lfm3", auto_approve_writes=True, enable_thinking=False)
    observed_prompts: List[str] = []

    res = await loop.run(
        ws_row["id"],
        "Create out.txt from in.txt",
        on_round_prompt=lambda _r, p: observed_prompts.append(p),
    )

    assert res.status == "completed"
    assert res.phase == "review"
    assert len(res.rounds) == 6
    assert len(res.deliverables) == 1
    assert res.deliverables[0].rel_path == "out.txt"
    assert res.deliverables[0].bytes == len("Total 205\n")
    assert res.deliverables[0].verified_read is True
    assert res.numeric_grounding.grounded_numbers == ["205"]
    assert res.numeric_grounding.grounded_ratio == 1.0

    # Verify strict append-only prefix property across all 6 rounds!
    gen_tail = "<|im_start|>assistant\n<think></think>"
    for idx in range(len(observed_prompts) - 1):
        prev_prefix = observed_prompts[idx][: -len(gen_tail)]
        assert observed_prompts[idx + 1].startswith(prev_prefix), f"Prefix mutated between round {idx + 1} and {idx + 2}!"

    # Verify early step blocking & demotion actually occurred in recorded prompts.
    assert "not allowed in phase 'plan'" in observed_prompts[1]
    assert "in_progress" in observed_prompts[2]  # Demoted item
    assert "FINISH REFUSED" in observed_prompts[4]

    store.close()


# ---------------------------------------------------------------------------
# 2. Desk Task End-to-End & Adversarial Rejection Fixtures
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_missions_desk_task_and_adversarial_fixtures(tmp_path: Path):
    # Case A: Clean desk_task end-to-end pass + frozen spec + JSONL journal
    ws_ok = tmp_path / "ws_ok"
    spec_ok = setup_desk_task_workspace(ws_ok)
    sandbox_ok = WorkspaceSandbox(ws_ok)
    store = HarnessStore(tmp_path / "harness.db")
    ws_row_ok = store.register_workspace("desk_ok", ws_ok)

    def script_ok(step_idx: int, req) -> str:
        if step_idx == 0:
            return _xml_call("todo_write", {"items": [{"id": "1", "title": "Read and summarize", "status": "in_progress"}]})
        if step_idx == 1:
            return (
                _xml_call("read_file", {"path": "inbox/q1_sales.txt"})
                + "\n"
                + _xml_call("read_file", {"path": "inbox/q2_sales.txt"})
                + "\n"
                + _xml_call("read_file", {"path": "inbox/notes.txt"})
            )
        if step_idx == 2:
            return _xml_call(
                "write_file",
                {
                    "path": "reports/desk_summary.md",
                    "content": "# Desk Summary\n- North: 120 units, 3600 USD\n- South: 85 units, 2550 USD\n- Total Units: 205\n- Total Revenue: 6150\n",
                },
            )
        if step_idx == 3:
            return _xml_call("read_file", {"path": "reports/desk_summary.md"})
        return _xml_call(
            "finish",
            {
                "summary": "Generated reports/desk_summary.md with Total Units 205 and Total Revenue 6150.",
                "deliverables": ["reports/desk_summary.md"],
            },
        )

    runner_ok = MissionRunner(ScriptedDaemonClient(script_ok), sandbox_ok, store, journal_dir=tmp_path / "journals", family="lfm3")
    res_ok = await runner_ok.run(ws_row_ok["id"], spec_ok)
    assert res_ok.status == "passed"
    assert res_ok.attempts == 1
    assert res_ok.spec_sha256 == spec_ok.sha256()
    journal_lines = [json.loads(line) for line in res_ok.journal_path.read_text(encoding="utf-8").splitlines()]
    assert [ev["event"] for ev in journal_lines] == ["mission_started", "attempt_started", "attempt_verified", "mission_passed"]

    # Case B: Adversarial path escape + missing deliverable refusal
    ws_esc = tmp_path / "ws_esc"
    spec_esc = setup_desk_task_workspace(ws_esc, max_attempts=1, max_rounds_per_attempt=4)
    ws_row_esc = store.register_workspace("desk_esc", ws_esc)

    def script_escape(step_idx: int, req) -> str:
        if step_idx == 0:
            return _xml_call("todo_write", {"items": [{"id": "1", "title": "Try escape", "status": "in_progress"}]})
        if step_idx == 1:
            return _xml_call("write_file", {"path": "../pwned.txt", "content": "evil"})
        return _xml_call("finish", {"summary": "Done", "deliverables": []})

    res_esc = await MissionRunner(
        ScriptedDaemonClient(script_escape), WorkspaceSandbox(ws_esc), store, journal_dir=tmp_path / "journals"
    ).run(ws_row_esc["id"], spec_esc)
    assert res_esc.status == "failed"
    assert not (tmp_path / "pwned.txt").exists()

    # Case C: Wrong-content retry (attempt 1 has wrong number 999 & TODO; attempt 2 fixes and passes; spec SHA unchanged)
    ws_retry = tmp_path / "ws_retry"
    spec_retry = setup_desk_task_workspace(ws_retry, max_attempts=2, max_rounds_per_attempt=6)
    ws_row_retry = store.register_workspace("desk_retry", ws_retry)
    attempt_calls = {"n": 0}

    def script_retry(step_idx: int, req) -> str:
        idx = step_idx
        attempt_calls["n"] += 1
        # AgentLoop resets round count per attempt; detect attempt #2 via prompt failure feedback.
        is_second_attempt = "Previous attempt failed" in req.prompt
        if not is_second_attempt:
            if idx == 0:
                return _xml_call("todo_write", {"items": [{"id": "1", "title": "Bad draft", "status": "in_progress"}]})
            if idx == 1:
                return _xml_call("write_file", {"path": "reports/desk_summary.md", "content": "TODO: North 999"})
            if idx == 2:
                return _xml_call("read_file", {"path": "reports/desk_summary.md"})
            return _xml_call("finish", {"summary": "Draft 999", "deliverables": ["reports/desk_summary.md"]})
        # Attempt 2 writes correct numbers.
        if "Previous attempt failed" in req.prompt and "phase=plan" in req.prompt and "reports/desk_summary.md" not in req.prompt[-200:]:
            pass
        if "<tool_response>" not in req.prompt.split("Previous attempt failed")[-1]:
            return _xml_call("todo_write", {"items": [{"id": "1", "title": "Fix summary", "status": "in_progress"}]})
        tail = req.prompt.split("<|im_start|>tool")[-1]
        if '"status": "written_to_workspace"' in tail:
            return _xml_call("read_file", {"path": "reports/desk_summary.md"})
        if '"sha256":' in tail and "6150" in tail:
            return _xml_call("finish", {"summary": "Fixed 205 and 6150.", "deliverables": ["reports/desk_summary.md"]})
        return _xml_call(
            "write_file",
            {
                "path": "reports/desk_summary.md",
                "content": "North: 120, 3600\nSouth: 85, 2550\nTotal Units: 205\nTotal Revenue: 6150\n",
            },
        )

    res_retry = await MissionRunner(
        ScriptedDaemonClient(script_retry), WorkspaceSandbox(ws_retry), store, journal_dir=tmp_path / "journals"
    ).run(ws_row_retry["id"], spec_retry)
    assert res_retry.status == "passed"
    assert res_retry.attempts == 2
    assert res_retry.reports[0].passed is False
    assert res_retry.reports[1].passed is True
    assert res_retry.spec_sha256 == spec_retry.sha256()

    # Case D: Semantic missions never auto-pass -> strictly human_review_required
    def script_semantic(step_idx: int, req) -> str:
        if step_idx == 0:
            return _xml_call("todo_write", {"items": [{"id": "1", "title": "Write report", "status": "in_progress"}]})
        if step_idx == 1:
            return _xml_call("write_file", {"path": "reports/screen_qa.md", "content": "Dialog shows OK."})
        if step_idx == 2:
            return _xml_call("read_file", {"path": "reports/screen_qa.md"})
        return _xml_call("finish", {"summary": "Done", "deliverables": ["reports/screen_qa.md"]})

    res_sem1 = await MissionRunner(
        ScriptedDaemonClient(script_semantic),
        WorkspaceSandbox(ws_ok),
        store,
        journal_dir=tmp_path / "journals",
        capabilities=Capabilities(vision=True),
    ).run(ws_row_ok["id"], make_screen_qa_spec())
    assert res_sem1.status == "human_review_required"

    res_sem2 = await MissionRunner(
        ScriptedDaemonClient(script_semantic), WorkspaceSandbox(ws_ok), store, journal_dir=tmp_path / "journals"
    ).run(ws_row_ok["id"], make_doc_extract_spec())
    assert res_sem2.status == "human_review_required"

    # Case E: json_schema & numeric_consistency standalone check verification
    schema = {
        "type": "object",
        "required": ["total", "items"],
        "properties": {"total": {"type": "integer"}, "items": {"type": "array", "minItems": 2}},
    }
    assert verify_json_schema_stdlib({"total": 205, "items": [1, 2]}, schema) == []
    assert len(verify_json_schema_stdlib({"total": "bad", "items": [1]}, schema)) >= 1

    (ws_ok / "out.json").write_text(json.dumps({"total": 205, "items": ["North", "South"]}), encoding="utf-8")
    spec_json = AcceptanceSpec(
        name="json_check",
        description="Schema check",
        prompt="x",
        required_deliverables=("out.json",),
        checks=(
            AcceptanceCheck(kind="json_schema", path="out.json", schema=schema),
            AcceptanceCheck(kind="numeric_consistency", path="out.json", expected_numbers=("205",)),
        ),
    )
    assert verify_acceptance_spec(WorkspaceSandbox(ws_ok), spec_json).passed is True
    store.close()


# ---------------------------------------------------------------------------
# 3. LFM3 Single-Quoted List Fallback & </tool_call> Stop Re-attachment
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_lfm3_single_quoted_todo_and_stop_reattach(tmp_path: Path):
    from src.lfm3_harness.protocol import parse_assistant_output

    parsed_sq = parse_assistant_output(
        "<tool_call>\n<function=todo_write>\n<parameter=items>\n[{'id': '1', 'title': 'Read notes', 'status': 'pending'}]\n</parameter>\n</function>\n</tool_call>",
        family="lfm3",
    )
    assert len(parsed_sq.tool_calls) == 1
    assert parsed_sq.tool_calls[0].name == "todo_write"
    assert parsed_sq.tool_calls[0].arguments["items"] == [{"id": "1", "title": "Read notes", "status": "pending"}]

    ws = tmp_path / "ws_reattach"
    ws.mkdir()
    (ws / "in.txt").write_text("Total=42\n", encoding="utf-8")
    sandbox = WorkspaceSandbox(ws)
    store = HarnessStore(tmp_path / "harness.db")
    ws_row = store.register_workspace("ws_reattach", ws)

    class _StopReattachClient:
        def __init__(self):
            self.requests: List[Any] = []

        async def stream_generate(self, req, *, cancel_event=None, read_timeout_s=120.0, session_busy_check=None):
            step = len(self.requests)
            self.requests.append(req)
            yield DaemonEvent("prefill_done", {"prompt_tokens": 100 + step * 20, "cached_tokens": 80 if step else 0})
            if step == 0:
                yield DaemonEvent("token", {"text": "First step.", "thinking": True})
                yield DaemonEvent("token", {"text": "<tool_call>\n<function=todos>\n<parameter=items>\n['Read in.txt', 'Write out.txt']\n</parameter>\n</function>", "thinking": False})
                yield DaemonEvent("done", {"prompt_tokens": 100, "cached_tokens": 0, "stopped_by": "</tool_call>"})
            elif step == 1:
                yield DaemonEvent("token", {"text": "<tool_call>\n<function=write_file>\n<parameter=path>\nout.txt\n</parameter>\n<parameter=content>\nTotal=42\n</parameter>\n</function>", "thinking": False})
                yield DaemonEvent("done", {"prompt_tokens": 120, "cached_tokens": 80, "stopped_by": "</tool_call>"})
            elif step == 2:
                yield DaemonEvent("token", {"text": "<tool_call>\n<function=read_file>\n<parameter=path>\nout.txt\n</parameter>\n</function>", "thinking": False})
                yield DaemonEvent("done", {"prompt_tokens": 140, "cached_tokens": 80, "stopped_by": "</tool_call>"})
            else:
                yield DaemonEvent("token", {"text": "<tool_call>\n<function=finish>\n<parameter=summary>\nTotal 42\n</parameter>\n<parameter=deliverables>\nout.txt\n</parameter>\n</function>", "thinking": False})
                yield DaemonEvent("done", {"prompt_tokens": 160, "cached_tokens": 80, "stopped_by": "</tool_call>"})

    client = _StopReattachClient()
    loop = AgentLoop(client, sandbox, store, family="lfm3", auto_approve_writes=True, enable_thinking=True, think_budget=64)
    seen_prompts: List[str] = []
    res = await loop.run(ws_row["id"], "Write out.txt", on_round_prompt=lambda _r, p: seen_prompts.append(p))

    assert res.status == "completed"
    assert client.requests[0].think_budget == 64
    assert "</tool_call>" in client.requests[0].stop
    assert res.todos[0]["title"] == "Read in.txt"
    gen_tail = "<|im_start|>assistant\n"
    for idx in range(len(seen_prompts) - 1):
        prev_prefix = seen_prompts[idx][: -len(gen_tail)]
        assert seen_prompts[idx + 1].startswith(prev_prefix)

    missing_res = loop.executor.execute("read_file", {})
    assert not missing_res.ok and "Missing required 'path'" in (missing_res.error or "")
    nf_res = loop.executor.execute("read_file", {"path": "no_such.txt"})
    assert not nf_res.ok and "Existing workspace files:" in (nf_res.error or "") and "in.txt" in (nf_res.error or "")
    esc_res = loop.executor.execute("read_file", {"path": "../outside.txt"})
    assert not esc_res.ok and "Existing workspace files:" not in (esc_res.error or "")
    store.close()


# ---------------------------------------------------------------------------
# 4. Duplicate-Call Loop Breaker, Aliasing (todos_write), and Post-Write Re-read
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_duplicate_ro_loop_breaker_and_post_write_reread(tmp_path: Path):
    ws = tmp_path / "ws_dup"
    ws.mkdir()
    (ws / "in.txt").write_text("Val=77\n", encoding="utf-8")
    sandbox = WorkspaceSandbox(ws)
    store = HarnessStore(tmp_path / "harness.db")
    ws_row = store.register_workspace("ws_dup", ws)

    def scripted_dup(step_idx: int, req) -> str:
        if step_idx == 0:
            return _xml_call("todos_write", {"items": [{"id": "1", "title": "Do task", "status": "in_progress"}]})
        if step_idx == 1:
            return _xml_call("read_file", {"path": "in.txt"})
        if step_idx == 2:
            # Repeat exact read_file -> must return compact already_executed hint!
            return _xml_call("read_file", {"path": "in.txt"})
        if step_idx == 3:
            return _xml_call("write_file", {"path": "out.txt", "content": "Val=77\n"})
        if step_idx == 4:
            # Post-write read_file on same target must execute fresh after write cleared seen_ro_calls
            return _xml_call("read_file", {"path": "out.txt"})
        return _xml_call("finish", {"summary": "Verified Val 77", "deliverables": ["out.txt"]})

    client = ScriptedDaemonClient(scripted_dup)
    loop = AgentLoop(client, sandbox, store, family="lfm3", auto_approve_writes=True, enable_thinking=False)
    res = await loop.run(ws_row["id"], "Write out.txt")
    assert res.status == "completed"
    assert "already_executed" in client.prompts[3] and "DUPLICATE CALL BLOCKED" in client.prompts[3]
    assert "read_files=['in.txt']" in client.prompts[3]
    assert "next=call finish now" in client.prompts[5]
    store.close()


# ---------------------------------------------------------------------------
# 5. LFM3 Force-Tool-Call Prefill & Stray XML Tag Sanitization
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_lfm3_force_tool_call_prefill_and_tag_sanitization(tmp_path: Path):
    ws = tmp_path / "ws_force"
    ws.mkdir()
    (ws / "in.txt").write_text("Val=88\n", encoding="utf-8")
    sandbox = WorkspaceSandbox(ws)
    store = HarnessStore(tmp_path / "harness.db")
    ws_row = store.register_workspace("ws_force", ws)

    class _SuffixClient:
        def __init__(self):
            self.requests: List[Any] = []

        async def stream_generate(self, req, *, cancel_event=None, read_timeout_s=120.0, session_busy_check=None):
            step = len(self.requests)
            self.requests.append(req)
            yield DaemonEvent("prefill_done", {"prompt_tokens": 100 + step * 20, "cached_tokens": 80 if step else 0})
            if step == 0:
                chunk = "todos>\n<parameter=items>\n['Read in.txt']\n</parameter>\n</function>\n"
            elif step == 1:
                chunk = "write_file>\n<parameter=path>\nout.txt\n</parameter>\n<parameter=content>\nVal=88\n</parameter>\n</function>\n"
            elif step == 2:
                chunk = "read_file>\n<parameter=path>\nout.txt\n</parameter>\n</function>\n"
            else:
                chunk = 'finish>\n<parameter=summary>\nVerified Val 88\n</parameter>\n<parameter=deliverables>\n["out.txt"]\n</deliverables>\n</parameter>\n</function>\n'
            yield DaemonEvent("token", {"text": chunk, "thinking": False})
            yield DaemonEvent("done", {"prompt_tokens": 100 + step * 20, "cached_tokens": 80 if step else 0, "stopped_by": "</tool_call>"})

    client = _SuffixClient()
    loop = AgentLoop(client, sandbox, store, family="lfm3", auto_approve_writes=True, enable_thinking=False, force_tool_call=True)
    seen_prompts: List[str] = []
    res = await loop.run(ws_row["id"], "Write out.txt", on_round_prompt=lambda _r, p: seen_prompts.append(p))

    assert res.status == "completed"
    assert res.deliverables[0].rel_path == "out.txt"
    assert all(req.prompt.endswith("<tool_call>\n<function=") for req in client.requests)
    for idx in range(len(seen_prompts) - 1):
        assert seen_prompts[idx + 1].startswith(seen_prompts[idx]), f"Full-prompt prefix mutated at round {idx + 1}!"
    store.close()


# ---------------------------------------------------------------------------
# 6. LFM3 Hybrid Think-in-Plan + Force-Tool-Call Work/Review
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_lfm3_hybrid_think_in_plan_then_force_tool_call(tmp_path: Path):
    ws = tmp_path / "ws_hybrid"
    ws.mkdir()
    (ws / "in.txt").write_text("Val=99\n", encoding="utf-8")
    sandbox = WorkspaceSandbox(ws)
    store = HarnessStore(tmp_path / "harness.db")
    ws_row = store.register_workspace("ws_hybrid", ws)

    class _HybridClient:
        def __init__(self):
            self.requests: List[Any] = []

        async def stream_generate(self, req, *, cancel_event=None, read_timeout_s=120.0, session_busy_check=None):
            step = len(self.requests)
            self.requests.append(req)
            yield DaemonEvent("prefill_done", {"prompt_tokens": 120 + step * 20, "cached_tokens": 100 if step else 0})
            if step == 0:
                yield DaemonEvent("token", {"text": "Plan steps carefully.", "thinking": True})
                yield DaemonEvent("token", {"text": '<tool_call>\n<function=todo_write>\n<parameter=items>\n[{"id": "1", "title": "Write out.txt", "status": "in_progress"}]\n</parameter>\n</function>', "thinking": False})
            elif step == 1:
                yield DaemonEvent("token", {"text": "write_file>\n<parameter=path>\nout.txt\n</parameter>\n<parameter=content>\nVal=99\n</parameter>\n</function>\n", "thinking": False})
            elif step == 2:
                yield DaemonEvent("token", {"text": "read_file>\n<parameter=path>\nout.txt\n</parameter>\n</function>\n", "thinking": False})
            else:
                yield DaemonEvent("token", {"text": 'finish>\n<parameter=summary>\nVerified Val 99\n</parameter>\n<parameter=deliverables>\n["out.txt"]\n</parameter>\n</function>\n', "thinking": False})
            yield DaemonEvent("done", {"prompt_tokens": 120 + step * 20, "cached_tokens": 100 if step else 0, "stopped_by": "</tool_call>"})

    client = _HybridClient()
    loop = AgentLoop(
        client,
        sandbox,
        store,
        family="lfm3",
        auto_approve_writes=True,
        enable_thinking=False,
        force_tool_call=True,
        think_in_plan=True,
        plan_think_budget=112,
    )
    seen_prompts: List[str] = []
    res = await loop.run(ws_row["id"], "Write out.txt", on_round_prompt=lambda _r, p: seen_prompts.append(p))

    assert res.status == "completed"
    assert res.deliverables[0].rel_path == "out.txt"
    assert client.requests[0].no_think is False
    assert client.requests[0].think_budget == 112
    assert not client.requests[0].prompt.endswith("<tool_call>\n<function=")
    for req in client.requests[1:]:
        assert req.no_think is True
        assert req.prompt.endswith("<tool_call>\n<function=")
    r0_base = seen_prompts[0].rsplit("<|im_start|>assistant\n", 1)[0] + "<|im_start|>assistant\n"
    assert seen_prompts[1].startswith(r0_base)
    assert seen_prompts[2].startswith(seen_prompts[1])
    store.close()
