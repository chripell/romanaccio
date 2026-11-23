"""Bounded 3-phase (plan / work / review) agent execution loop with strict prefix-cache stability."""

import ast
import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .approvals import ApprovalManager
from .backend import BackendError, DaemonClient, GenerateRequest
from .config import MAX_PLAN_STEPS, Capabilities, ModelFamily, SafetyLimits
from .paths import PathSandboxError, WorkspaceSandbox
from .protocol import ParsedAssistantTurn, parse_assistant_output, render_prompt
from .store import HarnessStore
from .tools import StagedWriteProposal, ToolExecutor, get_tool_schemas

_READ_ONLY_TOOLS = frozenset({"list_files", "read_file", "search_files", "related_files", "look", "transcribe"})
_PLAN_PHASE_TOOLS = frozenset({"todo_write", "list_files", "read_file", "search_files", "related_files"})
_WORK_PHASE_TOOLS = frozenset(
    {"todo_write", "list_files", "read_file", "search_files", "related_files", "write_file", "edit_file", "finish", "look", "transcribe"}
)
_SALIENT_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.-])(\d+(?:\.\d+)?)(?![A-Za-z0-9_.])")


@dataclass(frozen=True)
class TodoCheck:
    """Deterministic workspace check attached to a plan item."""

    kind: str  # "file_exists" | "dir_exists" | "content_matches"
    path: str
    pattern: str = ""


@dataclass
class TodoItem:
    """Mutable checklist item maintained by todo_write and verified deterministically."""

    id: str
    title: str
    status: str  # "pending" | "in_progress" | "done"
    check: Optional[TodoCheck] = None
    demoted_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "title": self.title, "status": self.status}
        if self.check is not None:
            d["check"] = asdict(self.check)
        if self.demoted_reason:
            d["demoted_reason"] = self.demoted_reason
        return d


@dataclass(frozen=True)
class DeliverableProof:
    """Measured workspace proof recorded when a job deliverable passes finish()."""

    rel_path: str
    bytes: int
    sha256: str
    verified_read: bool = True


@dataclass(frozen=True)
class NumericGroundingReport:
    """Verification of salient numeric claims in finish summary against observed workspace text."""

    cited_numbers: List[str]
    grounded_numbers: List[str]
    ungrounded_numbers: List[str]
    grounded_ratio: float


@dataclass(frozen=True)
class RoundTelemetry:
    """Per-round daemon cache and timing snapshot inside AgentLoop."""

    round_num: int
    phase: str
    prompt_tokens: int
    cached_tokens: int
    cache_status: str
    cache_tier: str
    cache_restore_ms: float
    prefill_ms: float
    decode_ms: float
    tool_calls: List[str]


@dataclass(frozen=True)
class JobRunResult:
    """Terminal result of a bounded AgentLoop job execution."""

    job_id: str
    status: str  # "completed" | "awaiting_approval" | "failed" | "cancelled" | "max_rounds"
    phase: str
    summary: str
    deliverables: List[DeliverableProof]
    numeric_grounding: NumericGroundingReport
    rounds: List[RoundTelemetry]
    total_tool_calls: int
    pending_action_ids: List[str]
    todos: List[Dict[str, Any]]
    error: str = ""


def evaluate_todo_check(sandbox: WorkspaceSandbox, check: TodoCheck) -> Tuple[bool, str]:
    """Evaluate a deterministic workspace check for a todo item."""
    kind = (check.kind or "").strip()
    try:
        if kind == "dir_exists":
            p = sandbox.resolve_dir(check.path, must_exist=True)
            return True, f"Directory exists: {sandbox.relativize(p)}"
        if kind == "file_exists":
            p = sandbox.resolve_read(check.path)
            return True, f"File exists ({p.stat().st_size} B): {sandbox.relativize(p)}"
        if kind == "content_matches":
            p = sandbox.resolve_read(check.path)
            text = p.read_text(encoding="utf-8", errors="replace")
            if check.pattern and check.pattern in text:
                return True, f"Pattern matched in {sandbox.relativize(p)}"
            return False, f"Pattern {check.pattern!r} not found in {check.path}"
        return False, f"Unsupported todo check kind: {kind}"
    except PathSandboxError as exc:
        return False, f"Workspace check failed ({exc.reason}): {exc}"
    except OSError as exc:
        return False, f"Workspace check I/O failed: {exc}"


def check_numeric_grounding(summary: str, observed_texts: Sequence[str]) -> NumericGroundingReport:
    """Check whether multi-digit/decimal numbers cited in summary exist in observed texts."""
    seen: Set[str] = set()
    cited: List[str] = []
    for match in _SALIENT_NUM_RE.finditer(summary or ""):
        num = match.group(1)
        if ("." not in num and int(num) < 10) or num in seen:
            continue
        seen.add(num)
        cited.append(num)

    corpus = "\n".join(t for t in observed_texts if t)
    grounded: List[str] = []
    ungrounded: List[str] = []
    for num in cited:
        pat = re.compile(rf"(?<![0-9.]){re.escape(num)}(?![0-9])")
        if pat.search(corpus):
            grounded.append(num)
        else:
            ungrounded.append(num)

    ratio = (len(grounded) / float(len(cited))) if cited else 1.0
    return NumericGroundingReport(
        cited_numbers=cited,
        grounded_numbers=grounded,
        ungrounded_numbers=ungrounded,
        grounded_ratio=round(ratio, 4),
    )


def _format_todo_compact(todos: Sequence[TodoItem]) -> str:
    if not todos:
        return "none"
    return "; ".join(f"[{t.status}] {t.id}:{t.title}" for t in todos)


_TOOL_NAME_ALIASES: Dict[str, str] = {
    "todos": "todo_write",
    "todo": "todo_write",
    "todo_list": "todo_write",
    "to_do_list": "todo_write",
    "write_todo": "todo_write",
    "write_todos": "todo_write",
    "update_todo": "todo_write",
    "update_todos": "todo_write",
    "create_todo": "todo_write",
    "create_todos": "todo_write",
    "todos_write": "todo_write",
    "read": "read_file",
    "open_file": "read_file",
    "view_file": "read_file",
    "write": "write_file",
    "create_file": "write_file",
    "save_file": "write_file",
    "edit": "edit_file",
    "patch_file": "edit_file",
    "replace_in_file": "edit_file",
    "list": "list_files",
    "ls": "list_files",
    "list_dir": "list_files",
    "list_directory": "list_files",
    "list_file": "list_files",
    "list_fields": "list_files",
    "search": "search_files",
    "grep": "search_files",
    "find_files": "search_files",
    "search_file": "search_files",
    "search_fields": "search_files",
    "done": "finish",
    "complete": "finish",
    "submit": "finish",
}


def _normalize_tool_name(name: str, allowed_set: Set[str]) -> str:
    """Resolve exact tool name, CamelCase variant, or common synonym."""
    raw = (name or "").strip()
    if raw in allowed_set:
        return raw
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw).lower()
    if snake in allowed_set:
        return snake
    compact = raw.replace("_", "").lower()
    for cand in allowed_set:
        if cand.replace("_", "").lower() == compact:
            return cand
    for alias_key in (snake, raw.lower()):
        mapped = _TOOL_NAME_ALIASES.get(alias_key)
        if mapped and mapped in allowed_set:
            return mapped
    return raw


def _build_job_system_contract(family: ModelFamily) -> str:
    if family == "lfm2":
        syntax = (
            "Emit tool calls using exact lowercase tool names in bracket syntax:\n"
            "<|tool_call_start|>[tool_name(param=\"value\")]<|tool_call_end|>"
        )
    else:
        syntax = (
            "Keep <think> to 1-2 short sentences, then emit ONE XML tool call and stop immediately.\n"
            "Example:\n"
            "<tool_call>\n<function=todo_write>\n<parameter=items>\n"
            "[{\"id\": \"1\", \"title\": \"Read inbox files\", \"status\": \"in_progress\"}]\n"
            "</parameter>\n</function>\n</tool_call>"
        )
    return (
        "You are a local workspace execution agent operating under strict 3-phase rules.\n"
        "Phase 1 (plan): First call todo_write with 3-4 brief steps ({id, title, status} only).\n"
        "Phase 2 (work): Read source files carefully with list_files/read_file/search_files, then create/edit files with write_file/edit_file.\n"
        "Phase 3 (review): After ANY file write or edit, call read_file on that modified file BEFORE calling finish.\n"
        "Keep arguments concise, never invent numbers or paths, and wait for each tool response.\n"
        f"{syntax}"
    )


def _parse_container_arg(raw: Any) -> Any:
    """Decode JSON or single-quoted Python list/dict strings emitted inside XML parameters."""
    if not isinstance(raw, str):
        return raw
    trimmed = re.sub(r"\s*</[A-Za-z0-9_]+>\s*$", "", raw.strip())
    if not ((trimmed.startswith("[") and trimmed.endswith("]")) or (trimmed.startswith("{") and trimmed.endswith("}"))):
        return raw
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(trimmed)
        except (ValueError, SyntaxError):
            return raw


class AgentLoop:
    """Bounded plan/work/review loop maintaining strictly append-only prefix-cached prompts."""

    def __init__(
        self,
        client: DaemonClient,
        sandbox: WorkspaceSandbox,
        store: HarnessStore,
        *,
        capabilities: Optional[Capabilities] = None,
        limits: Optional[SafetyLimits] = None,
        family: ModelFamily = "lfm3",
        approval_manager: Optional[ApprovalManager] = None,
        auto_approve_writes: bool = False,
        enable_thinking: bool = False,
        think_budget: int = 0,
        force_tool_call: bool = False,
        think_in_plan: bool = False,
        plan_think_budget: int = 112,
        temperature: float = 0.0,
        max_tokens_per_round: int = 512,
    ):
        self.client = client
        self.sandbox = sandbox
        self.store = store
        self.capabilities = capabilities or Capabilities()
        self.limits = limits or SafetyLimits()
        self.family: ModelFamily = family
        self.approval_manager = approval_manager
        self.auto_approve_writes = auto_approve_writes
        self.enable_thinking = enable_thinking
        self.think_budget = max(0, int(think_budget))
        self.force_tool_call = bool(force_tool_call)
        self.think_in_plan = bool(think_in_plan)
        self.plan_think_budget = max(0, int(plan_think_budget))
        self.temperature = temperature
        self.max_tokens_per_round = max_tokens_per_round
        self.executor = ToolExecutor(self.sandbox, self.capabilities, self.limits)
        self.system_contract = _build_job_system_contract(self.family)
        self.tool_schemas = get_tool_schemas(self.capabilities, include_job_tools=True)

    async def run(
        self,
        workspace_id: str,
        goal: str,
        *,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        required_deliverables: Sequence[str] = (),
        max_rounds: Optional[int] = None,
        cancel_event: Optional[asyncio.Event] = None,
        on_round_prompt: Optional[Callable[[int, str], None]] = None,
        on_round_complete: Optional[Callable[[RoundTelemetry, str], None]] = None,
    ) -> JobRunResult:
        """Execute a bounded 3-phase job and persist full progress in SQLite."""
        job_row = self.store.create_job(workspace_id, title or goal[:80], session_id=session_id, phase="plan", status="running")
        job_id = str(job_row["id"])
        cache_key = prompt_cache_key or f"job:{job_id}"
        max_r = max(1, min(self.limits.max_job_turns, int(max_rounds or self.limits.max_tool_rounds)))

        req_delivs: List[str] = []
        for raw_p in required_deliverables:
            norm = self.sandbox.relativize(self.sandbox.resolve_write(raw_p))
            if norm not in req_delivs:
                req_delivs.append(norm)

        phase = "plan"
        todos: List[TodoItem] = []
        written_files: Set[str] = set()
        read_after_write: Dict[str, bool] = {}
        observed_texts: Dict[str, str] = {}
        rounds: List[RoundTelemetry] = []
        pending_action_ids: List[str] = []
        total_calls = 0
        nudges = 0
        last_prompt_tokens = 0
        seen_ro_calls: Set[Tuple[str, str]] = set()
        last_todo_sig: Optional[str] = None

        # Messages stay immutable once appended so Round r+1 extends Round r cleanly in prefix cache.
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": f"{goal.strip()}\n\n<harness_state>\nphase=plan\ntodo=none\n</harness_state>",
            }
        ]

        for round_num in range(1, max_r + 1):
            if cancel_event is not None and cancel_event.is_set():
                self.store.update_job(job_id, phase=phase, status="cancelled", todo=[t.to_dict() for t in todos])
                return JobRunResult(
                    job_id=job_id,
                    status="cancelled",
                    phase=phase,
                    summary="",
                    deliverables=[],
                    numeric_grounding=check_numeric_grounding("", []),
                    rounds=rounds,
                    total_tool_calls=total_calls,
                    pending_action_ids=pending_action_ids,
                    todos=[t.to_dict() for t in todos],
                    error="Cancelled",
                )

            round_enable_thinking = self.enable_thinking or (self.think_in_plan and phase == "plan")
            rendered = render_prompt(
                messages,
                system_contract=self.system_contract,
                workspace_root=self.sandbox.root,
                tool_schemas=self.tool_schemas,
                family=self.family,
                enable_thinking=round_enable_thinking,
                preserve_thinking=True,
            )
            tool_lead = "<tool_call>\n<function=" if (self.force_tool_call and self.family == "lfm3" and not round_enable_thinking) else ""
            round_prompt = rendered.prompt + tool_lead
            if on_round_prompt is not None:
                on_round_prompt(round_num, round_prompt)

            stop_seqs = list(rendered.stop_sequences)
            if self.family == "lfm3" and "</tool_call>" not in stop_seqs:
                stop_seqs.append("</tool_call>")

            round_think_budget = (
                self.plan_think_budget
                if (self.think_in_plan and phase == "plan" and self.think_budget == 0)
                else self.think_budget
            )
            if self.family == "lfm3" and round_think_budget > 0 and last_prompt_tokens >= 6800:
                round_think_budget = min(round_think_budget, 64)

            raw_answer_parts: List[str] = []
            raw_think_parts: List[str] = []
            done_payload: Dict[str, Any] = {}
            try:
                async for ev in self.client.stream_generate(
                    GenerateRequest(
                        prompt=round_prompt,
                        max_tokens=self.max_tokens_per_round,
                        temperature=self.temperature,
                        think_budget=round_think_budget,
                        no_think=not round_enable_thinking,
                        raw=True,
                        cache=True,
                        prompt_cache_key=cache_key,
                        pin_blocks_up_to=rendered.pin_blocks_up_to,
                        stop=stop_seqs,
                    ),
                    cancel_event=cancel_event,
                ):
                    if ev.type == "token":
                        piece = str(ev.payload.get("text", ""))
                        if ev.payload.get("thinking"):
                            raw_think_parts.append(piece)
                        else:
                            raw_answer_parts.append(piece)
                    elif ev.type == "done":
                        done_payload = dict(ev.payload)
            except (BackendError, OSError) as exc:
                self.store.update_job(job_id, phase=phase, status="failed", todo=[t.to_dict() for t in todos])
                return JobRunResult(
                    job_id=job_id,
                    status="failed",
                    phase=phase,
                    summary="",
                    deliverables=[],
                    numeric_grounding=check_numeric_grounding("", []),
                    rounds=rounds,
                    total_tool_calls=total_calls,
                    pending_action_ids=pending_action_ids,
                    todos=[t.to_dict() for t in todos],
                    error=f"Backend failure: {exc}",
                )

            raw_text = "".join(raw_answer_parts)
            if tool_lead and not raw_text.lstrip().startswith("<tool_call>"):
                raw_text = tool_lead + raw_text
            if self.family == "lfm3" and done_payload.get("stopped_by") == "</tool_call>" and "</tool_call>" not in raw_text:
                raw_text += "</tool_call>"
            think_text = "".join(raw_think_parts).strip() or None
            parsed: ParsedAssistantTurn = parse_assistant_output(raw_text, family=self.family, thinking_override=think_text)

            call_names = [tc.name for tc in parsed.tool_calls]
            r_tel = RoundTelemetry(
                round_num=round_num,
                phase=phase,
                prompt_tokens=int(done_payload.get("prompt_tokens", 0)),
                cached_tokens=int(done_payload.get("cached_tokens", 0)),
                cache_status=str(done_payload.get("cache_status", "UNKNOWN")),
                cache_tier=str(done_payload.get("cache_tier", "unknown")),
                cache_restore_ms=float(done_payload.get("cache_restore_ms", 0.0)),
                prefill_ms=float(done_payload.get("prefill_ms", 0.0)),
                decode_ms=float(done_payload.get("decode_ms", 0.0)),
                tool_calls=call_names,
            )
            last_prompt_tokens = r_tel.prompt_tokens
            rounds.append(r_tel)
            if on_round_complete is not None:
                on_round_complete(r_tel, raw_text)

            asst_msg: Dict[str, Any] = {
                "role": "assistant",
                "thinking": parsed.thinking,
                "content": parsed.content,
            }
            if parsed.tool_calls:
                asst_msg["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in parsed.tool_calls
                ]
            messages.append(asst_msg)

            if not parsed.tool_calls:
                nudges += 1
                if nudges > 3:
                    self.store.update_job(job_id, phase=phase, status="failed", todo=[t.to_dict() for t in todos])
                    return JobRunResult(
                        job_id=job_id,
                        status="failed",
                        phase=phase,
                        summary=parsed.content,
                        deliverables=[],
                        numeric_grounding=check_numeric_grounding(parsed.content, list(observed_texts.values())),
                        rounds=rounds,
                        total_tool_calls=total_calls,
                        pending_action_ids=pending_action_ids,
                        todos=[t.to_dict() for t in todos],
                        error="Actor did not emit tool calls after repeated nudges",
                    )
                if self.family == "lfm2":
                    fmt_hint = "<|tool_call_start|>[todo_write(items=[...])]<|tool_call_end|>" if phase == "plan" else "<|tool_call_start|>[read_file(path=\"inbox/notes.txt\")]<|tool_call_end|>"
                elif phase == "plan":
                    fmt_hint = (
                        "<tool_call>\n<function=todo_write>\n<parameter=items>\n"
                        "[{\"id\": \"1\", \"title\": \"Read inbox files\", \"status\": \"in_progress\"}]\n"
                        "</parameter>\n</function>\n</tool_call>"
                    )
                else:
                    fmt_hint = (
                        "<tool_call>\n<function=read_file>\n<parameter=path>\n"
                        "inbox/notes.txt\n"
                        "</parameter>\n</function>\n</tool_call>"
                    )
                nudge_text = (
                    f"Action required: call a valid tool in phase '{phase}' using exact syntax:\n{fmt_hint}"
                    if phase != "plan"
                    else f"Action required: start in phase 'plan' by calling todo_write using exact syntax:\n{fmt_hint}"
                )
                if parsed.parse_errors:
                    nudge_text += f" Parse error on previous output: {'; '.join(parsed.parse_errors[:2])}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"nudge_{round_num}",
                        "content": f"{nudge_text}\n\n<harness_state>\nphase={phase}\ntodo={_format_todo_compact(todos)}\n</harness_state>",
                    }
                )
                continue

            allowed = _PLAN_PHASE_TOOLS if phase == "plan" else _WORK_PHASE_TOOLS
            tool_outputs: List[Tuple[str, str]] = []
            finish_outcome: Optional[JobRunResult] = None
            awaiting_approval = False

            all_known = _PLAN_PHASE_TOOLS | _WORK_PHASE_TOOLS
            for tc in parsed.tool_calls:
                total_calls += 1
                t_name = _normalize_tool_name(tc.name, all_known)
                if t_name not in allowed:
                    err_msg = (
                        f"Unknown tool '{t_name}'. Call one of: {sorted(allowed)}."
                        if t_name not in all_known
                        else f"Tool '{t_name}' not allowed in phase '{phase}'. Complete planning with todo_write first. Allowed now: {sorted(allowed)}."
                    )
                    tool_outputs.append(
                        (
                            tc.call_id,
                            json.dumps({"ok": False, "error": err_msg}),
                        )
                    )
                    continue

                if t_name == "todo_write":
                    todo_sig = json.dumps(tc.arguments, sort_keys=True, default=str)
                    if phase != "plan" and last_todo_sig == todo_sig:
                        tool_outputs.append(
                            (
                                tc.call_id,
                                json.dumps(
                                    {
                                        "ok": True,
                                        "already_recorded": True,
                                        "hint": "Checklist already recorded. Proceed to read remaining files, write deliverable, or call finish.",
                                    }
                                ),
                            )
                        )
                        continue
                    ok_todo, msg_todo, new_todos = self._handle_todo_write(tc.arguments)
                    if ok_todo:
                        last_todo_sig = todo_sig
                        todos = new_todos
                        if phase == "plan":
                            phase = "work"
                        self.store.update_job(job_id, phase=phase, todo=[t.to_dict() for t in todos])
                    tool_outputs.append((tc.call_id, msg_todo))
                    continue

                if t_name == "finish":
                    ok_fin, fin_msg, deliverables_list, grounding = self._handle_finish(
                        tc.arguments,
                        req_delivs=req_delivs,
                        written_files=written_files,
                        read_after_write=read_after_write,
                        todos=todos,
                        observed_texts=observed_texts,
                    )
                    tool_outputs.append((tc.call_id, fin_msg))
                    if ok_fin:
                        phase = "review"
                        summary_str = str(tc.arguments.get("summary") or "").strip()
                        self.store.update_job(
                            job_id,
                            phase="review",
                            status="completed",
                            todo=[t.to_dict() for t in todos],
                            deliverable={
                                "summary": summary_str,
                                "files": [asdict(d) for d in deliverables_list],
                                "numeric_grounding": asdict(grounding),
                            },
                        )
                        self.store.log_event(
                            "job",
                            job_id,
                            "job_completed",
                            {"deliverables": [d.rel_path for d in deliverables_list], "rounds": len(rounds)},
                        )
                        finish_outcome = JobRunResult(
                            job_id=job_id,
                            status="completed",
                            phase="review",
                            summary=summary_str,
                            deliverables=deliverables_list,
                            numeric_grounding=grounding,
                            rounds=rounds,
                            total_tool_calls=total_calls,
                            pending_action_ids=pending_action_ids,
                            todos=[t.to_dict() for t in todos],
                        )
                        break
                    else:
                        self.store.update_job(job_id, phase=phase, todo=[t.to_dict() for t in todos])
                    continue

                if t_name in ("write_file", "edit_file"):
                    res = self.executor.execute(t_name, tc.arguments, allowed_tools=tuple(allowed))
                    if not res.ok or res.staged_proposal is None:
                        tool_outputs.append((tc.call_id, json.dumps({"ok": False, "error": res.error or "Staging failed"})))
                        continue

                    prop: StagedWriteProposal = res.staged_proposal
                    if self.auto_approve_writes:
                        try:
                            receipt = self.sandbox.atomic_write_text(
                                prop.rel_path,
                                prop.new_content,
                                expected_sha256=prop.base_sha256,
                                max_bytes=self.limits.max_write_bytes,
                            )
                            written_files.add(receipt.rel_path)
                            read_after_write[receipt.rel_path] = False
                            seen_ro_calls.clear()
                            phase = "review"
                            self.store.update_job(job_id, phase=phase, todo=[t.to_dict() for t in todos])
                            tool_outputs.append(
                                (
                                    tc.call_id,
                                    json.dumps(
                                        {
                                            "ok": True,
                                            "status": "written_to_workspace",
                                            "path": receipt.rel_path,
                                            "bytes": receipt.bytes_written,
                                            "sha256": receipt.sha256,
                                            "next_required": f"Call read_file(path='{receipt.rel_path}') before finish().",
                                        }
                                    ),
                                )
                            )
                        except PathSandboxError as exc:
                            tool_outputs.append((tc.call_id, json.dumps({"ok": False, "error": str(exc)})))
                    else:
                        if self.approval_manager is None:
                            tool_outputs.append(
                                (tc.call_id, json.dumps({"ok": False, "error": "No ApprovalManager configured"}))
                            )
                            continue
                        stage = await self.approval_manager.stage_proposal(
                            session_id or job_id,
                            f"{job_id}_r{round_num}",
                            goal,
                            prop,
                        )
                        if stage.held:
                            tool_outputs.append(
                                (
                                    tc.call_id,
                                    json.dumps({"ok": False, "status": "held_by_write_gate", "revision_note": stage.revision_note}),
                                )
                            )
                        elif stage.action is not None:
                            pending_action_ids.append(str(stage.action["id"]))
                            awaiting_approval = True
                            tool_outputs.append(
                                (
                                    tc.call_id,
                                    json.dumps(
                                        {"ok": True, "status": "awaiting_approval", "action_id": stage.action["id"], "path": prop.rel_path}
                                    ),
                                )
                            )
                    continue

                # Read-only or sense tools
                ro_sig = (t_name, json.dumps(tc.arguments, sort_keys=True, default=str))
                if t_name in _READ_ONLY_TOOLS and ro_sig in seen_ro_calls:
                    tool_outputs.append(
                        (
                            tc.call_id,
                            json.dumps(
                                {
                                    "ok": False,
                                    "already_executed": True,
                                    "tool": t_name,
                                    "read_files": [k for k in sorted(observed_texts) if not k.startswith("__")],
                                    "written_files": sorted(written_files),
                                    "error": f"DUPLICATE CALL BLOCKED: already called {t_name} with these exact arguments. Read any unread files, call write_file to create deliverable, or call finish after reading back deliverable.",
                                }
                            ),
                        )
                    )
                    continue
                res = self.executor.execute(t_name, tc.arguments, allowed_tools=tuple(allowed))
                if not res.ok:
                    tool_outputs.append((tc.call_id, json.dumps({"ok": False, "error": res.error})))
                else:
                    if t_name in _READ_ONLY_TOOLS:
                        seen_ro_calls.add(ro_sig)
                    if t_name == "read_file":
                        try:
                            parsed_rd = json.loads(res.content)
                            norm_path = str(parsed_rd.get("path", ""))
                            if norm_path:
                                observed_texts[norm_path] = str(parsed_rd.get("content", ""))
                                if norm_path in written_files or norm_path in req_delivs:
                                    read_after_write[norm_path] = True
                        except Exception:
                            pass
                    elif t_name == "search_files":
                        observed_texts[f"__search_{round_num}"] = res.content
                    tool_outputs.append((tc.call_id, res.content))

            # Append tool messages in order, attaching the round's <harness_state> to the last one only.
            for idx_out, (cid, out_body) in enumerate(tool_outputs):
                body = out_body
                if idx_out == len(tool_outputs) - 1 and finish_outcome is None:
                    read_list = [k for k in sorted(observed_texts) if not k.startswith("__")]
                    state_parts = [f"phase={phase}", f"todo={_format_todo_compact(todos)}"]
                    if read_list:
                        state_parts.append(f"read_files={read_list}")
                    if written_files:
                        state_parts.append(f"written_files={sorted(written_files)}")
                    if phase == "review":
                        ready_paths = sorted(set(written_files) | set(req_delivs))
                        if ready_paths and all(read_after_write.get(p, False) for p in ready_paths):
                            state_parts.append(f"verified_deliverables={ready_paths}")
                            state_parts.append("next=call finish now")
                    body += "\n\n<harness_state>\n" + "\n".join(state_parts) + "\n</harness_state>"
                messages.append({"role": "tool", "tool_call_id": cid, "content": body})

            if finish_outcome is not None:
                return finish_outcome

            if awaiting_approval:
                self.store.update_job(job_id, phase=phase, status="awaiting_approval", todo=[t.to_dict() for t in todos])
                return JobRunResult(
                    job_id=job_id,
                    status="awaiting_approval",
                    phase=phase,
                    summary="Staged proposed file changes awaiting human approval.",
                    deliverables=[],
                    numeric_grounding=check_numeric_grounding("", []),
                    rounds=rounds,
                    total_tool_calls=total_calls,
                    pending_action_ids=pending_action_ids,
                    todos=[t.to_dict() for t in todos],
                )

        self.store.update_job(job_id, phase=phase, status="failed", todo=[t.to_dict() for t in todos])
        return JobRunResult(
            job_id=job_id,
            status="max_rounds",
            phase=phase,
            summary="",
            deliverables=[],
            numeric_grounding=check_numeric_grounding("", list(observed_texts.values())),
            rounds=rounds,
            total_tool_calls=total_calls,
            pending_action_ids=pending_action_ids,
            todos=[t.to_dict() for t in todos],
            error=f"Reached max_rounds ({max_r}) without passing finish()",
        )

    def _handle_todo_write(self, arguments: Dict[str, Any]) -> Tuple[bool, str, List[TodoItem]]:
        raw_items = _parse_container_arg(arguments.get("items"))
        if not isinstance(raw_items, list) or len(raw_items) == 0:
            return (
                False,
                json.dumps(
                    {
                        "ok": False,
                        "error": 'todo_write requires a non-empty \'items\' array, e.g. [{"id":"1","title":"Read files","status":"in_progress"}].',
                    }
                ),
                [],
            )
        if len(raw_items) > self.limits.max_plan_steps:
            return (
                False,
                json.dumps({"ok": False, "error": f"Too many todo items ({len(raw_items)} > {self.limits.max_plan_steps})"}),
                [],
            )

        items: List[TodoItem] = []
        demotions: List[str] = []
        for idx, raw in enumerate(raw_items, start=1):
            if isinstance(raw, str) and raw.strip():
                raw = {"id": f"step_{idx}", "title": raw.strip(), "status": "pending"}
            elif not isinstance(raw, dict):
                return False, json.dumps({"ok": False, "error": f"Todo item #{idx} must be an object"}), []
            tid = str(raw.get("id") or f"step_{idx}").strip()
            title = str(raw.get("title") or "").strip()
            if not title:
                return False, json.dumps({"ok": False, "error": f"Todo item '{tid}' requires non-empty title"}), []
            status = str(raw.get("status") or "pending").strip().lower()
            if status not in ("pending", "in_progress", "done"):
                status = "pending"

            chk: Optional[TodoCheck] = None
            raw_chk = raw.get("check")
            if isinstance(raw_chk, dict) and raw_chk.get("kind") and raw_chk.get("path"):
                chk = TodoCheck(
                    kind=str(raw_chk["kind"]).strip(),
                    path=str(raw_chk["path"]).strip(),
                    pattern=str(raw_chk.get("pattern") or ""),
                )

            demoted_reason = ""
            if status == "done" and chk is not None:
                passed, msg = evaluate_todo_check(self.sandbox, chk)
                if not passed:
                    status = "in_progress"
                    demoted_reason = msg
                    demotions.append(f"{tid}: {msg}")

            items.append(TodoItem(id=tid, title=title, status=status, check=chk, demoted_reason=demoted_reason))

        payload: Dict[str, Any] = {
            "ok": True,
            "count": len(items),
            "items": [it.to_dict() for it in items],
        }
        if demotions:
            payload["demotions"] = demotions
        return True, json.dumps(payload), items

    def _handle_finish(
        self,
        arguments: Dict[str, Any],
        *,
        req_delivs: Sequence[str],
        written_files: Set[str],
        read_after_write: Dict[str, bool],
        todos: List[TodoItem],
        observed_texts: Dict[str, str],
    ) -> Tuple[bool, str, List[DeliverableProof], NumericGroundingReport]:
        summary = str(arguments.get("summary") or "").strip()
        raw_delivs = _parse_container_arg(arguments.get("deliverables"))
        if not summary:
            return (
                False,
                json.dumps({"ok": False, "error": "finish() requires a non-empty 'summary'"}),
                [],
                check_numeric_grounding("", []),
            )
        if isinstance(raw_delivs, str) and raw_delivs.strip():
            raw_delivs = [raw_delivs.strip()]
        elif not isinstance(raw_delivs, list):
            raw_delivs = []

        merged_paths: List[str] = []
        for p in [*raw_delivs, *req_delivs]:
            s = str(p or "").strip()
            if not s:
                continue
            try:
                norm = self.sandbox.relativize(self.sandbox.resolve_write(s))
            except PathSandboxError as exc:
                return (
                    False,
                    json.dumps({"ok": False, "error": f"Invalid deliverable path '{s}': {exc}"}),
                    [],
                    check_numeric_grounding("", []),
                )
            if norm not in merged_paths:
                merged_paths.append(norm)

        if not merged_paths and written_files:
            return (
                False,
                json.dumps({"ok": False, "error": "finish() must list created/modified files in 'deliverables'"}),
                [],
                check_numeric_grounding("", []),
            )
        if not merged_paths and req_delivs:
            return (
                False,
                json.dumps({"ok": False, "error": f"Missing required deliverables: {list(req_delivs)}"}),
                [],
                check_numeric_grounding("", []),
            )

        # Check all written files & deliverables exist on disk and were read AFTER latest write.
        to_verify = sorted(set(merged_paths) | set(written_files))
        proofs: List[DeliverableProof] = []
        deliv_contents: List[str] = []
        for rel_p in to_verify:
            try:
                fpath = self.sandbox.resolve_read(rel_p)
            except PathSandboxError:
                return (
                    False,
                    json.dumps({"ok": False, "error": f"FINISH REFUSED: deliverable '{rel_p}' does not exist on disk. Write it first."}),
                    [],
                    check_numeric_grounding("", []),
                )
            if not read_after_write.get(rel_p, False):
                return (
                    False,
                    json.dumps(
                        {
                            "ok": False,
                            "error": f"FINISH REFUSED: deliverable '{rel_p}' has not been verified via read_file() after its latest write. Call read_file(path='{rel_p}') first.",
                        }
                    ),
                    [],
                    check_numeric_grounding("", []),
                )
            text = fpath.read_text(encoding="utf-8", errors="replace")
            deliv_contents.append(text)
            if rel_p in merged_paths:
                proofs.append(
                    DeliverableProof(
                        rel_path=rel_p,
                        bytes=fpath.stat().st_size,
                        sha256=self.sandbox.sha256_file(fpath),
                        verified_read=True,
                    )
                )

        # Re-run deterministic todo checks cleanly.
        failed_checks: List[str] = []
        for item in todos:
            if item.check is not None:
                ok_c, msg_c = evaluate_todo_check(self.sandbox, item.check)
                if not ok_c:
                    item.status = "in_progress"
                    item.demoted_reason = msg_c
                    failed_checks.append(f"{item.id}: {msg_c}")
        if failed_checks:
            return (
                False,
                json.dumps({"ok": False, "error": f"FINISH REFUSED: todo checks failed: {'; '.join(failed_checks)}"}),
                [],
                check_numeric_grounding("", []),
            )

        grounding = check_numeric_grounding(summary, [*observed_texts.values(), *deliv_contents])
        return (
            True,
            json.dumps(
                {
                    "ok": True,
                    "status": "completed",
                    "deliverables": [asdict(d) for d in proofs],
                    "numeric_grounding": asdict(grounding),
                }
            ),
            proofs,
            grounding,
        )
