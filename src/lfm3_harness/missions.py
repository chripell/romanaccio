"""Isolated Verified Tasks (Missions) runner with frozen SHA-256 AcceptanceSpecs and JSONL journaling."""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from .agent_loop import AgentLoop, JobRunResult, RoundTelemetry
from .backend import DaemonClient
from .config import Capabilities, ModelFamily, SafetyLimits
from .paths import PathSandboxError, WorkspaceSandbox
from .store import HarnessStore


@dataclass(frozen=True)
class AcceptanceCheck:
    """Single deterministic check evaluated against workspace files."""

    kind: Literal["file_exists", "dir_exists", "content_matches", "content_absent", "json_schema", "numeric_consistency"]
    path: str
    pattern: str = ""
    schema: Optional[Dict[str, Any]] = None
    expected_numbers: Tuple[str, ...] = ()
    case_sensitive: bool = False


@dataclass(frozen=True)
class AcceptanceSpec:
    """Immutable mission verification contract canonicalized and SHA-256 hashed before run 1."""

    name: str
    description: str
    prompt: str
    required_deliverables: Tuple[str, ...]
    checks: Tuple[AcceptanceCheck, ...]
    max_attempts: int = 3
    max_rounds_per_attempt: int = 12
    semantic_mode: Literal["deterministic", "human_review_only"] = "deterministic"

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single AcceptanceCheck evaluation."""

    kind: str
    path: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class AcceptanceReport:
    """Aggregate pass/fail report for a frozen AcceptanceSpec."""

    passed: bool
    results: List[CheckResult]

    def failure_summary(self) -> str:
        fails = [f"- [{r.kind}] {r.path}: {r.detail}" for r in self.results if not r.passed]
        return "\n".join(fails)


@dataclass(frozen=True)
class MissionRunResult:
    """Terminal result returned by MissionRunner."""

    run_id: str
    spec_id: str
    spec_sha256: str
    status: str  # "passed" | "failed" | "human_review_required"
    attempts: int
    reports: List[AcceptanceReport]
    job_results: List[JobRunResult]
    journal_path: Path


def verify_json_schema_stdlib(data: Any, schema: Dict[str, Any], loc: str = "$") -> List[str]:
    """Pure-stdlib JSON Schema subset validator (type, required, properties, items, minItems, minLength, enum)."""
    errors: List[str] = []
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(data, dict):
            return [f"{loc}: expected object, got {type(data).__name__}"]
        for req_k in schema.get("required", []):
            if req_k not in data:
                errors.append(f"{loc}: missing required key '{req_k}'")
        props = schema.get("properties") or {}
        for k, sub_schema in props.items():
            if k in data and isinstance(sub_schema, dict):
                errors.extend(verify_json_schema_stdlib(data[k], sub_schema, f"{loc}.{k}"))
    elif expected_type == "array":
        if not isinstance(data, list):
            return [f"{loc}: expected array, got {type(data).__name__}"]
        min_items = schema.get("minItems")
        if min_items is not None and len(data) < int(min_items):
            errors.append(f"{loc}: len {len(data)} < minItems {min_items}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(data):
                errors.extend(verify_json_schema_stdlib(item, item_schema, f"{loc}[{idx}]"))
    elif expected_type == "string":
        if not isinstance(data, str):
            return [f"{loc}: expected string, got {type(data).__name__}"]
        min_len = schema.get("minLength")
        if min_len is not None and len(data) < int(min_len):
            errors.append(f"{loc}: length {len(data)} < minLength {min_len}")
    elif expected_type == "integer":
        if isinstance(data, bool) or not isinstance(data, int):
            return [f"{loc}: expected integer, got {type(data).__name__}"]
    elif expected_type == "number":
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            return [f"{loc}: expected number, got {type(data).__name__}"]
    elif expected_type == "boolean":
        if not isinstance(data, bool):
            return [f"{loc}: expected boolean, got {type(data).__name__}"]
    elif expected_type == "null":
        if data is not None:
            return [f"{loc}: expected null, got {type(data).__name__}"]

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{loc}: value {data!r} not in enum {schema['enum']!r}")
    return errors


def evaluate_acceptance_check(sandbox: WorkspaceSandbox, chk: AcceptanceCheck) -> CheckResult:
    """Evaluate a single deterministic acceptance check strictly inside WorkspaceSandbox."""
    try:
        if chk.kind == "dir_exists":
            target = sandbox.resolve_dir(chk.path, must_exist=True)
            return CheckResult(chk.kind, chk.path, True, f"Directory exists: {sandbox.relativize(target)}")
        if chk.kind == "file_exists":
            target = sandbox.resolve_read(chk.path)
            return CheckResult(chk.kind, chk.path, True, f"File exists ({target.stat().st_size} B)")

        target = sandbox.resolve_read(chk.path)
        text = target.read_text(encoding="utf-8", errors="replace")

        if chk.kind == "content_matches":
            flags = 0 if chk.case_sensitive else re.IGNORECASE
            matched = bool(re.search(chk.pattern, text, flags))
            return CheckResult(
                chk.kind,
                chk.path,
                matched,
                f"Pattern {chk.pattern!r} matched" if matched else f"Pattern {chk.pattern!r} missing",
            )
        if chk.kind == "content_absent":
            flags = 0 if chk.case_sensitive else re.IGNORECASE
            hit = re.search(chk.pattern, text, flags)
            return CheckResult(
                chk.kind,
                chk.path,
                hit is None,
                f"Forbidden pattern {chk.pattern!r} absent" if hit is None else f"Found forbidden pattern {chk.pattern!r}",
            )
        if chk.kind == "json_schema":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                return CheckResult(chk.kind, chk.path, False, f"Invalid JSON: {exc}")
            errs = verify_json_schema_stdlib(parsed, chk.schema or {})
            return CheckResult(
                chk.kind,
                chk.path,
                len(errs) == 0,
                "JSON schema valid" if not errs else "; ".join(errs[:3]),
            )
        if chk.kind == "numeric_consistency":
            missing: List[str] = []
            for num in chk.expected_numbers:
                if not re.search(rf"(?<![0-9.]){re.escape(str(num))}(?![0-9])", text):
                    missing.append(str(num))
            return CheckResult(
                chk.kind,
                chk.path,
                len(missing) == 0,
                "All expected numbers verified" if not missing else f"Missing numbers: {missing}",
            )
        return CheckResult(chk.kind, chk.path, False, f"Unknown check kind: {chk.kind}")
    except PathSandboxError as exc:
        return CheckResult(chk.kind, chk.path, False, f"Sandbox error ({exc.reason}): {exc}")
    except OSError as exc:
        return CheckResult(chk.kind, chk.path, False, f"I/O error: {exc}")


def verify_acceptance_spec(sandbox: WorkspaceSandbox, spec: AcceptanceSpec) -> AcceptanceReport:
    """Evaluate all checks of an immutable AcceptanceSpec."""
    results = [evaluate_acceptance_check(sandbox, c) for c in spec.checks]
    return AcceptanceReport(passed=all(r.passed for r in results) and len(results) > 0, results=results)


class MissionJournal:
    """Durable append-only JSON-lines event log for isolated mission verification runs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, run_id: str, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "ts": round(time.time(), 4),
            "run_id": run_id,
            "event": event,
            "data": data or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    def read_events(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MissionRunner:
    """Executes isolated Verified Tasks against frozen, tamper-checked AcceptanceSpecs."""

    def __init__(
        self,
        client: DaemonClient,
        sandbox: WorkspaceSandbox,
        store: HarnessStore,
        *,
        journal_dir: Optional[Path] = None,
        capabilities: Optional[Capabilities] = None,
        limits: Optional[SafetyLimits] = None,
        family: ModelFamily = "lfm3",
        enable_thinking: Optional[bool] = None,
        think_budget: Optional[int] = None,
        force_tool_call: Optional[bool] = None,
        on_round_complete: Optional[Callable[[RoundTelemetry, str], None]] = None,
    ):
        self.client = client
        self.sandbox = sandbox
        self.store = store
        self.journal_dir = Path(journal_dir) if journal_dir else (Path(store.db_path).parent / "mission_journals")
        self.capabilities = capabilities or Capabilities()
        self.limits = limits or SafetyLimits()
        self.family: ModelFamily = family
        self._think_budget_explicit: bool = think_budget is not None
        self.enable_thinking: bool = False if enable_thinking is None else bool(enable_thinking)
        self.think_budget: int = (144 if self.enable_thinking else 0) if think_budget is None else max(0, int(think_budget))
        self.force_tool_call: bool = False if force_tool_call is None else bool(force_tool_call)
        self.think_in_plan: bool = False
        self.plan_think_budget: int = 0
        self.on_round_complete = on_round_complete

    def _assert_spec_unmodified(self, spec_id: str, run_id: str, expected_sha: str) -> None:
        spec_row = self.store.get_acceptance_spec(spec_id)
        run_row = self.store.get_mission_run(run_id)
        if spec_row is None or run_row is None:
            raise RuntimeError("Missing acceptance_specs or mission_runs record")
        actual_sha = hashlib.sha256(str(spec_row["canonical_json"]).encode("utf-8")).hexdigest()
        if actual_sha != expected_sha or str(run_row["spec_sha256"]) != expected_sha:
            raise RuntimeError(f"AcceptanceSpec tamper detected! expected={expected_sha} actual={actual_sha}")

    async def run(
        self,
        workspace_id: str,
        spec: AcceptanceSpec,
        *,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> MissionRunResult:
        """Run a mission up to spec.max_attempts while verifying the frozen SHA-256 contract."""
        canonical_json = spec.canonical_json()
        frozen_sha = spec.sha256()
        spec_row, run_row = self.store.create_mission_run(
            workspace_id,
            canonical_json,
            frozen_sha,
            status="running",
            max_attempts=spec.max_attempts,
        )
        spec_id = str(spec_row["id"])
        run_id = str(run_row["id"])
        journal = MissionJournal(self.journal_dir / f"{run_id}.jsonl")
        journal.write(run_id, "mission_started", {"name": spec.name, "spec_sha256": frozen_sha, "mode": spec.semantic_mode})

        # Semantic missions cannot self-certify: mark explicitly as human_review_required.
        if spec.semantic_mode == "human_review_only":
            loop = AgentLoop(
                self.client,
                self.sandbox,
                self.store,
                capabilities=self.capabilities,
                limits=self.limits,
                family=self.family,
                auto_approve_writes=True,
                enable_thinking=self.enable_thinking,
                think_budget=self.think_budget,
                force_tool_call=self.force_tool_call,
                think_in_plan=self.think_in_plan,
                plan_think_budget=self.plan_think_budget,
                max_tokens_per_round=384,
            )
            job_res = await loop.run(
                workspace_id,
                spec.prompt,
                title=spec.name,
                prompt_cache_key=f"mission:{run_id}:semantic",
                required_deliverables=spec.required_deliverables,
                max_rounds=spec.max_rounds_per_attempt,
                cancel_event=cancel_event,
                on_round_complete=self.on_round_complete,
            )
            self._assert_spec_unmodified(spec_id, run_id, frozen_sha)
            rep = verify_acceptance_spec(self.sandbox, spec) if spec.checks else AcceptanceReport(passed=False, results=[])
            self.store.update_mission_run(
                run_id,
                status="human_review_required",
                attempt=1,
                result={"reason": "semantic_mission_requires_human_review", "job_status": job_res.status},
            )
            journal.write(run_id, "mission_human_review_required", {"job_status": job_res.status})
            return MissionRunResult(
                run_id=run_id,
                spec_id=spec_id,
                spec_sha256=frozen_sha,
                status="human_review_required",
                attempts=1,
                reports=[rep],
                job_results=[job_res],
                journal_path=journal.path,
            )

        reports: List[AcceptanceReport] = []
        job_results: List[JobRunResult] = []
        last_failure_note = ""

        for attempt in range(1, max(1, spec.max_attempts) + 1):
            self._assert_spec_unmodified(spec_id, run_id, frozen_sha)
            self.store.update_mission_run(run_id, attempt=attempt)
            journal.write(run_id, "attempt_started", {"attempt": attempt})

            attempt_enable_thinking = self.enable_thinking
            attempt_think_budget = self.think_budget
            attempt_temp = (0.0 if attempt == 1 else 0.15) if self.family == "lfm3" else 0.0
            attempt_loop = AgentLoop(
                self.client,
                self.sandbox,
                self.store,
                capabilities=self.capabilities,
                limits=self.limits,
                family=self.family,
                auto_approve_writes=True,
                enable_thinking=attempt_enable_thinking,
                think_budget=attempt_think_budget,
                force_tool_call=self.force_tool_call,
                think_in_plan=self.think_in_plan,
                plan_think_budget=self.plan_think_budget,
                temperature=attempt_temp,
                max_tokens_per_round=384,
            )

            attempt_prompt = spec.prompt
            if last_failure_note:
                attempt_prompt += f"\n\nPrevious attempt failed deterministic acceptance checks:\n{last_failure_note}\nFix these exact issues cleanly."

            job_res = await attempt_loop.run(
                workspace_id,
                attempt_prompt,
                title=f"{spec.name}#a{attempt}",
                prompt_cache_key=f"mission:{run_id}",
                required_deliverables=spec.required_deliverables,
                max_rounds=spec.max_rounds_per_attempt,
                cancel_event=cancel_event,
                on_round_complete=self.on_round_complete,
            )
            job_results.append(job_res)

            self._assert_spec_unmodified(spec_id, run_id, frozen_sha)
            report = verify_acceptance_spec(self.sandbox, spec)
            reports.append(report)
            journal.write(
                run_id,
                "attempt_verified",
                {
                    "attempt": attempt,
                    "job_status": job_res.status,
                    "passed": report.passed,
                    "checks": [asdict(r) for r in report.results],
                },
            )

            if report.passed and job_res.status == "completed":
                self._assert_spec_unmodified(spec_id, run_id, frozen_sha)
                self.store.update_mission_run(
                    run_id,
                    status="passed",
                    attempt=attempt,
                    result={"checks": [asdict(r) for r in report.results], "summary": job_res.summary},
                )
                journal.write(run_id, "mission_passed", {"attempt": attempt, "spec_sha256": frozen_sha})
                return MissionRunResult(
                    run_id=run_id,
                    spec_id=spec_id,
                    spec_sha256=frozen_sha,
                    status="passed",
                    attempts=attempt,
                    reports=reports,
                    job_results=job_results,
                    journal_path=journal.path,
                )

            last_failure_note = report.failure_summary() or job_res.error or f"Job status={job_res.status}"

        self.store.update_mission_run(
            run_id,
            status="failed",
            attempt=len(job_results),
            result={"failure_summary": last_failure_note},
        )
        journal.write(run_id, "mission_failed", {"attempts": len(job_results), "failure_summary": last_failure_note})
        return MissionRunResult(
            run_id=run_id,
            spec_id=spec_id,
            spec_sha256=frozen_sha,
            status="failed",
            attempts=len(job_results),
            reports=reports,
            job_results=job_results,
            journal_path=journal.path,
        )


# ---------------------------------------------------------------------------
# Built-In Deterministic Desk Task & Semantic Fixtures
# ---------------------------------------------------------------------------
def setup_desk_task_workspace(root: Path, *, max_attempts: int = 3, max_rounds_per_attempt: int = 12) -> AcceptanceSpec:
    """Populate a clean desk_task workspace with deterministic source files and return its frozen spec."""
    inbox = root / "inbox"
    reports = root / "reports"
    inbox.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    (inbox / "q1_sales.txt").write_text(
        "Region: North\nUnits Sold: 120\nRevenue USD: 3600\nLead: Alice\n",
        encoding="utf-8",
    )
    (inbox / "q2_sales.txt").write_text(
        "Region: South\nUnits Sold: 85\nRevenue USD: 2550\nLead: Bob\n",
        encoding="utf-8",
    )
    (inbox / "notes.txt").write_text(
        "Action: Write concise markdown summary to reports/desk_summary.md\n"
        "Include both regions (North, South), exact Total Units = 205 (120 + 85), and Total Revenue = 6150 (3600 + 2550).\n"
        "Do NOT leave any TODO or PLACEHOLDER markers.\n",
        encoding="utf-8",
    )

    return AcceptanceSpec(
        name="desk_task",
        description="Aggregate sales records from inbox/ into verified reports/desk_summary.md.",
        prompt=(
            "Complete the desk task in this workspace.\n"
            "1. Call todo_write first with short steps.\n"
            "2. Call read_file on inbox/q1_sales.txt, inbox/q2_sales.txt, and inbox/notes.txt.\n"
            "3. Call write_file to create reports/desk_summary.md containing North (120 units, 3600), South (85 units, 2550), Total Units: 205, Total Revenue: 6150, with zero TODO or PLACEHOLDER text.\n"
            "4. Call read_file on reports/desk_summary.md to verify it on disk.\n"
            "5. Call finish with deliverables=['reports/desk_summary.md'] and a short grounded summary."
        ),
        required_deliverables=("reports/desk_summary.md",),
        checks=(
            AcceptanceCheck(kind="file_exists", path="reports/desk_summary.md"),
            AcceptanceCheck(kind="content_matches", path="reports/desk_summary.md", pattern=r"\bNorth\b"),
            AcceptanceCheck(kind="content_matches", path="reports/desk_summary.md", pattern=r"\bSouth\b"),
            AcceptanceCheck(
                kind="numeric_consistency",
                path="reports/desk_summary.md",
                expected_numbers=("120", "3600", "85", "2550", "205", "6150"),
            ),
            AcceptanceCheck(kind="content_absent", path="reports/desk_summary.md", pattern=r"\b(TODO|PLACEHOLDER)\b"),
        ),
        max_attempts=max_attempts,
        max_rounds_per_attempt=max_rounds_per_attempt,
        semantic_mode="deterministic",
    )


def make_screen_qa_spec(image_rel_path: str = "screenshot.png") -> AcceptanceSpec:
    """Create a screen_qa semantic mission strictly gated to human_review_only."""
    return AcceptanceSpec(
        name="screen_qa",
        description="Inspect screenshot and answer visual QA checklist (human_review_only).",
        prompt=f"Inspect {image_rel_path} using look and write reports/screen_qa.md.",
        required_deliverables=("reports/screen_qa.md",),
        checks=(AcceptanceCheck(kind="file_exists", path="reports/screen_qa.md"),),
        max_attempts=1,
        max_rounds_per_attempt=4,
        semantic_mode="human_review_only",
    )


def make_doc_extract_spec(doc_rel_path: str = "inbox/notes.txt") -> AcceptanceSpec:
    """Create a doc_extract semantic mission strictly gated to human_review_only."""
    return AcceptanceSpec(
        name="doc_extract",
        description="Extract unstructured key points into markdown (human_review_only).",
        prompt=f"Read {doc_rel_path} and write reports/extracted.md.",
        required_deliverables=("reports/extracted.md",),
        checks=(AcceptanceCheck(kind="file_exists", path="reports/extracted.md"),),
        max_attempts=1,
        max_rounds_per_attempt=4,
        semantic_mode="human_review_only",
    )
