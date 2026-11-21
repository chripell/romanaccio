"""Two-stage diff approval ledger (pending -> approved -> applied | failed, denied, expired)."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import SafetyLimits
from .gates import WriteGate, WriteGateVerdict
from .paths import WorkspaceSandbox, WriteReceipt
from .store import HarnessStore
from .tools import StagedWriteProposal


@dataclass(frozen=True)
class StageOutcome:
    """Result of staging a write proposal after running the preflight WriteGate."""

    held: bool
    gate_verdict: WriteGateVerdict
    action: Optional[Dict[str, Any]]
    revision_note: str = ""


@dataclass(frozen=True)
class ApprovalOutcome:
    """Deterministic terminal outcome of resolving a staged approval action."""

    ok: bool
    status: str
    action_id: str
    receipt: Optional[WriteReceipt] = None
    error: str = ""


class ApprovalManager:
    """Manages human-in-the-loop single-use diff approvals with strict TTL and TOCTOU safety."""

    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        store: HarnessStore,
        *,
        write_gate: Optional[WriteGate] = None,
        limits: Optional[SafetyLimits] = None,
    ):
        self.sandbox = sandbox
        self.store = store
        self.limits = limits or SafetyLimits()
        self.write_gate = write_gate or WriteGate(None, limits=self.limits)

    async def stage_proposal(
        self,
        session_id: str,
        message_id: str,
        user_request: str,
        proposal: StagedWriteProposal,
        *,
        ttl_s: Optional[int] = None,
        prompt_cache_key: str = "d1:write_gate",
    ) -> StageOutcome:
        """Evaluate WriteGate and persist a pending approval card unless confidently held."""
        verdict = await self.write_gate.evaluate(user_request, proposal, prompt_cache_key=prompt_cache_key)
        self.store.log_event(
            "session",
            session_id,
            "write_gate_evaluated",
            {
                "rel_path": proposal.rel_path,
                "tool_name": proposal.tool_name,
                **verdict.to_dict(),
            },
        )

        if verdict.outcome == "hold":
            self.store.log_event(
                "session",
                session_id,
                "write_gate_hold",
                {"rel_path": proposal.rel_path, "revision_note": verdict.revision_note},
            )
            return StageOutcome(
                held=True,
                gate_verdict=verdict,
                action=None,
                revision_note=verdict.revision_note,
            )

        action = self.store.create_action(
            session_id,
            message_id,
            tool_name=proposal.tool_name,
            rel_path=proposal.rel_path,
            base_sha256=proposal.base_sha256,
            new_content=proposal.new_content,
            diff_text=proposal.diff,
            reason=proposal.reason,
            ttl_s=self.limits.approval_ttl_s if ttl_s is None else int(ttl_s),
        )
        self.store.log_event(
            "action",
            action["id"],
            "action_staged",
            {"session_id": session_id, "rel_path": proposal.rel_path, "base_sha256": proposal.base_sha256},
        )
        return StageOutcome(held=False, gate_verdict=verdict, action=action, revision_note="")

    def allow_once(self, action_id: str, *, now: Optional[float] = None) -> ApprovalOutcome:
        """Approve a pending action once and atomically apply it with TOCTOU SHA-256 checks."""
        action = self.store.get_action(action_id)
        if action is None:
            return ApprovalOutcome(ok=False, status="not_found", action_id=action_id, error="Action not found")

        cur_status = str(action["status"])
        if cur_status != "pending":
            return ApprovalOutcome(
                ok=False,
                status=cur_status,
                action_id=action_id,
                error=f"Action cannot be approved from state '{cur_status}'",
            )

        ts = time.time() if now is None else float(now)
        if ts >= float(action["expires_at"]):
            self.store.transition_action(action_id, expected_status="pending", new_status="expired")
            self.store.log_event("action", action_id, "action_expired", {"rel_path": action["rel_path"]})
            return ApprovalOutcome(ok=False, status="expired", action_id=action_id, error="Approval card expired")

        if not self.store.transition_action(action_id, expected_status="pending", new_status="approved"):
            latest = self.store.get_action(action_id)
            st = str(latest["status"]) if latest else "unknown"
            return ApprovalOutcome(ok=False, status=st, action_id=action_id, error="Concurrent state change")

        try:
            receipt = self.sandbox.atomic_write_text(
                action["rel_path"],
                action["new_content"],
                expected_sha256=action["base_sha256"],
                max_bytes=self.limits.max_write_bytes,
            )
        except Exception as exc:
            err_msg = str(exc)
            self.store.transition_action(action_id, expected_status="approved", new_status="failed", error=err_msg)
            self.store.log_event("action", action_id, "action_failed", {"rel_path": action["rel_path"], "error": err_msg})
            return ApprovalOutcome(ok=False, status="failed", action_id=action_id, error=err_msg)

        self.store.transition_action(action_id, expected_status="approved", new_status="applied")
        self.store.log_event(
            "action",
            action_id,
            "action_applied",
            {"rel_path": receipt.rel_path, "bytes": receipt.bytes_written, "sha256": receipt.sha256},
        )
        return ApprovalOutcome(ok=True, status="applied", action_id=action_id, receipt=receipt)

    def deny(self, action_id: str, *, reason: str = "User denied", now: Optional[float] = None) -> ApprovalOutcome:
        """Reject a pending approval card cleanly without touching disk."""
        action = self.store.get_action(action_id)
        if action is None:
            return ApprovalOutcome(ok=False, status="not_found", action_id=action_id, error="Action not found")

        cur_status = str(action["status"])
        if cur_status != "pending":
            return ApprovalOutcome(
                ok=False,
                status=cur_status,
                action_id=action_id,
                error=f"Action cannot be denied from state '{cur_status}'",
            )

        ts = time.time() if now is None else float(now)
        if ts >= float(action["expires_at"]):
            self.store.transition_action(action_id, expected_status="pending", new_status="expired")
            self.store.log_event("action", action_id, "action_expired", {"rel_path": action["rel_path"]})
            return ApprovalOutcome(ok=False, status="expired", action_id=action_id, error="Approval card expired")

        if not self.store.transition_action(action_id, expected_status="pending", new_status="denied", error=reason):
            latest = self.store.get_action(action_id)
            st = str(latest["status"]) if latest else "unknown"
            return ApprovalOutcome(ok=False, status=st, action_id=action_id, error="Concurrent state change")

        self.store.log_event("action", action_id, "action_denied", {"rel_path": action["rel_path"], "reason": reason})
        return ApprovalOutcome(ok=True, status="denied", action_id=action_id, error=reason)
