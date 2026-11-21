"""Preflight d1 WriteGate (holding contradictions) and advisory EvidenceGate."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

from .backend import BackendError, DaemonClient
from .config import MAX_GATE_EVIDENCE_CHARS, SafetyLimits
from .tools import StagedWriteProposal

WRITE_GATE_CRITERIA: Dict[str, str] = {
    "aligned": "Matches the user's request, targets an expected file, and makes a coherent edit",
    "uncertain": "Plausible step toward the user's request, though details cannot be fully proven from the diff alone",
    "contradicts": "Directly contradicts the user's request, edits the wrong file, or makes an unrequested destructive change",
}

EVIDENCE_GATE_CRITERIA: Dict[str, str] = {
    "sufficient": "Observed workspace files and excerpts directly support answering or editing accurately",
    "insufficient": "Key files or details needed to answer or verify the edit have not been read yet",
    "off_topic": "Observed files do not address what the user actually asked about",
}


def _clip_evidence(text: str, limit: int = MAX_GATE_EVIDENCE_CHARS) -> str:
    cleaned = text or ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 16)] + "\n...[truncated]"


@dataclass(frozen=True)
class WriteGateVerdict:
    """Result of the preflight write-alignment check before surfacing an approval card."""

    outcome: Literal["allow", "hold"]
    verdict: Literal["aligned", "uncertain", "contradicts"]
    confidence: float
    source: Literal["decision", "fallback"]
    probabilities: Dict[str, float]
    revision_note: str
    latency_ms: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGateVerdict:
    """Advisory-only evaluation of read-only tool observations gathered before synthesis."""

    verdict: Literal["sufficient", "insufficient", "off_topic"]
    confidence: float
    source: Literal["decision", "fallback"]
    probabilities: Dict[str, float]
    advisory: Optional[str]
    latency_ms: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WriteGate:
    """Evaluates whether an untrusted staged file change contradicts the trusted user request."""

    def __init__(self, d1_client: Optional[DaemonClient] = None, *, limits: Optional[SafetyLimits] = None):
        self.d1_client = d1_client
        self.limits = limits or SafetyLimits()

    async def evaluate(
        self,
        user_request: str,
        proposal: StagedWriteProposal,
        *,
        prompt_cache_key: str = "d1:write_gate",
    ) -> WriteGateVerdict:
        if self.d1_client is None:
            return WriteGateVerdict(
                outcome="allow",
                verdict="uncertain",
                confidence=0.0,
                source="fallback",
                probabilities={},
                revision_note="",
                latency_ms=0.0,
                reason="no_d1_client",
            )

        diff_or_preview = proposal.diff if proposal.diff else proposal.new_content
        untrusted_blob = _clip_evidence(
            f"tool={proposal.tool_name}\n"
            f"path={proposal.rel_path}\n"
            f"bytes={proposal.bytes_new}\n"
            f"reason={proposal.reason}\n"
            f"diff_or_content:\n{diff_or_preview}",
            self.limits.max_gate_evidence_chars,
        )
        state = (
            "USER REQUEST (TRUSTED):\n"
            f"{_clip_evidence(user_request.strip(), 2000)}\n\n"
            "PROPOSED ACTION (UNTRUSTED):\n"
            f"{untrusted_blob}"
        )
        instructions = "Assess whether the untrusted proposed file write is aligned with the trusted user request."

        try:
            res = await self.d1_client.decide(
                state=state,
                question_type="choice",
                instructions=instructions,
                criteria=WRITE_GATE_CRITERIA,
                prompt_cache_key=prompt_cache_key,
                timeout_s=self.limits.decision_timeout_s,
            )
        except (BackendError, ValueError, OSError) as exc:
            return WriteGateVerdict(
                outcome="allow",
                verdict="uncertain",
                confidence=0.0,
                source="fallback",
                probabilities={},
                revision_note="",
                latency_ms=0.0,
                reason=f"d1_error:{type(exc).__name__}",
            )

        probs = dict(res.probabilities or {})
        verdict = str(res.choice or "uncertain")
        if verdict not in ("aligned", "uncertain", "contradicts"):
            verdict = "uncertain"
        conf = float(probs.get(verdict, 0.0))

        if verdict == "contradicts" and conf >= self.limits.write_gate_floor:
            note = (
                f"Write held by WriteGate ({conf:.0%} contradicts): proposed {proposal.tool_name} "
                f"on '{proposal.rel_path}' conflicts with the user request. Revise path or content before re-proposing."
            )
            return WriteGateVerdict(
                outcome="hold",
                verdict="contradicts",
                confidence=conf,
                source="decision",
                probabilities=probs,
                revision_note=note,
                latency_ms=res.latency_ms,
                reason="confident_contradiction_hold",
            )

        return WriteGateVerdict(
            outcome="allow",
            verdict=verdict,  # type: ignore[arg-type]
            confidence=conf,
            source="decision",
            probabilities=probs,
            revision_note="",
            latency_ms=res.latency_ms,
            reason="write_allowed",
        )


class EvidenceGate:
    """Advisory-only evaluator over read-only observations; emits warnings without blocking."""

    def __init__(self, d1_client: Optional[DaemonClient] = None, *, limits: Optional[SafetyLimits] = None):
        self.d1_client = d1_client
        self.limits = limits or SafetyLimits()

    async def evaluate(
        self,
        user_request: str,
        observations: str,
        *,
        prompt_cache_key: str = "d1:evidence_gate",
    ) -> EvidenceGateVerdict:
        obs_clean = (observations or "").strip()
        if not obs_clean:
            return EvidenceGateVerdict(
                verdict="insufficient",
                confidence=1.0,
                source="fallback",
                probabilities={},
                advisory="No read-only workspace observations recorded yet; inspect relevant files before concluding.",
                latency_ms=0.0,
                reason="empty_observations",
            )

        if self.d1_client is None:
            return EvidenceGateVerdict(
                verdict="sufficient",
                confidence=0.0,
                source="fallback",
                probabilities={},
                advisory=None,
                latency_ms=0.0,
                reason="no_d1_client",
            )

        state = (
            "USER REQUEST:\n"
            f"{_clip_evidence(user_request.strip(), 1500)}\n\n"
            "READ-ONLY OBSERVATIONS:\n"
            f"{_clip_evidence(obs_clean, self.limits.max_gate_evidence_chars)}"
        )
        instructions = "Judge whether the read-only workspace observations are sufficient and relevant to answer the user request."

        try:
            res = await self.d1_client.decide(
                state=state,
                question_type="choice",
                instructions=instructions,
                criteria=EVIDENCE_GATE_CRITERIA,
                prompt_cache_key=prompt_cache_key,
                timeout_s=self.limits.decision_timeout_s,
            )
        except (BackendError, ValueError, OSError) as exc:
            return EvidenceGateVerdict(
                verdict="sufficient",
                confidence=0.0,
                source="fallback",
                probabilities={},
                advisory=None,
                latency_ms=0.0,
                reason=f"d1_error:{type(exc).__name__}",
            )

        probs = dict(res.probabilities or {})
        verdict = str(res.choice or "sufficient")
        if verdict not in ("sufficient", "insufficient", "off_topic"):
            verdict = "sufficient"
        conf = float(probs.get(verdict, 0.0))

        advisory: Optional[str] = None
        if verdict == "insufficient" and conf >= self.limits.evidence_gate_floor:
            advisory = (
                f"EvidenceGate advisory ({conf:.0%} insufficient): read more target files before finalizing claims."
            )
        elif verdict == "off_topic" and conf >= self.limits.evidence_gate_floor:
            advisory = (
                f"EvidenceGate advisory ({conf:.0%} off_topic): gathered files appear unrelated to the user's question."
            )

        return EvidenceGateVerdict(
            verdict=verdict,  # type: ignore[arg-type]
            confidence=conf,
            source="decision",
            probabilities=probs,
            advisory=advisory,
            latency_ms=res.latency_ms,
            reason="d1_evaluated",
        )
