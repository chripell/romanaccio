"""Hybrid 7-intent router combining calibrated single-pass d1 decisions with deterministic rules."""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from .backend import BackendError, DaemonClient
from .config import Capabilities, SafetyLimits

INTENTS: Tuple[str, ...] = ("general", "explore", "summarize", "related", "report", "vision", "voice")

INTENT_CRITERIA: Dict[str, str] = {
    "general": "General Q&A, coding explanation, or conversation not asking to browse or edit files",
    "explore": "List directories, find files, search workspace code, or inspect specific files",
    "summarize": "Summarize, recap, or give a concise overview of a file, module, or folder",
    "related": "Find related, dependent, importing, or similar files across the workspace",
    "report": "Write, edit, create, draft, refactor, update, or save a file or report in the workspace",
    "vision": "Inspect or analyze an image, screenshot, diagram, or photo",
    "voice": "Transcribe an audio recording, voice memo, or spoken clip",
}

_EXT_RE = re.compile(
    r"(?:^|[\s'\"(])[\w./-]*\.(?:py|c|h|cpp|hpp|md|json|txt|yaml|yml|toml|sh|js|ts|css|html|rs|jinja)\b",
    re.IGNORECASE,
)
_VISION_RE = re.compile(
    r"\b(image|screenshot|photo|picture|diagram|chart|png|jpg|jpeg|webp|gif|bmp|look\s+at)\b",
    re.IGNORECASE,
)
_VOICE_RE = re.compile(
    r"\b(audio|voice|transcribe|transcript|recording|dictation|spoken|wav|mp3|flac|m4a|ogg)\b",
    re.IGNORECASE,
)
_RELATED_RE = re.compile(
    r"\b(related|dependencies|dependents|references|referencing|callers|callees|imports|imported\s+by|similar\s+to|what\s+uses)\b",
    re.IGNORECASE,
)
_SUMMARIZE_RE = re.compile(
    r"\b(summarize|summary|tldr|tl;dr|recap|overview|condense|digest|brief\s+me\s+on)\b",
    re.IGNORECASE,
)
_REPORT_RE = re.compile(
    r"\b(write_file|edit_file|create|save|edit|modify|update|patch|refactor|draft|report)\b"
    r"|\bwrite\s+(?:a|an|the|to|out|notes?|file|doc|docs|summary|report|test|tests|code|script|readme)\b",
    re.IGNORECASE,
)
_EXPLORE_RE = re.compile(
    r"\b(list|find|search|grep|locate|inspect|read|open|show|browse|directory|directories|folder|folders|files|workspace|codebase)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    """Explicitly attributed routing decision for UI badges, audit logs, and prompt suffixes."""

    intent: str
    confidence: float
    source: Literal["decision", "heuristic"]
    heuristic_intent: str
    decision_intent: Optional[str]
    decision_confidence: Optional[float]
    probabilities: Dict[str, float]
    reason: str
    latency_ms: float
    route_hint: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def heuristic_route(message: str, capabilities: Optional[Capabilities] = None) -> Tuple[str, float, str]:
    """Deterministic keyword and file-extension classifier for fallback and safety guards."""
    text = (message or "").strip()
    caps = capabilities or Capabilities()
    if not text:
        return "general", 0.60, "empty_message"

    if caps.vision and _VISION_RE.search(text):
        return "vision", 0.72, "keyword_vision"
    if caps.voice_asr and _VOICE_RE.search(text):
        return "voice", 0.72, "keyword_voice"
    if _RELATED_RE.search(text):
        return "related", 0.72, "keyword_related"
    if _SUMMARIZE_RE.search(text):
        return "summarize", 0.72, "keyword_summarize"
    if _REPORT_RE.search(text):
        return "report", 0.72, "keyword_report"
    if _EXPLORE_RE.search(text) or _EXT_RE.search(text):
        return "explore", 0.72, "keyword_explore"
    return "general", 0.60, "default_general"


class IntentRouter:
    """Routes user prompts via single-pass d1 classification with transparent heuristic fallback."""

    def __init__(
        self,
        d1_client: Optional[DaemonClient] = None,
        *,
        capabilities: Optional[Capabilities] = None,
        limits: Optional[SafetyLimits] = None,
        min_chars: int = 24,
    ):
        self.d1_client = d1_client
        self.capabilities = capabilities or Capabilities()
        self.limits = limits or SafetyLimits()
        self.min_chars = min_chars

    def _active_criteria(self) -> Dict[str, str]:
        criteria: Dict[str, str] = {}
        for intent in INTENTS:
            if intent == "vision" and not self.capabilities.vision:
                continue
            if intent == "voice" and not self.capabilities.voice_asr:
                continue
            criteria[intent] = INTENT_CRITERIA[intent]
        return criteria

    async def route(self, message: str, *, prompt_cache_key: str = "d1:router") -> RouteDecision:
        """Classify message intent, never masking fallback provenance."""
        cleaned = (message or "").strip()
        h_intent, h_conf, h_reason = heuristic_route(cleaned, self.capabilities)

        if len(cleaned) < self.min_chars:
            return RouteDecision(
                intent=h_intent,
                confidence=h_conf,
                source="heuristic",
                heuristic_intent=h_intent,
                decision_intent=None,
                decision_confidence=None,
                probabilities={},
                reason="short_message_skip",
                latency_ms=0.0,
                route_hint=f"{h_intent} ({h_conf:.2f}, heuristic)",
            )

        if self.d1_client is None:
            return RouteDecision(
                intent=h_intent,
                confidence=h_conf,
                source="heuristic",
                heuristic_intent=h_intent,
                decision_intent=None,
                decision_confidence=None,
                probabilities={},
                reason=f"no_d1_client:{h_reason}",
                latency_ms=0.0,
                route_hint=f"{h_intent} ({h_conf:.2f}, heuristic)",
            )

        criteria = self._active_criteria()
        state = f"USER_MESSAGE:\n{cleaned[:2000]}"
        instructions = "Classify the single primary user intent for this local workspace assistant request."

        try:
            res = await self.d1_client.decide(
                state=state,
                question_type="choice",
                instructions=instructions,
                criteria=criteria,
                prompt_cache_key=prompt_cache_key,
                timeout_s=self.limits.decision_timeout_s,
            )
        except (BackendError, ValueError, OSError) as exc:
            return RouteDecision(
                intent=h_intent,
                confidence=h_conf,
                source="heuristic",
                heuristic_intent=h_intent,
                decision_intent=None,
                decision_confidence=None,
                probabilities={},
                reason=f"d1_error:{type(exc).__name__}",
                latency_ms=0.0,
                route_hint=f"{h_intent} ({h_conf:.2f}, heuristic)",
            )

        probs = dict(res.probabilities or {})
        d_intent = str(res.choice or "general")
        d_conf = float(probs.get(d_intent, 0.0))

        # Safety floor 1: confidence below 0.55 cannot drive routing.
        if d_conf < self.limits.route_min_confidence:
            return RouteDecision(
                intent=h_intent,
                confidence=h_conf,
                source="heuristic",
                heuristic_intent=h_intent,
                decision_intent=d_intent,
                decision_confidence=d_conf,
                probabilities=probs,
                reason="below_confidence_floor",
                latency_ms=res.latency_ms,
                route_hint=f"{h_intent} ({h_conf:.2f}, heuristic)",
            )

        # Safety floor 2: a fuzzy 'general' choice never overrides a deterministic workspace keyword.
        if d_intent == "general" and h_intent != "general":
            return RouteDecision(
                intent=h_intent,
                confidence=h_conf,
                source="heuristic",
                heuristic_intent=h_intent,
                decision_intent="general",
                decision_confidence=d_conf,
                probabilities=probs,
                reason="general_cannot_override_keyword",
                latency_ms=res.latency_ms,
                route_hint=f"{h_intent} ({h_conf:.2f}, heuristic)",
            )

        return RouteDecision(
            intent=d_intent,
            confidence=d_conf,
            source="decision",
            heuristic_intent=h_intent,
            decision_intent=d_intent,
            decision_confidence=d_conf,
            probabilities=probs,
            reason="d1_accepted",
            latency_ms=res.latency_ms,
            route_hint=f"{d_intent} ({d_conf:.2f}, decision)",
        )
