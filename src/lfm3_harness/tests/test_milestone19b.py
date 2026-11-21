"""Fast host pytest suite for Milestone 19B: IntentRouter, WriteGate, EvidenceGate, and ApprovalManager."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness import (
    ApprovalManager,
    BackendDisconnectError,
    Capabilities,
    DecisionResult,
    EvidenceGate,
    HarnessStore,
    IntentRouter,
    MAX_GATE_EVIDENCE_CHARS,
    StagedWriteProposal,
    ToolExecutor,
    WorkspaceSandbox,
    WriteGate,
)


class FakeD1Client:
    """Deterministic stub mimicking DaemonClient.decide() for unit tests."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    async def decide(
        self,
        state: Optional[str],
        question_type: str,
        instructions: str,
        *,
        criteria: Any = None,
        prompt_cache_key: str = "d1:default",
        timeout_s: float = 12.0,
        image_bytes: Optional[bytes] = None,
    ) -> DecisionResult:
        self.calls.append(
            {
                "state": state,
                "question_type": question_type,
                "instructions": instructions,
                "criteria": criteria,
                "prompt_cache_key": prompt_cache_key,
            }
        )
        return self.handler(state, question_type, criteria)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# 1. IntentRouter (7 Intents, Floor, General Guard, Capability Gating, Traceability)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_intent_router_guards_fallbacks_and_capability_gating():
    def handler(state, qtype, criteria):
        return DecisionResult(
            question_type=qtype,
            instructions="",
            latency_ms=12.4,
            choice="summarize",
            probabilities={"general": 0.10, "explore": 0.12, "summarize": 0.78},
        )

    fake_d1 = FakeD1Client(handler)
    router = IntentRouter(fake_d1, capabilities=Capabilities(vision=False, voice_asr=False))

    # 1. Short message (< 24 chars) skips d1 completely -> heuristic trace.
    short_dec = await router.route("read README.md please")
    assert short_dec.source == "heuristic"
    assert short_dec.reason == "short_message_skip"
    assert short_dec.intent == "explore"
    assert len(fake_d1.calls) == 0

    # 2. Confident d1 decision (>= 0.55) accepted with capability-gated criteria (no vision/voice).
    good_dec = await router.route("Could you give a high level recap of how authentication works?")
    assert good_dec.source == "decision"
    assert good_dec.intent == "summarize"
    assert good_dec.confidence == pytest.approx(0.78)
    assert good_dec.reason == "d1_accepted"
    sent_criteria = fake_d1.calls[-1]["criteria"]
    assert "vision" not in sent_criteria and "voice" not in sent_criteria

    # 3. Low-confidence d1 (< 0.55) falls back cleanly with heuristic source + preserved decision fields.
    fake_d1.handler = lambda s, q, c: DecisionResult(
        question_type=q,
        instructions="",
        latency_ms=10.0,
        choice="related",
        probabilities={"related": 0.41, "explore": 0.35, "general": 0.24},
    )
    low_dec = await router.route("Can you read config.json and find the port setting?")
    assert low_dec.source == "heuristic"
    assert low_dec.reason == "below_confidence_floor"
    assert low_dec.intent == "explore"
    assert low_dec.decision_intent == "related"
    assert low_dec.decision_confidence == pytest.approx(0.41)

    # 4. d1 picking 'general' MUST NOT override strong deterministic workspace keyword.
    fake_d1.handler = lambda s, q, c: DecisionResult(
        question_type=q,
        instructions="",
        latency_ms=9.8,
        choice="general",
        probabilities={"general": 0.89, "explore": 0.05, "report": 0.06},
    )
    no_ovr = await router.route("Please inspect notes.md and edit the title line inside.")
    assert no_ovr.source == "heuristic"
    assert no_ovr.reason == "general_cannot_override_keyword"
    assert no_ovr.intent == "report"
    assert no_ovr.decision_intent == "general"

    # 5. Capability-gated senses enabled -> criteria includes vision/voice.
    router_av = IntentRouter(fake_d1, capabilities=Capabilities(vision=True, voice_asr=True))
    fake_d1.handler = lambda s, q, c: DecisionResult(
        question_type=q,
        instructions="",
        latency_ms=11.0,
        choice="vision",
        probabilities={"vision": 0.86, "voice": 0.04, "general": 0.10},
    )
    vis_dec = await router_av.route("Look at screenshot.png and tell me what error is shown.")
    assert "vision" in fake_d1.calls[-1]["criteria"] and "voice" in fake_d1.calls[-1]["criteria"]
    assert vis_dec.source == "decision" and vis_dec.intent == "vision"

    # 6. d1 failure degrades gracefully with explicit d1_error reason.
    def broken_handler(s, q, c):
        raise BackendDisconnectError("socket closed")

    fake_d1.handler = broken_handler
    err_dec = await router.route("Search the src directory for TODO comments across python modules.")
    assert err_dec.source == "heuristic"
    assert err_dec.intent == "explore"
    assert err_dec.reason == "d1_error:BackendDisconnectError"


# ---------------------------------------------------------------------------
# 2. Preflight WriteGate & Advisory EvidenceGate
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_write_gate_framing_hold_and_evidence_gate_advisory():
    prop = StagedWriteProposal(
        tool_name="write_file",
        rel_path="wrong_target.txt",
        base_sha256="",
        new_content="X" * 10_000,
        diff="--- a/wrong\n+++ b/wrong\n+" + ("Y" * 10_000),
        bytes_new=10_000,
        reason="overwrite",
    )

    def wg_hold(state, qtype, criteria):
        return DecisionResult(
            question_type=qtype,
            instructions="",
            latency_ms=14.0,
            choice="contradicts",
            probabilities={"aligned": 0.05, "uncertain": 0.10, "contradicts": 0.85},
        )

    fake_d1 = FakeD1Client(wg_hold)
    wg = WriteGate(fake_d1)

    v_hold = await wg.evaluate("Update README.md only", prop)
    assert v_hold.outcome == "hold"
    assert v_hold.verdict == "contradicts"
    assert "wrong_target.txt" in v_hold.revision_note
    sent_state = fake_d1.calls[-1]["state"]
    assert "USER REQUEST (TRUSTED):" in sent_state
    assert "PROPOSED ACTION (UNTRUSTED):" in sent_state
    # Untrusted payload clipped at MAX_GATE_EVIDENCE_CHARS (+ trusted header)
    untrusted_section = sent_state.split("PROPOSED ACTION (UNTRUSTED):\n", 1)[1]
    assert len(untrusted_section) <= MAX_GATE_EVIDENCE_CHARS

    # Low-confidence contradicts (< 0.75) allows approval card with explicit uncertainty.
    fake_d1.handler = lambda s, q, c: DecisionResult(
        question_type=q,
        instructions="",
        latency_ms=12.0,
        choice="contradicts",
        probabilities={"aligned": 0.15, "uncertain": 0.25, "contradicts": 0.60},
    )
    v_low = await wg.evaluate("Update README.md only", prop)
    assert v_low.outcome == "allow"

    # EvidenceGate: empty observations fallback, confident insufficient advisory, confident sufficient None.
    eg = EvidenceGate(fake_d1)
    e_empty = await eg.evaluate("How does main work?", "")
    assert e_empty.verdict == "insufficient" and e_empty.advisory is not None

    fake_d1.handler = lambda s, q, c: DecisionResult(
        question_type=q,
        instructions="",
        latency_ms=10.5,
        choice="insufficient",
        probabilities={"sufficient": 0.10, "insufficient": 0.80, "off_topic": 0.10},
    )
    e_insuff = await eg.evaluate("How does main work?", "Read only partial README header.")
    assert e_insuff.advisory is not None and "insufficient" in e_insuff.advisory

    fake_d1.handler = lambda s, q, c: DecisionResult(
        question_type=q,
        instructions="",
        latency_ms=10.5,
        choice="sufficient",
        probabilities={"sufficient": 0.88, "insufficient": 0.08, "off_topic": 0.04},
    )
    e_suff = await eg.evaluate("How does main work?", "def main(): return 42")
    assert e_suff.verdict == "sufficient" and e_suff.advisory is None


# ---------------------------------------------------------------------------
# 3. ApprovalManager (Hold Suppression, Single-Use Apply, TTL Expiry, TOCTOU Failure, No Allow Always)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_approval_manager_lifecycle_ttl_toctou_and_hold(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")

    sandbox = WorkspaceSandbox(ws)
    store = HarnessStore(tmp_path / "harness.db")
    ws_row = store.register_workspace("demo", ws)
    sess = store.create_session(ws_row["id"], "M19B Test")
    msg = store.add_message(sess["id"], "assistant", "Proposed change")

    tools = ToolExecutor(sandbox)
    staged_ok = tools.execute(
        "edit_file",
        {"path": "app.py", "target": "VALUE = 1", "replacement": "VALUE = 2", "reason": "bump value"},
    )
    assert staged_ok.ok and staged_ok.staged_proposal is not None

    # 1. Hold suppresses pending approval creation cleanly.
    d1_hold = FakeD1Client(
        lambda s, q, c: DecisionResult(
            question_type=q,
            instructions="",
            latency_ms=9.0,
            choice="contradicts",
            probabilities={"aligned": 0.02, "uncertain": 0.08, "contradicts": 0.90},
        )
    )
    mgr_hold = ApprovalManager(sandbox, store, write_gate=WriteGate(d1_hold))
    assert not hasattr(mgr_hold, "allow_always")

    held_res = await mgr_hold.stage_proposal(sess["id"], msg["id"], "Do not touch app.py", staged_ok.staged_proposal)
    assert held_res.held is True and held_res.action is None
    assert store.list_actions(sess["id"]) == []

    # 2. Aligned write creates pending card and allow_once applies pending -> approved -> applied.
    d1_allow = FakeD1Client(
        lambda s, q, c: DecisionResult(
            question_type=q,
            instructions="",
            latency_ms=8.5,
            choice="aligned",
            probabilities={"aligned": 0.92, "uncertain": 0.06, "contradicts": 0.02},
        )
    )
    mgr = ApprovalManager(sandbox, store, write_gate=WriteGate(d1_allow))
    staged_1 = await mgr.stage_proposal(sess["id"], msg["id"], "Change VALUE to 2 in app.py", staged_ok.staged_proposal)
    assert not staged_1.held and staged_1.action is not None
    aid1 = staged_1.action["id"]

    applied = mgr.allow_once(aid1)
    assert applied.ok and applied.status == "applied"
    assert applied.receipt is not None and applied.receipt.bytes_written == len("VALUE = 2\n")
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert store.get_action(aid1)["status"] == "applied"

    # Repeat allow_once refused cleanly.
    repeat = mgr.allow_once(aid1)
    assert not repeat.ok and repeat.status == "applied"

    # 3. TOCTOU external modification between stage & allow_once transitions approved -> failed (no purgatory!).
    prop_2 = tools.execute(
        "edit_file",
        {"path": "app.py", "target": "VALUE = 2", "replacement": "VALUE = 3", "reason": "bump again"},
    ).staged_proposal
    assert prop_2 is not None
    staged_2 = await mgr.stage_proposal(sess["id"], msg["id"], "Change VALUE to 3", prop_2)
    assert staged_2.action is not None
    aid2 = staged_2.action["id"]

    # Simulate external out-of-band file change before user clicks Allow.
    target.write_text("VALUE = 999\n", encoding="utf-8")
    toctou_res = mgr.allow_once(aid2)
    assert not toctou_res.ok
    assert toctou_res.status == "failed"
    assert store.get_action(aid2)["status"] == "failed"
    assert "TOCTOU" in (store.get_action(aid2)["error"] or "")
    assert target.read_text(encoding="utf-8") == "VALUE = 999\n"

    # 4. TTL expiration transitions pending -> expired without applying.
    prop_3 = tools.execute("write_file", {"path": "new.txt", "content": "fresh\n"}).staged_proposal
    assert prop_3 is not None
    staged_3 = await mgr.stage_proposal(sess["id"], msg["id"], "Create new.txt", prop_3, ttl_s=10)
    assert staged_3.action is not None
    aid3 = staged_3.action["id"]

    exp_res = mgr.allow_once(aid3, now=time.time() + 15.0)
    assert not exp_res.ok and exp_res.status == "expired"
    assert store.get_action(aid3)["status"] == "expired"
    assert not (ws / "new.txt").exists()

    # 5. User denial marks denied cleanly without writing.
    staged_4 = await mgr.stage_proposal(sess["id"], msg["id"], "Create new.txt", prop_3, ttl_s=600)
    assert staged_4.action is not None
    aid4 = staged_4.action["id"]
    deny_res = mgr.deny(aid4, reason="Not needed")
    assert deny_res.ok and deny_res.status == "denied"
    assert store.get_action(aid4)["status"] == "denied"
    assert not (ws / "new.txt").exists()

    store.close()
