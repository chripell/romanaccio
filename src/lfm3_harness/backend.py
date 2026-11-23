"""Pure-stdlib async TCP client and tokenizer/d1 readout engine for NPU daemons."""

import asyncio
import gc
import json
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .config import DECISION_TIMEOUT_S

BOS = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Verified exact token IDs shared across LFM3-1.7B, d1-2B, and LFM2.5-8B-A1B vocabularies.
_BUILTIN_LFM_TOKENS: Dict[str, int] = {
    "yes": 11683,
    "Yes": 12447,
    " yes": 18171,
    " Yes": 17550,
    " YES": 119000,
    "no": 2243,
    "No": 4547,
    "NO": 19598,
    " no": 794,
    " No": 2752,
    " NO": 14580,
    **{chr(ord("A") + i): 41 + i for i in range(26)},
    **{str(i): 24 + i for i in range(10)},
}


class BackendError(RuntimeError):
    """Base error raised by resident NPU daemon communication."""

    status_code: int = 500


class BackendUnavailableError(BackendError):
    status_code: int = 503


class BackendDisconnectError(BackendError):
    status_code: int = 502


class BackendTimeoutError(BackendError):
    status_code: int = 504


class SessionBusyError(BackendError):
    status_code: int = 409


class TokenizerLookup:
    """Zero-dependency single-token ID lookup loaded from HuggingFace tokenizer.json."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None, added_tokens: Optional[Dict[str, int]] = None):
        self._vocab: Dict[str, int] = dict(_BUILTIN_LFM_TOKENS)
        if vocab:
            self._vocab.update(vocab)
        if added_tokens:
            self._vocab.update(added_tokens)

    @classmethod
    def from_file(cls, tokenizer_json_path: Union[str, Path]) -> "TokenizerLookup":
        path = Path(tokenizer_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = dict(data.get("model", {}).get("vocab", {}))
        added = {item["content"]: int(item["id"]) for item in data.get("added_tokens", []) if "content" in item and "id" in item}
        del data
        gc.collect()
        return cls(vocab=vocab, added_tokens=added)

    def single_token_id(self, piece: str) -> Optional[int]:
        """Resolve exact single-token ID for a readout option string (mapping leading space -> 'Ġ')."""
        if piece in self._vocab:
            return self._vocab[piece]
        if piece.startswith(" "):
            gpt2_piece = "\u0120" + piece[1:]
            if gpt2_piece in self._vocab:
                return self._vocab[gpt2_piece]
        return None

    def single_token_ids(self, pieces: Sequence[str]) -> List[int]:
        seen = set()
        out: List[int] = []
        for p in pieces:
            tid = self.single_token_id(p)
            if tid is not None and tid not in seen:
                seen.add(tid)
                out.append(tid)
        return out


class Calibration:
    """Temperature scaling and piecewise linear calibration tables (pure stdlib)."""

    def __init__(self, calib_dict: Optional[Dict[str, Any]] = None):
        calib_dict = calib_dict or {}
        self.default_temp = float(calib_dict.get("default", 1.337398453966105))
        self.temperatures = calib_dict.get("temperatures", {})
        self.tables = calib_dict.get("tables", {})

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Calibration":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def temperature(self, q_type: str, cardinality: int = 2) -> float:
        if q_type == "noul":
            bucket = "noul"
        elif q_type == "score":
            bucket = f"score:{cardinality}"
        elif cardinality <= 3:
            bucket = "choice:small"
        elif cardinality <= 8:
            bucket = "choice:mid"
        elif cardinality <= 26:
            bucket = "choice:large"
        else:
            bucket = "choice:inventory"
        return float(self.temperatures.get(bucket, self.default_temp))

    def through(self, name: str, value: float) -> float:
        knots = self.tables.get(name)
        if not knots:
            return value
        idx = bisect_left([raw for raw, _ in knots], value)
        if idx == 0:
            return float(knots[0][1])
        if idx == len(knots):
            return float(knots[-1][1])
        (a, pa), (b, pb) = knots[idx - 1], knots[idx]
        if b == a:
            return float(pa)
        return float(pa + (value - a) * (pb - pa) / (b - a))


@dataclass(frozen=True)
class GenerateRequest:
    """Request payload sent to a resident C++ daemon."""

    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.1
    top_k: int = 50
    repeat_penalty: float = 1.1
    think_budget: int = 0
    no_think: bool = True
    raw: bool = True
    cache: bool = True
    prompt_cache_key: str = ""
    pin_blocks_up_to: int = 0
    stop: List[str] = field(default_factory=list)
    image_bytes: Optional[bytes] = None
    audio_bytes: Optional[bytes] = None
    embed_token_id: Optional[int] = None
    suppress_tokens: Optional[Tuple[int, ...]] = None
    session_id: Optional[str] = None


@dataclass(frozen=True)
class DaemonEvent:
    """Single JSON line event emitted by `lfm3_v2` or `lfm2_moe`."""

    type: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class DecisionResult:
    """Single-pass d1 probability/choice/score result (0 decode tokens)."""

    question_type: str
    instructions: str
    latency_ms: float
    decision: Optional[bool] = None
    probability: Optional[float] = None
    choice: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    level: Optional[str] = None
    level_index: Optional[int] = None
    distribution: Optional[List[float]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class DaemonClient:
    """Serialized async client over line-delimited JSON TCP to a resident C++ NPU daemon."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8088,
        *,
        tokenizer: Optional[TokenizerLookup] = None,
        calibration: Optional[Calibration] = None,
        connect_timeout_s: float = 5.0,
        assistant_lead: str = "<think></think>",
    ):
        self.host = host
        self.port = port
        self.tokenizer = tokenizer or TokenizerLookup()
        self.calibration = calibration or Calibration()
        self.connect_timeout_s = connect_timeout_s
        self.assistant_lead = assistant_lead
        self._lock = asyncio.Lock()
        self._active_sessions: set[str] = set()

    async def _open(
        self,
        payload: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        *,
        audio_bytes: Optional[bytes] = None,
        embed_token_id: Optional[int] = None,
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.connect_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise BackendTimeoutError(f"Timed out connecting to backend {self.host}:{self.port}") from exc
        except (OSError, ConnectionError) as exc:
            raise BackendUnavailableError(f"Backend unreachable at {self.host}:{self.port}: {exc}") from exc

        try:
            body = dict(payload)
            media_bytes = audio_bytes if audio_bytes is not None else image_bytes
            if embed_token_id is not None:
                body["embed_token_id"] = int(embed_token_id)
            if media_bytes:
                if audio_bytes is not None or body.get("embed_token_id") == 125031:
                    body["audio_bytes"] = len(media_bytes)
                    body["embed_token_id"] = int(body.get("embed_token_id") or 125031)
                else:
                    body["image_bytes"] = len(media_bytes)
            writer.write((json.dumps(body) + "\n").encode("utf-8"))
            if media_bytes:
                writer.write(media_bytes)
            await writer.drain()
            return reader, writer
        except (OSError, ConnectionError) as exc:
            await self._close_quietly(writer)
            raise BackendDisconnectError(f"Failed sending request to {self.host}:{self.port}: {exc}") from exc

    @staticmethod
    async def _close_quietly(writer: asyncio.StreamWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    @staticmethod
    async def _read_event(reader: asyncio.StreamReader) -> Optional[DaemonEvent]:
        while True:
            line = await reader.readline()
            if not line:
                return None
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            return DaemonEvent(type=str(obj.get("type", "")), payload=obj)

    async def command(self, cmd_payload: Dict[str, Any], *, timeout_s: float = 5.0) -> Dict[str, Any]:
        """Run a short control command (`cache_stats`, `cache_clear`) under the daemon lock."""
        async with self._lock:
            reader, writer = await self._open(cmd_payload)
            try:
                ev = await asyncio.wait_for(self._read_event(reader), timeout=timeout_s)
                if ev is None:
                    raise BackendDisconnectError(f"Empty response from {self.host}:{self.port}")
                if ev.type == "error":
                    raise BackendError(ev.payload.get("message", "Engine command error"))
                return ev.payload
            except asyncio.TimeoutError as exc:
                raise BackendTimeoutError(f"Backend command {cmd_payload.get('cmd')} timed out") from exc
            except (OSError, ConnectionError) as exc:
                raise BackendDisconnectError(f"Backend {self.host}:{self.port} disconnected: {exc}") from exc
            finally:
                await self._close_quietly(writer)

    async def cache_stats(self, timeout_s: float = 5.0) -> Dict[str, Any]:
        return await self.command({"cmd": "cache_stats"}, timeout_s=timeout_s)

    async def cache_clear(self, include_pinned: bool = False, timeout_s: float = 5.0) -> Dict[str, Any]:
        return await self.command({"cmd": "cache_clear", "include_pinned": bool(include_pinned)}, timeout_s=timeout_s)

    async def stream_generate(
        self,
        req: GenerateRequest,
        *,
        cancel_event: Optional[asyncio.Event] = None,
        read_timeout_s: float = 120.0,
        session_busy_check: Optional[Callable[[str], bool]] = None,
    ) -> AsyncIterator[DaemonEvent]:
        """Stream daemon events sequentially and close socket immediately upon cancellation."""
        if req.session_id and (
            req.session_id in self._active_sessions
            or (session_busy_check is not None and session_busy_check(req.session_id))
        ):
            raise SessionBusyError(f"Session '{req.session_id}' already has an active stream")

        if req.session_id:
            self._active_sessions.add(req.session_id)
        try:
            async with self._lock:
                payload: Dict[str, Any] = {
                    "prompt": req.prompt,
                    "max_tokens": int(req.max_tokens),
                    "temperature": float(req.temperature),
                    "top_k": int(req.top_k),
                    "repeat_penalty": float(req.repeat_penalty),
                    "think_budget": int(req.think_budget),
                    "no_think": bool(req.no_think),
                    "raw": bool(req.raw),
                    "cache": bool(req.cache),
                    "pin_blocks_up_to": int(req.pin_blocks_up_to),
                }
                if req.prompt_cache_key:
                    payload["prompt_cache_key"] = req.prompt_cache_key
                if req.stop:
                    payload["stop"] = list(req.stop)
                if req.embed_token_id is not None:
                    payload["embed_token_id"] = int(req.embed_token_id)
                if req.suppress_tokens:
                    payload["suppress_tokens"] = [int(t) for t in req.suppress_tokens]

                reader, writer = await self._open(
                    payload,
                    image_bytes=req.image_bytes,
                    audio_bytes=req.audio_bytes,
                    embed_token_id=req.embed_token_id,
                )
                try:
                    while True:
                        if cancel_event is not None and cancel_event.is_set():
                            return

                        read_task = asyncio.create_task(self._read_event(reader))
                        if cancel_event is None:
                            try:
                                ev = await asyncio.wait_for(read_task, timeout=read_timeout_s)
                            except asyncio.TimeoutError as exc:
                                read_task.cancel()
                                raise BackendTimeoutError(f"Backend {self.host}:{self.port} read timed out") from exc
                        else:
                            cancel_task = asyncio.create_task(cancel_event.wait())
                            done_tasks, pending_tasks = await asyncio.wait(
                                {read_task, cancel_task},
                                timeout=read_timeout_s,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for p in pending_tasks:
                                p.cancel()
                            if not done_tasks:
                                raise BackendTimeoutError(f"Backend {self.host}:{self.port} read timed out")
                            if cancel_task in done_tasks and cancel_event.is_set():
                                return
                            ev = await read_task

                        if ev is None:
                            return
                        if ev.type == "error":
                            raise BackendError(ev.payload.get("message", "Engine generation error"))
                        yield ev
                        if ev.type == "done":
                            return
                except (OSError, ConnectionError) as exc:
                    raise BackendDisconnectError(f"Backend {self.host}:{self.port} disconnected: {exc}") from exc
                finally:
                    await self._close_quietly(writer)
        finally:
            if req.session_id:
                self._active_sessions.discard(req.session_id)

    def _render_d1_prompt(
        self,
        state: Optional[str],
        question_type: str,
        instructions: str,
        criteria: Any = None,
    ) -> Tuple[str, List[List[int]], List[str]]:
        state_str = (state or "").strip()
        pfx = f"{IM_START}user\nSTATE:\n{state_str}\n\nQUESTION:\n" if state_str else f"{IM_START}user\n"
        labels: List[str] = []
        groups: List[List[int]] = []

        if question_type == "noul":
            extra = ""
            if isinstance(criteria, dict):
                extra = f"\nYes: {criteria.get('true', '')}\nNo: {criteria.get('false', '')}"
            body = f"{instructions}{extra}\n\nReply with yes or no only."
            labels = ["no", "yes"]
            groups.append(self.tokenizer.single_token_ids(["no", "No", "NO", " no", " No", " NO"]))
            groups.append(self.tokenizer.single_token_ids(["yes", "Yes", "YES", " yes", " Yes", " YES"]))
        elif question_type == "choice":
            if isinstance(criteria, (list, tuple)):
                crit_dict = {str(c): str(c) for c in criteria}
            elif isinstance(criteria, dict):
                crit_dict = {str(k): str(v) for k, v in criteria.items()}
            else:
                crit_dict = {"A": "Option A", "B": "Option B"}
            opt_lines: List[str] = []
            for idx, (label, desc) in enumerate(crit_dict.items()):
                code = chr(ord("A") + idx) if idx < 26 else str(idx)
                labels.append(label)
                opt_lines.append(f"{code}. {desc or label}")
                groups.append(self.tokenizer.single_token_ids([code, " " + code]))
            body = f"{instructions}\n\nOptions:\n" + "\n".join(opt_lines) + "\n\nReply with the option code only."
        elif question_type == "score":
            levels = list(criteria) if isinstance(criteria, (list, tuple)) else ["1", "2", "3", "4", "5"]
            legend = "\n".join(f"{i} {lvl}" for i, lvl in enumerate(levels))
            body = f"{instructions}\n\n{legend}\n\nReply with a single digit 0-{len(levels) - 1} only."
            labels = [str(lvl) for lvl in levels]
            for i in range(len(levels)):
                groups.append(self.tokenizer.single_token_ids([str(i), " " + str(i)]))
        else:
            raise ValueError(f"Unsupported question_type: {question_type}")

        for g in groups:
            if not g:
                raise ValueError(f"No single-token readout IDs resolved for question_type={question_type}")

        prompt = f"{BOS}{pfx}{body}{IM_END}\n{IM_START}assistant\n{self.assistant_lead}"
        return prompt, groups, labels

    async def decide(
        self,
        state: Optional[str],
        question_type: str,
        instructions: str,
        *,
        criteria: Any = None,
        prompt_cache_key: str = "d1:default",
        timeout_s: float = DECISION_TIMEOUT_S,
        image_bytes: Optional[bytes] = None,
    ) -> DecisionResult:
        """Execute a single-forward-pass typed d1 decision (0 decode tokens)."""
        prompt, token_groups, labels = self._render_d1_prompt(state, question_type, instructions, criteria)
        requested = sorted({tid for group in token_groups for tid in group})

        async with self._lock:
            payload: Dict[str, Any] = {
                "prompt": prompt,
                "max_tokens": 0,
                "return_logits": True,
                "requested_tokens": requested,
                "raw": True,
                "cache": True,
                "prompt_cache_key": prompt_cache_key,
            }
            reader, writer = await self._open(payload, image_bytes=image_bytes)
            try:
                ev = await asyncio.wait_for(self._read_event(reader), timeout=timeout_s)
                if ev is None:
                    raise BackendDisconnectError(f"Empty logits response from {self.host}:{self.port}")
                if ev.type == "error":
                    raise BackendError(ev.payload.get("message", "Decision error"))
                if ev.type != "logits":
                    raise BackendError(f"Unexpected decision response type '{ev.type}'")
            except asyncio.TimeoutError as exc:
                raise BackendTimeoutError(f"d1 decision timed out after {timeout_s} s") from exc
            except (OSError, ConnectionError) as exc:
                raise BackendDisconnectError(f"Backend {self.host}:{self.port} disconnected during decide: {exc}") from exc
            finally:
                await self._close_quietly(writer)

        raw_logits = {int(k): float(v) for k, v in (ev.payload.get("logits") or {}).items()}
        latency_ms = float(ev.payload.get("prefill_ms", 0.0))

        scores = [max(raw_logits.get(tid, -1e9) for tid in group) for group in token_groups]
        temp = self.calibration.temperature(question_type, len(labels))
        scaled = [s / temp for s in scores]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        denom = sum(exps)
        probs = [e / denom for e in exps]

        details = {
            "temperature": round(temp, 4),
            "cache_status": ev.payload.get("cache_status"),
            "cached_tokens": ev.payload.get("cached_tokens"),
        }

        if question_type == "noul":
            p_no, p_yes = probs[0], probs[1]
            return DecisionResult(
                question_type="noul",
                instructions=instructions,
                latency_ms=round(latency_ms, 2),
                decision=bool(p_yes >= 0.5),
                probability=round(p_yes, 4),
                details={**details, "p_yes": round(p_yes, 4), "p_no": round(p_no, 4)},
            )
        if question_type == "choice":
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            return DecisionResult(
                question_type="choice",
                instructions=instructions,
                latency_ms=round(latency_ms, 2),
                choice=labels[best_idx],
                probabilities={lab: round(p, 4) for lab, p in zip(labels, probs)},
                details=details,
            )

        cum = 0.0
        median_idx = 0
        for idx, p in enumerate(probs):
            cum += p
            if cum >= 0.5:
                median_idx = idx
                break
        return DecisionResult(
            question_type="score",
            instructions=instructions,
            latency_ms=round(latency_ms, 2),
            level=labels[median_idx],
            level_index=median_idx,
            distribution=[round(p, 4) for p in probs],
            details=details,
        )
