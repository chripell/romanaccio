"""Prefix-cache-first prompt rendering and resilient LFM3/LFM2.5 tool-call parsing."""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import ModelFamily

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_LFM3_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_LFM3_FUNC_RE = re.compile(r"<function=([^\s>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_LFM3_PARAM_RE = re.compile(r"<parameter=([^\s>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_LFM2_CALL_BLOCK_RE = re.compile(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    """Single parsed function invocation emitted by an actor model."""

    name: str
    arguments: Dict[str, Any]
    raw: str = ""
    call_id: str = ""


@dataclass(frozen=True)
class ParsedAssistantTurn:
    """Normalized assistant turn extracted from raw daemon output."""

    content: str
    thinking: str
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt string plus KV prefix-pinning and stop-sequence metadata."""

    prompt: str
    pinned_prefix_text: str
    pin_blocks_up_to: int
    estimated_prefix_tokens: int
    stop_sequences: List[str]


def _render_lfm3_param_value(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _format_lfm2_arg_value(val: Any) -> str:
    if isinstance(val, str):
        return "'" + val + "'"
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, bool):
        return "True" if val else "False"
    return str(val)


def _canonical_tool_fn(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return {k: v for k, v in fn.items() if k != "defer_loading"}


def _sort_tool_schemas(tool_schemas: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not tool_schemas:
        return []
    fns = [_canonical_tool_fn(dict(t)) for t in tool_schemas]
    return sorted(fns, key=lambda item: str(item.get("name", "")))


def _render_tool_calls(tool_calls: Sequence[Dict[str, Any]], family: ModelFamily) -> str:
    if family == "lfm2":
        parts: List[str] = []
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = str(fn.get("name", ""))
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            arg_str = ", ".join(f"{k}={_format_lfm2_arg_value(v)}" for k, v in args.items())
            parts.append(f"{name}({arg_str})")
        return "<|tool_call_start|>[" + ", ".join(parts) + "]<|tool_call_end|>"

    rendered: List[str] = []
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = str(fn.get("name", ""))
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        lines = [f"<tool_call>\n<function={name}>\n"]
        if isinstance(args, dict):
            for k, v in args.items():
                lines.append(f"<parameter={k}>\n{_render_lfm3_param_value(v)}\n</parameter>\n")
        lines.append("</function>\n</tool_call>")
        rendered.append("".join(lines))
    return "\n\n".join(rendered)


def build_pinned_system_prefix(
    system_contract: str,
    workspace_root: Path | str,
    tool_schemas: Optional[Sequence[Dict[str, Any]]] = None,
    family: ModelFamily = "lfm3",
) -> str:
    """Build the immutable `<|startoftext|><|im_start|>system...<|im_end|>\\n` block.

    All per-turn dynamic hints (route, phase, todo, advisories) are strictly excluded
    so Chained Hash128 KV blocks 0..K remain invariant across routes, tools, and turns.
    """
    ws_line = f"Workspace root: {Path(workspace_root).as_posix()}"
    base = system_contract.strip()
    sys_body = f"{base}\n{ws_line}" if base else ws_line

    sorted_fns = _sort_tool_schemas(tool_schemas)
    if sorted_fns:
        serialized = [json.dumps(fn, ensure_ascii=False) for fn in sorted_fns]
        if family == "lfm2":
            sys_body += "\nList of tools: [" + ", ".join(serialized) + "]"
        else:
            sys_body += "\n\n<tools>\n" + "\n".join(serialized) + "\n</tools>"

    return f"<|startoftext|><|im_start|>system\n{sys_body}<|im_end|>\n"


def _strip_think_from_content(text: str) -> str:
    if "<think>" not in text:
        return text
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def _format_suffix_note(
    route_hint: Optional[str] = None,
    phase: Optional[str] = None,
    todo_summary: Optional[str] = None,
    advisories: Optional[Sequence[str]] = None,
) -> str:
    fields: List[str] = []
    if route_hint:
        fields.append(f"route={route_hint.strip()}")
    if phase:
        fields.append(f"phase={phase.strip()}")
    if todo_summary:
        fields.append(f"todo={todo_summary.strip()}")
    if advisories:
        joined = " | ".join(a.strip() for a in advisories if a and a.strip())
        if joined:
            fields.append(f"advisories={joined}")
    if not fields:
        return ""
    return "\n\n<harness_state>\n" + "\n".join(fields) + "\n</harness_state>"


def render_prompt(
    messages: Sequence[Dict[str, Any]],
    *,
    system_contract: str,
    workspace_root: Path | str,
    tool_schemas: Optional[Sequence[Dict[str, Any]]] = None,
    family: ModelFamily = "lfm3",
    enable_thinking: bool = False,
    preserve_thinking: bool = False,
    route_hint: Optional[str] = None,
    phase: Optional[str] = None,
    todo_summary: Optional[str] = None,
    advisories: Optional[Sequence[str]] = None,
) -> RenderedPrompt:
    """Render ChatML conversation preserving an immutable pinned system+tools prefix."""
    pinned_prefix = build_pinned_system_prefix(
        system_contract=system_contract,
        workspace_root=workspace_root,
        tool_schemas=tool_schemas,
        family=family,
    )
    est_tokens = max(1, len(pinned_prefix.encode("utf-8")) // 4)
    pin_blocks = est_tokens // 64

    filtered = [m for m in messages if (m.get("role") or "").strip().lower() != "system"]
    last_user_idx = -1
    for idx, msg in enumerate(filtered):
        if (msg.get("role") or "").strip().lower() == "user":
            last_user_idx = idx

    # Identify tail message index to receive compact suffix state (if any).
    suffix_note = _format_suffix_note(route_hint, phase, todo_summary, advisories)
    tail_target_idx = len(filtered) - 1 if (filtered and suffix_note) else -1

    parts: List[str] = [pinned_prefix]
    idx = 0
    while idx < len(filtered):
        msg = filtered[idx]
        role = (msg.get("role") or "user").strip().lower()

        # Merge consecutive tool responses sharing a non-empty tool_call_id for LFM3 parity.
        if family == "lfm3" and role == "tool" and msg.get("tool_call_id"):
            tc_id = msg.get("tool_call_id")
            merged_items: List[str] = []
            end_idx = idx
            while end_idx < len(filtered):
                cand = filtered[end_idx]
                if (cand.get("role") or "").strip().lower() == "tool" and cand.get("tool_call_id") == tc_id:
                    c_txt = str(cand.get("content") or "")
                    if end_idx == tail_target_idx and suffix_note:
                        c_txt += suffix_note
                    if c_txt:
                        merged_items.append(c_txt)
                    end_idx += 1
                else:
                    break
            joined_body = "\n".join(merged_items)
            if "<tool_response>" not in joined_body:
                joined_body = f"<tool_response>\n{joined_body}\n</tool_response>"
            parts.append(f"<|im_start|>tool\n{joined_body}<|im_end|>\n")
            idx = end_idx
            continue

        parts.append(f"<|im_start|>{role}\n")
        if role == "assistant":
            keep_think = preserve_thinking or (idx > last_user_idx)
            thinking = msg.get("thinking") or msg.get("reasoning") or msg.get("reasoning_content")
            content = str(msg.get("content") or "")

            if family == "lfm3":
                if thinking is not None and keep_think:
                    parts.append(f"<think>{thinking}</think>")
                elif keep_think and "</think>" not in content:
                    parts.append("<think></think>")
                if not keep_think:
                    content = "<think></think>" + _strip_think_from_content(content)
            else:
                if thinking and keep_think:
                    parts.append(f"<think>{thinking}</think>")
                if not keep_think:
                    content = _strip_think_from_content(content)

            if idx == tail_target_idx and suffix_note:
                content += suffix_note
            parts.append(content)
            tcalls = msg.get("tool_calls")
            if tcalls:
                parts.append(_render_tool_calls(tcalls, family))
            parts.append("<|im_end|>\n")
        elif role == "tool":
            body = str(msg.get("content") or "")
            if idx == tail_target_idx and suffix_note:
                body += suffix_note
            if family == "lfm3" and "<tool_response>" not in body:
                body = f"<tool_response>\n{body}\n</tool_response>"
            parts.append(f"{body}<|im_end|>\n")
        else:
            body = str(msg.get("content") or "")
            if idx == tail_target_idx and suffix_note:
                body += suffix_note
            parts.append(f"{body}<|im_end|>\n")

        idx += 1

    parts.append("<|im_start|>assistant\n")
    if family == "lfm3":
        parts.append("<think>" if enable_thinking else "<think></think>")

    stop_seqs = ["<tool_response>", "<|im_start|>"] if family == "lfm3" else ["<|im_start|>"]
    return RenderedPrompt(
        prompt="".join(parts),
        pinned_prefix_text=pinned_prefix,
        pin_blocks_up_to=pin_blocks,
        estimated_prefix_tokens=est_tokens,
        stop_sequences=stop_seqs,
    )


def _coerce_param_value(raw_val: str) -> Any:
    trimmed = raw_val.strip()
    if trimmed == "true":
        return True
    if trimmed == "false":
        return False
    if trimmed == "null":
        return None
    # Only JSON-decode structured containers or clean scalars, preserving multi-line code strings.
    cleaned = re.sub(r"\s*</[A-Za-z0-9_]+>\s*$", "", trimmed)
    if (cleaned.startswith("{") and cleaned.endswith("}")) or (cleaned.startswith("[") and cleaned.endswith("]")):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                parsed_py = ast.literal_eval(cleaned)
                if isinstance(parsed_py, (dict, list)):
                    return parsed_py
            except (ValueError, SyntaxError):
                pass
            return raw_val
    if trimmed and trimmed.isdigit() and not (trimmed.startswith("0") and len(trimmed) > 1):
        return int(trimmed)
    return raw_val


def _ast_node_to_py(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id == "null":
            return None
        raise ValueError(f"Unsupported identifier in tool arguments: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        val = node.operand.value
        if isinstance(val, (int, float)):
            return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.List):
        return [_ast_node_to_py(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return [_ast_node_to_py(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {_ast_node_to_py(k): _ast_node_to_py(v) for k, v in zip(node.keys, node.values) if k is not None}
    raise ValueError(f"Unsafe AST node in tool call arguments: {type(node).__name__}")


def _parse_lfm3_calls(text: str, calls: List[ParsedToolCall], errors: List[str]) -> str:
    def _consume(match: re.Match[str]) -> str:
        inner = match.group(1)
        fn_match = _LFM3_FUNC_RE.search(inner)
        if not fn_match:
            # Safe JSON fallback inside <tool_call>{"name": ..., "arguments": {...}}</tool_call>
            try:
                obj = json.loads(inner.strip())
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("name"), str) and obj["name"].strip():
                raw_args = obj.get("arguments") or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError:
                        raw_args = {}
                if isinstance(raw_args, dict):
                    calls.append(
                        ParsedToolCall(
                            name=obj["name"].strip(),
                            arguments=raw_args,
                            raw=match.group(0),
                            call_id=f"call_{len(calls) + 1}",
                        )
                    )
                    return ""
            errors.append(f"Malformed LFM3 <tool_call> missing <function=...>: {inner[:120]}")
            return ""
        fn_name = fn_match.group(1).strip()
        param_blob = fn_match.group(2)
        args: Dict[str, Any] = {}
        for pm in _LFM3_PARAM_RE.finditer(param_blob):
            pname = pm.group(1).strip()
            pval = _coerce_param_value(pm.group(2))
            args[pname] = pval
        calls.append(ParsedToolCall(name=fn_name, arguments=args, raw=match.group(0), call_id=f"call_{len(calls) + 1}"))
        return ""

    return _LFM3_TOOL_CALL_RE.sub(_consume, text)


def _parse_lfm2_calls(text: str, calls: List[ParsedToolCall], errors: List[str]) -> str:
    def _consume(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not inner:
            errors.append("Empty <|tool_call_start|> block")
            return ""
        try:
            tree = ast.parse(inner, mode="eval")
        except SyntaxError as exc:
            errors.append(f"Invalid LFM2 tool call syntax: {exc}")
            return ""

        body = tree.body
        nodes = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
        for node in nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                errors.append(f"Invalid LFM2 call AST node: {ast.dump(node)[:120]}")
                continue
            args: Dict[str, Any] = {}
            ok = True
            for kw in node.keywords:
                if kw.arg is None:
                    errors.append(f"Unsupported **kwargs in tool call {node.func.id}")
                    ok = False
                    break
                try:
                    args[kw.arg] = _ast_node_to_py(kw.value)
                except ValueError as exc:
                    errors.append(str(exc))
                    ok = False
                    break
            if ok:
                calls.append(
                    ParsedToolCall(name=node.func.id, arguments=args, raw=match.group(0), call_id=f"call_{len(calls) + 1}")
                )
        return ""

    return _LFM2_CALL_BLOCK_RE.sub(_consume, text)


def parse_assistant_output(
    raw_text: str,
    *,
    family: str = "auto",
    thinking_override: Optional[str] = None,
) -> ParsedAssistantTurn:
    """Extract thinking, clean visible prose, and typed tool calls without throwing."""
    extracted_thinks: List[str] = []
    if thinking_override:
        extracted_thinks.append(thinking_override)

    def _collect_think(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body:
            extracted_thinks.append(body)
        return ""

    remaining = _THINK_TAG_RE.sub(_collect_think, raw_text or "")
    # Handle unclosed trailing/leading </think> fragment safely.
    if "</think>" in remaining:
        before, after = remaining.split("</think>", 1)
        cleaned_before = before.replace("<think>", "").strip()
        if cleaned_before:
            extracted_thinks.append(cleaned_before)
        remaining = after.lstrip()

    calls: List[ParsedToolCall] = []
    errors: List[str] = []

    if family in ("auto", "lfm3") and ("<tool_call>" in remaining or "</tool_call>" in remaining):
        if "<tool_call>" in remaining and "</tool_call>" in remaining:
            remaining = _parse_lfm3_calls(remaining, calls, errors)
        else:
            errors.append("Unclosed <tool_call> tag detected")

    if family in ("auto", "lfm2") and ("<|tool_call_start|>" in remaining or "<|tool_call_end|>" in remaining):
        if "<|tool_call_start|>" in remaining and "<|tool_call_end|>" in remaining:
            remaining = _parse_lfm2_calls(remaining, calls, errors)
        else:
            errors.append("Unclosed <|tool_call_start|> block detected")

    visible = remaining.replace("<|im_end|>", "").strip()
    thinking = "\n".join(extracted_thinks).strip()
    return ParsedAssistantTurn(content=visible, thinking=thinking, tool_calls=calls, parse_errors=errors)
