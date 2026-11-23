"""FastAPI + zero-build SPA server for Romanaccio LFM3 / LFM2.5 NPU Harness."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .agent_loop import _normalize_tool_name
from .approvals import ApprovalManager
from .audio import AudioService
from .backend import (
    BackendDisconnectError,
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
    Calibration,
    DaemonClient,
    GenerateRequest,
    TokenizerLookup,
)
from .config import (
    BackendEndpoint,
    Capabilities,
    HarnessConfig,
    SafetyLimits,
    validate_workspace_root,
)
from .gates import EvidenceGate, WriteGate
from .missions import (
    AcceptanceSpec,
    MissionRunner,
    make_doc_extract_spec,
    make_screen_qa_spec,
    setup_desk_task_workspace,
)
from .paths import PathSandboxError, WorkspaceSandbox
from .protocol import ParsedToolCall, _format_suffix_note, parse_assistant_output, render_prompt
from .router import IntentRouter
from .store import HarnessStore
from .tools import ToolExecutor, get_tool_schemas
from .vision import ImageMetadata, VisionClient, build_image_metadata, format_vision_placeholder, inspect_image_bytes

WEB_DIR = Path(__file__).resolve().parent / "web"
_READ_ONLY_CHAT_TOOLS = frozenset({"list_files", "read_file", "search_files", "related_files", "look", "transcribe"})
_AUTH_COOKIE = "lfm3_auth"
_DEFAULT_SYSTEM_CONTRACT = (
    "You are a disciplined local workspace coding and analysis assistant.\n"
    "Use read-only tools cleanly before making claims about files.\n"
    "Use write_file or edit_file to propose file modifications (they require explicit human approval).\n"
    "Never fabricate file contents, numbers, or tool outputs."
)


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in ("127.0.0.1", "localhost", "::1")


def _read_proc_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024.0, 2)
    except OSError:
        pass
    return 0.0


def _read_sysfs_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _collect_system_health(active_chat: int, active_missions: int, audio_caps: Dict[str, Any]) -> Dict[str, Any]:
    thermals: List[Dict[str, Any]] = []
    t_root = Path("/sys/class/thermal")
    if t_root.exists():
        for tz in sorted(t_root.glob("thermal_zone*")):
            t_type = (tz / "type").read_text(encoding="utf-8", errors="ignore").strip() if (tz / "type").exists() else tz.name
            temp_mc = _read_sysfs_int(tz / "temp")
            if temp_mc is not None:
                thermals.append({"zone": tz.name, "type": t_type, "temp_c": round(temp_mc / 1000.0, 1)})
    return {
        "rss_mib": _read_proc_rss_mib(),
        "active_chat_streams": active_chat,
        "active_missions": active_missions,
        "npu_freq_hz": {
            "die0": _read_sysfs_int(Path("/sys/class/devfreq/51c00000.eswin-npu/cur_freq")),
            "die1": _read_sysfs_int(Path("/sys/class/devfreq/71c00000.eswin-npu/cur_freq")),
        },
        "thermals": thermals,
        "audio": audio_caps,
    }


def _stored_msgs_to_chatml(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert persisted SQLite chat_messages rows back into chronological ChatML dicts."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        role = str(r.get("role") or "user")
        tel: Dict[str, Any] = {}
        tel_raw = r.get("telemetry_json")
        if isinstance(tel_raw, str) and tel_raw.strip() and tel_raw.strip() != "{}":
            try:
                parsed_tel = json.loads(tel_raw)
                if isinstance(parsed_tel, dict):
                    tel = parsed_tel
            except json.JSONDecodeError:
                tel = {}
        if role == "tool":
            content_str = str(r.get("content") or "")
            advs = tel.get("advisories") if isinstance(tel.get("advisories"), list) else None
            if advs:
                content_str += _format_suffix_note(advisories=[str(a) for a in advs])
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(r.get("tool_call_id") or ""),
                    "content": content_str,
                }
            )
            continue
        content_str = str(r.get("content") or "")
        if role == "user":
            rh = str(tel.get("route_hint") or "").strip() or None
            if rh:
                content_str += _format_suffix_note(route_hint=rh)
        item: Dict[str, Any] = {"role": role, "content": content_str}
        if r.get("thinking"):
            item["thinking"] = str(r["thinking"])
        tcalls_raw = r.get("tool_calls_json")
        if isinstance(tcalls_raw, str) and tcalls_raw.strip() and tcalls_raw.strip() != "[]":
            try:
                tc_list = json.loads(tcalls_raw)
                if isinstance(tc_list, list) and tc_list:
                    item["tool_calls"] = tc_list
            except json.JSONDecodeError:
                pass
        out.append(item)
    return out


def _build_daemon_client(
    ep: Optional[BackendEndpoint],
    *,
    tokenizer_override: Optional[TokenizerLookup] = None,
    calibration_path: Optional[Path] = None,
    load_tokenizer: bool = True,
) -> Optional[DaemonClient]:
    if ep is None:
        return None
    tok = tokenizer_override
    if load_tokenizer and tok is None and ep.tokenizer_path and Path(ep.tokenizer_path).exists():
        tok = TokenizerLookup.from_file(ep.tokenizer_path)
    cal = Calibration.load(calibration_path) if calibration_path else Calibration()
    return DaemonClient(
        host=ep.host,
        port=ep.port,
        tokenizer=tok,
        calibration=cal,
        connect_timeout_s=5.0,
        assistant_lead="<think></think>" if ep.family == "lfm3" else "",
    )


def create_app(
    config: HarnessConfig,
    *,
    store: Optional[HarnessStore] = None,
    audio_service: Optional[AudioService] = None,
    vision_client: Optional[VisionClient] = None,
    calibration_path: Optional[Path] = None,
    allow_no_auth: bool = False,
) -> FastAPI:
    """Create the configured FastAPI application for the Romanaccio LFM3/LFM2.5 Harness."""
    ws_root = validate_workspace_root(config.workspace_root)
    state_store = store or HarnessStore(config.db_path)
    ws_row = state_store.register_workspace(ws_root.name or "workspace", ws_root)
    sandbox = WorkspaceSandbox(ws_root)

    audio = audio_service or AudioService()
    audio_caps = audio.capabilities()
    v_client: Optional[VisionClient] = vision_client or (VisionClient(config.vision_url) if config.vision_url else None)
    caps = Capabilities(
        vision=bool(config.capabilities.vision or config.vision_url or v_client is not None),
        voice_asr=bool(config.capabilities.voice_asr or audio_caps["voice_asr"]),
        voice_tts=bool(config.capabilities.voice_tts or audio_caps["voice_tts"]),
        tts_engine=config.capabilities.tts_engine or audio_caps["tts_engine"],
    )

    clients: Dict[str, DaemonClient] = {}
    endpoints: Dict[str, BackendEndpoint] = {}
    actor_ep = config.actor
    actor_native_vision = bool(actor_ep.supports_vision)
    if actor_ep.family == "lfm3" and caps.vision and not actor_ep.supports_vision:
        actor_ep = BackendEndpoint(
            name=actor_ep.name,
            host=actor_ep.host,
            port=actor_ep.port,
            family=actor_ep.family,
            supports_thinking=actor_ep.supports_thinking,
            supports_vision=True,
            default_repeat_penalty=actor_ep.default_repeat_penalty,
            tokenizer_path=actor_ep.tokenizer_path,
        )
    endpoints[actor_ep.name] = actor_ep
    d1_ep = config.d1
    d1_separate = bool(d1_ep is not None and (d1_ep.host != actor_ep.host or d1_ep.port != actor_ep.port))

    c_primary = _build_daemon_client(actor_ep, calibration_path=calibration_path, load_tokenizer=not d1_separate)
    if c_primary is not None:
        clients[actor_ep.name] = c_primary

    if config.secondary_actor is not None:
        sec_ep = config.secondary_actor
        endpoints[sec_ep.name] = sec_ep
        c_sec = _build_daemon_client(sec_ep, calibration_path=calibration_path, load_tokenizer=False)
        if c_sec is not None:
            clients[sec_ep.name] = c_sec

    d1_client = (
        _build_daemon_client(d1_ep, calibration_path=calibration_path, load_tokenizer=True)
        if d1_separate
        else c_primary
    )
    if actor_native_vision:
        vision_daemon_client = clients[actor_ep.name]
        vision_daemon_name = actor_ep.name
    elif d1_separate and d1_client is not None:
        vision_daemon_client = d1_client
        vision_daemon_name = d1_ep.name if d1_ep is not None else "d1-2b"
    else:
        vision_daemon_client = clients[actor_ep.name]
        vision_daemon_name = actor_ep.name

    router = IntentRouter(d1_client, capabilities=caps, limits=config.limits)
    write_gate = WriteGate(d1_client, limits=config.limits)
    evidence_gate = EvidenceGate(d1_client, limits=config.limits)
    approval_mgr = ApprovalManager(sandbox, state_store, write_gate=write_gate, limits=config.limits)

    auth_token: Optional[str] = None if allow_no_auth else config.auth_token
    auth_enabled = bool(auth_token)

    active_chat_cancels: Dict[str, asyncio.Event] = {}
    active_mission_cancels: Dict[str, asyncio.Event] = {}
    mission_notifiers: Dict[str, asyncio.Condition] = {}
    mission_tasks: Dict[str, asyncio.Task] = {}
    mission_logs: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    audio_tmp_dir = state_store.db_path.parent / "audio_uploads"
    image_tmp_dir = state_store.db_path.parent / "image_uploads"
    uploaded_images: Dict[str, Tuple[ImageMetadata, Path]] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for ev in list(active_chat_cancels.values()) + list(active_mission_cancels.values()):
            ev.set()
        for t in list(mission_tasks.values()):
            if not t.done():
                t.cancel()
        if store is None:
            state_store.close()

    app = FastAPI(title="Romanaccio Harness Web UI", version=__version__, lifespan=lifespan)
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    def _is_public(path: str) -> bool:
        return (
            path in ("/", "/healthz", "/favicon.ico", "/api/auth/status", "/api/auth/login", "/api/auth/logout")
            or path.startswith("/static/")
        )

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        path = request.url.path
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and path.startswith("/api/"):
            origin = request.headers.get("origin")
            host = request.headers.get("host")
            if origin:
                parsed = urllib.parse.urlparse(origin)
                if not host or parsed.netloc.lower() != host.lower():
                    return JSONResponse({"detail": "CSRF Origin mismatch"}, status_code=403)

        if auth_enabled and not _is_public(path):
            bearer = ""
            auth_hdr = request.headers.get("authorization") or ""
            if auth_hdr.lower().startswith("bearer "):
                bearer = auth_hdr[7:].strip()
            cookie_tok = request.cookies.get(_AUTH_COOKIE, "")
            supplied = bearer or cookie_tok
            if not supplied or not hmac.compare_digest(supplied, auth_token or ""):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        return await call_next(request)

    async def _json_body(request: Request, *, required: bool = False) -> Dict[str, Any]:
        raw = await request.body()
        if not raw:
            if required:
                raise HTTPException(status_code=400, detail="Request body must not be empty")
            return {}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(obj, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        return obj

    def _resolve_model(model_name: Optional[str]) -> Tuple[BackendEndpoint, DaemonClient]:
        if model_name and model_name in clients:
            return endpoints[model_name], clients[model_name]
        if model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")
        return actor_ep, clients[actor_ep.name]

    @app.get("/")
    async def index() -> Response:
        idx = WEB_DIR / "index.html"
        if not idx.is_file():
            return JSONResponse({"status": "ok", "note": "SPA index.html not found"})
        return FileResponse(idx)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__, "workspace": ws_row["root_path"], "auth_enabled": auth_enabled}

    @app.get("/api/auth/status")
    async def auth_status(request: Request) -> Dict[str, Any]:
        authed = not auth_enabled
        if auth_enabled:
            tok = request.cookies.get(_AUTH_COOKIE, "")
            hdr = request.headers.get("authorization") or ""
            bearer = hdr[7:].strip() if hdr.lower().startswith("bearer ") else ""
            cand = bearer or tok
            authed = bool(cand and hmac.compare_digest(cand, auth_token or ""))
        return {"auth_enabled": auth_enabled, "authenticated": authed}

    @app.post("/api/auth/login")
    async def auth_login(request: Request, response: Response) -> Dict[str, Any]:
        if not auth_enabled:
            return {"ok": True, "auth_enabled": False}
        body = await _json_body(request, required=True)
        tok = str(body.get("token") or "").strip()
        if not tok or not hmac.compare_digest(tok, auth_token or ""):
            raise HTTPException(status_code=401, detail="Invalid access token")
        response.set_cookie(
            key=_AUTH_COOKIE,
            value=auth_token or "",
            httponly=True,
            samesite="strict",
            path="/",
        )
        return {"ok": True, "auth_enabled": True}

    @app.post("/api/auth/logout")
    async def auth_logout(response: Response) -> Dict[str, Any]:
        response.delete_cookie(key=_AUTH_COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/models")
    async def get_models() -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        primary_reachable = False
        for name, ep in endpoints.items():
            cli = clients[name]
            reachable = False
            stats: Dict[str, Any] = {}
            try:
                stats = await asyncio.wait_for(cli.cache_stats(timeout_s=1.2), timeout=1.5)
                reachable = True
            except Exception:
                reachable = False
            if name == actor_ep.name:
                primary_reachable = reachable
            out.append(
                {
                    "name": ep.name,
                    "host": ep.host,
                    "port": ep.port,
                    "family": ep.family,
                    "supports_thinking": ep.supports_thinking,
                    "supports_vision": ep.supports_vision,
                    "reachable": reachable,
                    "cache": stats,
                }
            )

        d1_reachable = primary_reachable
        d1_stats: Dict[str, Any] = {}
        if d1_separate and d1_client is not None:
            d1_reachable = False
            try:
                d1_stats = await asyncio.wait_for(d1_client.cache_stats(timeout_s=1.2), timeout=1.5)
                d1_reachable = True
            except Exception:
                d1_reachable = False

        vision_reachable = False
        vision_model = ""
        if v_client is not None:
            try:
                vh = await asyncio.wait_for(v_client.health(), timeout=2.0)
                vision_reachable = bool(vh.get("status") == "ok")
                vision_model = str(vh.get("model") or "")
            except Exception:
                vision_reachable = False

        return {
            "default_model": actor_ep.name,
            "models": out,
            "d1": {
                "name": d1_ep.name if (d1_separate and d1_ep) else actor_ep.name,
                "host": d1_ep.host if d1_ep else actor_ep.host,
                "port": d1_ep.port if d1_ep else actor_ep.port,
                "separate_daemon": d1_separate,
                "reachable": d1_reachable,
                "cache": d1_stats,
            },
            "capabilities": {
                "vision": caps.vision,
                "vision_url": config.vision_url or (getattr(v_client, "base_url", None) if v_client else None),
                "vision_reachable": vision_reachable,
                "vision_model": vision_model,
                "voice_asr": caps.voice_asr,
                "voice_tts": caps.voice_tts,
                "tts_engine": audio.tts_engine,
                "ffmpeg_available": audio_caps["ffmpeg_available"],
            },
        }

    @app.get("/api/workspaces")
    async def get_workspaces() -> Dict[str, Any]:
        return {"default_workspace_id": ws_row["id"], "workspaces": state_store.list_workspaces()}

    @app.get("/api/workspaces/{ws_id}/files")
    async def get_workspace_files(ws_id: str, path: str = ".") -> Dict[str, Any]:
        if ws_id != ws_row["id"] and state_store.get_workspace(ws_id) is None:
            raise HTTPException(status_code=404, detail="Workspace not found")
        ex = ToolExecutor(sandbox, caps, config.limits)
        res = ex.execute("list_files", {"path": path, "recursive": False})
        if not res.ok:
            raise HTTPException(status_code=400, detail=res.error or "Failed listing files")
        return json.loads(res.content)

    @app.get("/api/sessions")
    async def list_sessions(workspace_id: Optional[str] = None) -> Dict[str, Any]:
        sessions = state_store.list_sessions(workspace_id)
        for s in sessions:
            s["active_stream"] = s["id"] in active_chat_cancels
        return {"sessions": sessions}

    @app.post("/api/sessions")
    async def create_session(request: Request) -> Dict[str, Any]:
        body = await _json_body(request, required=False)
        title = str(body.get("title") or "New Chat").strip()[:120] or "New Chat"
        model_name = str(body.get("model_name") or actor_ep.name).strip()
        ep, _ = _resolve_model(model_name)
        sess = state_store.create_session(ws_row["id"], title, model_name=ep.name, status="idle")
        return {"session": sess}

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str) -> Dict[str, Any]:
        sess = state_store.get_session(sid)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = state_store.list_messages(sid)
        actions = state_store.list_actions(sid)
        now = time.time()
        for act in actions:
            act["expires_in_s"] = max(0, int(round(float(act["expires_at"]) - now)))
        return {
            "session": {**sess, "active_stream": sid in active_chat_cancels},
            "messages": msgs,
            "actions": actions,
        }

    @app.patch("/api/sessions/{sid}")
    async def patch_session(sid: str, request: Request) -> Dict[str, Any]:
        sess = state_store.get_session(sid)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found")
        body = await _json_body(request, required=True)
        if "title" in body:
            t = str(body["title"] or "").strip()[:120]
            if t:
                state_store.update_session_title(sid, t)
        if "model_name" in body:
            ep, _ = _resolve_model(str(body["model_name"] or "").strip())
            state_store.update_session_model(sid, ep.name)
        return {"session": state_store.get_session(sid)}

    @app.delete("/api/sessions/{sid}")
    async def delete_session(sid: str) -> Dict[str, Any]:
        if sid in active_chat_cancels:
            raise HTTPException(status_code=409, detail="Cannot delete session while streaming")
        ok = state_store.delete_session(sid)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}

    @app.post("/api/sessions/{sid}/stop")
    async def stop_session_stream(sid: str) -> Dict[str, Any]:
        sess = state_store.get_session(sid)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found")
        ev = active_chat_cancels.get(sid)
        if ev is not None:
            ev.set()
            return {"ok": True, "stopping": True}
        return {"ok": True, "stopping": False}

    @app.post("/api/sessions/{sid}/messages")
    async def post_session_message(sid: str, request: Request):
        sess = state_store.get_session(sid)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if sid in active_chat_cancels or sess.get("status") == "streaming":
            raise HTTPException(status_code=409, detail=f"Session '{sid}' already has an active stream")

        body = await _json_body(request, required=True)
        content = str(body.get("content") or "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="Message 'content' must not be empty")

        req_model = str(body.get("model_name") or sess.get("model_name") or actor_ep.name).strip()
        ep, client = _resolve_model(req_model)
        if req_model != sess.get("model_name"):
            state_store.update_session_model(sid, ep.name)

        enable_thinking = bool(body.get("enable_thinking", False)) and ep.supports_thinking
        think_budget = max(0, int(body.get("think_budget", 160 if enable_thinking else 0)))
        temperature = max(0.0, min(2.0, float(body.get("temperature", 0.1))))
        max_tokens = max(1, min(4096, int(body.get("max_tokens", 1024))))
        max_rounds = max(1, min(config.limits.max_tool_rounds, int(body.get("max_rounds", config.limits.max_tool_rounds))))

        image_id = str(body.get("image_id") or "").strip()
        turn_image_bytes: Optional[bytes] = None
        attached_img_meta: Optional[ImageMetadata] = None
        vision_prefix = ""
        stored_user_content = content
        if image_id:
            if image_id not in uploaded_images:
                raise HTTPException(status_code=404, detail=f"Unknown image_id '{image_id}'")
            if ep.family != "lfm3":
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{ep.name}' does not support direct vision input; switch to '{actor_ep.name}' or use workspace look()",
                )
            img_meta, img_embed_path = uploaded_images[image_id]
            attached_img_meta = img_meta
            turn_image_bytes = img_embed_path.read_bytes()
            vision_prefix = format_vision_placeholder(img_meta.num_tokens) + "\n"
            stored_user_content = (
                f"[Image attached: {img_meta.filename} ({img_meta.width}x{img_meta.height}, {img_meta.num_tokens} vis tok)]\n"
                f"{content}"
            )

        prior_rows = state_store.list_messages(sid)
        if not prior_rows and (sess.get("title") or "").strip() in ("", "New Chat"):
            state_store.update_session_title(sid, content.splitlines()[0][:64])

        user_row = state_store.add_message(sid, "user", stored_user_content)
        asst_placeholder = state_store.add_message(sid, "assistant", "", status="streaming")
        cancel_ev = asyncio.Event()
        active_chat_cancels[sid] = cancel_ev
        state_store.update_session_status(sid, "streaming")

        async def event_stream():
            final_text_parts: List[str] = []
            final_think_parts: List[str] = []
            all_tool_calls: List[Dict[str, Any]] = []
            telemetry_summary: Dict[str, Any] = {}
            terminal_status = "completed"

            async def _watch_disconnect() -> None:
                while not cancel_ev.is_set():
                    try:
                        if await request.is_disconnected():
                            cancel_ev.set()
                            return
                    except Exception:
                        return
                    await asyncio.sleep(0.25)

            watcher = asyncio.create_task(_watch_disconnect())
            try:
                # 1. Single-pass 7-intent routing
                route = await router.route(content, prompt_cache_key=f"d1:router:{sid}")
                state_store.log_event("session", sid, "route_decision", route.to_dict())
                yield _sse("route", route.to_dict())

                stored_user_txt = stored_user_content
                active_image_bytes = turn_image_bytes
                user_prompt_content = f"{vision_prefix}{content}" if vision_prefix else content
                if (
                    turn_image_bytes is not None
                    and attached_img_meta is not None
                    and not actor_native_vision
                    and d1_separate
                    and d1_client is not None
                ):
                    vis_prompt = (
                        f"<|startoftext|><|im_start|>user\n"
                        f"{vision_prefix}{content}"
                        f"<|im_end|>\n<|im_start|>assistant\n<think></think>"
                    )
                    vis_req = GenerateRequest(
                        prompt=vis_prompt,
                        max_tokens=192,
                        temperature=0.1,
                        repeat_penalty=actor_ep.default_repeat_penalty,
                        think_budget=0,
                        no_think=True,
                        raw=True,
                        cache=True,
                        prompt_cache_key=f"vision:{sid}",
                        image_bytes=turn_image_bytes,
                        session_id=f"{sid}:img",
                    )
                    vis_ans: List[str] = []
                    try:
                        async for vev in vision_daemon_client.stream_generate(vis_req, cancel_event=cancel_ev, read_timeout_s=60.0):
                            if vev.type == "token" and not vev.payload.get("thinking"):
                                vis_ans.append(str(vev.payload.get("text") or ""))
                    except (BackendUnavailableError, BackendDisconnectError, BackendTimeoutError, BackendError) as exc:
                        terminal_status = "error"
                        yield _sse("error", {"status_code": getattr(exc, "status_code", 502), "message": str(exc)})
                        return
                    vis_obs = "".join(vis_ans).strip()
                    yield _sse(
                        "vision_decode",
                        {
                            "vision_actor": vision_daemon_name,
                            "filename": attached_img_meta.filename,
                            "visual_tokens": attached_img_meta.num_tokens,
                            "observation": vis_obs,
                        },
                    )
                    stored_user_txt = (
                        f"[Image attached: {attached_img_meta.filename} ({attached_img_meta.width}x{attached_img_meta.height}, "
                        f"{attached_img_meta.num_tokens} vis tok) | {vision_daemon_name} vision: {vis_obs}]\n{content}"
                    )
                    user_prompt_content = stored_user_txt
                    active_image_bytes = None

                if route.route_hint or stored_user_txt != stored_user_content:
                    state_store.update_message(
                        user_row["id"],
                        content=stored_user_txt,
                        telemetry={"route_hint": route.route_hint} if route.route_hint else None,
                    )
                user_note = _format_suffix_note(route_hint=route.route_hint)

                history = _stored_msgs_to_chatml(prior_rows)
                history.append({"role": "user", "content": f"{user_prompt_content}{user_note}"})

                tool_schemas = get_tool_schemas(caps, include_job_tools=False)
                executor = ToolExecutor(sandbox, caps, config.limits)
                allowed_names = {s["function"]["name"] for s in tool_schemas if isinstance(s.get("function"), dict)}
                ro_observations: List[str] = []
                evidence_sent = False
                last_tool_row: Optional[Dict[str, Any]] = None

                for round_idx in range(1, max_rounds + 1):
                    if cancel_ev.is_set():
                        terminal_status = "cancelled"
                        break

                    if round_idx >= 2 and ro_observations and not evidence_sent:
                        ev_verdict = await evidence_gate.evaluate(
                            content,
                            "\n\n".join(ro_observations),
                            prompt_cache_key=f"d1:evidence:{sid}",
                        )
                        state_store.log_event("session", sid, "evidence_gate", ev_verdict.to_dict())
                        yield _sse("evidence_gate", ev_verdict.to_dict())
                        evidence_sent = True
                        if ev_verdict.advisory:
                            adv_note = _format_suffix_note(advisories=[ev_verdict.advisory])
                            if adv_note and history:
                                history[-1]["content"] = f"{str(history[-1].get('content') or '')}{adv_note}"
                            if last_tool_row is not None:
                                state_store.update_message(
                                    last_tool_row["id"],
                                    content=str(last_tool_row.get("content") or ""),
                                    telemetry={"advisories": [ev_verdict.advisory]},
                                )

                    rendered = render_prompt(
                        history,
                        system_contract=_DEFAULT_SYSTEM_CONTRACT,
                        workspace_root=sandbox.root,
                        tool_schemas=tool_schemas,
                        family=ep.family,
                        enable_thinking=enable_thinking,
                    )
                    gen_req = GenerateRequest(
                        prompt=rendered.prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        repeat_penalty=ep.default_repeat_penalty,
                        think_budget=think_budget if enable_thinking else 0,
                        no_think=not enable_thinking,
                        raw=True,
                        cache=True,
                        prompt_cache_key=f"chat:{sid}:{ep.name}",
                        pin_blocks_up_to=rendered.pin_blocks_up_to,
                        stop=rendered.stop_sequences,
                        image_bytes=active_image_bytes if round_idx == 1 else None,
                        session_id=f"{sid}:r{round_idx}",
                    )

                    round_raw: List[str] = []
                    round_think: List[str] = []
                    round_done: Dict[str, Any] = {}

                    try:
                        async for ev in client.stream_generate(gen_req, cancel_event=cancel_ev, read_timeout_s=180.0):
                            if ev.type == "prefill_done":
                                yield _sse("prefill_done", {"round": round_idx, **ev.payload})
                            elif ev.type == "token":
                                txt = str(ev.payload.get("text") or "")
                                if ev.payload.get("thinking"):
                                    round_think.append(txt)
                                    yield _sse("thinking", {"round": round_idx, "text": txt})
                                else:
                                    round_raw.append(txt)
                                    yield _sse("token", {"round": round_idx, "text": txt})
                            elif ev.type == "done":
                                round_done = dict(ev.payload)
                    except (BackendUnavailableError, BackendDisconnectError, BackendTimeoutError, BackendError) as exc:
                        terminal_status = "error"
                        yield _sse("error", {"status_code": getattr(exc, "status_code", 502), "message": str(exc)})
                        return

                    if cancel_ev.is_set():
                        terminal_status = "cancelled"
                        final_text_parts.extend(round_raw)
                        final_think_parts.extend(round_think)
                        yield _sse("done", {"status": "cancelled", "round": round_idx, "message_id": asst_placeholder["id"]})
                        return

                    raw_text = "".join(round_raw)
                    think_text = "".join(round_think)
                    parsed = parse_assistant_output(raw_text, family=ep.family)
                    if parsed.thinking and not think_text:
                        think_text = parsed.thinking

                    if think_text:
                        final_think_parts.append(think_text)

                    telemetry_summary = {
                        "model": ep.name,
                        "rounds": round_idx,
                        "prompt_tokens": round_done.get("prompt_tokens", 0),
                        "cached_tokens": round_done.get("cached_tokens", 0),
                        "cache_status": round_done.get("cache_status", ""),
                        "cache_tier": round_done.get("cache_tier", ""),
                        "cache_restore_ms": round_done.get("cache_restore_ms", 0.0),
                        "prefill_ms": round_done.get("prefill_ms", 0.0),
                        "decode_ms": round_done.get("decode_ms", 0.0),
                        "completion_tokens": round_done.get("completion_tokens", 0),
                        "route": route.to_dict(),
                    }

                    if not parsed.tool_calls:
                        final_text_parts.append(parsed.content)
                        yield _sse("done", {"status": "completed", "message_id": asst_placeholder["id"], "telemetry": telemetry_summary})
                        return

                    # Normalize tool calls and execute read-only tools or stage write proposals
                    tc_dicts: List[Dict[str, Any]] = []
                    norm_calls: List[ParsedToolCall] = []
                    for idx_tc, tc in enumerate(parsed.tool_calls, start=1):
                        n_name = _normalize_tool_name(tc.name, allowed_names)
                        cid = tc.call_id or f"call_r{round_idx}_{idx_tc}"
                        ntc = ParsedToolCall(name=n_name, arguments=tc.arguments, raw=tc.raw, call_id=cid)
                        norm_calls.append(ntc)
                        tc_entry = {"id": cid, "name": n_name, "arguments": tc.arguments}
                        tc_dicts.append(tc_entry)
                        all_tool_calls.append(tc_entry)
                        yield _sse("tool_call", {"round": round_idx, "call_id": cid, "name": n_name, "arguments": tc.arguments})

                    history.append(
                        {
                            "role": "assistant",
                            "content": parsed.content,
                            "thinking": think_text,
                            "tool_calls": [
                                {"id": tc["id"], "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                for tc in tc_dicts
                            ],
                        }
                    )

                    awaiting_approval = False
                    for ntc in norm_calls:
                        if ntc.name in ("write_file", "edit_file"):
                            res = executor.execute(ntc.name, ntc.arguments, allowed_tools=tuple(allowed_names))
                            if not res.ok or res.staged_proposal is None:
                                body_out = json.dumps({"ok": False, "error": res.error or "Staging failed"})
                                yield _sse("tool_result", {"round": round_idx, "call_id": ntc.call_id, "name": ntc.name, "ok": False, "result": body_out})
                                history.append({"role": "tool", "tool_call_id": ntc.call_id, "content": body_out})
                                continue

                            stage = await approval_mgr.stage_proposal(
                                sid,
                                asst_placeholder["id"],
                                content,
                                res.staged_proposal,
                                prompt_cache_key=f"d1:write_gate:{sid}",
                            )
                            if stage.held:
                                hold_payload = {
                                    "ok": False,
                                    "status": "held_by_write_gate",
                                    "revision_note": stage.revision_note,
                                    "verdict": stage.gate_verdict.to_dict(),
                                }
                                yield _sse("write_gate_hold", {"round": round_idx, "call_id": ntc.call_id, **hold_payload})
                                history.append({"role": "tool", "tool_call_id": ntc.call_id, "content": json.dumps(hold_payload)})
                            elif stage.action is not None:
                                act_payload = dict(stage.action)
                                act_payload["expires_in_s"] = max(0, int(round(float(act_payload["expires_at"]) - time.time())))
                                yield _sse("approval_card", {"round": round_idx, "action": act_payload, "gate": stage.gate_verdict.to_dict()})
                                state_store.add_message(
                                    sid,
                                    "tool",
                                    json.dumps({"ok": True, "status": "awaiting_approval", "action_id": stage.action["id"], "path": res.staged_proposal.rel_path}),
                                    tool_call_id=ntc.call_id,
                                )
                                awaiting_approval = True
                        elif ntc.name == "look" and v_client is not None and caps.vision:
                            rel_img = str(ntc.arguments.get("path") or "").strip()
                            v_question = str(
                                ntc.arguments.get("question")
                                or "Describe this image accurately, noting shapes, colors, layout, and any readable text."
                            ).strip()
                            look_ok = False
                            try:
                                target_img = sandbox.resolve_read(rel_img)
                                raw_img = target_img.read_bytes()
                                img_meta_tool = build_image_metadata(raw_img, target_img.name)
                                emb_bytes, n_vis_tok, _vinfo = await v_client.encode(raw_img, filename=target_img.name)
                                vis_prompt = (
                                    f"<|startoftext|><|im_start|>user\n"
                                    f"{format_vision_placeholder(n_vis_tok)}\n{v_question}"
                                    f"<|im_end|>\n<|im_start|>assistant\n<think></think>"
                                )
                                vis_req = GenerateRequest(
                                    prompt=vis_prompt,
                                    max_tokens=256,
                                    temperature=0.1,
                                    repeat_penalty=actor_ep.default_repeat_penalty,
                                    think_budget=0,
                                    no_think=True,
                                    raw=True,
                                    cache=True,
                                    prompt_cache_key=f"vision:{sid}",
                                    image_bytes=emb_bytes,
                                    session_id=f"{sid}:r{round_idx}:look",
                                )
                                vis_ans: List[str] = []
                                async for vev in vision_daemon_client.stream_generate(vis_req, cancel_event=cancel_ev, read_timeout_s=60.0):
                                    if vev.type == "token" and not vev.payload.get("thinking"):
                                        vis_ans.append(str(vev.payload.get("text") or ""))
                                look_ok = True
                                out_str = json.dumps(
                                    {
                                        "status": "ok",
                                        "path": sandbox.relativize(target_img),
                                        "bytes": img_meta_tool.bytes,
                                        "sha256": img_meta_tool.sha256,
                                        "format": img_meta_tool.format,
                                        "width": img_meta_tool.width,
                                        "height": img_meta_tool.height,
                                        "visual_tokens": n_vis_tok,
                                        "vision_actor": vision_daemon_name,
                                        "description": "".join(vis_ans).strip(),
                                    },
                                    indent=2,
                                )
                            except PathSandboxError as exc:
                                out_str = json.dumps({"ok": False, "error": f"Sandbox violation ({exc.reason}): {exc}"})
                            except Exception as exc:
                                out_str = json.dumps({"ok": False, "error": f"Vision look failed: {exc}"})
                            if look_ok:
                                ro_observations.append(f"[look]\n{out_str}")
                            yield _sse("tool_result", {"round": round_idx, "call_id": ntc.call_id, "name": ntc.name, "ok": look_ok, "result": out_str})
                            history.append({"role": "tool", "tool_call_id": ntc.call_id, "content": out_str})
                            last_tool_row = state_store.add_message(sid, "tool", out_str, tool_call_id=ntc.call_id)
                        elif ntc.name in _READ_ONLY_CHAT_TOOLS:
                            res = executor.execute(ntc.name, ntc.arguments, allowed_tools=tuple(allowed_names))
                            out_str = res.content if res.ok else json.dumps({"ok": False, "error": res.error})
                            if res.ok:
                                ro_observations.append(f"[{ntc.name}]\n{out_str}")
                            yield _sse("tool_result", {"round": round_idx, "call_id": ntc.call_id, "name": ntc.name, "ok": res.ok, "result": out_str})
                            history.append({"role": "tool", "tool_call_id": ntc.call_id, "content": out_str})
                            last_tool_row = state_store.add_message(sid, "tool", out_str, tool_call_id=ntc.call_id)
                        else:
                            err_out = json.dumps({"ok": False, "error": f"Unknown/disabled tool '{ntc.name}'"})
                            yield _sse("tool_result", {"round": round_idx, "call_id": ntc.call_id, "name": ntc.name, "ok": False, "result": err_out})
                            history.append({"role": "tool", "tool_call_id": ntc.call_id, "content": err_out})

                    if awaiting_approval:
                        if parsed.content:
                            final_text_parts.append(parsed.content)
                        else:
                            final_text_parts.append("Proposed file modification staged for approval.")
                        yield _sse("done", {"status": "awaiting_approval", "message_id": asst_placeholder["id"], "telemetry": telemetry_summary})
                        return

                yield _sse("done", {"status": "max_rounds", "message_id": asst_placeholder["id"], "telemetry": telemetry_summary})
            finally:
                watcher.cancel()
                try:
                    await watcher
                except BaseException:
                    pass
                active_chat_cancels.pop(sid, None)
                state_store.update_message(
                    asst_placeholder["id"],
                    content="\n\n".join(p for p in final_text_parts if p).strip(),
                    thinking="\n\n".join(p for p in final_think_parts if p).strip(),
                    status=terminal_status,
                    telemetry={**telemetry_summary, "tool_calls": all_tool_calls},
                )
                if all_tool_calls:
                    with state_store.conn:
                        state_store.conn.execute(
                            "UPDATE chat_messages SET tool_calls_json = ? WHERE id = ?",
                            (json.dumps(all_tool_calls), asst_placeholder["id"]),
                        )
                state_store.update_session_status(sid, "idle")

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/actions/{action_id}/allow")
    async def allow_action(action_id: str) -> Dict[str, Any]:
        outcome = approval_mgr.allow_once(action_id)
        if outcome.status == "not_found":
            raise HTTPException(status_code=404, detail=outcome.error)
        if outcome.status == "expired":
            raise HTTPException(status_code=410, detail=outcome.error)
        if not outcome.ok:
            raise HTTPException(status_code=409, detail=outcome.error)
        return {
            "ok": True,
            "status": outcome.status,
            "action_id": action_id,
            "receipt": asdict(outcome.receipt) if outcome.receipt else None,
            "action": state_store.get_action(action_id),
        }

    @app.post("/api/actions/{action_id}/deny")
    async def deny_action(action_id: str, request: Request) -> Dict[str, Any]:
        body = await _json_body(request, required=False)
        reason = str(body.get("reason") or "User denied").strip()
        outcome = approval_mgr.deny(action_id, reason=reason)
        if outcome.status == "not_found":
            raise HTTPException(status_code=404, detail=outcome.error)
        if outcome.status == "expired":
            raise HTTPException(status_code=410, detail=outcome.error)
        if not outcome.ok:
            raise HTTPException(status_code=409, detail=outcome.error)
        return {"ok": True, "status": outcome.status, "action_id": action_id, "action": state_store.get_action(action_id)}

    @app.get("/api/missions/templates")
    async def mission_templates() -> Dict[str, Any]:
        return {
            "templates": [
                {"id": "desk_task", "name": "Desk Task Summary", "semantic_mode": "deterministic", "checks": 5},
                {"id": "screen_qa", "name": "Screenshot QA", "semantic_mode": "human_review_only", "requires_capability": "vision"},
                {"id": "doc_extract", "name": "Document Extract", "semantic_mode": "human_review_only"},
            ]
        }

    @app.get("/api/missions")
    async def list_missions(workspace_id: Optional[str] = None) -> Dict[str, Any]:
        runs = state_store.list_mission_runs(workspace_id)
        for r in runs:
            r["active"] = r["id"] in active_mission_cancels
        return {"missions": runs}

    @app.get("/api/missions/{run_id}")
    async def get_mission(run_id: str) -> Dict[str, Any]:
        row = state_store.get_mission_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Mission run not found")
        spec_row = state_store.get_acceptance_spec(row["spec_id"])
        row["spec"] = json.loads(spec_row["canonical_json"]) if spec_row else None
        row["active"] = run_id in active_mission_cancels
        return {"mission": row}

    @app.post("/api/missions/run")
    async def run_mission(request: Request) -> Dict[str, Any]:
        body = await _json_body(request, required=True)
        tmpl = str(body.get("template") or "desk_task").strip()
        ep, client = _resolve_model(str(body.get("model_name") or actor_ep.name).strip())
        max_attempts = max(1, min(5, int(body.get("max_attempts", 3))))
        max_rounds = max(3, min(20, int(body.get("max_rounds_per_attempt", 12))))

        isolated = bool(body.get("isolated_subdir", True))
        if isolated:
            m_ws = ws_root / ".lfm3_missions" / f"run_{int(time.time())}_{secrets.token_hex(3)}"
            m_ws.mkdir(parents=True, exist_ok=True)
        else:
            m_ws = ws_root
        m_sandbox = WorkspaceSandbox(m_ws)

        if tmpl == "desk_task":
            spec = setup_desk_task_workspace(m_ws, max_attempts=max_attempts, max_rounds_per_attempt=max_rounds)
        elif tmpl == "screen_qa":
            spec = make_screen_qa_spec()
        elif tmpl == "doc_extract":
            spec = make_doc_extract_spec()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mission template '{tmpl}'")

        cancel_ev = asyncio.Event()
        started_fut: asyncio.Future[Tuple[str, str]] = asyncio.get_running_loop().create_future()
        run_id_holder: Dict[str, str] = {}
        loop = asyncio.get_running_loop()

        def _push_mission_ev(run_id: str, ev_type: str, data: Dict[str, Any]) -> None:
            mission_logs.setdefault(run_id, []).append((ev_type, data))
            cond = mission_notifiers.get(run_id)
            if cond is not None:
                async def _wake() -> None:
                    async with cond:
                        cond.notify_all()
                loop.create_task(_wake())

        def _on_started(run_id: str, spec_sha: str) -> None:
            run_id_holder["id"] = run_id
            active_mission_cancels[run_id] = cancel_ev
            mission_notifiers[run_id] = asyncio.Condition()
            _push_mission_ev(run_id, "mission_started", {"run_id": run_id, "template": tmpl, "spec_sha256": spec_sha, "model": ep.name})
            if not started_fut.done():
                started_fut.set_result((run_id, spec_sha))

        def _on_round(rt, _raw_text: str) -> None:
            rid = run_id_holder.get("id")
            if rid:
                _push_mission_ev(rid, "round", asdict(rt))

        runner = MissionRunner(
            client,
            m_sandbox,
            state_store,
            journal_dir=state_store.db_path.parent / "mission_journals",
            capabilities=caps,
            limits=config.limits,
            family=ep.family,
            on_round_complete=_on_round,
        )

        async def _bg_runner() -> None:
            try:
                res = await runner.run(ws_row["id"], spec, cancel_event=cancel_ev, on_started=_on_started)
                last_rep = res.reports[-1] if res.reports else None
                last_job = res.job_results[-1] if res.job_results else None
                payload = {
                    "run_id": res.run_id,
                    "status": res.status,
                    "attempts": res.attempts,
                    "spec_sha256": res.spec_sha256,
                    "checks": [asdict(c) for c in (last_rep.results if last_rep else [])],
                    "deliverables": [asdict(d) for d in (last_job.deliverables if last_job else [])],
                    "numeric_grounding": asdict(last_job.numeric_grounding) if last_job else None,
                    "rounds": [asdict(r) for r in (last_job.rounds if last_job else [])],
                }
                _push_mission_ev(res.run_id, "mission_done", payload)
            except Exception as exc:
                rid = run_id_holder.get("id")
                if not started_fut.done():
                    started_fut.set_exception(exc)
                if rid:
                    state_store.update_mission_run(rid, status="failed", result={"error": str(exc)})
                    _push_mission_ev(rid, "error", {"run_id": rid, "message": str(exc)})
            finally:
                rid = run_id_holder.get("id")
                if rid:
                    active_mission_cancels.pop(rid, None)

        task = asyncio.create_task(_bg_runner())
        run_id, spec_sha = await started_fut
        mission_tasks[run_id] = task
        if body.get("wait"):
            await task
        return {
            "run_id": run_id,
            "spec_sha256": spec_sha,
            "template": tmpl,
            "model": ep.name,
            "mission": state_store.get_mission_run(run_id),
        }

    @app.get("/api/missions/{run_id}/stream")
    async def stream_mission(run_id: str):
        row = state_store.get_mission_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Mission run not found")

        async def _gen():
            cursor = 0
            cond = mission_notifiers.get(run_id)
            while True:
                hist = mission_logs.get(run_id, [])
                while cursor < len(hist):
                    ev_t, data = hist[cursor]
                    cursor += 1
                    yield _sse(ev_t, data)
                    if ev_t in ("mission_done", "error"):
                        return
                if cond is None or run_id not in active_mission_cancels:
                    latest = state_store.get_mission_run(run_id)
                    if latest and latest["status"] != "running":
                        yield _sse("mission_done", {"run_id": run_id, "status": latest["status"], "result": json.loads(latest["result_json"] or "{}")})
                        return
                if cond is None:
                    return
                async with cond:
                    if cursor < len(mission_logs.get(run_id, [])):
                        continue
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=30.0)
                    except asyncio.TimeoutError:
                        yield _sse("ping", {"ts": time.time()})

        return StreamingResponse(_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/missions/{run_id}/stop")
    async def stop_mission(run_id: str) -> Dict[str, Any]:
        ev = active_mission_cancels.get(run_id)
        if ev is not None:
            ev.set()
            return {"ok": True, "stopping": True}
        return {"ok": True, "stopping": False}

    @app.get("/api/jobs")
    async def list_jobs(workspace_id: Optional[str] = None) -> Dict[str, Any]:
        return {"jobs": state_store.list_jobs(workspace_id)}

    @app.get("/api/cache/stats")
    async def cache_stats() -> Dict[str, Any]:
        per_model: Dict[str, Any] = {}
        for name, cli in clients.items():
            try:
                per_model[name] = {"reachable": True, **(await asyncio.wait_for(cli.cache_stats(timeout_s=2.0), timeout=2.5))}
            except Exception as exc:
                per_model[name] = {"reachable": False, "error": str(exc)}
        if d1_separate and d1_client is not None:
            try:
                per_model["d1"] = {"reachable": True, **(await asyncio.wait_for(d1_client.cache_stats(timeout_s=2.0), timeout=2.5))}
            except Exception as exc:
                per_model["d1"] = {"reachable": False, "error": str(exc)}
        return {"models": per_model}

    @app.post("/api/cache/clear")
    async def cache_clear(request: Request) -> Dict[str, Any]:
        body = await _json_body(request, required=False)
        target = str(body.get("model") or "all").strip()
        include_pinned = bool(body.get("include_pinned", False))
        results: Dict[str, Any] = {}
        all_targets: List[Tuple[str, DaemonClient]] = list(clients.items())
        if d1_separate and d1_client is not None:
            all_targets.append(("d1", d1_client))

        if target == "all":
            targets = all_targets
        else:
            targets = [(n, c) for n, c in all_targets if n == target]
            if not targets:
                raise HTTPException(status_code=404, detail=f"Unknown model '{target}'")
        for name, cli in targets:
            try:
                results[name] = await asyncio.wait_for(cli.cache_clear(include_pinned=include_pinned, timeout_s=5.0), timeout=5.5)
            except Exception as exc:
                results[name] = {"ok": False, "error": str(exc)}
        return {"include_pinned": include_pinned, "results": results}

    @app.get("/api/system/health")
    async def system_health() -> Dict[str, Any]:
        return _collect_system_health(len(active_chat_cancels), len(active_mission_cancels), audio.capabilities())

    @app.post("/api/images/upload")
    async def upload_image(request: Request, filename: str = "image.png", max_dim: int = 512) -> Dict[str, Any]:
        if v_client is None or not caps.vision:
            raise HTTPException(status_code=503, detail="Vision sidecar is not configured (--vision-url)")
        ctype = (request.headers.get("content-type") or "").lower()
        chosen_name = filename
        if "multipart/form-data" in ctype:
            try:
                form = await request.form()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Multipart upload unavailable: {exc}") from exc
            up = form.get("file") or form.get("image")
            if up is None:
                raise HTTPException(status_code=400, detail="Missing 'file' form field")
            chosen_name = getattr(up, "filename", None) or filename
            raw_bytes = await up.read()
        else:
            raw_bytes = await request.body()

        try:
            pre_meta = build_image_metadata(raw_bytes, chosen_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            embed_bytes, n_tok, vinfo = await v_client.encode(raw_bytes, filename=pre_meta.filename, max_dim=max_dim)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        image_tmp_dir.mkdir(parents=True, exist_ok=True)
        image_id = f"img_{secrets.token_hex(6)}"
        raw_path = (image_tmp_dir / f"{image_id}_{pre_meta.filename}").resolve()
        embed_path = (image_tmp_dir / f"{image_id}.f32").resolve()
        raw_path.write_bytes(raw_bytes)
        embed_path.write_bytes(embed_bytes)

        meta = ImageMetadata(
            filename=pre_meta.filename,
            bytes=pre_meta.bytes,
            sha256=pre_meta.sha256,
            format=pre_meta.format,
            width=pre_meta.width,
            height=pre_meta.height,
            num_tokens=n_tok,
            saved_path=str(raw_path),
        )
        uploaded_images[image_id] = (meta, embed_path)
        return {"ok": True, "image_id": image_id, "metadata": meta.to_dict(), "vision_info": vinfo}

    @app.post("/api/audio/upload")
    async def upload_audio(request: Request, filename: str = "clip.wav") -> Dict[str, Any]:
        ctype = (request.headers.get("content-type") or "").lower()
        raw_bytes: bytes = b""
        chosen_name = filename
        if "multipart/form-data" in ctype:
            try:
                form = await request.form()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Multipart upload unavailable: {exc}") from exc
            up = form.get("file") or form.get("audio")
            if up is None:
                raise HTTPException(status_code=400, detail="Missing 'file' form field")
            chosen_name = getattr(up, "filename", None) or filename
            raw_bytes = await up.read()
        else:
            raw_bytes = await request.body()

        try:
            meta = await audio.inspect_and_normalize(raw_bytes, chosen_name, audio_tmp_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        transcript, engine = await audio.transcribe(meta)
        return {
            "ok": True,
            "metadata": meta.to_dict(),
            "transcript": transcript,
            "asr_engine": engine,
            "asr_available": engine is not None,
        }

    @app.post("/api/tts")
    async def tts_speak(request: Request) -> Response:
        body = await _json_body(request, required=True)
        text = str(body.get("text") or "").strip()
        speed = float(body.get("speed", 1.0))
        if not text:
            raise HTTPException(status_code=400, detail="TTS 'text' must not be empty")
        try:
            wav_bytes, eng, lat_ms = await audio.synthesize_tts(text, speed=speed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-TTS-Engine": eng, "X-TTS-Latency-Ms": f"{lat_ms:.2f}"},
        )

    return app


def _ensure_self_signed_tls(tls_dir: Path) -> Tuple[Path, Path]:
    cert = tls_dir / "cert.pem"
    key = tls_dir / "key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    openssl = shutil.which("openssl")
    if not openssl:
        raise RuntimeError("openssl binary not found; cannot generate self-signed TLS certificate")
    tls_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "365",
            "-subj",
            "/CN=roma.local",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.chmod(key, 0o600)
    return cert, key


def build_config_from_args(ns: argparse.Namespace) -> Tuple[HarnessConfig, bool, Optional[str], Optional[Path], Optional[Path]]:
    ws = validate_workspace_root(Path(ns.workspace_root))
    state_dir = Path(ns.state_dir).expanduser().resolve() if ns.state_dir else (Path.home() / ".local" / "state" / "romanaccio_harness")
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "harness.db"

    primary = BackendEndpoint(
        name="lfm3-1.7b",
        host=ns.lfm3_host,
        port=ns.lfm3_port,
        family="lfm3",
        supports_thinking=True,
        supports_vision=False,
        tokenizer_path=Path(ns.lfm3_tokenizer) if ns.lfm3_tokenizer else None,
    )
    secondary = None
    if ns.moe_port > 0:
        secondary = BackendEndpoint(
            name="lfm2.5-8b-a1b",
            host=ns.moe_host,
            port=ns.moe_port,
            family="lfm2",
            supports_thinking=False,
            tokenizer_path=Path(ns.moe_tokenizer) if ns.moe_tokenizer else None,
        )
    d1_ep = None
    if ns.d1_port > 0:
        d1_tok = ns.d1_tokenizer or ns.lfm3_tokenizer
        d1_ep = BackendEndpoint(
            name="d1-2b",
            host=ns.d1_host,
            port=ns.d1_port,
            family="lfm3",
            supports_thinking=True,
            tokenizer_path=Path(d1_tok) if d1_tok else None,
        )

    cal_file: Optional[Path] = Path(ns.calibration_json) if ns.calibration_json else None
    if cal_file is None:
        default_d1_cal = Path("/home/chri/models/d1-2B_TP2/calibration.json")
        if default_d1_cal.is_file():
            cal_file = default_d1_cal

    tok: Optional[str] = ns.auth_token
    if ns.no_auth:
        tok = None
    elif not tok and not _is_loopback(ns.bind):
        tok = secrets.token_urlsafe(24)

    ssl_cert: Optional[Path] = Path(ns.ssl_certfile) if ns.ssl_certfile else None
    ssl_key: Optional[Path] = Path(ns.ssl_keyfile) if ns.ssl_keyfile else None
    if ns.self_signed_tls and (ssl_cert is None or ssl_key is None):
        ssl_cert, ssl_key = _ensure_self_signed_tls(state_dir / "tls")

    audio_svc = AudioService(
        piper_bin=ns.piper_bin or None,
        piper_model=ns.piper_model or None,
    )
    audio_caps = audio_svc.capabilities()
    caps = Capabilities(
        vision=bool(ns.vision_url),
        voice_asr=bool(audio_caps["voice_asr"]),
        voice_tts=bool(audio_caps["voice_tts"]),
        tts_engine=audio_caps["tts_engine"],
    )
    cfg = HarnessConfig(
        workspace_root=ws,
        db_path=db_path,
        actor=primary,
        secondary_actor=secondary,
        d1=d1_ep,
        vision_url=ns.vision_url or None,
        capabilities=caps,
        limits=SafetyLimits(),
        auth_token=tok,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
    )
    return cfg, bool(ns.no_auth), ns.piper_bin or None, (Path(ns.piper_model) if ns.piper_model else None), cal_file


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Romanaccio Zero-Build LAN Harness Web Server (Milestone 19E)")
    parser.add_argument("--workspace-root", default="/home/chri/workspace", help="Confined root directory for tools/edits")
    parser.add_argument("--state-dir", default="", help="SQLite/TLS/audio state directory")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind host (127.0.0.1 or 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP/HTTPS listen port")
    parser.add_argument("--lfm3-host", default="127.0.0.1")
    parser.add_argument("--lfm3-port", type=int, default=8088)
    parser.add_argument("--lfm3-tokenizer", default="")
    parser.add_argument("--moe-host", default="127.0.0.1")
    parser.add_argument("--moe-port", type=int, default=8089, help="Secondary LFM2.5 MoE port (0 to disable)")
    parser.add_argument("--moe-tokenizer", default="")
    parser.add_argument("--d1-host", default="127.0.0.1")
    parser.add_argument("--d1-port", type=int, default=0, help="Dedicated d1 port (0 = reuse lfm3 actor)")
    parser.add_argument("--d1-tokenizer", default="", help="Optional tokenizer.json for dedicated d1-2B package")
    parser.add_argument("--vision-url", default="")
    parser.add_argument("--calibration-json", default="")
    parser.add_argument("--piper-bin", default="")
    parser.add_argument("--piper-model", default="")
    parser.add_argument("--auth-token", default="", help="Optional fixed Bearer/cookie auth token")
    parser.add_argument("--no-auth", action="store_true", help="Explicitly disable LAN bearer auth")
    parser.add_argument("--ssl-certfile", default="")
    parser.add_argument("--ssl-keyfile", default="")
    parser.add_argument("--self-signed-tls", action="store_true", help="Generate/use self-signed TLS cert for mic Secure Context")
    ns = parser.parse_args(argv)

    cfg, allow_no_auth, piper_bin, piper_model, cal_path = build_config_from_args(ns)
    audio_svc = AudioService(piper_bin=piper_bin, piper_model=piper_model)
    app = create_app(cfg, audio_service=audio_svc, calibration_path=cal_path, allow_no_auth=allow_no_auth)

    scheme = "https" if (cfg.ssl_certfile and cfg.ssl_keyfile) else "http"
    print(f"[romanaccio-harness] {scheme}://{ns.bind}:{ns.port} | workspace={cfg.workspace_root}")
    print(f"[romanaccio-harness] actors: {cfg.actor.name}={cfg.actor.host}:{cfg.actor.port}"
          + (f", {cfg.secondary_actor.name}={cfg.secondary_actor.host}:{cfg.secondary_actor.port}" if cfg.secondary_actor else "")
          + (f", {cfg.d1.name}={cfg.d1.host}:{cfg.d1.port}" if cfg.d1 else ""))
    print(f"[romanaccio-harness] vision={cfg.vision_url or 'disabled'} | audio={audio_svc.capabilities()}")
    if cfg.auth_token:
        print(f"[romanaccio-harness] auth_token={cfg.auth_token}")
    else:
        print("[romanaccio-harness] auth=disabled")
    sys.stdout.flush()

    import uvicorn

    uv_kw: Dict[str, Any] = {"host": ns.bind, "port": ns.port, "log_level": "info"}
    if cfg.ssl_certfile and cfg.ssl_keyfile:
        uv_kw["ssl_certfile"] = str(cfg.ssl_certfile)
        uv_kw["ssl_keyfile"] = str(cfg.ssl_keyfile)
    uvicorn.run(app, **uv_kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
