"""SQLite WAL-mode persistence for workspaces, sessions, actions, jobs, and missions."""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    title TEXT NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'idle',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    thinking TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    tool_call_id TEXT NOT NULL DEFAULT '',
    telemetry_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'completed',
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_actions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    base_sha256 TEXT NOT NULL,
    new_content TEXT NOT NULL,
    diff_text TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    resolved_at REAL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    session_id TEXT,
    title TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'plan',
    status TEXT NOT NULL DEFAULT 'running',
    todo_json TEXT NOT NULL DEFAULT '[]',
    deliverable_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS acceptance_specs (
    id TEXT PRIMARY KEY,
    spec_sha256 TEXT NOT NULL UNIQUE,
    canonical_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mission_runs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    spec_id TEXT NOT NULL,
    spec_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    attempt INTEGER NOT NULL DEFAULT 1,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (spec_id) REFERENCES acceptance_specs(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_actions_session ON chat_actions(session_id, status);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id);
"""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class HarnessStore:
    """Thread-safe SQLite WAL persistence with automatic startup crash reconciliation."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.execute("PRAGMA busy_timeout=5000;")
        self.conn.executescript(_SCHEMA_SQL)
        self.reconcile_interrupted_runs()

    def close(self) -> None:
        self.conn.close()

    def journal_mode(self) -> str:
        row = self.conn.execute("PRAGMA journal_mode;").fetchone()
        return str(row[0]).lower() if row else ""

    def log_event(self, entity_type: str, entity_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO events (entity_type, entity_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (entity_type, entity_id, event_type, json.dumps(payload or {}), now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def reconcile_interrupted_runs(self, *, now: Optional[float] = None) -> Dict[str, int]:
        """Mark interrupted streaming/running entities cancelled and expire stale pending approvals."""
        ts = now if now is not None else time.time()
        counts = {"sessions": 0, "messages": 0, "jobs": 0, "missions": 0, "expired_actions": 0}

        with self.conn:
            for table, active_states, new_state, key, etype in [
                ("chat_sessions", ("streaming", "running"), "cancelled", "sessions", "session"),
                ("chat_messages", ("streaming", "running"), "interrupted", "messages", "message"),
                ("jobs", ("running", "streaming"), "cancelled", "jobs", "job"),
                ("mission_runs", ("running", "streaming"), "interrupted", "missions", "mission"),
            ]:
                placeholders = ",".join("?" for _ in active_states)
                rows = self.conn.execute(
                    f"SELECT id, status FROM {table} WHERE status IN ({placeholders})",
                    active_states,
                ).fetchall()
                if rows:
                    self.conn.execute(
                        f"UPDATE {table} SET status = ? WHERE status IN ({placeholders})",
                        (new_state, *active_states),
                    )
                    if table in ("chat_sessions", "jobs", "mission_runs"):
                        self.conn.execute(
                            f"UPDATE {table} SET updated_at = ? WHERE id IN ({','.join('?' for _ in rows)})",
                            (ts, *(r["id"] for r in rows)),
                        )
                    for r in rows:
                        self.conn.execute(
                            "INSERT INTO events (entity_type, entity_id, event_type, payload_json, created_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (
                                etype,
                                r["id"],
                                "startup_reconciled",
                                json.dumps({"from": r["status"], "to": new_state}),
                                ts,
                            ),
                        )
                    counts[key] = len(rows)

            expired = self.conn.execute(
                "SELECT id FROM chat_actions WHERE status = 'pending' AND expires_at <= ?",
                (ts,),
            ).fetchall()
            if expired:
                self.conn.execute(
                    "UPDATE chat_actions SET status = 'expired', resolved_at = ? WHERE status = 'pending' AND expires_at <= ?",
                    (ts, ts),
                )
                for r in expired:
                    self.conn.execute(
                        "INSERT INTO events (entity_type, entity_id, event_type, payload_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ("action", r["id"], "startup_expired", json.dumps({"to": "expired"}), ts),
                    )
                counts["expired_actions"] = len(expired)

        return counts

    # --- Workspaces & Sessions ---
    def register_workspace(self, name: str, root_path: Path | str) -> Dict[str, Any]:
        resolved = str(Path(root_path).resolve())
        existing = self.conn.execute("SELECT * FROM workspaces WHERE root_path = ?", (resolved,)).fetchone()
        if existing:
            return dict(existing)
        ws_id = _new_id("ws")
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO workspaces (id, name, root_path, created_at) VALUES (?, ?, ?, ?)",
                (ws_id, name, resolved, now),
            )
        return dict(self.conn.execute("SELECT * FROM workspaces WHERE id = ?", (ws_id,)).fetchone())

    def create_session(self, workspace_id: str, title: str, model_name: str = "lfm3-1.7b", status: str = "idle") -> Dict[str, Any]:
        sid = _new_id("sess")
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_sessions (id, workspace_id, title, model_name, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, workspace_id, title, model_name, status, now, now),
            )
        return dict(self.conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (sid,)).fetchone())

    def update_session_status(self, session_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE chat_sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), session_id),
            )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        thinking: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: str = "",
        telemetry: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ) -> Dict[str, Any]:
        mid = _new_id("msg")
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_messages (id, session_id, role, content, thinking, tool_calls_json, "
                "tool_call_id, telemetry_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mid,
                    session_id,
                    role,
                    content,
                    thinking,
                    json.dumps(tool_calls or []),
                    tool_call_id,
                    json.dumps(telemetry or {}),
                    status,
                    now,
                ),
            )
            self.conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        return dict(self.conn.execute("SELECT * FROM chat_messages WHERE id = ?", (mid,)).fetchone())

    def update_message(self, message_id: str, *, content: str, thinking: str = "", status: str = "completed", telemetry: Optional[Dict[str, Any]] = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE chat_messages SET content = ?, thinking = ?, status = ?, telemetry_json = ? WHERE id = ?",
                (content, thinking, status, json.dumps(telemetry or {}), message_id),
            )

    def list_messages(self, session_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Actions (Approvals) ---
    def create_action(
        self,
        session_id: str,
        message_id: str,
        *,
        tool_name: str,
        rel_path: str,
        base_sha256: str,
        new_content: str,
        diff_text: str,
        reason: str = "",
        ttl_s: int = 600,
    ) -> Dict[str, Any]:
        aid = _new_id("act")
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_actions (id, session_id, message_id, tool_name, rel_path, base_sha256, "
                "new_content, diff_text, reason, status, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (aid, session_id, message_id, tool_name, rel_path, base_sha256, new_content, diff_text, reason, now, now + ttl_s),
            )
        return dict(self.conn.execute("SELECT * FROM chat_actions WHERE id = ?", (aid,)).fetchone())

    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM chat_actions WHERE id = ?", (action_id,)).fetchone()
        return dict(row) if row else None

    def list_actions(self, session_id: str, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status is not None:
            rows = self.conn.execute(
                "SELECT * FROM chat_actions WHERE session_id = ? AND status = ? ORDER BY created_at ASC, rowid ASC",
                (session_id, status),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM chat_actions WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def transition_action(self, action_id: str, *, expected_status: str, new_status: str, error: str = "") -> bool:
        now = time.time()
        with self.conn:
            cur = self.conn.execute(
                "UPDATE chat_actions SET status = ?, error = ?, resolved_at = ? WHERE id = ? AND status = ?",
                (new_status, error, now, action_id, expected_status),
            )
            return cur.rowcount == 1

    # --- Jobs & Missions ---
    def create_job(self, workspace_id: str, title: str, *, session_id: Optional[str] = None, phase: str = "plan", status: str = "running") -> Dict[str, Any]:
        jid = _new_id("job")
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO jobs (id, workspace_id, session_id, title, phase, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (jid, workspace_id, session_id, title, phase, status, now, now),
            )
        return dict(self.conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone())

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def update_job(
        self,
        job_id: str,
        *,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        todo: Optional[List[Dict[str, Any]]] = None,
        deliverable: Optional[Dict[str, Any]] = None,
    ) -> None:
        sets: List[str] = ["updated_at = ?"]
        vals: List[Any] = [time.time()]
        if phase is not None:
            sets.append("phase = ?")
            vals.append(phase)
        if status is not None:
            sets.append("status = ?")
            vals.append(status)
        if todo is not None:
            sets.append("todo_json = ?")
            vals.append(json.dumps(todo))
        if deliverable is not None:
            sets.append("deliverable_json = ?")
            vals.append(json.dumps(deliverable))
        vals.append(job_id)
        with self.conn:
            self.conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", tuple(vals))

    def get_acceptance_spec(self, spec_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM acceptance_specs WHERE id = ?", (spec_id,)).fetchone()
        return dict(row) if row else None

    def get_mission_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM mission_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def update_mission_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        attempt: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        sets: List[str] = ["updated_at = ?"]
        vals: List[Any] = [time.time()]
        if status is not None:
            sets.append("status = ?")
            vals.append(status)
        if attempt is not None:
            sets.append("attempt = ?")
            vals.append(int(attempt))
        if result is not None:
            sets.append("result_json = ?")
            vals.append(json.dumps(result))
        vals.append(run_id)
        with self.conn:
            self.conn.execute(f"UPDATE mission_runs SET {', '.join(sets)} WHERE id = ?", tuple(vals))

    def create_mission_run(
        self,
        workspace_id: str,
        canonical_spec_json: str,
        spec_sha256: str,
        *,
        status: str = "running",
        max_attempts: int = 3,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        now = time.time()
        with self.conn:
            spec_row = self.conn.execute("SELECT * FROM acceptance_specs WHERE spec_sha256 = ?", (spec_sha256,)).fetchone()
            if spec_row:
                spec_id = spec_row["id"]
            else:
                spec_id = _new_id("spec")
                self.conn.execute(
                    "INSERT INTO acceptance_specs (id, spec_sha256, canonical_json, created_at) VALUES (?, ?, ?, ?)",
                    (spec_id, spec_sha256, canonical_spec_json, now),
                )
                spec_row = self.conn.execute("SELECT * FROM acceptance_specs WHERE id = ?", (spec_id,)).fetchone()
            m_id = _new_id("mis")
            self.conn.execute(
                "INSERT INTO mission_runs (id, workspace_id, spec_id, spec_sha256, status, attempt, max_attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (m_id, workspace_id, spec_id, spec_sha256, status, max_attempts, now, now),
            )
            run_row = self.conn.execute("SELECT * FROM mission_runs WHERE id = ?", (m_id,)).fetchone()
        return dict(spec_row), dict(run_row)

    def list_events(self, entity_type: Optional[str] = None, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if entity_type and entity_id:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE entity_type = ? AND entity_id = ? ORDER BY id ASC",
                (entity_type, entity_id),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM events ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]
