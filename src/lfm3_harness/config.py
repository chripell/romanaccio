"""Configuration and safety constants for the lfm3_harness package."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

ROUTE_MIN_CONFIDENCE: float = 0.55
EVIDENCE_GATE_FLOOR: float = 0.65
WRITE_GATE_FLOOR: float = 0.75
DECISION_TIMEOUT_S: float = 12.0

MAX_TOOL_ROUNDS: int = 12
MAX_PLAN_STEPS: int = 8
MAX_JOB_TURNS: int = 64
APPROVAL_TTL_S: int = 600

MAX_READ_BYTES: int = 1_048_576
MAX_WRITE_BYTES: int = 1_048_576
MAX_OUTPUT_CHARS: int = 32_000
MAX_LIST_ENTRIES: int = 500
MAX_GATE_EVIDENCE_CHARS: int = 7_000
MAX_SEARCH_RESULTS: int = 40

ModelFamily = Literal["lfm3", "lfm2"]


def _is_same_or_child(candidate: Path, forbidden: Path) -> bool:
    try:
        candidate.relative_to(forbidden)
        return True
    except ValueError:
        return False


def validate_workspace_root(path: Path) -> Path:
    """Resolve and verify a workspace root directory against forbidden system paths."""
    if path is None or str(path).strip() == "":
        raise ValueError("Workspace root path cannot be empty")
    raw_str = str(path)
    if "\x00" in raw_str:
        raise ValueError("Workspace root path contains NUL byte")

    raw_path = Path(raw_str).expanduser()
    if raw_path.is_symlink():
        raise ValueError(f"Workspace root cannot itself be a symlink: {raw_path}")
    resolved = raw_path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Workspace root is not a directory: {resolved}")

    if resolved == Path("/"):
        raise ValueError("Refusing filesystem root '/' as workspace")

    forbidden_prefixes = [
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev"),
        Path("/root"),
        Path.home().resolve() / ".ssh",
        Path("/root/.ssh"),
    ]
    for prefix in forbidden_prefixes:
        if _is_same_or_child(resolved, prefix):
            raise ValueError(f"Refusing sensitive system directory as workspace root: {resolved}")

    return resolved


@dataclass(frozen=True)
class SafetyLimits:
    """Execution bounds, confidence floors, and I/O caps."""

    route_min_confidence: float = ROUTE_MIN_CONFIDENCE
    evidence_gate_floor: float = EVIDENCE_GATE_FLOOR
    write_gate_floor: float = WRITE_GATE_FLOOR
    decision_timeout_s: float = DECISION_TIMEOUT_S
    max_tool_rounds: int = MAX_TOOL_ROUNDS
    max_plan_steps: int = MAX_PLAN_STEPS
    max_job_turns: int = MAX_JOB_TURNS
    approval_ttl_s: int = APPROVAL_TTL_S
    max_read_bytes: int = MAX_READ_BYTES
    max_write_bytes: int = MAX_WRITE_BYTES
    max_output_chars: int = MAX_OUTPUT_CHARS
    max_list_entries: int = MAX_LIST_ENTRIES
    max_gate_evidence_chars: int = MAX_GATE_EVIDENCE_CHARS
    max_search_results: int = MAX_SEARCH_RESULTS


@dataclass(frozen=True)
class BackendEndpoint:
    """TCP endpoint metadata for a resident NPU inference daemon."""

    name: str
    host: str = "127.0.0.1"
    port: int = 8088
    family: ModelFamily = "lfm3"
    supports_thinking: bool = True
    supports_vision: bool = False
    default_repeat_penalty: float = 1.1
    tokenizer_path: Optional[Path] = None
    supports_audio: bool = False


@dataclass(frozen=True)
class Capabilities:
    """Explicitly enabled multimodal capabilities (strictly gated in tool schemas)."""

    vision: bool = False
    voice_asr: bool = False
    voice_tts: bool = False
    tts_engine: Optional[str] = None


@dataclass(frozen=True)
class HarnessConfig:
    """Top-level configuration container for the local harness."""

    workspace_root: Path
    db_path: Path
    actor: BackendEndpoint = field(default_factory=lambda: BackendEndpoint(name="lfm3-1.7b", port=8088, family="lfm3"))
    secondary_actor: Optional[BackendEndpoint] = None
    d1: Optional[BackendEndpoint] = None
    vision_url: Optional[str] = None
    audio_daemon: Optional[BackendEndpoint] = None
    audio_url: Optional[str] = None
    native_tts_default: bool = False
    capabilities: Capabilities = field(default_factory=Capabilities)
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    auth_token: Optional[str] = None
    ssl_certfile: Optional[Path] = None
    ssl_keyfile: Optional[Path] = None
