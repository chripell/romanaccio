"""Workspace path confinement, symlink rejection, and atomic TOCTOU-checked writes."""

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import MAX_WRITE_BYTES, validate_workspace_root

_RESERVED_DEVICE_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


class PathSandboxError(ValueError):
    """Raised when a relative workspace path violates sandbox confinement rules."""

    def __init__(self, reason: str, rel_path: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.rel_path = rel_path


@dataclass(frozen=True)
class WriteReceipt:
    """Verified outcome of an atomic workspace file write."""

    rel_path: str
    path: Path
    bytes_written: int
    sha256: str


class WorkspaceSandbox:
    """Component-by-component confined filesystem accessor rooted at a single directory."""

    def __init__(self, root: Path):
        self.root: Path = validate_workspace_root(Path(root))

    def _validate_components(self, rel_path: str, *, allow_empty_or_dot: bool) -> List[str]:
        if not isinstance(rel_path, str):
            raise PathSandboxError("invalid_type", str(rel_path), "Path must be a string")
        if "\x00" in rel_path:
            raise PathSandboxError("nul_byte", rel_path, "Path contains NUL byte")

        stripped = rel_path.strip()
        if not stripped:
            if allow_empty_or_dot:
                return []
            raise PathSandboxError("empty_path", rel_path, "Empty path is not allowed")

        if stripped.startswith("/") or stripped.startswith("\\"):
            raise PathSandboxError("absolute_path", rel_path, f"Absolute paths are forbidden: {rel_path}")
        if len(stripped) >= 2 and stripped[0].isalpha() and stripped[1] == ":":
            raise PathSandboxError("drive_path", rel_path, f"Drive-prefixed paths are forbidden: {rel_path}")
        if "\\" in stripped:
            raise PathSandboxError("backslash", rel_path, f"Backslashes are not allowed in paths: {rel_path}")

        parts: List[str] = []
        for seg in stripped.split("/"):
            if seg == "" or seg == ".":
                continue
            if seg == "..":
                raise PathSandboxError("traversal", rel_path, f"Parent traversal '..' is forbidden: {rel_path}")
            stem = seg.split(".", 1)[0].upper()
            if stem in _RESERVED_DEVICE_STEMS:
                raise PathSandboxError("reserved_name", rel_path, f"Reserved device name '{seg}' is forbidden")
            parts.append(seg)

        if not parts and not allow_empty_or_dot:
            raise PathSandboxError("root_target", rel_path, "Cannot target workspace root directory for file operation")

        return parts

    def _walk_no_symlinks(self, parts: List[str], rel_path: str) -> Path:
        cur = self.root
        for seg in parts:
            cur = cur / seg
            if cur.is_symlink():
                raise PathSandboxError("symlink", rel_path, f"Symlink component rejected: {cur.name}")

        # Verify resolved deepest existing ancestor still lies strictly inside root.
        probe = cur
        while not probe.exists() and probe != self.root:
            probe = probe.parent
        resolved_probe = probe.resolve(strict=True)
        try:
            resolved_probe.relative_to(self.root)
        except ValueError as exc:
            raise PathSandboxError("escape", rel_path, f"Resolved path escapes workspace root: {rel_path}") from exc

        return cur

    def relativize(self, path: Path) -> str:
        """Convert an in-workspace Path back to normalized POSIX relative notation."""
        return path.relative_to(self.root).as_posix()

    def resolve_dir(self, rel_path: str = ".", *, must_exist: bool = True) -> Path:
        """Resolve a directory inside the workspace ('.' or '' refers to root)."""
        parts = self._validate_components(rel_path, allow_empty_or_dot=True)
        target = self._walk_no_symlinks(parts, rel_path)
        if must_exist:
            if not target.exists():
                raise PathSandboxError("not_found", rel_path, f"Directory does not exist: {rel_path}")
            if not target.is_dir():
                raise PathSandboxError("not_a_dir", rel_path, f"Path is not a directory: {rel_path}")
        return target

    def resolve_read(self, rel_path: str) -> Path:
        """Resolve an existing regular file inside the workspace for reading."""
        parts = self._validate_components(rel_path, allow_empty_or_dot=False)
        target = self._walk_no_symlinks(parts, rel_path)
        if not target.exists():
            raise PathSandboxError("not_found", rel_path, f"File not found: {rel_path}")
        if not target.is_file():
            raise PathSandboxError("not_a_file", rel_path, f"Target is not a regular file: {rel_path}")
        return target

    def resolve_write(self, rel_path: str) -> Path:
        """Resolve a file target inside the workspace for writing (need not exist yet)."""
        parts = self._validate_components(rel_path, allow_empty_or_dot=False)
        target = self._walk_no_symlinks(parts, rel_path)
        if target.exists() and target.is_dir():
            raise PathSandboxError("is_directory", rel_path, f"Cannot overwrite directory with file: {rel_path}")
        return target

    @staticmethod
    def sha256_file(path: Path) -> str:
        """Return lowercase hex SHA-256 of an existing file."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(65_536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def atomic_write_text(
        self,
        rel_path: str,
        content: str,
        *,
        expected_sha256: Optional[str] = None,
        max_bytes: int = MAX_WRITE_BYTES,
    ) -> WriteReceipt:
        """Write UTF-8 text via temp file + fsync + os.replace with optional TOCTOU SHA-256 check."""
        raw_bytes = content.encode("utf-8")
        if len(raw_bytes) > max_bytes:
            raise PathSandboxError(
                "size_limit",
                rel_path,
                f"Content size {len(raw_bytes)} B exceeds max_write_bytes ({max_bytes} B)",
            )

        target = self.resolve_write(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-validate full hierarchy after mkdir in case an external race introduced a symlink.
        target = self.resolve_write(rel_path)

        if expected_sha256 is not None:
            if expected_sha256 == "":
                if target.exists():
                    raise PathSandboxError(
                        "toctou_mismatch",
                        rel_path,
                        f"TOCTOU conflict: {rel_path} did not exist when staged but exists now",
                    )
            else:
                if not target.exists():
                    raise PathSandboxError(
                        "toctou_mismatch",
                        rel_path,
                        f"TOCTOU conflict: {rel_path} was removed since staging",
                    )
                current_sha = self.sha256_file(target)
                if current_sha != expected_sha256:
                    raise PathSandboxError(
                        "toctou_mismatch",
                        rel_path,
                        f"TOCTOU conflict on {rel_path}: expected {expected_sha256[:12]}, got {current_sha[:12]}",
                    )

        tmp_name = f".lfm3_tmp_{os.getpid()}_{uuid.uuid4().hex[:10]}"
        tmp_path = target.parent / tmp_name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        fd = -1
        try:
            fd = os.open(tmp_path, flags, 0o644)
            view = memoryview(raw_bytes)
            written = 0
            while written < len(raw_bytes):
                n = os.write(fd, view[written:])
                if n <= 0:
                    raise OSError("Short write to temp file")
                written += n
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(tmp_path, target)
        except Exception:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                if tmp_path.exists() or tmp_path.is_symlink():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

        digest = hashlib.sha256(raw_bytes).hexdigest()
        return WriteReceipt(
            rel_path=self.relativize(target),
            path=target,
            bytes_written=len(raw_bytes),
            sha256=digest,
        )
