"""Path confinement helpers for Workbench / trusted-deployment surfaces."""
from __future__ import annotations

from pathlib import Path


class PathOutsideWorkspaceError(PermissionError):
    """Raised when a filesystem path escapes the allowed workspace/cwd roots."""


def resolve_allowed_path(
    path: str | Path,
    *,
    workspace_dir: str | Path,
    cwd: str | Path | None = None,
) -> Path:
    """Resolve ``path`` and ensure it lies under cwd and/or the workspace root.

    Matches the policy used by ``/api/dataset/inspect-file`` and ``/api/files/browse``.
    """
    target = Path(path).expanduser().resolve()
    base_dir = Path(cwd or Path.cwd()).resolve()
    ws_path = Path(workspace_dir).resolve()
    allowed = (
        target == base_dir
        or target.is_relative_to(base_dir)
        or target == ws_path
        or target.is_relative_to(ws_path)
    )
    if not allowed:
        raise PathOutsideWorkspaceError(
            f"Access denied: path outside workspace/cwd: {path}"
        )
    return target


def is_path_allowed(
    path: str | Path,
    *,
    workspace_dir: str | Path,
    cwd: str | Path | None = None,
) -> bool:
    try:
        resolve_allowed_path(path, workspace_dir=workspace_dir, cwd=cwd)
        return True
    except PathOutsideWorkspaceError:
        return False
