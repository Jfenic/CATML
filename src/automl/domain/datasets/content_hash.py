"""Cryptographic content fingerprint for dataset files (recipe reproducibility)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_content_hash(path: str | Path, *, chunk_size: int = 65536) -> str:
    """Return lowercase hex SHA-256 of file bytes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash missing dataset file: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def try_compute_file_content_hash(path: str | Path | None) -> str | None:
    """Best-effort hash; returns None when path is missing or unreadable."""
    if path is None:
        return None
    try:
        return compute_file_content_hash(path)
    except (OSError, ValueError):
        return None
