"""Paths. pathlib only; nothing machine-specific is hard-coded.

Roots (environment variable first, then the default next to the repository):
  CCP_DATA         prepared datasets (docs/data.md)                        default <repo>/data
  CCP_CKPT         checkpoint search roots, os.pathsep-separated            default <repo>/checkpoints, <repo>/release_assets/checkpoints
  CCP_ASSETS       unpacked release assets (banks, score caches, npz)       default <repo>/release_assets
  CCP_THIRD_PARTY  pinned third-party clones (third_party/clone.py)         default <repo>/third_party
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


def data_root() -> Path:
    return _env_path("CCP_DATA", REPO / "data")


def assets_root() -> Path:
    return _env_path("CCP_ASSETS", REPO / "release_assets")


def third_party_root() -> Path:
    return _env_path("CCP_THIRD_PARTY", REPO / "third_party")


def ckpt_roots() -> list[Path]:
    v = os.environ.get("CCP_CKPT")
    if v:
        return [Path(p).expanduser().resolve() for p in v.split(os.pathsep) if p]
    return [REPO / "checkpoints", assets_root() / "checkpoints"]


def resolve_ckpt(rel: str | Path) -> Path:
    """First existing <root>/<rel> over ckpt_roots(); raises FileNotFoundError listing the roots searched."""
    rel = Path(rel)
    if rel.is_absolute():
        if rel.exists():
            return rel
        raise FileNotFoundError(rel)
    for r in ckpt_roots():
        p = r / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"{rel} not found under any checkpoint root {[str(r) for r in ckpt_roots()]} "
                            "(set CCP_CKPT; see docs/checkpoints.md)")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str | Path) -> dict:
    import yaml
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_sha256(path: str | Path, expected: str, what: str = "") -> None:
    """Raises ValueError (not assert: survives python -O) if the file's sha256 differs from `expected`."""
    got = sha256_file(path)
    if got != expected:
        raise ValueError(f"sha256 mismatch for {what or path}: file {got}, expected {expected}")
