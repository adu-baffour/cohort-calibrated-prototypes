"""Helpers shared by the round drivers: provenance guards, output paths, registered bundle blocks and recorded inputs."""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from ccp import recorded  # noqa: E402
from ccp.hosts import load_host_config  # noqa: E402

GENERATED = REPO / "results" / "generated" / "rounds"
CODE_DIRS = ("ccp", "scripts", "configs")
BUNDLES = {  # round -> (registration document holding the bundle block, fenced block name)
    "baselines": ("docs/preregistration/FINAL_BUNDLE_baselines.md", "yaml-fsor-baselines"),
    "variants": ("docs/preregistration/FINAL_BUNDLE_variants.md", "yaml-fsor-variants"),
    "final_round": ("docs/preregistration/FINAL_BUNDLE_final_round.md", "yaml-fsor-final-round"),
    "foundation_round": ("docs/preregistration/FINAL_BUNDLE_foundation_round.md", "yaml-fsor-foundation-round"),
}


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def git_head(repo: Path = REPO) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def require_clean_code(repo: Path = REPO):
    """Provenance: refuse if any tracked file under ccp/, scripts/ or configs/ differs from HEAD (staged or not); outputs record git HEAD."""
    r = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", *CODE_DIRS], cwd=repo, capture_output=True, text=True, check=True)
    if r.stdout.strip():
        raise SystemExit("uncommitted changes in tracked code / configuration (commit them first; the output records git HEAD): "
                         + r.stdout.replace("\n", " "))


def out_dir(round_name: str) -> Path:
    d = GENERATED / round_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")


def record(rel: str, paper_names: bool = True):
    """A recorded validation or test record (checked against results/PROVENANCE.md; keys in the paper's names by default)."""
    return recorded.load(rel, paper_names)


def generated_or_recorded(round_name: str, name: str, recorded_rel: str):
    """A record re-made in results/generated/rounds/<round>/ if present (paper names), else the recorded one (translated)."""
    p = GENERATED / round_name / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else record(recorded_rel)


def bundle_block(round_name: str, paper_names: bool = True) -> tuple[dict, str]:
    """The registered bundle block of a round (parsed YAML, translated to the paper's names) and its token (the first 16 hex characters
    of the sha256 of the block's text)."""
    import yaml
    rel, fence = BUNDLES[round_name]
    txt = recorded.read_bytes(rel).decode("utf-8")
    m = re.search(rf"```{re.escape(fence)}\n(.*?)\n```", txt, re.S)
    block = yaml.safe_load(m.group(1))
    return (recorded.translate(block) if paper_names else block), hashlib.sha256(m.group(1).encode("utf-8")).hexdigest()[:16]


def cfg(h: str) -> dict:
    return load_host_config(h)


def kind(h: str) -> str:
    return cfg(h)["kind"]


def trained(h: str) -> bool:
    return kind(h) in ("snatcher_f", "dnpg")


def ccp_weight(h: str, shot: int) -> float | None:
    c = cfg(h)
    return float(c["fusion"]["frozen_weights"][f"{shot}shot"]["w"]) if c["kind"] in ("snatcher_f", "dnpg") else None


def fmt(x: dict, k: str) -> str:
    """'mean +- CI' of a summary, or 'mean +- std' of a seed summary."""
    v = x[k]
    if isinstance(v, dict):
        return f"{v['mean']:.2f} ± {v['std']:.2f}"
    return f"{v:.2f} ± {x[k + '_ci95']:.2f}"


def worst_diff(got: dict, ref: dict, metrics=("auroc", "acc")) -> tuple[float, int]:
    """Max |difference| and the number of compared values over the summaries present in `ref`."""
    worst, n = 0.0, 0
    for k, x in ref.items():
        for m in metrics:
            if m in x:
                worst = max(worst, abs(got[k][m] - x[m]))
                n += 1
    return worst, n
