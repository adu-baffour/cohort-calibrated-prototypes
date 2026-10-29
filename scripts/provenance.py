"""Verbatim copies from release 1.5.0 of the authors' earlier release repository (private) and their provenance.

    python scripts/provenance.py check [--source <checkout>]   # every listed file has its recorded sha256 (and equals its source)
    python scripts/provenance.py add-check results/checks/<file> --source "<script and arguments that wrote it>"

The verbatim sets (VERBATIM) are recorded results, registration documents, bundles, markers and episode manifests. They were copied
byte for byte from the files tracked at the source commit and are never edited; results/PROVENANCE.md lists each with its sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "PROVENANCE.md"
SOURCE_COMMIT = "96cae35"   # release 1.5.0 of the earlier release repository
SOURCE_TAG = "release-1.5.0"
# (path prefix in the source, destination prefix here); every tracked file under the prefix was copied
VERBATIM = [
    ("results/recorded/", "results/recorded/"),
    ("results/baselines/", "results/baselines/"),
    ("results/variants/", "results/variants/"),
    ("results/final_round/", "results/final_round/"),
    ("results/foundation_round/", "results/foundation_round/"),
    ("results/paper_prep/paired_derived.json", "results/derived/paired_derived.json"),
    ("results/paper_prep/bank_geometry_exploratory.json", "results/derived/bank_geometry_exploratory.json"),
    ("results/paper_prep/cost_snatcher_valA.json", "results/derived/cost_snatcher_valA.json"),
    ("results/paper_prep/environment.json", "results/derived/environment.json"),
    ("bundles/", "bundles/"),
    ("manifests/", "manifests/"),
    ("docs/preregistration/", "docs/preregistration/"),
    ("docs/prereg_variants.md", "docs/prereg_variants.md"),
    ("figures/src/mini_class_names.tsv", "configs/datasets/mini_class_names.tsv"),
    ("third_party/fonts/", "third_party/fonts/"),
    ("third_party/patches/", "third_party/patches/"),
]
RELEASE_SECTION = "\n## Release assets"       # the release archives, written when they were packed
CHECKS_SECTION = "## 2.0.0 checks"             # files written by this repository's checks (results/checks/), pinned by add-check


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob(src: Path, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=src, capture_output=True, check=True).stdout


def check_rows() -> list[tuple[str, str, str]]:
    """(file, sha256, source) rows of the "2.0.0 checks" section of the current file."""
    if not OUT.exists():
        return []
    txt = OUT.read_text(encoding="utf-8")
    if CHECKS_SECTION not in txt:
        return []
    sec = txt[txt.index(CHECKS_SECTION):].split("\n## ", 1)[0]
    return re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \| `([^`]+)` \|$", sec, re.M)


def checks_block(rows=None) -> list[str]:
    rows = check_rows() if rows is None else rows
    if not rows:
        return []
    return [f"{CHECKS_SECTION} ({len(rows)} file{'s' if len(rows) != 1 else ''})", "",
            "Not copied from release 1.5.0: validation checks run for this release by its own scripts and written by them into "
            "results/checks/. They are pinned here like the verbatim files and never edited; a new run writes a new file.", "",
            "| File | sha256 | Source |", "|---|---|---|"] + [f"| `{d}` | `{h}` | `{s}` |" for d, h, s in rows] + [""]


def add_check(path: str, source: str):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    if not rel.startswith("results/checks/"):
        raise SystemExit("a 2.0.0 check lives in results/checks/")
    rows = [r for r in check_rows() if r[0] != rel] + [(rel, sha((ROOT / rel).read_bytes()), source)]
    txt = OUT.read_text(encoding="utf-8")
    if CHECKS_SECTION in txt:
        i = txt.index(CHECKS_SECTION)
        j = txt.find("\n## ", i + 1)
        txt = txt[:i] + "\n".join(checks_block(sorted(rows))) + "\n" + txt[j + 1:]
    else:
        i = txt.index("## Line endings")
        txt = txt[:i] + "\n".join(checks_block(sorted(rows))) + "\n" + txt[i:]
    OUT.write_text(txt, encoding="utf-8", newline="\n")
    print(f"listed {rel} ({rows[-1][1]}) in {OUT.relative_to(ROOT).as_posix()}")


def listed() -> list[tuple[str, str, str]]:
    """(file, sha256, source) rows of the file tables (the line-ending notes are not file rows)."""
    txt = OUT.read_text(encoding="utf-8").split("\n## Line endings")[0].split(RELEASE_SECTION)[0]
    return re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \| `([^`]+)` \|$", txt, re.M)


def check(src: Path | None) -> int:
    """Each listed file has its sha256; with --source, each also equals the committed source file."""
    bad = 0
    rows = listed()
    for d, h, s in rows:
        p = ROOT / d
        if not p.exists() or sha(p.read_bytes()) != h:
            print("MISMATCH", d)
            bad += 1
            continue
        if src is not None and s.startswith(f"{SOURCE_TAG}@"):
            if sha(blob(src, s.split(":", 1)[1])) != h:
                print("DIFFERS FROM SOURCE", d)
                bad += 1
    print(f"{len(rows)} files listed; {bad} problems")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "add-check"))
    ap.add_argument("file", nargs="?", help="add-check: the check file under results/checks/")
    ap.add_argument("--source", default=None, help="check: path of a checkout of the earlier release repository; add-check: what wrote the file")
    a = ap.parse_args()
    if a.cmd == "add-check":
        if not a.file or not a.source:
            raise SystemExit("add-check needs the file and --source")
        add_check(a.file, a.source)
        return
    src = Path(a.source).resolve() if a.source else None
    sys.exit(1 if check(src) else 0)


if __name__ == "__main__":
    main()
