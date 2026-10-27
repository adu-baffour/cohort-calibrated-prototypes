"""Verbatim copies from release 1.5.0 of the authors' earlier release repository (private) and their provenance.

    python scripts/provenance.py copy --source <checkout>     # copy the verbatim sets, verify, write results/PROVENANCE.md
    python scripts/provenance.py check [--source <checkout>]   # every listed file has its recorded sha256 (and equals its source)
    python scripts/provenance.py add-check results/checks/<file> --source "<script and arguments that wrote it>"

The verbatim sets (VERBATIM) are recorded results, registration documents, bundles, markers and episode manifests. They are copied
byte for byte from the files tracked at the source commit and are never edited. Each copy is checked against the file in the source
working tree and against the blob at the source commit. Figure 1 is copied from the paper zip (configs/paper.yaml).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "PROVENANCE.md"
SOURCE_COMMIT = "96cae35"   # release 1.5.0 of the earlier release repository
SOURCE_TAG = "release-1.5.0"
# (path prefix in the source, destination prefix here); every tracked file under the prefix is copied
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
    ("figures/src/mini_class_names.tsv", "figures/src/mini_class_names.tsv"),
    ("third_party/fonts/", "third_party/fonts/"),
    ("third_party/patches/", "third_party/patches/"),
]
CATEGORY = [("results/recorded/", "recorded results (rounds (i)-(iv) and validation records)"),
            ("results/baselines/", "round (v) records"), ("results/variants/", "round (vi) records"),
            ("results/final_round/", "round (vii) records"), ("results/foundation_round/", "round (viii) records"),
            ("results/derived/", "derived and measured records"), ("bundles/", "bundles and markers"),
            ("manifests/", "episode manifests"), ("docs/", "registration documents"),
            ("figures/src/Figure1_", "not redistributed: Figure 1 (contains mini-ImageNet images; kept locally, git-ignored, checked when present)"),
            ("figures/", "figure sources"),
            ("third_party/", "third-party files")]
CRLF_NOTES: list[tuple[str, str, str]] = []   # (source path, sha256 of the CRLF working-tree bytes, sha256 of the committed bytes)
PAPER_FIG1 = "figs/fig1_method.pdf"
FIG1_DEST = "figures/src/Figure1_CCP-A_overview.pdf"
DRAWIO_DEST = "figures/src/Figure1_CCP-A_overview.drawio"
LOCAL_ONLY = (FIG1_DEST, DRAWIO_DEST)          # pinned here, never tracked (dataset images; scripts/release_scan.py)
RELEASE_SECTION = "\n## Release assets"       # written by scripts/release_manifest.py and kept when this file is rewritten
CHECKS_SECTION = "## 2.0.0 checks"             # files written by this repository's checks (results/checks/), pinned by add-check


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tracked(src: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=src, capture_output=True, check=True).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def blob(src: Path, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=src, capture_output=True, check=True).stdout


def paper_zip() -> tuple[Path, str]:
    cfg = (ROOT / "configs" / "paper.yaml").read_text(encoding="utf-8")
    z = Path(re.search(r"^zip_path: (.+)$", cfg, re.M).group(1).strip())
    z = z if z.is_absolute() else (ROOT / z).resolve()
    return z, re.search(r"^zip_sha256: ([0-9a-f]{64})$", cfg, re.M).group(1)


def copy(src: Path) -> list[tuple[str, str, str]]:
    files = tracked(src)
    rows = []
    for s_pref, d_pref in VERBATIM:
        hits = [f for f in files if f == s_pref or (s_pref.endswith("/") and f.startswith(s_pref))]
        if not hits:
            raise SystemExit(f"nothing tracked under {s_pref}")
        for f in hits:
            data = blob(src, f)                     # the committed content is the published 1.5.0 file
            work = (src / f).read_bytes()
            if work != data:
                # allowed only for line endings: a Windows checkout can hold CRLF bytes that git normalised to LF on commit
                if work.replace(b"\r\n", b"\n") != data:
                    raise SystemExit(f"{f}: working tree differs from {SOURCE_COMMIT} beyond line endings")
                CRLF_NOTES.append((f, sha(work), sha(data)))
            dest = d_pref + f[len(s_pref):] if s_pref.endswith("/") else d_pref
            p = ROOT / dest
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            rows.append((dest, sha(data), f"{SOURCE_TAG}@{SOURCE_COMMIT}:{f}"))
    z, zsha = paper_zip()
    zb = z.read_bytes()
    if sha(zb) != zsha:
        raise SystemExit(f"paper zip sha256 differs from configs/paper.yaml: {sha(zb)}")
    with zipfile.ZipFile(z) as zf:
        data = zf.read(PAPER_FIG1)
    (ROOT / FIG1_DEST).write_bytes(data)
    rows.append((FIG1_DEST, sha(data), f"paper zip {z.name} (sha256 {zsha[:16]}):{PAPER_FIG1}"))
    drawio = ROOT / DRAWIO_DEST
    if drawio.exists():   # extracted once from the PDF's metadata (draw.io stores the diagram URL-encoded in the Subject field)
        rows.append((DRAWIO_DEST, sha(drawio.read_bytes()), f"draw.io XML embedded in {FIG1_DEST} (PDF Subject, URL-decoded)"))
    return sorted(rows)


def write(rows):
    L = ["# Provenance of the verbatim files", "",
         "Generated by `scripts/provenance.py copy`; checked by `scripts/provenance.py check` and `tests/test_provenance.py`.", "",
         f"Every file below is a byte-for-byte copy of the file tracked in release 1.5.0 (commit `{SOURCE_COMMIT}`) of the authors' earlier release "
         f"repository, which is private (source column "
         f"`{SOURCE_TAG}@<commit>:<path>`), "
         "or of the paper's Figure 1, and is never edited. Paths, names and keys inside these files are the original ones; "
         "`docs/naming.md` maps them to the paper's names. `results/recorded/PROVENANCE.md` is the 1.x provenance of the recorded files "
         "(their origin in the research repository) and is itself a verbatim copy. The section \"2.0.0 checks\" lists the files written by this "
         "release's own checks (results/checks/); \"Release assets\" lists the release archives.", ""]
    placed = set()
    for pref, title in CATEGORY:
        sel = [r for r in rows if r[0].startswith(pref) and r[0] not in placed]
        placed.update(r[0] for r in sel)
        if not sel:
            continue
        L += [f"## {title} ({len(sel)} file{'s' if len(sel) != 1 else ''})", "", "| File | sha256 | Source |", "|---|---|---|"]
        L += [f"| `{d}` | `{h}` | `{s}` |" for d, h, s in sel]
        L.append("")
    L += checks_block()
    if CRLF_NOTES:
        L += ["## Line endings", "",
              "The source is the committed blob (the file as published in release 1.5.0). For the files below, the Windows working tree "
              "of the source held CRLF line endings that git normalised to LF on commit; the content is otherwise identical. No registration "
              "document, bundle or marker records either sha256.", "",
              "| Source file | sha256, CRLF working-tree bytes | sha256, committed bytes (copied) |", "|---|---|---|"]
        L += [f"| `{f}` | `{a}` | `{b}` |" for f, a, b in CRLF_NOTES]
        L.append("")
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    tail = old[old.index(RELEASE_SECTION):] if RELEASE_SECTION in old else ""
    OUT.write_text("\n".join(L) + tail, encoding="utf-8", newline="\n")


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
        txt = txt[:i] + "\n".join(checks_block(sorted(rows))) + txt[j + 1:]
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
        if d in LOCAL_ONLY and not p.exists():
            print("absent (not redistributed; checked when present):", d)
            continue
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
    ap.add_argument("cmd", choices=("copy", "check", "add-check"))
    ap.add_argument("file", nargs="?", help="add-check: the check file under results/checks/")
    ap.add_argument("--source", default=None, help="copy / check: path of a checkout of the earlier release repository; add-check: what wrote the file")
    a = ap.parse_args()
    if a.cmd == "add-check":
        if not a.file or not a.source:
            raise SystemExit("add-check needs the file and --source")
        add_check(a.file, a.source)
        return
    a.source = Path(a.source) if a.source else None
    if a.cmd == "copy":
        if a.source is None:
            raise SystemExit("copy needs --source")
        rows = copy(a.source.resolve())
        write(rows)
        print(f"copied {len(rows)} files; wrote {OUT.relative_to(ROOT).as_posix()}")
    else:
        sys.exit(1 if check(a.source.resolve() if a.source else None) else 0)


if __name__ == "__main__":
    main()
