"""Pin the parts of the paper that the repository checks against, each by sha256, and report any that change.

    python scripts/paper_pin.py extract     # unzip the pinned paper zip into paper/ (git-ignored)
    python scripts/paper_pin.py pin         # (re)write configs/paper_pins.json from paper/
    python scripts/paper_pin.py check       # compare paper/ with the pins; exit 1 if a checked item changed

Checked items (a change fails the check): tables/T1-T8.tex, figs/*.pdf, sections/*.tex, every figure caption and its
\\includegraphics line, and the abstract. Reported only: manuscript.tex as a whole (its end matter - data statement, declarations,
CRediT - is expected to change without affecting any table, figure or number check) and the zip itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PINS = ROOT / "configs" / "paper_pins.json"


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


def zip_info() -> tuple[Path, str]:
    cfg = (ROOT / "configs" / "paper.yaml").read_text(encoding="utf-8")
    z = Path(re.search(r"^zip_path: (.+)$", cfg, re.M).group(1).strip())
    return (z if z.is_absolute() else (ROOT / z).resolve()), re.search(r"^zip_sha256: ([0-9a-f]{64})$", cfg, re.M).group(1)


def braced(s: str, i: int) -> str:
    """The content of the brace group that opens at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j]
    raise ValueError("unbalanced braces")


def figures(tex: str) -> dict[str, dict]:
    out = {}
    for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
        body = m.group(1)
        label = re.search(r"\\label\{([^}]*)\}", body).group(1)
        c = body.index("\\caption{")
        out[label] = {"caption": braced(body, c + len("\\caption")),
                      "includegraphics": re.search(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", body).group(0)}
    return out


def abstract(tex: str) -> str:
    return re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).strip()


def collect() -> dict:
    if not PAPER.exists():
        print("paper/ missing: run `python scripts/paper_pin.py extract` (skipped)")
        sys.exit(3)
    items = {}
    for p in sorted((PAPER / "tables").glob("T*.tex")) + sorted((PAPER / "figs").glob("*.pdf")) + sorted((PAPER / "sections").glob("*.tex")):
        items[p.relative_to(PAPER).as_posix()] = sha(p.read_bytes())
    for p in sorted((PAPER / "sections").glob("*.tex")):
        for label, f in figures(p.read_text(encoding="utf-8")).items():
            items[f"caption:{label}"] = sha(f["caption"])
            items[f"includegraphics:{label}"] = sha(f["includegraphics"])
    man = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    items["abstract"] = sha(abstract(man))
    return items


def report_only() -> dict:
    z, _ = zip_info()
    out = {"manuscript.tex": sha((PAPER / "manuscript.tex").read_bytes())}
    if z.exists():
        out["zip"] = sha(z.read_bytes())
    return out


def extract():
    z, zsha = zip_info()
    data = z.read_bytes()
    if sha(data) != zsha:
        print(f"note: the zip's sha256 {sha(data)} differs from configs/paper.yaml (a new paper version); run `pin` after review")
    PAPER.mkdir(exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(PAPER)
    print(f"extracted {z.name} into paper/")


def check() -> int:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    now, ro = collect(), report_only()
    changed = sorted(k for k in pins["checked"] if now.get(k) != pins["checked"][k])
    added = sorted(set(now) - set(pins["checked"]))
    for k in changed:
        print(f"CHANGED (checked): {k}")
    for k in added:
        print(f"NEW (not pinned): {k}")
    for k, v in pins["report_only"].items():
        if ro.get(k) != v:
            print(f"changed (report only): {k}")
    print(f"paper pins: {len(pins['checked'])} checked items, {len(changed)} changed, {len(added)} new")
    return 1 if changed or added else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("extract", "pin", "check"))
    a = ap.parse_args()
    if a.cmd == "extract":
        extract()
    elif a.cmd == "pin":
        PINS.write_text(json.dumps({"checked": collect(), "report_only": report_only()}, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {PINS.relative_to(ROOT).as_posix()}")
    else:
        sys.exit(check())


if __name__ == "__main__":
    main()
