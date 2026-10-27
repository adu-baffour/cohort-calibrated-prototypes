"""Clone the third-party code this release builds on, at the pinned commits, and apply our patches (third_party/patches/).

    python third_party/clone.py                 # all
    python third_party/clone.py SnaTCHer DNPG   # only what evaluation needs
    python third_party/clone.py DNPG --source <local clone or mirror URL>

Nothing from these repositories is copied into this repository; see THIRD_PARTY_LICENSES.md (MIT, DINOv2 Apache-2.0, except RFS, whose
repository has no licence file: it is only fetched here and imported at run time, never redistributed). A clone whose HEAD is not the pinned commit, or whose
patch does not apply cleanly, is an error.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOS = {
    "FEAT": ("https://github.com/Sha-Lab/FEAT.git", "47bdc7c1672e00b027c67469d0291e7502918950", None),
    "SnaTCHer": ("https://github.com/MinkiJ/SnaTCHer.git", "335c42469f0a7ad72c5c3480c8effc8c293823e0", None),
    "TANE": ("https://github.com/shiyuanh/TANE.git", "fe089d477ba9387dd2d008d8872cdc84f8ad77ed", None),
    # DNPG moved from github.com/iCGY96/DNPG (the URL used for the research clone; it now redirects) to github.com/gary23ai/DNPG
    "DNPG": ("https://github.com/gary23ai/DNPG.git", "38b8fd13656ae66c87488415223a8df24ad67c80", "dnpg.patch"),
    # RFS (Tian et al., ECCV 2020): models/resnet.py is the CIFAR-FS backbone of the final round (ccp/hosts/rfs_resnet12.py)
    "RFS": ("https://github.com/WangYueFt/rfs.git", "f8c837ba93c62dd0ac68a2f4019c619aa86b8421", None),
    # foundation round (docs/preregistration/foundation_round.md): frozen image encoders, code only (weights are downloaded and pinned separately)
    "CLIP": ("https://github.com/openai/CLIP.git", "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6", None),
    "dinov2": ("https://github.com/facebookresearch/dinov2.git", "7764ea0f912e53c92e82eb78a2a1631e92725fc8", None),
}
# FRN (github.com/Tsingularity/FRN): not cloned; its head is re-implemented in scripts/train/frn_head.py (MIT, attributed).


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def clone(name: str, source: str | None = None, root: Path = HERE) -> Path:
    url, commit, patch = REPOS[name]
    dst = root / name
    if not (dst / ".git").exists():
        run(["git", "clone", "--no-checkout", source or url, dst])
    run(["git", "checkout", "--force", commit], cwd=dst)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dst, capture_output=True, text=True, check=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"{name}: HEAD {head} != pinned {commit}")
    if patch:
        diff = subprocess.run(["git", "diff", "--quiet"], cwd=dst)
        if diff.returncode == 0:
            run(["git", "apply", "--whitespace=nowarn", HERE / "patches" / patch], cwd=dst)
        else:
            run(["git", "apply", "--check", "--reverse", HERE / "patches" / patch], cwd=dst)  # already applied: must be exactly our patch
    print(f"{name}: {dst} @ {commit}" + (f" + {patch}" if patch else ""), flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=list(REPOS))
    ap.add_argument("--source", default=None, help="clone from this URL / local path instead of GitHub (one repo at a time)")
    ap.add_argument("--root", type=Path, default=HERE)
    a = ap.parse_args()
    if a.source and len(a.names) != 1:
        sys.exit("--source needs exactly one repository name")
    for n in a.names:
        clone(n, a.source, a.root)


if __name__ == "__main__":
    main()
