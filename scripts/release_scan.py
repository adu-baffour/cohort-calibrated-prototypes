"""Release scan over the files git would commit (tracked, and untracked but not ignored). Run before every push.

    python scripts/release_scan.py [--names NAME ...]
    python scripts/release_scan.py --assets [DIR]       # the release archives (default release_assets/upload/)

Fails on:
  - dataset images: a raster image file other than the listed plots (PLOT_IMAGES), a PDF that embeds raster images other than the listed
    rasterized plots (RASTER_PDFS), and a text file that embeds an image (data:image). Figures 1 and 4 contain mini-ImageNet images and
    are never tracked; a tracked copy fails here even if it is listed in .gitignore;
  - machine-specific or private data: absolute user paths, e-mail addresses other than GitHub no-reply addresses, credential-like tokens
    (GitHub, Hugging Face, OpenAI / Anthropic, AWS, Slack, Google API keys, private keys), wandb entities / keys, the given user / machine
    names; files over 5 MB; model, feature or array files (.pth, .pt, .npy, .npz, .pkl);
  - mentions of AI tools or AI attribution lines;
  - registration-era labels in the public prose (Markdown and the generated LaTeX): the paper's names are used everywhere except the
    verbatim recorded files and registration documents and the files that map the two (EXEMPT_NAMES). Code may use recorded keys, and
    prose may quote them as inline code (`d1_val.json`).
--assets lists every member of the release archives (*.tar) and fails on any image file (by its magic bytes, whatever its name) and on any
array that looks like image data: a uint8 array of shape (..., H, W, 3) or (..., 3, H, W) in an .npy / .npz member (headers only), or a
uint8 tensor of that shape in a PyTorch checkpoint (tensor dtypes read from the checkpoint's pickle; uint8 tensors are then loaded and
their shapes checked). Every uint8 array is listed, image-shaped or not.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERNS = {
    "absolute Windows user path": r"[A-Za-z]:[\\/]+Users[\\/]",
    "absolute Unix home path": r"(?<![\w.])/(home|Users)/[A-Za-z0-9_.-]+/",
    "e-mail address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|org|net|edu|io|ai|cn)\b",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "OpenAI / Anthropic key": r"\bsk-(ant-)?[A-Za-z0-9_-]{20,}",
    "AWS key": r"\bAKIA[0-9A-Z]{16}\b",
    "Slack token": r"\bxox[abpr]-[A-Za-z0-9-]{10,}",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "wandb": r"(?i)\bwandb\.(init|login)\b|WANDB_(API_KEY|ENTITY|PROJECT)",
    "AI tool or attribution": r"(?i)\b(claude|anthropic|chatgpt|gpt-[345]|copilot|gemini|llm-generated)\b|co-authored-by:|generated with \[",
    "embedded image": r"data:image/",
}
OLD_NAMES = {   # registration-era labels and their paper names (docs/naming.md)
    "D1 (SCG)": r"\bD1\b|\bd1_(?:val|test|valA|valB)\b(?!\.)",
    "H2b (H2)": r"\bH2b\b|\bH2b?-s\b",
    "V1-V4 (CCP-QW, full AS-norm, shared-cohort correction, fusion-weight factor)": r"\bV[1-4][- ](?:QW|ASK|SHC|FW)\b|\bV1fused\b",
    "B1-B9 (baseline names)": r"\bB[1-9][ab]?-(?:QTERM|KNN|RMD|CL2N|ZN|NNN|QB|TEEN|CSLS)",
    "G1-G4 (criteria (i)-(iv))": r"\bG[1-4]\b",
    "tvA / tvB (valA / valB)": r"\btv[AB]\b",
}
EXEMPT_NAMES = ("results/", "bundles/", "manifests/", "docs/preregistration/", "docs/prereg_variants.md", "docs/naming.md", "ccp/naming.py",
                "scripts/rounds/render_recorded.py", "CHANGELOG.md", "PAPER_EDITS.md", "scripts/release_scan.py")
PROSE = {".md", ".tex", ".cff", ".txt"}
ALLOWED_EMAIL = re.compile(r"^\d+\+[\w-]+@users\.noreply\.github\.com$|^[\w.-]+@users\.noreply\.github\.com$")
RASTER = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
PLOT_IMAGES = {   # raster files that are plots of recorded numbers (no dataset image)
    "results/final_round/d1_test.png", "results/final_round/d1_val.png", "results/foundation_round/d1_test.png",
    "results/foundation_round/d1_val.png", "results/variants/d1/d1_panel.png",
    "docs/results/figures/episode_gains.png", "docs/results/figures/gain_vs_host.png", "docs/results/figures/sensitivity.png"}
RASTER_PDFS = {"paper_assets/figures/fig2_mechanism.pdf": 1}   # path -> number of raster images (Figure 2(a): the rasterized scatter)
BINARY = RASTER | {".pdf", ".ttf", ".drawio"}
FORBIDDEN_EXT = {".pth", ".pt", ".npy", ".npz", ".pkl", ".pickle"}
DATASET_IMAGE_FILES = ("paper_assets/figures/fig1_method.pdf", "paper_assets/figures/fig4_examples.pdf",
                       "figures/src/Figure1_CCP-A_overview.pdf", "figures/src/Figure1_CCP-A_overview.drawio")


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return sorted({f for f in out.split("\n") if f and (REPO / f).exists()})


def pdf_raster_images(data: bytes) -> int:
    """Image XObjects of a PDF (raw objects and inflated object streams), not counting soft masks (an image's alpha channel)."""
    parts = [data]
    for m in re.finditer(rb"stream\r?\n", data):
        end = data.find(b"endstream", m.end())
        try:
            parts.append(zlib.decompress(data[m.end():end]))
        except zlib.error:
            continue
    images = sum(len(re.findall(rb"/Subtype\s*/Image\b", p)) for p in parts)
    masks = sum(len(re.findall(rb"/SMask\s+\d+\s+0\s+R", p)) for p in parts)
    return images - masks


def scan(names=()) -> list[str]:
    pats = dict(PATTERNS)
    for n in names:
        pats[f"name '{n}'"] = rf"(?i)\b{re.escape(n)}\b"
    problems = []
    for rel in files():
        f = REPO / rel
        ext = f.suffix.lower()
        if rel in DATASET_IMAGE_FILES:
            problems.append(f"{rel}: contains dataset images (never tracked; git rm --cached)")
        if ext in FORBIDDEN_EXT:
            problems.append(f"{rel}: model / array file type in git")
        if f.stat().st_size > 5 * 2**20:
            problems.append(f"{rel}: over 5 MB")
        if ext in RASTER and rel not in PLOT_IMAGES:
            problems.append(f"{rel}: raster image that is not a listed plot (dataset images are never tracked)")
        if ext == ".pdf":
            n = pdf_raster_images(f.read_bytes())
            if n != RASTER_PDFS.get(rel, 0):
                problems.append(f"{rel}: embeds {n} raster image(s); allowed {RASTER_PDFS.get(rel, 0)} (dataset images are never tracked)")
        if ext in BINARY and ext != ".drawio":
            continue
        try:
            txt = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        checks = dict(pats)
        if ext in PROSE and not rel.startswith(EXEMPT_NAMES):
            checks.update({f"registration label {k}": v for k, v in OLD_NAMES.items()})
        prose = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), txt)     # inline code quotes recorded keys and file names
        for label, p in checks.items():
            if rel == "scripts/release_scan.py" and not label.startswith("name "):
                continue
            for m in re.finditer(p, prose if label.startswith("registration label") else txt):
                if label == "e-mail address" and ALLOWED_EMAIL.match(m.group(0)):
                    continue
                problems.append(f"{rel}:{txt.count(chr(10), 0, m.start()) + 1}: {label}: {m.group(0)[:60]!r}")
    return problems


IMAGE_MAGIC = {b"\x89PNG": "PNG", b"\xff\xd8\xff": "JPEG", b"GIF8": "GIF", b"BM": "BMP", b"II*\x00": "TIFF", b"MM\x00*": "TIFF",
               b"%PDF": "PDF", b"<svg": "SVG"}


def image_shaped(dtype, shape) -> bool:
    if str(dtype) not in ("uint8", "torch.uint8") or len(shape) < 3:
        return False
    return (shape[-1] == 3 and min(shape[-3:-1]) >= 8) or (shape[-3] == 3 and min(shape[-2:]) >= 8)


def npy_header(f):
    import numpy as np
    major, _ = np.lib.format.read_magic(f)
    hdr = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
    shape, _, dtype = hdr(f)
    return dtype, shape


def scan_member(name: str, data: bytes) -> tuple[list[str], list[str]]:
    """(problems, uint8 arrays) of one archive member."""
    import io
    import zipfile
    problems, u8 = [], []
    head = data[:16].lstrip()
    for magic, kind in IMAGE_MAGIC.items():
        if head.startswith(magic):
            problems.append(f"{name}: {kind} image file")
    if name.endswith(".npy"):
        dt, sh = npy_header(io.BytesIO(data))
        arrays = {name: (dt, sh)}
    elif name.endswith(".npz"):
        arrays = {}
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                with z.open(n) as f:
                    first = f.read(6)
                if first != b"\x93NUMPY":
                    problems.append(f"{name}/{n}: not an array")
                    continue
                with z.open(n) as f:
                    arrays[f"{name}/{n}"] = npy_header(f)
    elif name.endswith((".pth", ".pt")):
        arrays = {}
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            pkl = [n for n in z.namelist() if n.endswith("data.pkl")]
            for n in z.namelist():                       # images stored inside the checkpoint archive
                with z.open(n) as f:
                    first = f.read(16)
                for magic, kind in IMAGE_MAGIC.items():
                    if first.startswith(magic):
                        problems.append(f"{name}/{n}: {kind} image file")
            if any(b"ByteStorage" in z.read(n) or b"uint8" in z.read(n) for n in pkl):
                import torch
                obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
                stack = [("", obj)]
                while stack:
                    k, o = stack.pop()
                    if isinstance(o, torch.Tensor):
                        arrays[f"{name}:{k}"] = (o.dtype, tuple(o.shape))
                    elif isinstance(o, dict):
                        stack += [(f"{k}/{kk}", v) for kk, v in o.items()]
                    elif isinstance(o, (list, tuple)):
                        stack += [(f"{k}[{i}]", v) for i, v in enumerate(o)]
    else:
        arrays = {}
    for k, (dt, sh) in arrays.items():
        if str(dt) in ("uint8", "torch.uint8"):
            u8.append(f"{k}: {dt} {tuple(sh)}")
            if image_shaped(dt, sh):
                problems.append(f"{k}: uint8 array of shape {tuple(sh)} (image-like)")
    return problems, u8


def scan_assets(folder: Path) -> int:
    import tarfile
    tars = sorted(folder.glob("*.tar"))
    if not tars:
        print(f"no archives in {folder}")
        return 1
    n_bad = 0
    for t in tars:
        with tarfile.open(t) as tf:
            members = [m for m in tf if m.isfile()]
            print(f"{t.name}: {len(members)} members")
            for m in members:
                problems, u8 = scan_member(m.name, tf.extractfile(m).read())
                print(f"  {m.name} ({m.size:,} bytes)" + ("" if not problems else "  <- " + "; ".join(problems)))
                for x in u8:
                    print(f"      uint8 array: {x}")
                n_bad += len(problems)
    print(f"asset scan: {len(tars)} archives, {n_bad} problems")
    return n_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="*", default=[], help="user / machine names that must not appear (case-insensitive, whole word)")
    ap.add_argument("--assets", nargs="?", const=REPO / "release_assets" / "upload", type=Path, default=None,
                    help="scan the members of the release archives instead of the repository")
    a = ap.parse_args()
    if a.assets is not None:
        sys.exit(1 if scan_assets(a.assets) else 0)
    problems = scan(a.names)
    for p in problems:
        print(" ", p)
    print(f"release scan: {len(files())} files, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
