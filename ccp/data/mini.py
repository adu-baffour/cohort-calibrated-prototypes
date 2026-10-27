"""mini-ImageNet (Ravi & Larochelle 64 / 16 / 20 split, as shipped by FEAT; checked file by file against FEAT's split CSVs).

Source images: the original-resolution JPEGs in <image_root>/{train,val,test}/<wnid>/<file>.jpg (docs/data.md). No image is
redistributed. Two 84 x 84 RGB views are prepared once:
  eval84   PIL Resize(92) on the short side (aspect kept, bilinear) + CenterCrop(84)  (FEAT / FRN test geometry; frozen FEAT,
           FRN, SnaTCHer-F hosts)
  whole84  whole image, PIL LANCZOS resize to 84 x 84 (the RFS-pickle geometry of TANE / DNPG; TANE and DNPG hosts)
Normalisation for every mini host: FEAT's ResNet-12 constants (FEAT_MEAN / FEAT_STD below; identical to TANE / DNPG's).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
FEAT_MEAN = tuple(x / 255.0 for x in (120.39586422, 115.59361427, 104.54012653))
FEAT_STD = tuple(x / 255.0 for x in (70.68188272, 68.27635443, 72.54505529))
CSV_URL = "https://raw.githubusercontent.com/Sha-Lab/FEAT/47bdc7c1672e00b027c67469d0291e7502918950/data/miniimagenet/split/{}.csv"


def index(image_root: Path) -> dict:
    """{'paths', 'split', 'labels', 'wnids'} in the global order: split (train, val, test), then sorted wnid, then sorted file name."""
    image_root = Path(image_root)
    paths, split, labels, wnids = [], [], [], []
    for s in SPLITS:
        for w in sorted(p.name for p in (image_root / s).iterdir() if p.is_dir()):
            wnids.append(w)
            for f in sorted(p.name for p in (image_root / s / w).iterdir()):
                paths.append(str(image_root / s / w / f))
                split.append(s)
                labels.append(len(wnids) - 1)
    return {"paths": paths, "split": np.array(split), "labels": np.array(labels), "wnids": wnids}


def verify_split(idx: dict, csv_dir: Path) -> dict:
    """Compares the folder layout with FEAT's split CSVs (filename,label rows)."""
    out = {}
    for s in SPLITS:
        rows = list(csv.reader(open(Path(csv_dir) / f"{s}.csv", encoding="utf-8")))[1:]
        csv_set = {(r[1], r[0]) for r in rows}
        here = {(Path(q).parent.name, Path(q).name) for q, sp in zip(idx["paths"], idx["split"]) if sp == s}
        out[s] = {"csv": len(csv_set), "folder": len(here), "identical": csv_set == here}
    return out


def eval_view(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 92 / min(w, h)
    im = im.resize((max(92, round(w * s)), max(92, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = int(round((w - 84) / 2.0)), int(round((h - 84) / 2.0))
    return np.asarray(im.crop((left, top, left + 84, top + 84)), dtype=np.uint8)


def whole_view(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((84, 84), Image.LANCZOS), dtype=np.uint8)


VIEWS = {"eval84": eval_view, "whole84": whole_view}
