"""Prepare CUB-200-2011 for the cross-domain round (docs/preregistration/final_round.md, (B)): whole images (no bounding boxes) in mini-ImageNet's
two test views, class split by the FRN rule.

    python scripts/prepare_data/cub.py [--archive $CCP_DATA/raw/CUB_200_2011.tgz]

Split rule (FRN, Wertheimer et al., CVPR 2021; 1-based class id i from classes.txt): i % 2 == 0 -> train (100 classes), i % 4 == 1 -> val
(50), i % 4 == 3 -> test (50). Global order: split (train, val, test), then class id, then image id (images.txt). Views, exactly mini's
(ccp.data.mini.VIEWS): eval84 = PIL Resize(92) short side bilinear + CenterCrop(84) (frozen FEAT, SnaTCHer-F); whole84 = whole image, PIL
LANCZOS 84 x 84 (DNPG). Images are decoded with PIL and converted to RGB (a few CUB files are greyscale / CMYK).
Foundation round (docs/preregistration/foundation_round.md), analysis view only: eval224 = PIL Resize(256, bicubic) on the short side
(long side int(256 * long / short), torchvision's rule) + CenterCrop(224) (offsets int(round((side - 224) / 2))); `--views eval224`.
Output: $CCP_DATA/cub/prepared/{eval84,whole84}_{train,val,test}.npy (+ .json sidecars with sha256), index.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccp.data import mini  # noqa: E402
from ccp.data.images import CUB_SIZES  # noqa: E402
from ccp.utils.paths import data_root, load_yaml, sha256_file, REPO  # noqa: E402

SPLITS = ("train", "val", "test")


def eval224_view(path: str) -> np.ndarray:
    """Foundation-round analysis view: PIL Resize(256, BICUBIC) on the short side (torchvision rule for the long side) + CenterCrop(224)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = int(round((nw - 224) / 2.0)), int(round((nh - 224) / 2.0))
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


VIEW_FN = {"eval84": mini.VIEWS["eval84"], "whole84": mini.VIEWS["whole84"], "eval224": eval224_view}
SIDE = {"eval84": 84, "whole84": 84, "eval224": 224}
TRANSFORM = {"eval84": "PIL Resize(92) short side bilinear + CenterCrop(84)", "whole84": "PIL LANCZOS resize to 84x84",
             "eval224": "PIL Resize(256) short side bicubic (long side int(256 * long / short)) + CenterCrop(224)"}


def split_of(class_id: int) -> str | None:
    """FRN rule on the 1-based class id."""
    if class_id % 2 == 0:
        return "train"
    return "val" if class_id % 4 == 1 else "test"


def read_index(root: Path) -> dict:
    """{'paths', 'split', 'labels', 'class_ids', 'image_ids', 'class_names'} in the global order."""
    classes = {int(a): b for a, b in (line.split(maxsplit=1) for line in (root / "classes.txt").read_text().splitlines() if line.strip())}
    images = {int(a): b for a, b in (line.split(maxsplit=1) for line in (root / "images.txt").read_text().splitlines() if line.strip())}
    lab = {int(a): int(b) for a, b in (line.split() for line in (root / "image_class_labels.txt").read_text().splitlines() if line.strip())}
    rows = sorted(((SPLITS.index(split_of(lab[i])), lab[i], i) for i in images), key=lambda r: r)
    return {"paths": [str(root / "images" / images[i]) for _, _, i in rows], "split": np.array([SPLITS[s] for s, _, _ in rows]),
            "labels": np.array([c - 1 for _, c, _ in rows]), "class_ids": [c for _, c, _ in rows], "image_ids": [i for _, _, i in rows],
            "class_names": classes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--views", nargs="+", default=["eval84", "whole84"], choices=list(VIEW_FN))
    a = ap.parse_args()
    cfg = load_yaml(REPO / "configs" / "datasets" / "cub.yaml")
    archive = a.archive or data_root() / "raw" / "CUB_200_2011.tgz"
    got = sha256_file(archive)
    if got != cfg["archive"]["sha256"]:
        raise SystemExit(f"{archive}: sha256 {got} != {cfg['archive']['sha256']} (configs/datasets/cub.yaml)")
    raw = data_root() / "raw"
    root = raw / "CUB_200_2011"
    if not (root / "images.txt").exists():
        with tarfile.open(archive) as t:
            bad = [m.name for m in t.getmembers() if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts]
            if bad:
                raise SystemExit(f"unsafe paths in {archive}: {bad[:3]}")
            t.extractall(raw)   # the archive's sha256 is pinned and its member paths are checked above
    idx = read_index(root)
    sizes = {s: int((idx["split"] == s).sum()) for s in SPLITS}
    n_cls = {s: len({c for c, sp in zip(idx["class_ids"], idx["split"]) if sp == s}) for s in SPLITS}
    print("sizes", sizes, "classes", n_cls, flush=True)
    if n_cls != {"train": 100, "val": 50, "test": 50} or sizes != CUB_SIZES:
        raise SystemExit(f"split sizes {sizes} / classes {n_cls} differ from ccp.data.images.CUB_SIZES {CUB_SIZES} and 100 / 50 / 50")
    out = data_root() / "cub" / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"labels": idx["labels"].tolist(), "class_ids": idx["class_ids"], "image_ids": idx["image_ids"], "sizes": sizes,
               "class_names": {str(k): v for k, v in idx["class_names"].items()},
               "order": "split (train, val, test) by the FRN rule, then class id, then image id; label = class id - 1"},
              open(out / "index.json", "w"), indent=0)
    for view in a.views:
        fn = VIEW_FN[view]
        for s in SPLITS:
            p = out / f"{view}_{s}.npy"
            paths = [q for q, sp in zip(idx["paths"], idx["split"]) if sp == s]
            if not p.exists():
                with Pool(a.workers) as pool:
                    arr = np.stack(pool.map(fn, paths, chunksize=64))
                tmp = p.with_suffix(".tmp.npy")
                np.save(tmp, arr)
                tmp.replace(p)
            arr = np.load(p, mmap_mode="r")
            assert arr.shape == (sizes[s], SIDE[view], SIDE[view], 3) and arr.dtype == np.uint8
            side = {"dataset": "cub", "split": s, "view": view, "channel_order": "RGB", "n": int(arr.shape[0]), "sha256": sha256_file(p),
                    "transform": TRANSFORM[view]}
            json.dump(side, open(p.with_suffix(".json"), "w"), indent=1)
            print(view, s, side["n"], side["sha256"], flush=True)


if __name__ == "__main__":
    main()
