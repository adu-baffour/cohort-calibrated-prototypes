"""Prepared image arrays and the global index.

Layout under CCP_DATA (written by scripts/prepare_data/{mini,tiered}.py):
  mini/prepared/{eval84,whole84}_{train,val,test}.npy + .json      RGB uint8
  mini/prepared/index.json                                          wnids, labels, split sizes, global offsets
  tiered/prepared/{train,val,test}_bgr84.npy + .json                BGR uint8
  tiered/prepared/{train,val,test}_labels.npy                       label_specific from the learn2learn pickles
  cifarfs/prepared/eval32_{train,val,test}.npy + .json, index.json   RGB uint8 32 x 32 (scripts/prepare_data/cifarfs.py)
  cub/prepared/{eval84,whole84}_{train,val,test}.npy + .json, index.json   RGB uint8 84 x 84, mini's two test views (scripts/prepare_data/cub.py)

Global index (the index space of the episode manifests):
  mini    train 0..38,399 | val 38,400..47,999 | test 48,000..59,999 (split order, then sorted wnid, then sorted file name)
  tiered  train 0..448,694 | val 448,695..572,955 | test 572,956..779,164 (label-pickle order within each split)
  cifarfs train 0..38,399 | val 38,400..47,999 | test 48,000..59,999 (split order, then the split's class-list order, then CIFAR-100 train
          images followed by CIFAR-100 test images in archive order)
  cub     train | val | test by the FRN class rule (split order, then class id, then image id); sizes in CUB_SIZES
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ccp.utils.paths import data_root

SPLITS = ("train", "val", "test")
MINI_SIZES = {"train": 38400, "val": 9600, "test": 12000}
TIERED_SIZES = {"train": 448695, "val": 124261, "test": 206209}
CIFARFS_SIZES = {"train": 38400, "val": 9600, "test": 12000}
CUB_SIZES = {"train": 5903, "val": 2927, "test": 2958}   # FRN class rule on CUB-200-2011 (verified by scripts/prepare_data/cub.py)
SIZES = {"mini": MINI_SIZES, "tiered": TIERED_SIZES, "cifarfs": CIFARFS_SIZES, "cub": CUB_SIZES}
SIDE = {"mini": 84, "tiered": 84, "cifarfs": 32, "cub": 84}
SIDE_BY_VIEW = {"eval224": 224}   # CUB eval224: analysis view of the foundation round (docs/preregistration/foundation_round.md)
INDEX_LABELS = ("mini", "cifarfs", "cub")   # labels stored in prepared/index.json (global order)


def offsets(dataset: str) -> dict:
    s = SIZES[dataset]
    return {"train": 0, "val": s["train"], "test": s["train"] + s["val"]}


@dataclass
class ImageSet:
    """One prepared split: `arr` (n, 84, 84, 3) uint8 (memory-mapped), its channel order and its global offset."""
    arr: np.ndarray
    channel_order: str
    offset: int
    split: str
    dataset: str
    view: str

    def __len__(self):
        return self.arr.shape[0]

    def rows(self, global_idx: np.ndarray) -> np.ndarray:
        r = np.asarray(global_idx) - self.offset
        if r.min() < 0 or r.max() >= len(self):
            raise IndexError(f"global indices outside {self.dataset}/{self.split} [{self.offset}, {self.offset + len(self)})")
        return r

    def as_order(self, x: np.ndarray, want: str) -> np.ndarray:
        """Return `x` (..., 3) in channel order `want` (exact channel flip if needed)."""
        if want not in ("RGB", "BGR") or self.channel_order not in ("RGB", "BGR"):
            raise ValueError(f"unknown channel order {self.channel_order} -> {want}")
        return x if want == self.channel_order else x[..., ::-1]


def open_split(dataset: str, split: str, view: str = "eval84", root: Path | None = None) -> ImageSet:
    """view: mini / cub 'eval84' (Resize 92 + CenterCrop 84) or 'whole84' (whole image LANCZOS 84 x 84); tiered 'bgr84'; cifarfs 'eval32'."""
    root = Path(root) if root else data_root()
    if dataset in INDEX_LABELS:
        p = root / dataset / "prepared" / f"{view}_{split}.npy"
    elif dataset == "tiered":
        p = root / "tiered" / "prepared" / f"{split}_{view}.npy"
    else:
        raise ValueError(dataset)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/prepare_data/{dataset}.py (docs/data.md)")
    side = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    arr = np.load(p, mmap_mode="r")
    px = SIDE_BY_VIEW.get(view, SIDE[dataset])
    want = (SIZES[dataset][split], px, px, 3)
    if arr.shape != want or arr.dtype != np.uint8:
        raise ValueError(f"{p}: shape {arr.shape} {arr.dtype}, expected {want} uint8")
    return ImageSet(arr, side["channel_order"], offsets(dataset)[split], split, dataset, view)


def labels(dataset: str, split: str, root: Path | None = None) -> np.ndarray:
    root = Path(root) if root else data_root()
    if dataset in INDEX_LABELS:
        idx = json.loads((root / dataset / "prepared" / "index.json").read_text(encoding="utf-8"))
        o = offsets(dataset)[split]
        return np.asarray(idx["labels"][o:o + SIZES[dataset][split]], dtype=np.int64)
    return np.load(root / "tiered" / "prepared" / f"{split}_labels.npy")


def class_to_indices(dataset: str, split: str, root: Path | None = None) -> dict:
    """{class id: global indices} for the episode generator."""
    lab = labels(dataset, split, root)
    off = offsets(dataset)[split]
    return {int(c): np.where(lab == c)[0] + off for c in np.unique(lab)}
