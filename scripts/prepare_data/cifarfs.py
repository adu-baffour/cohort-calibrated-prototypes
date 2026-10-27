"""Prepare CIFAR-FS (Bertinetto et al., ICLR 2019; 64 / 16 / 20 CIFAR-100 classes) from the official CIFAR-100 python archive.

    python scripts/prepare_data/cifarfs.py [--archive $CCP_DATA/raw/cifar-100-python.tar.gz]

Class lists: configs/datasets/cifarfs_split/{train,val,test}.txt, the r2d2 lists as mirrored by learn2learn (zenodo 7978538 cifar100.zip,
cifar100/splits/bertinetto/*.txt; sha256 pinned in configs/datasets/cifarfs.yaml). Every class keeps all 600 images (500 CIFAR-100 train +
100 CIFAR-100 test), as in CIFAR-FS. Global order: split (train, val, test), then the split's class-list order, then the CIFAR-100 train
images of the class in archive order followed by its test images in archive order. Label = position of the class in the concatenated
train + val + test lists (train 0-63, val 64-79, test 80-99). View eval32: the original 32 x 32 RGB pixels (no resampling).
Output: $CCP_DATA/cifarfs/prepared/eval32_{train,val,test}.npy (+ .json sidecars with sha256), index.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccp.data.images import CIFARFS_SIZES  # noqa: E402
from ccp.utils.paths import data_root, load_yaml, sha256_file, REPO  # noqa: E402

SPLITS = ("train", "val", "test")


def class_lists() -> dict:
    cfg = load_yaml(REPO / "configs" / "datasets" / "cifarfs.yaml")
    out = {}
    for s in SPLITS:
        p = REPO / "configs" / "datasets" / "cifarfs_split" / f"{s}.txt"
        if sha256_file(p) != cfg["class_lists"][s]["sha256"]:
            raise SystemExit(f"{p}: sha256 differs from configs/datasets/cifarfs.yaml")
        out[s] = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [len(out[s]) for s in SPLITS] == [64, 16, 20] and len(set(sum(out.values(), []))) == 100
    return out


def load_cifar100(archive: Path) -> dict:
    """{'names': fine label names, 'train': (data uint8 (50000,32,32,3), labels), 'test': (...)} straight from the archive."""
    with tarfile.open(archive) as t:
        def member(n):
            return pickle.load(t.extractfile(f"cifar-100-python/{n}"), encoding="bytes")
        names = [x.decode() for x in member("meta")[b"fine_label_names"]]
        out = {"names": names}
        for part in ("train", "test"):
            d = member(part)
            out[part] = (np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.asarray(d[b"fine_labels"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=None)
    a = ap.parse_args()
    cfg = load_yaml(REPO / "configs" / "datasets" / "cifarfs.yaml")
    archive = a.archive or data_root() / "raw" / "cifar-100-python.tar.gz"
    if sha256_file(archive) != cfg["archive"]["sha256"]:
        raise SystemExit(f"{archive}: sha256 differs from configs/datasets/cifarfs.yaml")
    lists = class_lists()
    c = load_cifar100(archive)
    name_to_id = {n: i for i, n in enumerate(c["names"])}
    out = data_root() / "cifarfs" / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    labels, classes = [], []
    for s in SPLITS:
        imgs = []
        for cname in lists[s]:
            fid = name_to_id[cname]
            parts = [c[p][0][c[p][1] == fid] for p in ("train", "test")]
            assert [len(x) for x in parts] == [500, 100], (cname, [len(x) for x in parts])
            imgs.append(np.concatenate(parts))
            labels += [len(classes)] * 600
            classes.append(cname)
        arr = np.ascontiguousarray(np.concatenate(imgs))
        assert arr.shape == (CIFARFS_SIZES[s], 32, 32, 3) and arr.dtype == np.uint8
        p = out / f"eval32_{s}.npy"
        np.save(p, arr)
        side = {"dataset": "cifarfs", "split": s, "view": "eval32", "channel_order": "RGB", "n": int(arr.shape[0]), "sha256": sha256_file(p),
                "transform": "none: the original CIFAR-100 32 x 32 RGB pixels"}
        json.dump(side, open(p.with_suffix(".json"), "w"), indent=1)
        print(s, side["n"], side["sha256"], flush=True)
    json.dump({"classes": classes, "labels": labels, "sizes": CIFARFS_SIZES, "cifar100_fine_ids": [name_to_id[x] for x in classes],
               "order": "split (train, val, test), class-list order, CIFAR-100 train images then test images in archive order"},
              open(out / "index.json", "w"), indent=0)


if __name__ == "__main__":
    main()
