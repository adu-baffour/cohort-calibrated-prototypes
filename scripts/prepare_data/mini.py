"""Prepare mini-ImageNet: verify the split against FEAT's CSVs, write the eval84 and whole84 views (RGB uint8) and the index.

    python scripts/prepare_data/mini.py --image-root <dir with train/ val/ test/> [--csv-dir <FEAT split csv dir>]

Output: $CCP_DATA/mini/prepared/{eval84,whole84}_{train,val,test}.npy (+ .json sidecars with sha256), index.json.
Expected counts: train 38,400 (64 classes), val 9,600 (16), test 12,000 (20); sha256 of every array in docs/data.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccp.data import mini  # noqa: E402
from ccp.data.images import MINI_SIZES  # noqa: E402
from ccp.utils.paths import data_root, sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-root", type=Path, default=None, help="default $CCP_DATA/mini/raw/miniImageNet")
    ap.add_argument("--csv-dir", type=Path, default=None, help="FEAT split CSVs; downloaded (pinned commit) if missing")
    ap.add_argument("--out", type=Path, default=None, help="default $CCP_DATA/mini/prepared")
    ap.add_argument("--views", nargs="+", default=["eval84", "whole84"], choices=list(mini.VIEWS))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    root = data_root() / "mini"
    image_root = a.image_root or root / "raw" / "miniImageNet"
    csv_dir = a.csv_dir or root / "split"
    out = a.out or root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    for s in mini.SPLITS:
        if not (csv_dir / f"{s}.csv").exists():
            urllib.request.urlretrieve(mini.CSV_URL.format(s), csv_dir / f"{s}.csv")
    idx = mini.index(image_root)
    chk = mini.verify_split(idx, csv_dir)
    print("split check vs FEAT CSVs:", chk, flush=True)
    if not all(v["identical"] for v in chk.values()):
        raise SystemExit("folder layout differs from FEAT's split CSVs")
    for s in mini.SPLITS:
        n = int((idx["split"] == s).sum())
        if n != MINI_SIZES[s]:
            raise SystemExit(f"{s}: {n} images, expected {MINI_SIZES[s]}")
    json.dump({"wnids": idx["wnids"], "labels": idx["labels"].tolist(), "sizes": MINI_SIZES,
               "order": "split (train, val, test), sorted wnid, sorted file name"}, open(out / "index.json", "w"), indent=0)
    for view in a.views:
        fn = mini.VIEWS[view]
        for s in mini.SPLITS:
            p = out / f"{view}_{s}.npy"
            paths = [q for q, sp in zip(idx["paths"], idx["split"]) if sp == s]
            if not p.exists():
                with Pool(a.workers) as pool:
                    arr = np.stack(pool.map(fn, paths, chunksize=256))
                tmp = p.with_suffix(".tmp.npy")
                np.save(tmp, arr)
                tmp.replace(p)
            arr = np.load(p, mmap_mode="r")
            assert arr.shape == (MINI_SIZES[s], 84, 84, 3) and arr.dtype == np.uint8
            n_zero = int((np.asarray(arr).reshape(arr.shape[0], -1).max(1) == 0).sum())
            if n_zero:
                raise SystemExit(f"{p}: {n_zero} all-zero images")
            side = {"dataset": "mini", "split": s, "view": view, "channel_order": "RGB", "n": int(arr.shape[0]), "sha256": sha256_file(p),
                    "transform": {"eval84": "PIL Resize(92) short side bilinear + CenterCrop(84)", "whole84": "PIL LANCZOS resize to 84x84"}[view]}
            json.dump(side, open(p.with_suffix(".json"), "w"), indent=1)
            print(view, s, side["n"], side["sha256"], flush=True)


if __name__ == "__main__":
    main()
