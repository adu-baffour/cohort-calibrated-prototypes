"""Prepare tieredImageNet: md5-check the learn2learn PNG pickles, decode them and write BGR uint8 arrays and labels.

    python scripts/prepare_data/tiered.py [--raw <dir with tiered-imagenet-*.pkl>]

Output: $CCP_DATA/tiered/prepared/{train,val,test}_bgr84.npy (+ .json sidecars with sha256), {split}_labels.npy.
Expected counts: train 448,695 (351 classes), val 124,261 (97), test 206,209 (160); sha256 of every array in docs/data.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccp.data import tiered  # noqa: E402
from ccp.data.images import TIERED_SIZES  # noqa: E402
from ccp.utils.paths import data_root, sha256_file  # noqa: E402

N_CLASSES = {"train": 351, "val": 97, "test": 160}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=None, help="default $CCP_DATA/tiered/raw")
    ap.add_argument("--out", type=Path, default=None, help="default $CCP_DATA/tiered/prepared")
    ap.add_argument("--splits", nargs="+", default=list(tiered.SPLITS), choices=list(tiered.SPLITS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-md5", action="store_true")
    a = ap.parse_args()
    raw = a.raw or data_root() / "tiered" / "raw"
    out = a.out or data_root() / "tiered" / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    for s in a.splits:
        pk = raw / f"tiered-imagenet-{s}_images_png.pkl"
        if not a.skip_md5:
            got = md5(pk)
            if got != tiered.MD5[f"{s}_images_png"]:
                raise SystemExit(f"{pk}: md5 {got} != {tiered.MD5[f'{s}_images_png']}")
            print(s, "md5 ok", flush=True)
        lab = tiered.labels(raw, s)
        if len(lab) != TIERED_SIZES[s] or len(np.unique(lab)) != N_CLASSES[s]:
            raise SystemExit(f"{s}: {len(lab)} labels / {len(np.unique(lab))} classes, expected {TIERED_SIZES[s]} / {N_CLASSES[s]}")
        np.save(out / f"{s}_labels.npy", lab.astype(np.int64))
        p = out / f"{s}_bgr84.npy"
        if not p.exists():
            data = tiered.images_pickle(raw, s)
            if len(data) != TIERED_SIZES[s]:
                raise SystemExit(f"{s}: {len(data)} images, expected {TIERED_SIZES[s]}")
            tmp = p.with_suffix(".tmp.npy")
            arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(data), 84, 84, 3))
            with Pool(a.workers) as pool:
                for i in range(0, len(data), 20000):
                    arr[i:i + 20000] = np.stack(pool.map(tiered.decode_bgr, data[i:i + 20000], chunksize=500))
            arr.flush()
            del arr, data
            tmp.replace(p)
        arr = np.load(p, mmap_mode="r")
        assert arr.shape == (TIERED_SIZES[s], 84, 84, 3) and arr.dtype == np.uint8
        n_zero = sum(int((np.asarray(arr[i:i + 20000]).reshape(-1, 84 * 84 * 3).max(1) == 0).sum()) for i in range(0, arr.shape[0], 20000))
        if n_zero:
            raise SystemExit(f"{p}: {n_zero} all-zero images")
        side = {"dataset": "tiered", "split": s, "view": "bgr84", "channel_order": "BGR", "n": int(arr.shape[0]), "sha256": sha256_file(p),
                "labels_sha256": sha256_file(out / f"{s}_labels.npy"), "transform": "PIL decode of the 84x84 PNG, RGB -> BGR channel flip"}
        json.dump(side, open(p.with_suffix(".json"), "w"), indent=1)
        print(s, side["n"], side["sha256"], flush=True)


if __name__ == "__main__":
    main()
