"""DNPG training wrapper (documented; NOT run by the release verification). Reproduces how our DNPG checkpoints were trained
(docs/checkpoints.md; research record: DNPG_SETUP.md, docs/preregistration/DNPG_SETUP_sections_9-11.md).

    python scripts/train/train_dnpg.py data --dataset mini|tiered            # write DNPG's data folder (train + val ONLY)
    python scripts/train/train_dnpg.py run  --dataset mini   --seed 0|1|2    # Phase 2 -> meta 1-shot -> meta 5-shot, strictly sequential
    python scripts/train/train_dnpg.py run  --dataset tiered --seed 0
    python scripts/train/train_dnpg.py launch {phase2|train|test} [--torch_seed N] -- <DNPG script arguments>

Code: third_party/DNPG @ 38b8fd1 + third_party/patches/dnpg.patch (python third_party/clone.py DNPG). Measured wall time on one RTX 4090
under Windows (num_workers 0): mini 4.47 h per seed (Phase 2 0.75 h, meta 1-shot 1.5 h, meta 5-shot 2.3 h); tiered seed 0 12.1 h
(5-shot with gradient accumulation). Deviations from the released code / README, all recorded:
  1. Validation-based selection: DNPG's evaluation loaders are hard-wired to the 'test' partition; our data folder makes 'test' a HARD LINK
     of the validation file and this launcher refuses to run otherwise (no test image can be read during training or selection).
  2. Per-epoch selection episodes: --n_test_runs 600 (README 3,000; 70-85% of the run time).
  3. Class-count patch (Phase 2 RPL / GCPL centres use the dataset's class count, not FC100's 60) and save-path patches (dnpg.patch).
  4. --num_workers 0 (Windows DataLoader spawn cost); TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (torch >= 2.6 cannot load DNPG's own checkpoints).
  5. Import shims (third_party/shims: mkl, h5py, pandas, tensorboardX) appended to the END of sys.path.
  6. Seeds: seed 0 = the code's torch.manual_seed(0) (no option); seeds 1, 2: --torch_seed N seeds torch and numpy once and maps DNPG's
     torch.manual_seed(0) to N. Meta-training episodes are drawn by np.random.seed(item) and are the SAME for every seed.
  7. Resize: whole image, PIL LANCZOS to 84 x 84 from the original JPEGs (the RFS pickles used another resize).
  8. tieredImageNet: BGR arrays (as the released tiered backbone expects); zero word-embedding placeholders (never used by DNPG's
     classifier; the check is recorded in results/recorded/validation/dnpg_placeholder_check.json).
  9. tieredImageNet 5-shot only: gradient accumulation (DNPG_ACCUM_EPISODES=1: micro-batch 1 episode, 2 micro-batches per optimiser step,
     BatchNorm statistics per micro-batch), because the 2-episode step needs more than 24 GB.
Phase 2 checkpoint fed to meta-learning: mini_last.pth (epoch 50); headline meta checkpoint: max_acc.pth (selected on validation classes).
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ccp.data.images import labels as split_labels, open_split  # noqa: E402
from ccp.utils.paths import data_root, resolve_ckpt, sha256_file, third_party_root  # noqa: E402

DNPG = third_party_root() / "DNPG"
SHIMS = REPO / "third_party" / "shims"
TARGETS = {"phase2": (DNPG / "pretrain" / "pretrain_openW", "batch_process.py"), "train": (DNPG, "train.py"), "test": (DNPG, "test.py")}
PRE = {"mini": "third_party/tane/miniImageNet_pre.pth", "tiered": "third_party/tane/tieredImageNet_pre.pth"}


def dnpg_data_root(dataset: str) -> Path:
    return data_root() / "dnpg" / dataset


# ------------------------------------------------------------------------------------------------------------------ data
def write_data(dataset: str):
    import pickle
    root = dnpg_data_root(dataset)
    man = {"files": {}}
    if dataset == "mini":
        out = root / "miniImageNet"
        out.mkdir(parents=True, exist_ok=True)
        idx = json.loads((data_root() / "mini" / "prepared" / "index.json").read_text(encoding="utf-8"))
        wn = idx["wnids"]
        for name, split in (("train_phase_train", "train"), ("val", "val")):
            arr = np.array(open_split("mini", split, "whole84").arr)
            lab = split_labels("mini", split).tolist()
            pack = {"data": arr, "labels": lab, "catname2label": {wn[c]: c for c in sorted(set(lab))}}
            p = out / f"miniImageNet_category_split_{name}.pickle"
            with open(p, "wb") as f:
                pickle.dump(pack, f, protocol=4)
            man["files"][p.name] = {"n": len(lab), "sha256": sha256_file(p)}
        val, test = out / "miniImageNet_category_split_val.pickle", out / "miniImageNet_category_split_test.pickle"
        test.unlink(missing_ok=True)
        os.link(val, test)  # the 'test' partition IS the validation file
        man["files"][test.name] = {"hard_link_to": val.name, "sha256": man["files"][val.name]["sha256"]}
        vec = resolve_ckpt("third_party/tane/miniImageNet_category_vector.pickle")  # TANE's class-name word vectors (unused by DNPG's classifier)
        shutil.copyfile(vec, out / vec.name)
        man["files"][vec.name] = {"sha256": sha256_file(out / vec.name)}
    else:
        out = root / "tieredImageNet"
        out.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            s = open_split("tiered", split, "bgr84")
            assert s.channel_order == "BGR"
            p = out / f"few-shot-{split}.npz"
            np.savez(p, features=np.array(s.arr), targets=split_labels("tiered", split).astype(np.int64))
            man["files"][p.name] = {"n": len(s), "sha256": sha256_file(p)}
        val, test = out / "few-shot-val.npz", out / "few-shot-test.npz"
        test.unlink(missing_ok=True)
        os.link(val, test)
        man["files"][test.name] = {"hard_link_to": val.name, "sha256": man["files"][val.name]["sha256"]}
        for split, n in (("train", 351), ("val", 97), ("test", 160)):
            p = out / f"few-shot-wordemb-{split}.npz"
            np.savez(p, features=np.zeros((n, 300), dtype=np.float32))
            man["files"][p.name] = {"placeholder": "zeros", "shape": [n, 300], "sha256": sha256_file(p)}
    (root / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man, indent=1))


def guard_data(args: list[str]):
    """Refuse to run unless DNPG's 'test' partition is the hard link of our validation file and no unlisted data file exists."""
    dr = Path(args[args.index("--data_root") + 1]).resolve()
    ds = args[args.index("--dataset") + 1]
    dataset = {"miniImageNet": "mini", "tieredImageNet": "tiered"}[ds]
    if dr != dnpg_data_root(dataset).resolve():
        raise SystemExit(f"--data_root must be {dnpg_data_root(dataset)}")
    man = json.loads((dr / "MANIFEST.json").read_text(encoding="utf-8"))
    d = dr / ds
    if dataset == "mini":
        tp, vp, ext = d / "miniImageNet_category_split_test.pickle", d / "miniImageNet_category_split_val.pickle", ".pickle"
    else:
        tp, vp, ext = d / "few-shot-test.npz", d / "few-shot-val.npz", ".npz"
    if not (tp.exists() and os.path.samefile(tp, vp)):
        raise SystemExit(f"{tp} is not the hard link of {vp}; refusing to run")
    for f in d.iterdir():
        if f.suffix == ext and f.name not in man["files"]:
            raise SystemExit(f"unexpected data file {f}; refusing to run")


# ------------------------------------------------------------------------------------------------------------------ launcher
def launch(which: str, rest: list[str], torch_seed: int | None):
    guard_data(rest)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cwd, script = TARGETS[which]
    os.chdir(cwd)
    sys.path.insert(0, str(cwd))
    sys.path.append(str(SHIMS))
    if torch_seed is not None:
        import torch
        orig = torch.manual_seed
        orig(torch_seed)
        np.random.seed(torch_seed)
        torch.manual_seed = lambda s: orig(torch_seed if s == 0 else s)
    sys.argv = [script] + rest
    print(f"[train_dnpg] {which}: cwd={cwd} argv={sys.argv}", flush=True)
    runpy.run_path(str(cwd / script), run_name="__main__")


def run_chain(dataset: str, seed: int, out_root: Path):
    ds = {"mini": "miniImageNet", "tiered": "tieredImageNet"}[dataset]
    common = ["--dataset", ds, "--data_root", str(dnpg_data_root(dataset)), "--num_workers", "0"]
    seed_arg = [] if seed == 0 else ["--torch_seed", str(seed)]
    p2_dir = out_root / f"phase2_seed{seed}"
    p2_ck = p2_dir / "EntropyRot" / f"{ds}_ResNet12_batch_128_lr_0.05" / "mini_last.pth"
    me = [sys.executable, str(Path(__file__).resolve()), "launch"]
    log = {}
    t0 = time.time()
    rc = subprocess.run(me + ["phase2"] + seed_arg + ["--"] + common + ["--model_path", str(p2_dir), "--pret_model_path", str(resolve_ckpt(PRE[dataset])),
                                                                        "--batch_size", "128"]).returncode
    log["phase2"] = {"rc": rc, "minutes": (time.time() - t0) / 60}
    if rc:
        raise SystemExit(f"phase2 failed ({rc})")
    meta = ["--logroot", str(out_root / f"meta_seed{seed}"), "--n_ways", "5", "--gpus", "0", "--pretrained_model_path", str(p2_ck), "--learning_rate", "0.07",
            "--tunefeat", "0.0001", "--tune_part", "4", "--cosine", "--train_weight_base", "1", "--n_test_runs", "600"]
    for s in (1, 5):   # one shot at a time: two meta runs in parallel exceed a 24 GB card
        env = dict(os.environ, DNPG_ACCUM_EPISODES="1") if (dataset == "tiered" and s == 5) else None
        t0 = time.time()
        rc = subprocess.run(me + ["train"] + seed_arg + ["--"] + common + meta + ["--n_shots", str(s)], env=env).returncode
        log[f"meta_{s}shot"] = {"rc": rc, "minutes": (time.time() - t0) / 60, "accumulation": env is not None}
        if rc:
            raise SystemExit(f"meta {s}-shot failed ({rc})")
    (out_root / f"train_log_seed{seed}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--dataset", choices=("mini", "tiered"), required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", choices=("mini", "tiered"), required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path, default=REPO / "runs" / "dnpg")
    ln = sub.add_parser("launch")
    ln.add_argument("which", choices=list(TARGETS))
    ln.add_argument("--torch_seed", type=int, default=None)
    ln.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "data":
        write_data(a.dataset)
    elif a.cmd == "run":
        a.out.mkdir(parents=True, exist_ok=True)
        run_chain(a.dataset, a.seed, a.out / a.dataset)
    else:
        rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
        launch(a.which, rest, a.torch_seed)


if __name__ == "__main__":
    main()
