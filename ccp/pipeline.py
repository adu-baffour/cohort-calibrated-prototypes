"""End-to-end evaluation pipeline shared by scripts/build_bank.py and scripts/evaluate.py (the registered test-pass code path).

1. Images: the prepared split arrays in the host's view and channel order (ccp.data).
2. Bank: base images embedded with the host's encoder (configs/hosts/*.yaml `bank.embed`):
     full_split   every base image is embedded (fixed batches of 500 from the first image) and the bank is the configured subset of
                  them; CCP-S statistics and class means use all base images (mini hosts; tiered frozen FEAT / TANE)
     subset_only  only the bank subset is embedded, as one array (tiered SnaTCHer-F and DNPG, as recorded; no CCP-S row)
3. Target features: EVERY image of the target split (val or test) is embedded once, in fixed batches of 500 from the first image of the
   split; per-query cohort statistics are computed row-wise over that matrix. A query's features therefore never depend on which
   episodes are evaluated (inductive, and bitwise reproducible).
4. Scores per episode (ccp.scores) and, for trained hosts, the host's own score and the fused rows (ccp.fusion).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ccp import metrics
from ccp.cohort import Bank, build_bank
from ccp.data.images import SIZES, labels as split_labels, offsets, open_split
from ccp.utils.paths import assets_root

TIERED_BANK_SEED, TIERED_BANK_SIZE = 20260926, 38400


def bank_rows(dataset: str) -> np.ndarray:
    """Rows of the base split that form the CCP-A bank: mini all 38,400; tiered the RandomState(20260926) subset of 38,400 (sorted)."""
    ntr = SIZES[dataset]["train"]
    if dataset in ("mini", "cifarfs", "cub"):   # cub: the target-domain bank of the cross-domain analysis row (all training-class images)
        return np.arange(ntr)
    return np.sort(np.random.RandomState(TIERED_BANK_SEED).choice(ntr, TIERED_BANK_SIZE, replace=False))


def host_images(host, split: str, dataset: str | None = None) -> np.ndarray:
    """Prepared split of `dataset` (default: the host's own) in the host's view, converted to the host's channel order (exact flip; a no-op
    for the registered data). A mini host on CUB (cross-domain) reads CUB in the host's mini view (eval84 or whole84)."""
    s = open_split(dataset or host.dataset, split, host.view)
    if s.channel_order != host.channel_order:
        return np.ascontiguousarray(s.arr[..., ::-1])
    return s.arr


def make_bank(host, log=print) -> Bank:
    import torch
    t0 = time.time()
    rows = bank_rows(host.dataset)
    embed_mode = host.bank_entry()["embed"]
    base = host_images(host, "train")
    lab = split_labels(host.dataset, "train")
    if embed_mode == "full_split":
        z0, zf = host.embed(base)
        use_rows, lab_all = rows, lab
    elif embed_mode == "subset_only":
        z0, zf = host.embed(base[rows])
        use_rows, lab_all = np.arange(len(rows)), lab[rows]
    else:
        raise ValueError(embed_mode)
    meta = {"host": host.name, "dataset": host.dataset, "shot": host.shot if host.trained else None, "embed": embed_mode, "view": host.view,
            "channel_order": host.channel_order, "bank_global_index": (offsets(host.dataset)["train"] + rows).tolist(),
            # Known issue (docs/preregistration/foundation_round.md, Known issues): this metadata field names the tiered subset seed for every
            # non-mini bank (CIFAR-FS, CUB, ...), where no subset is drawn. Nothing reads it. It is left unchanged because the pinned sha256 of
            # the CIFAR-FS bank (configs/hosts/cifarfs_rfs_seed0.yaml) covers these bytes.
            "subset": None if host.dataset == "mini" else {"seed": TIERED_BANK_SEED, "size": TIERED_BANK_SIZE},
            "ccps_stats_over": "all base images" if embed_mode == "full_split" else "the bank subset (CCP-S is not a registered row for this host)",
            "feature": "layer-4 GAP, bf16 autocast, fp16-rounded local map, flip views"}
    dev = host.device
    b = build_bank(z0.to(dev), zf.to(dev), lab_all, use_rows, meta, host.feature_dtype)
    b.bank_index = np.asarray(meta["bank_global_index"], dtype=np.int64)
    log(f"bank {host.name} shot={host.shot}: {b.raw_noflip.shape} {b.raw_noflip.dtype}, {time.time() - t0:.0f} s")
    del z0, zf
    torch.cuda.empty_cache()
    return b


def make_target_bank(host, dataset: str, log=print) -> Bank:
    """Cross-domain ANALYSIS bank (docs/preregistration/final_round.md, (B)): every training-class image of the target dataset (CUB), embedded
    with the host's own encoder in the host's view, flip views, the host's stored precision. It uses unlabelled target-domain images, so the
    row built on it is not the inductive cross-domain result."""
    import torch
    t0 = time.time()
    base = host_images(host, "train", dataset)
    rows = np.arange(base.shape[0])
    z0, zf = host.embed(base)
    meta = {"host": host.name, "dataset": dataset, "target_domain_bank": True, "shot": host.shot if host.trained else None,
            "embed": "full_split", "view": host.view, "channel_order": host.channel_order,
            "bank_global_index": (offsets(dataset)["train"] + rows).tolist(), "subset": None, "ccps_stats_over": "all target training-class images",
            "feature": "layer-4 GAP, bf16 autocast, fp16-rounded local map, flip views"}
    b = build_bank(z0.to(host.device), zf.to(host.device), split_labels(dataset, "train"), rows, meta, host.feature_dtype)
    b.bank_index = np.asarray(meta["bank_global_index"], dtype=np.int64)
    log(f"target bank {host.name} shot={host.shot} on {dataset}: {b.raw_noflip.shape} {b.raw_noflip.dtype}, {time.time() - t0:.0f} s")
    del z0, zf
    torch.cuda.empty_cache()
    return b


def target_bank_k(bank: Bank, k: int = 1000) -> int:
    """Registered K = 1000, or every bank image if the bank holds fewer."""
    return min(k, bank.raw_noflip.shape[0])


def bank_path(host) -> Path:
    return assets_root() / host.bank_entry()["file"]


def load_bank(host, allow_build: bool = False, log=print) -> Bank:
    p = bank_path(host)
    sha = host.bank_entry().get("sha256")
    if p.exists():
        return Bank.load(p, sha)
    if not allow_build:
        raise FileNotFoundError(f"{p} missing: download the release assets or run scripts/build_bank.py --host {host.name}")
    b = make_bank(host, log)
    got = b.save(p)
    if sha and got != sha:
        log(f"WARNING: rebuilt bank {p} sha256 {got} differs from the released {sha}")
    return b


class Target:
    """Embeddings of every image of one split for a host, with the global -> row map and (trained hosts) pooled embeddings."""

    def __init__(self, host, split: str, bank: Bank, need_pooled: bool = False, dataset: str | None = None):
        from ccp.scores import TargetFeatures
        self.split, self.host, self.dataset = split, host, dataset or host.dataset
        self.arr = host_images(host, split, self.dataset)
        self.offset = offsets(self.dataset)[split]
        z0, zf = host.embed(self.arr)
        self.tf = TargetFeatures(z0, zf, bank.to_device(host.device))
        self.E = host.pooled(self.arr) if need_pooled else None

    def rows(self, global_idx) -> np.ndarray:
        r = np.asarray(global_idx) - self.offset
        if r.min() < 0 or r.max() >= self.arr.shape[0]:
            raise IndexError(f"episode indices outside the {self.split} split")
        return r

    def image(self, global_idx: int) -> np.ndarray:
        return self.arr[int(global_idx) - self.offset]


def score_ccp(target: Target, eps, names: list[str]) -> dict:
    """{name: (scores (n, 150), accuracy (n,))} for the cohort scores `names` on Episodes `eps`."""
    from ccp.scores import EpisodeContext, score_episode
    S = {n: np.zeros((eps.n, 150)) for n in names}
    A = {n: np.zeros(eps.n) for n in names}
    for e in range(eps.n):
        ep = EpisodeContext(target.tf, target.rows(eps.indices(e)), eps.shot)
        for n, (s, acc, _) in score_episode(ep, names).items():
            S[n][e], A[n][e] = s, acc
    return {n: (S[n], A[n]) for n in names}


def score_host_own(host, target: Target, eps) -> tuple[dict, np.ndarray]:
    if host.kind == "snatcher_f":
        return host.own_scores(eps, target.rows, target.E)
    if host.kind == "dnpg":
        return host.own_scores(eps, target.image)
    raise ValueError(f"{host.kind} has no own open-set score")


def summarise_rows(rows: dict, accs: dict) -> dict:
    pe = {k: metrics.episode_auroc(v) for k, v in rows.items()}
    return {k: {**metrics.summarise(pe[k], accs[k])} for k in rows}, pe
