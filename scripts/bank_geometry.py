"""EXPLORATORY (not pre-registered, not a hypothesis, not used for any selection): geometry of the base cohort banks of the frozen backbones.
Base banks only (released bank files; no validation or test image, episode or manifest is read). GPU if available. Writes
results/paper_prep/bank_geometry_exploratory.json.

    python scripts/bank_geometry.py [--n 5000] [--seed 20261006]

Per frozen backbone (frozen FEAT mini and tiered, RFS CIFAR-FS, CLIP ViT-B/16 and DINOv2 ViT-S/B/L-14 on mini, tiered, CIFAR-FS, CUB):
  B = the unit flip-view bank (as CCP-A uses it), mu = mean of B, u = mu / ||mu||;
  mean_cos_to_bank_mean = mean over all bank rows of cos(b, u); ||mu|| is reported too;
  for n bank rows drawn with RandomState(seed): m_K(v) = mean of the K = 1000 largest cosines of v to B with v itself left out;
  r2_mK_on_cos_to_mean = squared Pearson correlation between m_K(v) and cos(v, u) over those rows (share of the variance of m_K explained
  by a linear function of the cosine to the bank mean).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ccp.cohort import REGISTERED_K, Bank, unit_flip  # noqa: E402
from ccp.utils.paths import assets_root  # noqa: E402

FROZEN = {"frozen FEAT mini": "mini/frozen_feat_mini", "frozen FEAT tiered": "tiered/frozen_feat_tiered", "RFS CIFAR-FS": "cifarfs/cifarfs_rfs_seed0",
          **{f"{b} {ds}": f"foundation/{ds}_{h}" for ds in ("mini", "tiered", "cifarfs", "cub")
             for h, b in (("clip_b16", "CLIP ViT-B/16"), ("dinov2_s14", "DINOv2 ViT-S/14"), ("dinov2_b14", "DINOv2 ViT-B/14"),
                          ("dinov2_l14", "DINOv2 ViT-L/14"))}}


def geometry(bank: Bank, n: int, seed: int, device: str) -> dict:
    import torch
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    B = unit_flip(t(bank.raw_noflip), t(bank.raw_flip))
    mu = B.mean(0)
    u = mu / mu.norm()
    c_all = B @ u
    idx = np.random.RandomState(seed).choice(B.shape[0], size=min(n, B.shape[0]), replace=False)
    m = []
    with torch.no_grad():
        for a in range(0, len(idx), 500):
            rows = torch.as_tensor(idx[a:a + 500], device=device)
            S = B[rows] @ B.T
            S[torch.arange(len(rows), device=device), rows] = -2.0          # leave v itself out
            m.append(S.topk(REGISTERED_K, dim=1).values.mean(1))
    m = torch.cat(m).double().cpu().numpy()
    c = c_all[torch.as_tensor(idx, device=device)].double().cpu().numpy()
    r = float(np.corrcoef(m, c)[0, 1])
    return {"n_bank": int(B.shape[0]), "dim": int(B.shape[1]), "norm_of_bank_mean": float(mu.norm()),
            "mean_cos_to_bank_mean": float(c_all.double().mean()), "sd_cos_to_bank_mean": float(c_all.double().std()),
            "n_sampled": int(len(idx)), "mean_mK": float(m.mean()), "sd_mK": float(m.std()), "pearson_r": r, "r2_mK_on_cos_to_mean": r * r}


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261006)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"label": "EXPLORATORY: not pre-registered, not a hypothesis, not used for any selection; base banks only",
           "definition": __doc__.split("\n\n")[2].strip(), "K": REGISTERED_K, "n": a.n, "seed": a.seed, "backbones": {}}
    for name, rel in FROZEN.items():
        p = assets_root() / "banks" / f"{rel}.npz"
        out["backbones"][name] = {"bank_file": f"banks/{rel}.npz", **geometry(Bank.load(p), a.n, a.seed, device)}
        g = out["backbones"][name]
        print(f"{name:28s} n={g['n_bank']:6d} mean cos to bank mean {g['mean_cos_to_bank_mean']:.3f}  R2(m_K ~ cos) {g['r2_mK_on_cos_to_mean']:.3f}")
        torch.cuda.empty_cache()
    out["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    f = REPO / "results" / "paper_prep" / "bank_geometry_exploratory.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
