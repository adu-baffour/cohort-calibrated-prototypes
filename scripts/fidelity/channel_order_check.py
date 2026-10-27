"""Channel-order check for the released tieredImageNet checkpoints (FEAT meta-trained feat-K-shot.pth): FEAT classifier accuracy
(slf_attn prototypes, -||q - proto||^2 / 64) on 200 closed-set 5-way episodes (K shots + 15 queries per class) drawn from the 351 BASE
(training) classes with RandomState(20260930 + K), feeding the images as RGB or as BGR. Validation of preprocessing only (base classes).
GPU.   python scripts/fidelity/channel_order_check.py

Recorded result (docs/results/implementation_checks.md): 1-shot RGB 86.27 vs BGR 93.64; 5-shot 94.96 vs 98.11 -> BGR. The prepared tiered arrays are BGR
(scripts/prepare_data/tiered.py), so "RGB" here is the exact channel flip of them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ccp.data.images import labels, open_split  # noqa: E402
from ccp.hosts import build_host  # noqa: E402


def sample_closed(lab: np.ndarray, shot: int, seed: int, n=200, q=15) -> np.ndarray:
    """(n, 5*shot + 5*q) row indices: support (class-major) then queries (class-major); the recorded sampler."""
    rs = np.random.RandomState(seed)
    cls = np.unique(lab)
    byc = {c: np.where(lab == c)[0] for c in cls}
    out = np.empty((n, 5 * (shot + q)), dtype=np.int64)
    for e in range(n):
        cs = rs.choice(cls, 5, replace=False)
        picks = [rs.choice(byc[c], shot + q, replace=False) for c in cs]
        out[e] = np.concatenate([np.concatenate([p[:shot] for p in picks]), np.concatenate([p[shot:] for p in picks])])
    return out


def main(n_eps: int = 200):
    import torch
    tr = open_split("tiered", "train", "bgr84")
    lab = labels("tiered", "train")
    out = {}
    for sh in (1, 5):
        eps = sample_closed(lab, sh, 20260930 + sh, n=n_eps)
        host = build_host("snatcher_f_tiered", shot=sh)
        tgt = torch.arange(5, device="cuda").repeat_interleave(15)
        r = {}
        for order in ("RGB", "BGR"):
            accs = []
            for e in range(n_eps):
                x = np.array(tr.arr[np.sort(eps[e])][np.argsort(np.argsort(eps[e]))])
                E = host.pooled(x if order == "BGR" else np.ascontiguousarray(x[..., ::-1]))
                bproto = E[:5 * sh].view(5, sh, -1).mean(1)
                with torch.no_grad():
                    proto = host.attn(bproto[None], bproto[None], bproto[None])[0][0]
                logits = -(E[5 * sh:, None, :] - proto[None]).pow(2).sum(2) / 64.0
                accs.append(float((logits.argmax(1) == tgt).float().mean() * 100))
            r[order] = {"acc": float(np.mean(accs)), "ci95": float(1.96 * np.std(accs) / np.sqrt(n_eps))}
        r["chosen"] = max(("RGB", "BGR"), key=lambda o: r[o]["acc"])
        out[f"{sh}shot"] = r
        print(sh, r, flush=True)
        del host
        torch.cuda.empty_cache()
    p = REPO / "results" / "generated" / "fidelity_channel_order.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
