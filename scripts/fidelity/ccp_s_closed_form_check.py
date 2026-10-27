"""CCP-S closed form (m(v) = v . b_bar, s(v) = sqrt(v^T C v) + 1e-6) vs the brute-force S-norm (mean and std, ddof 0, of the cosines of v
to every base image) on a host's bank. Validation only. GPU (or CPU with --device cpu).

    python scripts/fidelity/ccp_s_closed_form_check.py [--host frozen_feat_mini] [--split valA_fsor_5way_1shot]

Reports the max |difference| of m and s over every validation image and prototype of the manifest, and of the CCP-S AUROC per episode.
Recorded result (docs/results/implementation_checks.md): max |AUROC difference| 1.8e-5 over four mini backbones (tolerance 0.01). Synthetic check: tests/test_cohort.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ccp import metrics  # noqa: E402
from ccp.cohort import ccp_s_matrix, s_norm_stats  # noqa: E402
from ccp.episodes import Episodes  # noqa: E402
from ccp.hosts import build_host  # noqa: E402
from ccp.pipeline import Target, load_bank  # noqa: E402
from ccp.scores import EpisodeContext  # noqa: E402


def brute(V, B, chunk=1024):
    import torch
    ms, ss = [], []
    for a in range(0, V.shape[0], chunk):
        c = (V[a:a + chunk].double() @ B.double().T)
        ms.append(c.mean(1)); ss.append(c.std(1, unbiased=False) + 1e-6)
    return torch.cat(ms).float(), torch.cat(ss).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="frozen_feat_mini")
    ap.add_argument("--split", default="valA_fsor_5way_1shot")
    a = ap.parse_args()
    host = build_host(a.host)
    host.shot = None
    bank = load_bank(host)
    if bank.meta.get("ccps_stats_over") != "all base images" or bank.raw_noflip.shape[0] != bank.meta["n_all_embedded"]:
        raise SystemExit("brute force needs every base image in the bank (mini hosts)")
    tgt = Target(host, "val", bank)
    B = tgt.tf.bank.B
    eps = Episodes(host.dataset, a.split)
    mq_c, sq_c = s_norm_stats(tgt.tf.uf, tgt.tf.bank.bbar, tgt.tf.bank.C)
    mq_b, sq_b = brute(tgt.tf.uf, B)
    d_stats = max(float((mq_c - mq_b).abs().max()), float((sq_c - sq_b).abs().max()))
    au_c, au_b = [], []
    for e in range(eps.n):
        ep = EpisodeContext(tgt.tf, tgt.rows(eps.indices(e)), eps.shot)
        c = ep.cos(True)
        mp_c, sp_c = s_norm_stats(ep.pf, tgt.tf.bank.bbar, tgt.tf.bank.C)
        mp_b, sp_b = brute(ep.pf, B)
        d_stats = max(d_stats, float((mp_c - mp_b).abs().max()), float((sp_c - sp_b).abs().max()))
        r = ep.r_q
        au_c.append(metrics.auroc((-ccp_s_matrix(c, mq_c[r], sq_c[r], mp_c, sp_c).max(1).values).cpu().numpy()))
        au_b.append(metrics.auroc((-ccp_s_matrix(c, mq_b[r], sq_b[r], mp_b, sp_b).max(1).values).cpu().numpy()))
    res = {"host": a.host, "manifest": a.split, "max_abs_diff_m_s": d_stats, "auroc_closed_form": float(np.mean(au_c)), "auroc_brute_force": float(np.mean(au_b)),
           "max_abs_diff_episode_auroc": float(np.abs(np.array(au_c) - np.array(au_b)).max())}
    print(json.dumps(res, indent=1))
    p = REPO / "results" / "generated" / "fidelity_ccp_s.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
