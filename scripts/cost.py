"""Cost measurement (paper Table 8), validation episodes only (mini-ImageNet valA). GPU. Writes results/generated/cost.json; Table 8 is
rendered from the recorded measurement (results/recorded/validation/fsor_cost.json) by scripts/make_tables.py.

    python scripts/cost.py [--episodes 200] [--warmup 20]

Protocol (as recorded): frozen FEAT (eval view) for COSF / CCP-A / CCP-S; DNPG seed 0 max_acc.pth (whole view) for its own score D and for
DNPG + CCP. Per-episode times: mean over `--episodes` valA episodes per shot after `--warmup` warm-up episodes, torch.cuda.synchronize around
each timed block; extra memory = torch.cuda.max_memory_allocated inside the block minus the memory allocated before it. The shared backbone
forward (support + 150 queries, both flip views) is timed separately. Parameter counts from the DNPG model; one-off bank embedding times.
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
from ccp.cohort import closed_form_stats, m_topk, s_norm_stats, unit_flip  # noqa: E402
from ccp.episodes import Episodes  # noqa: E402
from ccp.hosts import build_host  # noqa: E402
from ccp.hosts.base import gap  # noqa: E402
from ccp.pipeline import host_images  # noqa: E402


def timed(fn):
    import torch
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
    return r, time.perf_counter() - t, torch.cuda.max_memory_allocated() - base


def main():
    import torch
    import torch.nn.functional as F
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    res = {"gpu": torch.cuda.get_device_name(0), "protocol": __doc__.split("\n\n")[1].strip()}
    host = build_host("frozen_feat_mini")
    base = host_images(host, "train")
    xb = torch.from_numpy(np.array(base)).cuda()
    (b0, bf), t_embed, _ = timed(lambda: (gap(host.encoder_forward, xb), gap(host.encoder_forward, xb.flip(2))))
    xb = None
    Bf = unit_flip(b0, bf)
    bbar, C = closed_form_stats(Bf)
    res["cohort"] = {"n_base": int(Bf.shape[0]), "embed_seconds_both_views": t_embed, "ccpA_bank_MB_fp16": Bf.numel() * 2 / 2**20,
                     "ccpA_bank_MB_fp32": Bf.numel() * 4 / 2**20, "ccpS_stats_MB_fp32": (bbar.numel() + C.numel()) * 4 / 2**20}
    val = host_images(host, "val")
    per = {}
    for sh in (1, 5):
        eps = Episodes("mini", f"valA_fsor_5way_{sh}shot")
        T = {k: [] for k in ("backbone", "COSF", "CCP-A", "CCP-S")}
        Mm = {k: 0 for k in T}
        for e in range(a.warmup + a.episodes):
            x = torch.from_numpy(np.array(val[eps.indices(e) - 38400])).cuda()
            (z0, zf), tb, mb = timed(lambda: (gap(host.encoder_forward, x), gap(host.encoder_forward, x.flip(2))))
            ns = 5 * sh
            q = z0 + zf
            P = F.normalize(q[:ns].view(5, sh, -1).mean(1), dim=1)
            U = F.normalize(q[ns:], dim=1)
            _, tc, mc = timed(lambda: (U @ P.T).max(1).values)
            _, ta, ma = timed(lambda: ((U @ P.T) - 0.5 * (m_topk(P, Bf)[None] + m_topk(U, Bf)[:, None])).max(1).values)

            def ccps():
                c = U @ P.T
                mq, sq = s_norm_stats(U, bbar, C)
                mp, sp = s_norm_stats(P, bbar, C)
                return (0.5 * ((c - mp[None]) / sp[None] + (c - mq[:, None]) / sq[:, None])).max(1).values
            _, ts, ms = timed(ccps)
            if e >= a.warmup:
                for k, v, m in (("backbone", tb, mb), ("COSF", tc, mc), ("CCP-A", ta, ma), ("CCP-S", ts, ms)):
                    T[k].append(v); Mm[k] = max(Mm[k], m)
        per[f"{sh}shot"] = {k: {"ms_per_episode": float(np.mean(v) * 1000), "peak_extra_MB": Mm[k] / 2**20} for k, v in T.items()}
    host = None
    torch.cuda.empty_cache()
    for sh in (1, 5):
        dn = build_host("dnpg_mini_seed0", shot=sh)
        whole = host_images(dn, "val")
        if sh == 1:
            res["dnpg_params"] = {"total": sum(p.numel() for p in dn.net.parameters()), "backbone_finetuned": sum(p.numel() for p in dn.net.feature.parameters()),
                                  "new_modules_trainable": sum(p.numel() for n, p in dn.net.named_parameters() if not n.startswith("feature.") and p.requires_grad)}
            wb = torch.from_numpy(np.array(host_images(dn, "train"))).cuda()
            (_, _), t_d, _ = timed(lambda: (gap(dn.encoder_forward, wb), gap(dn.encoder_forward, wb.flip(2))))
            db0, dbf = gap(dn.encoder_forward, wb), gap(dn.encoder_forward, wb.flip(2))
            del wb
            DBf = unit_flip(db0, dbf)
            res["cohort"]["dnpg_embed_seconds_both_views"] = t_d
        eps = Episodes("mini", f"valA_fsor_5way_{sh}shot").subset(a.warmup + a.episodes)
        Td, Tc, md, mc = [], [], 0, 0
        for e in range(eps.n):
            one = eps.subset(e + 1)
            for k in ("known", "support", "query", "unknown", "unk_query"):
                setattr(one, k, getattr(eps, k)[e:e + 1])
            one.n = 1
            _, td, mdd = timed(lambda: dn.own_scores(one, lambda g: whole[g - 38400]))
            x = torch.from_numpy(np.array(whole[eps.indices(e) - 38400])).cuda()

            def ccp_on_dnpg():
                z0, zf = gap(dn.encoder_forward, x), gap(dn.encoder_forward, x.flip(2))
                q = z0 + zf
                ns = 5 * sh
                P = F.normalize(q[:ns].view(5, sh, -1).mean(1), dim=1)
                U = F.normalize(q[ns:], dim=1)
                return ((U @ P.T) - 0.5 * (m_topk(P, DBf)[None] + m_topk(U, DBf)[:, None])).max(1).values
            _, tcc, mcc = timed(ccp_on_dnpg)
            if e >= a.warmup:
                Td.append(td); Tc.append(tcc); md = max(md, mdd); mc = max(mc, mcc)
        per[f"{sh}shot"]["DNPG (own score, incl. its forward with 5x support augmentation)"] = {"ms_per_episode": float(np.mean(Td) * 1000), "peak_extra_MB": md / 2**20}
        per[f"{sh}shot"]["CCP-A on DNPG features (extra over DNPG: flip-view forward + CCP-A)"] = {"ms_per_episode": float(np.mean(Tc) * 1000), "peak_extra_MB": mc / 2**20}
        dn = None
        torch.cuda.empty_cache()
    res["per_episode"] = per
    out = REPO / "results" / "generated" / "cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
