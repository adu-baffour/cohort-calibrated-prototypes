"""Test round (viii), the foundation-model round (paper Section 5.5; docs/preregistration/foundation_round.md): the scorer for the frozen
foundation-model hosts (CLIP ViT-B/16, DINOv2 ViT-S/14, ViT-B/14, ViT-L/14). The registered statistics (H1, H2, H3) are in ccp.stats.

score_frozen embeds each target split ONCE per host (the encoder is the same for both shots) and scores every requested manifest with the
rows of ccp.rounds.final_round.episode_rows (COS, COSF, CCP-A, CCP-QW and the nine baselines); per-episode metrics come from
ccp.rounds.final_round (per_episode, summary, paired_vs), so the arithmetic is the registered one.
"""
from __future__ import annotations

import numpy as np

from ccp.rounds import final_round as FR
from ccp.stats import N_CELLS, N_PERM, PERM_SEED, STRAT_SEED  # noqa: F401  (the registered constants of this round)


def score_frozen(host_name: str, jobs: list[tuple[int, str, str]], rows: list[str], evidence: set = frozenset(), need_scg: bool = True,
                 allow_test: bool = False, log=print) -> dict:
    """GPU. jobs: [(shot, dataset, manifest)]. Returns {(shot, dataset, manifest): {'rows', 'own': None, 'scg', 'n'}} in the format of
    ccp.rounds.final_round.score_host; each (dataset, split) is embedded once."""
    import torch

    from ccp.episodes import Episodes
    from ccp.hosts import build_host, load_host_config
    from ccp.pipeline import Target, load_bank
    from ccp.scores import EpisodeContext
    cfg = load_host_config(host_name)
    host = build_host(cfg)
    bank = load_bank(host, allow_build=False, log=log)
    out, targets = {}, {}
    tgt = torch.arange(FR.WAY, device=host.device).repeat_interleave(FR.NQ)
    for sh, ds, m in jobs:
        eps = Episodes(ds, m, allow_test=allow_test)
        split = "test" if eps.split == "test" else "val"
        if (ds, split) not in targets:
            for k in list(targets):
                del targets[k]
            torch.cuda.empty_cache()
            targets[(ds, split)] = Target(host, split, bank, dataset=ds)
        target = targets[(ds, split)]
        R = {n: {"u": np.zeros((eps.n, 150)), "correct": np.zeros((eps.n, 75), bool), "acc": np.zeros(eps.n)} for n in rows}
        X, M = (np.zeros((eps.n, 150)), np.zeros((eps.n, 150))) if need_scg else (None, None)
        with torch.no_grad():
            for e in range(eps.n):
                ep = EpisodeContext(target.tf, target.rows(eps.indices(e)), eps.shot)
                for n, (u, pred) in FR.episode_rows(ep, rows, set(evidence)).items():
                    R[n]["u"][e] = u.cpu().numpy()
                    if pred is not None:
                        hit = pred[:75] == tgt
                        R[n]["correct"][e] = hit.cpu().numpy()
                        R[n]["acc"][e] = hit.float().mean().item() * 100
                if need_scg:
                    X[e], M[e] = ep.cos(True).max(1).values.cpu().numpy(), ep.m_q().cpu().numpy()
        out[(sh, ds, m)] = {"rows": R, "own": None, "n": eps.n, "scg": {"x": X, "m": M} if need_scg else None}
        log(f"{host_name} shot {sh} {ds}/{m}: {len(rows)} rows")
    del targets, host
    torch.cuda.empty_cache()
    return out
