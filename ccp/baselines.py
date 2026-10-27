"""Training-free baselines (paper Section 4.4; docs/preregistration/baselines.md; registered names in docs/naming.md): grids in tie order, selection,
and the one scoring function shared by the validation selection and report (scripts/rounds/baselines.py), the bundle dry run and the
registered test pass of round (v).

Rows per host:
  frozen host: a matrix baseline alone (its own label); an evidence baseline (query cohort term, kNN) fused with COSF by fixed-weight
               fusion, COSF + w e (COSF label)
  trained host (SnaTCHer-F, DNPG): fused[b] = host score + w u_b (host label); u_b = -max_k s_k (matrix) or e (evidence)
  references: 'host' (COSF | S | D), 'COS' (frozen hosts), 'CCP' (CCP-A | H_fix with the frozen weights of configs/hosts)
Selection: per host and dataset, the setting with the highest mean valA AUROC over both shots (and DNPG seeds); ties within 1e-9 go to
the earlier setting of the grid. QB-Norm IS and QB-Norm DIS are selected separately (two families of the registration).
"""
from __future__ import annotations

import numpy as np

from ccp import fusion, metrics

# family -> settings in TIE ORDER (the first of equal values wins); registration order 1-9
FAMILIES = {
    "query cohort term": ["query cohort term"],
    "kNN": [f"kNN/k={k}/sign=+" for k in (10, 100, 1000)] + [f"kNN/k={k}/sign=-" for k in (10, 100, 1000)],
    "RMD": [f"RMD/delta={x}" for x in (0.1, 0.5, 1.0)],
    "CL2N": ["CL2N"],
    "ZN": ["ZN"],
    "NNN": [f"NNN/alpha={a}/k={k}" for a in (0.25, 0.5, 0.75, 1.0) for k in (100, 1000)],
    "QB-Norm IS": [f"QB-Norm IS/temperature={t}" for t in (10, 20, 50)],
    "QB-Norm DIS": [f"QB-Norm DIS/temperature={t}" for t in (10, 20, 50)],
    "TEEN": ["TEEN"],
    "episode CSLS": ["episode CSLS"],
}
TRANSDUCTIVE = {"episode CSLS"}
EVIDENCE = {"query cohort term"} | set(FAMILIES["kNN"])
ALL_SETTINGS = [v for vs in FAMILIES.values() for v in vs]
HOST_GROUPS = {"frozen FEAT mini": ["frozen_feat_mini"], "frozen FEAT tiered": ["frozen_feat_tiered"], "SnaTCHer-F mini": ["snatcher_f_mini"],
               "DNPG mini": ["dnpg_mini_seed0", "dnpg_mini_seed1", "dnpg_mini_seed2"]}


def val_sets(dataset: str, shot: int) -> dict:
    """Manifest names of valA and valB (tieredImageNet's files are named tvA / tvB)."""
    a, b = ("valA", "valB") if dataset == "mini" else ("tvA", "tvB")
    return {"A": f"{a}_fsor_5way_{shot}shot", "B": f"{b}_fsor_5way_{shot}shot"}


def raw_scores(host_name: str, shot: int, manifests: list[str], settings: list[str], allow_test: bool = False, log=print) -> dict:
    """GPU. For each manifest: {'own': (n,150) | None, 'own_acc', 'COSF': (score, acc), 'COS': ..., 'CCP-A': ..., '<setting>': (score, acc|None)}.
    Scores are unknown scores (higher = unknown) for matrix baselines and the evidence e for evidence baselines."""
    import torch

    from ccp.episodes import Episodes
    from ccp.hosts import build_host, load_host_config
    from ccp.pipeline import Target, load_bank, score_host_own
    from ccp.scores import EpisodeContext, get_evidence, get_score
    cfg = load_host_config(host_name)
    trained = cfg["kind"] in ("snatcher_f", "dnpg")
    host = build_host(cfg, shot=shot if trained else None)
    host.shot = shot
    bank = load_bank(host, allow_build=False, log=log)
    out = {}
    split = None
    target = None
    for m in manifests:
        eps = Episodes(cfg["dataset"], m, allow_test=allow_test)
        sp = "test" if eps.split == "test" else "val"
        if sp != split:
            del target
            target = Target(host, sp, bank, need_pooled=cfg["kind"] == "snatcher_f")
            split = sp
        res = {}
        if trained:
            own, acc = score_host_own(host, target, eps)
            res["own"], res["own_acc"] = own["S" if cfg["kind"] == "snatcher_f" else "D"], acc
        names = ["COS", "COSF", "CCP-A"] + [v for v in settings if v not in EVIDENCE]
        ev = [v for v in settings if v in EVIDENCE]
        S = {n: np.zeros((eps.n, 150)) for n in names + ev}
        A = {n: np.zeros(eps.n) for n in names}
        tgt = torch.arange(5, device=host.device).repeat_interleave(15)
        with torch.no_grad():
            for e in range(eps.n):
                ep = EpisodeContext(target.tf, target.rows(eps.indices(e)), eps.shot)
                for n in names:
                    M = get_score(n)(ep)
                    S[n][e] = (-M.max(1).values).cpu().numpy()
                    A[n][e] = (M.argmax(1)[:75] == tgt).float().mean().item() * 100
                for n in ev:
                    S[n][e] = get_evidence(n)(ep).cpu().numpy()
        for n in names:
            res[n] = (S[n], A[n])
        for n in ev:
            res[n] = (S[n], None)
        out[m] = res
        log(f"{host_name} shot {shot} {m}: scored {len(settings)} settings")
    del target, host
    torch.cuda.empty_cache()
    return out


def fit_weights(cfg_kind: str, rawA: dict, settings: list[str]) -> dict:
    """w_b on valA: sigma_A(reference) / sigma_A(u_b); reference = the host score (trained) or COSF (frozen; evidence baselines only)."""
    trained = cfg_kind in ("snatcher_f", "dnpg")
    w = {}
    for v in settings:
        if trained:
            w[v] = fusion.fit_weight(rawA["own"], rawA[v][0])["w"]
        elif v in EVIDENCE:
            w[v] = fusion.fit_weight(rawA["COSF"][0], rawA[v][0])["w"]
    return w


def rows(cfg_kind: str, raw: dict, settings: list[str], weights: dict, ccp_w: float | None) -> dict:
    """{row name: (score (n,150), acc (n,))} for the references and the baselines (the reported rows)."""
    trained = cfg_kind in ("snatcher_f", "dnpg")
    out = {}
    if trained:
        own, acc = raw["own"], raw["own_acc"]
        out["host"] = (own, acc)
        out["CCP"] = (fusion.h_fix(own, raw["CCP-A"][0], ccp_w), acc)
        for v in settings:
            out[v] = (fusion.h_fix(own, raw[v][0], weights[v]), acc)
    else:
        out["host"] = raw["COSF"]
        out["COS"] = raw["COS"]
        out["CCP"] = raw["CCP-A"]
        for v in settings:
            out[v] = (raw["COSF"][0] + weights[v] * raw[v][0], raw["COSF"][1]) if v in EVIDENCE else raw[v]
    return out


def summarise(rws: dict) -> tuple[dict, dict]:
    pe = {k: metrics.episode_auroc(s) for k, (s, _) in rws.items()}
    return {k: metrics.summarise(pe[k], rws[k][1]) for k in rws}, pe


def choose(valA_auroc: dict) -> dict:
    """valA_auroc: {setting: mean AUROC over shots (and seeds)} -> {family: chosen setting}; ties within 1e-9 go to the earlier setting."""
    chosen = {}
    for fam, vs in FAMILIES.items():
        best, bv = None, -1e18
        for v in vs:
            if v in valA_auroc and valA_auroc[v] > bv + 1e-9:
                best, bv = v, valA_auroc[v]
        chosen[fam] = best
    return chosen
