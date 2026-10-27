"""The four registered method variants of the variants round (paper Section 5.6, Table 7(b); docs/prereg_variants.md; registered names in
docs/naming.md): grids in tie order, the scoring shared by valA selection, the valB / valC reports and the bundle dry run
(scripts/rounds/variants.py), and the pure selection and acceptance rules (unit-tested on synthetic numbers).

  CCP-QW (Eq. 6)               u = lambda m_K(q) - max_k [c_k - beta m_K(p_k)], CCP-A's label; selected lambda = 1, beta = 1/2
  full AS-norm                 CCP-A additionally divided by the top-K standard deviations (own label)
  shared-cohort correction     the query correction reduced by gamma times the cohort overlap (own label); selected gamma = 0.25
  fusion-weight factor         trained hosts only: H = D + factor * w * u_CCP-A, or CCP-A alone; selected per host

Rows per configuration (the reference = CCP-A on frozen hosts, H_fix = D + w CCP-A with the frozen weights of configs/hosts on trained hosts):
  frozen hosts   CCP-QW: u_QW with the CCP-A label; full AS-norm, shared-cohort correction: their own label; fusion-weight factor: n/a
  trained hosts  CCP-QW, full AS-norm, shared-cohort correction: D + w_v u_v (w_v = sigma_A(D) / sigma_A(u_v) on valA per host, seed and
                 shot); fusion-weight factor: D + factor w u_CCP-A or u_CCP-A alone; accuracy = the host's classifier in every row
Acceptance criteria (Section 5.6; numbered as in the paper): (i) mean valB gain >= +0.30; (ii) no configuration below -0.20;
(iii) frozen-host accuracy change >= -0.20; (iv) on valC, mean gain >= +0.15 with no configuration below -0.20.
"""
from __future__ import annotations

import numpy as np

from ccp import fusion, metrics

# ---------------------------------------------------------------------------------------------------------------- grids in TIE ORDER
QW_GRID = [(0.5, 0.5)] + [(lam, beta) for lam in (0.5, 0.75, 1.0, 1.25, 1.5) for beta in (0.0, 0.5) if (lam, beta) != (0.5, 0.5)]
CCP_QW = [f"CCP-QW/lambda={lam}/beta={beta}" for lam, beta in QW_GRID]
CCP_QW_NULL = CCP_QW[0]                                  # lambda = beta = 1/2 is CCP-A
FULL_AS_NORM = ["full AS-norm"]
SHARED_COHORT = [f"shared-cohort correction/gamma={g}" for g in (0.25, 0.5, 1.0)]
FUSION_WEIGHT_FACTOR = ["fusion-weight factor/factor=1", "fusion-weight factor/factor=0.5", "fusion-weight factor/factor=2",
                        "fusion-weight factor/factor=4", "fusion-weight factor/CCP-A alone"]   # factor 1 (registered H_fix) first
FUSION_WEIGHT_FACTOR_NULL = FUSION_WEIGHT_FACTOR[0]
FAMILIES = {"CCP-QW": CCP_QW, "full AS-norm": FULL_AS_NORM, "shared-cohort correction": SHARED_COHORT,
            "fusion-weight factor": FUSION_WEIGHT_FACTOR}
SCORED = CCP_QW + FULL_AS_NORM + SHARED_COHORT          # computed per episode; the fusion-weight factor rows are derived from CCP-A
HOST_GROUPS = {"frozen FEAT mini": ["frozen_feat_mini"], "frozen FEAT tiered": ["frozen_feat_tiered"], "SnaTCHer-F mini": ["snatcher_f_mini"],
               "DNPG mini": ["dnpg_mini_seed0", "dnpg_mini_seed1", "dnpg_mini_seed2"]}
FROZEN_GROUPS = ["frozen FEAT mini", "frozen FEAT tiered"]
TRAINED_GROUPS = ["SnaTCHer-F mini", "DNPG mini"]
SHOTS = (1, 5)
CELLS = [(g, sh) for g in HOST_GROUPS for sh in SHOTS]                       # the 8 configurations of the round
FACTOR_CELLS = [(g, sh) for g in TRAINED_GROUPS for sh in SHOTS]             # the 4 trained-host configurations
TIE = 1e-9
# acceptance thresholds (AUROC / accuracy points): criteria (i), (ii), (iii), (iv)
MIN_MEAN_GAIN, MIN_CONFIG_GAIN, MIN_ACC_CHANGE, MIN_VALC_GAIN = 0.30, -0.20, -0.20, 0.15


def qw_params(name: str) -> tuple[float, float]:
    return QW_GRID[CCP_QW.index(name)]


def factor_of(name: str) -> float | None:
    return None if name.endswith("CCP-A alone") else float(name.split("=")[1])


def family_of(v: str) -> str:
    return next(f for f, vs in FAMILIES.items() if v in vs)


def val_sets(dataset: str, shot: int) -> dict:
    """Manifest names of valA, valB and valC (tieredImageNet's valA / valB files are named tvA / tvB)."""
    a, b = ("valA", "valB") if dataset == "mini" else ("tvA", "tvB")
    return {"A": f"{a}_fsor_5way_{shot}shot", "B": f"{b}_fsor_5way_{shot}shot", "C": f"valC_fsor_5way_{shot}shot"}


# ---------------------------------------------------------------------------------------------------------------- per-episode scoring (GPU or CPU)
def episode_scores(ep) -> dict:
    """{name: (unknown score (150,) tensor, predicted label (150,) tensor)} for CCP-A and every scored variant of one episode.
    Raises if the top-K statistics behind full AS-norm / the shared-cohort correction differ in any bit from the registered m_K, or if
    CCP-QW at its null setting differs from CCP-A."""
    import torch

    from ccp.cohort import REGISTERED_K, ccp_a_matrix, ccp_qw_matrix, cohort_overlap, full_as_norm_matrix, shared_cohort_matrix
    from ccp.scores.shared_cohort import GAMMAS
    c, mq, mp = ep.cos(True), ep.m_q(REGISTERED_K), ep.m_p(REGISTERED_K)
    tq, tp = ep.topk_q(REGISTERED_K), ep.topk_p(REGISTERED_K)
    if not (torch.equal(tq[0], mq) and torch.equal(tp[0], mp)):
        raise RuntimeError("top-K mean differs from the registered m_K (chunking or kernel changed)")
    A = ccp_a_matrix(c, mq, mp)
    lab = A.argmax(1)
    out = {"CCP-A": (-A.max(1).values, lab)}
    for name, (lam, beta) in zip(CCP_QW, QW_GRID):
        out[name] = (-ccp_qw_matrix(c, mq, mp, lam, beta).max(1).values, lab)
    if not torch.equal(out[CCP_QW_NULL][0], out["CCP-A"][0]):
        raise RuntimeError("CCP-QW at lambda = beta = 1/2 differs from CCP-A")
    M = full_as_norm_matrix(c, tq[0], tq[1], tp[0], tp[1])
    out["full AS-norm"] = (-M.max(1).values, M.argmax(1))
    o = cohort_overlap(tq[2], tp[2], ep.tf.bank.B.shape[0])
    for g in GAMMAS:
        M = shared_cohort_matrix(c, mq, mp, o, g)
        out[f"shared-cohort correction/gamma={g}"] = (-M.max(1).values, M.argmax(1))
    return out


def raw_scores(host_name: str, shot: int, manifests: list[str], allow_test: bool = False, log=print) -> dict:
    """GPU. {manifest: {'own': (n,150) | None, 'own_acc': (n,) | None, name: (score (n,150), acc (n,)) for CCP-A and SCORED}}."""
    import torch

    from ccp.episodes import Episodes
    from ccp.hosts import build_host, load_host_config
    from ccp.pipeline import Target, load_bank, score_host_own
    from ccp.scores import EpisodeContext
    cfg = load_host_config(host_name)
    trained = cfg["kind"] in ("snatcher_f", "dnpg")
    host = build_host(cfg, shot=shot if trained else None)
    host.shot = shot
    bank = load_bank(host, allow_build=False, log=log)
    out, split, target = {}, None, None
    names = ["CCP-A"] + SCORED
    for m in manifests:
        eps = Episodes(cfg["dataset"], m, allow_test=allow_test)
        sp = "test" if eps.split == "test" else "val"
        if sp != split:
            del target
            target = Target(host, sp, bank, need_pooled=cfg["kind"] == "snatcher_f")
            split = sp
        res = {"own": None, "own_acc": None}
        if trained:
            own, acc = score_host_own(host, target, eps)
            res["own"], res["own_acc"] = own["S" if cfg["kind"] == "snatcher_f" else "D"], acc
        S = {n: np.zeros((eps.n, 150)) for n in names}
        A = {n: np.zeros(eps.n) for n in names}
        tgt = torch.arange(5, device=host.device).repeat_interleave(15)
        with torch.no_grad():
            for e in range(eps.n):
                ep = EpisodeContext(target.tf, target.rows(eps.indices(e)), eps.shot)
                for n, (u, lab) in episode_scores(ep).items():
                    S[n][e] = u.cpu().numpy()
                    A[n][e] = (lab[:75] == tgt).float().mean().item() * 100
        for n in names:
            res[n] = (S[n], A[n])
        out[m] = res
        log(f"{host_name} shot {shot} {m}: scored CCP-A and {len(SCORED)} variants")
    del target, host
    torch.cuda.empty_cache()
    return out


def scg_raw(host_name: str, manifest: str, log=print) -> dict:
    """GPU, SCG inputs on validation episodes only: per query of every episode, x = max_k c_k (mirror views), m = m_K(q), y = 1 for unknown."""
    import torch

    from ccp.episodes import Episodes
    from ccp.hosts import build_host, load_host_config
    from ccp.pipeline import Target, load_bank
    from ccp.scores import EpisodeContext
    cfg = load_host_config(host_name)
    host = build_host(cfg)
    bank = load_bank(host, allow_build=False, log=log)
    eps = Episodes(cfg["dataset"], manifest)
    if eps.split != "val":
        raise ValueError("SCG uses validation episodes only")
    target = Target(host, "val", bank)
    X, M = np.zeros((eps.n, 150)), np.zeros((eps.n, 150))
    with torch.no_grad():
        for e in range(eps.n):
            ep = EpisodeContext(target.tf, target.rows(eps.indices(e)), eps.shot)
            X[e], M[e] = ep.cos(True).max(1).values.cpu().numpy(), ep.m_q().cpu().numpy()
    del target, host
    torch.cuda.empty_cache()
    return {"x": X, "m": M, "y": np.tile(np.r_[np.zeros(75), np.ones(75)], (eps.n, 1))}


# ---------------------------------------------------------------------------------------------------------------- rows, weights
def fit_weights(raw_A: dict, variants: list[str]) -> dict:
    """Trained hosts: w_v = sigma_A(D) / sigma_A(u_v) over every valA query score (the registered H_fix rule), for the scored variants."""
    return {v: fusion.fit_weight(raw_A["own"], raw_A[v][0])["w"] for v in variants if v in SCORED}


def rows(raw: dict, trained: bool, variants: list[str], weights: dict, ccp_w: float | None) -> dict:
    """{'reference' and each variant: (score (n,150), acc (n,))}."""
    if not trained:
        out = {"reference": raw["CCP-A"]}
        for v in variants:
            if v in FUSION_WEIGHT_FACTOR:
                continue
            out[v] = raw[v] if v not in CCP_QW else (raw[v][0], raw["CCP-A"][1])
        return out
    D, acc, U = raw["own"], raw["own_acc"], raw["CCP-A"][0]
    out = {"reference": (fusion.h_fix(D, U, ccp_w), acc)}
    for v in variants:
        if v in FUSION_WEIGHT_FACTOR:
            f = factor_of(v)
            out[v] = (U.copy(), acc) if f is None else (D + (f * ccp_w) * U, acc)
        else:
            out[v] = (fusion.h_fix(D, raw[v][0], weights[v]), acc)
    return out


def per_episode(rws: dict) -> dict:
    return {k: (metrics.episode_auroc(s), np.asarray(a, dtype=np.float64)) for k, (s, a) in rws.items()}


# ---------------------------------------------------------------------------------------------------------------- pure rules
def choose(means: dict, order: list[str]) -> str:
    """The variant with the highest value; ties within 1e-9 go to the earlier entry of `order`."""
    best, bv = None, -1e18
    for v in order:
        if v in means and means[v] > bv + TIE:
            best, bv = v, means[v]
    return best


def select_global(cell_auroc: dict, order: list[str], cells=CELLS) -> tuple[str, dict]:
    """cell_auroc: {variant: {(group, shot): valA AUROC (DNPG: mean over seeds)}} -> (choice by the mean over `cells`, the means)."""
    means = {v: float(np.mean([cell_auroc[v][c] for c in cells])) for v in order if v in cell_auroc}
    return choose(means, order), means


def select_factor(cell_auroc: dict) -> dict:
    """Per trained host group: the fusion-weight factor option with the highest mean valA AUROC over its two shots."""
    return {g: choose({v: float(np.mean([cell_auroc[v][(g, sh)] for sh in SHOTS])) for v in FUSION_WEIGHT_FACTOR}, FUSION_WEIGHT_FACTOR)
            for g in TRAINED_GROUPS}


def cell_paired(pe_var: list[np.ndarray], pe_ref: list[np.ndarray]) -> dict:
    """Per-episode AUROC difference variant - reference, averaged over the hosts of the configuration (DNPG: seeds)."""
    d = np.mean([a - b for a, b in zip(pe_var, pe_ref)], axis=0)
    return metrics.paired(d, np.zeros_like(d))


def criteria_valB(gain: dict, acc_change: dict, cells: list) -> dict:
    """Criteria (i)-(iii) on valB. gain: {configuration: mean paired AUROC change}; acc_change: {configuration: accuracy change, frozen hosts}."""
    g = [gain[c] for c in cells]
    frozen = [c for c in cells if c[0] in FROZEN_GROUPS]
    out = {"mean_gain": float(np.mean(g)), "min_cell_gain": float(min(g)),
           "min_frozen_acc_change": float(min(acc_change[c] for c in frozen)) if frozen else None}
    out["(i)"] = out["mean_gain"] >= MIN_MEAN_GAIN
    out["(ii)"] = out["min_cell_gain"] >= MIN_CONFIG_GAIN
    out["(iii)"] = True if not frozen else out["min_frozen_acc_change"] >= MIN_ACC_CHANGE
    out["pass_i_iii"] = out["(i)"] and out["(ii)"] and out["(iii)"]
    return out


def criterion_valC(gain: dict, cells: list) -> dict:
    """Criterion (iv) on valC: mean paired AUROC gain >= +0.15 and no configuration below -0.20."""
    g = [gain[c] for c in cells]
    out = {"mean_gain": float(np.mean(g)), "min_cell_gain": float(min(g))}
    out["(iv)"] = out["mean_gain"] >= MIN_VALC_GAIN and out["min_cell_gain"] >= MIN_CONFIG_GAIN
    return out


def candidate(results: dict, eligible: dict) -> str | None:
    """Among the families meeting (i)-(iii) (and eligible: CCP-QW is not, if its selection is CCP-A itself), the highest valB mean gain;
    ties to the earlier family. None if no family qualifies."""
    best, bv = None, -1e18
    for f in FAMILIES:
        if f in results and eligible.get(f, True) and results[f]["pass_i_iii"] and results[f]["mean_gain"] > bv + TIE:
            best, bv = f, results[f]["mean_gain"]
    return best


def promotion(cand: str | None, valC: dict) -> str | None:
    """The candidate is selected only if it meets (iv); otherwise nothing (no fallback to the next family)."""
    return cand if cand is not None and valC[cand]["(iv)"] else None
