"""Registered statistics of the foundation-model round (paper Section 4.5; docs/preregistration/foundation_round.md, Sections 7-8).

H1: one-sided Wilcoxon signed-rank p for "CCP-A - COSF AUROC > 0" per configuration, Holm step-down adjustment over the 32 configurations
and Bonferroni-simultaneous 95% CIs. H2 / H3 (H2b / H2 in the registration documents): Spearman rank correlation of SCG (validation /
test) with the test gain, one-sided permutation p with 100,000 permutations, overall and within the 8 dataset x shot groups. The
permutation p-value is (1 + #{rho_perm >= rho}) / (1 + n_perm), so its floor is 1 / (n_perm + 1), about 1e-5.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 32
PERM_SEED = 20261006            # H2, H3
STRAT_SEED = 20261007           # within-group H2, H3
N_PERM = 100_000


def wilcoxon_greater(d: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank p for median(d) > 0 (zero_method 'wilcox'); 1.0 if every difference is zero."""
    from scipy.stats import wilcoxon
    d = np.asarray(d, dtype=np.float64)
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue)


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order (ties keep the input order)."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def bonferroni_ci(d: np.ndarray, m: int = N_CELLS, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lower, upper): mean +- z_{1 - alpha / (2 m)} * std / sqrt(n) (std ddof 0, as ccp.metrics.ci95)."""
    from scipy.stats import norm
    d = np.asarray(d, dtype=np.float64)
    z = norm.ppf(1 - alpha / (2 * m))
    h = z * d.std() / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def spearman_perm(x, y, n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """Spearman rho and the one-sided permutation p for rho > 0: (1 + #{rho_perm >= rho}) / (1 + n_perm), y permuted."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rx, ry = rankdata(x), rankdata(y)

    def rho(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    r = rho(rx, ry)
    rng = np.random.RandomState(seed)
    ge = sum(rho(rx, ry[rng.permutation(len(ry))]) >= r - 1e-12 for _ in range(n_perm))
    return {"rho": r, "p_one_sided": (1 + ge) / (1 + n_perm), "n": int(len(x)), "n_perm": n_perm, "seed": seed}


def within_group_spearman_perm(x, y, groups, n_perm: int = N_PERM, seed: int = STRAT_SEED) -> dict:
    """The mean over groups of the within-group Spearman rho between x and y; null = independent permutations of y within each group
    (n_perm draws, RandomState(seed)); one-sided p = (1 + #{stat_perm >= stat}) / (1 + n_perm). Groups in order of first appearance."""
    from scipy.stats import rankdata
    x, y, groups = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)
    members = [np.where(groups == s)[0] for s in dict.fromkeys(groups.tolist())]

    def rho(a, b):
        a, b = rankdata(a) - (len(a) + 1) / 2, rankdata(b) - (len(b) + 1) / 2
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else 0.0
    per = [rho(x[g], y[g]) for g in members]
    stat = float(np.mean(per))
    rng = np.random.RandomState(seed)
    ge = 0
    for _ in range(n_perm):
        s = np.mean([rho(x[g], y[g][rng.permutation(len(g))]) for g in members])
        ge += s >= stat - 1e-12
    return {"stat_mean_within_rho": stat, "per_stratum_rho": per, "p_one_sided": (1 + ge) / (1 + n_perm), "n_strata": len(members),
            "n_perm": n_perm, "seed": seed}


def headroom_paired(a_row, a_cosf) -> dict:
    """Registered descriptive headroom ratio (amendment A5): per-episode h_e = (AUROC_row - AUROC_COSF) / (100 - AUROC_COSF), AUROC in %;
    episodes with AUROC_COSF = 100 are excluded and counted. Mean, 95% CI (1.96 std / sqrt(n), ddof 0), fraction with h > 0."""
    from ccp import metrics
    a_row, a_cosf = np.asarray(a_row, dtype=np.float64), np.asarray(a_cosf, dtype=np.float64)
    ok = a_cosf < 100.0
    h = (a_row[ok] - a_cosf[ok]) / (100.0 - a_cosf[ok])
    return {"mean": float(h.mean()) if h.size else float("nan"), "ci95": metrics.ci95(h) if h.size else float("nan"),
            "frac_gain": float((h > 0).mean()) if h.size else float("nan"), "n_used": int(ok.sum()), "n_excluded": int((~ok).sum())}


def headroom_cell(mean_row: float, mean_cosf: float) -> float:
    """Configuration-level headroom ratio: (mean AUROC_row - mean AUROC_COSF) / (100 - mean AUROC_COSF)."""
    return float((mean_row - mean_cosf) / (100.0 - mean_cosf))


def h1_table(cells: dict) -> dict:
    """cells: {configuration: per-episode CCP-A - COSF AUROC differences}. Per configuration: mean, 95% CI, Bonferroni CI, one-sided p,
    Holm-adjusted p."""
    from ccp import metrics
    names = list(cells)
    p = [wilcoxon_greater(cells[c]) for c in names]
    ph = holm(p)
    out = {}
    for c, pi, phi in zip(names, p, ph):
        d = np.asarray(cells[c], dtype=np.float64)
        mean, lo, hi = bonferroni_ci(d, len(names))
        out[c] = {"mean": mean, "ci95": metrics.ci95(d), "bonferroni_lo": lo, "bonferroni_hi": hi, "p_one_sided": pi, "p_holm": phi,
                  "frac_gain": float((d > 0).mean())}
    return {"cells": out, "n_holm_significant": int(sum(x["p_holm"] < 0.05 and x["mean"] > 0 for x in out.values())),
            "n_bonferroni_ci_excludes_zero": int(sum(x["bonferroni_lo"] > 0 for x in out.values())), "n_cells": len(names)}
