"""Metrics (paper Section 4.5). Episode layout everywhere: queries 0..74 known (class-major, 15 per class), 75..149 unknown.

  accuracy   fraction of known queries correctly classified before rejection (%)
  AUROC      per episode over all 150 queries, unknown = positive (%), the Mann-Whitney rank statistic with average ranks for ties
             (identical to sklearn.metrics.roc_auc_score; tests/test_metrics.py)
  OSCR       area under the CCR-FPR curve (Dhamija et al., 2018): a query is accepted when u(q) <= tau; equals AUROC when every known
             query is correct
  FPR95      fraction of unknown queries accepted at the smallest threshold that accepts at least 95% of the known queries: the 72nd
             smallest of the 75 known scores (ties accepted); lower is better
  CI         95% half-width over n episodes = 1.96 sd / sqrt(n), sd with ddof 0
  paired     per-episode differences of two rows on the same episodes: mean, 95% CI, two-sided Wilcoxon signed-rank p
             (scipy.stats.wilcoxon, zero_method="wilcox"), fraction of positive differences (and of ties)
  seeds      mean and sample standard deviation (ddof 1) over seeds
"""
from __future__ import annotations

import numpy as np

N_KNOWN = 75
N_UNKNOWN = 75


def auroc(scores: np.ndarray, labels: np.ndarray | None = None) -> float:
    """AUROC in percent, higher score = more likely unknown. labels: 1 = unknown (positive); default = the FSOR layout
    (first 75 known, last 75 unknown). Same arithmetic as the recorded runs: (R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg) * 100."""
    from scipy.stats import rankdata
    s = np.asarray(scores)
    y = np.r_[np.zeros(N_KNOWN, bool), np.ones(N_UNKNOWN, bool)] if labels is None else np.asarray(labels).astype(bool)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ")
    r = rankdata(s, method="average")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if labels is None:  # contiguous positives: the recorded code summed r[75:]
        r_pos = r[N_KNOWN:].sum()
    else:
        r_pos = r[y].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) * 100)


def oscr(scores: np.ndarray, correct_known: np.ndarray, unknown: np.ndarray | None = None) -> float:
    """OSCR in percent (Dhamija et al., NeurIPS 2018; docs/preregistration/final_round.md). A query is ACCEPTED as known at threshold tau
    when its open-set score u <= tau (higher u = more likely unknown). CCR(tau) = fraction of known queries that are accepted AND correctly
    labelled; FPR(tau) = fraction of unknown queries accepted. Thresholds run over -inf and every distinct score; the curve goes from
    (0, 0) to (1, accuracy) and its area over FPR is the trapezoidal sum (tied known / unknown scores give a straight segment, the same
    convention as the average-rank AUROC: with every known query correct, OSCR == AUROC).
    scores (n,), unknown (n,) bool (default: the FSOR layout, last 75 unknown), correct_known (n_known,) bool in the order of the known queries."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.r_[np.zeros(N_KNOWN, bool), np.ones(N_UNKNOWN, bool)] if unknown is None else np.asarray(unknown, bool)
    c = np.zeros(s.size, bool)
    c[~y] = np.asarray(correct_known, bool)
    n_k, n_u = int((~y).sum()), int(y.sum())
    order = np.argsort(s, kind="mergesort")
    s, y, c = s[order], y[order], c[order]
    last = np.r_[s[1:] != s[:-1], True]                 # end of each group of equal scores
    ccr = np.r_[0.0, np.cumsum(c)[last] / n_k]
    fpr = np.r_[0.0, np.cumsum(y)[last] / n_u]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (ccr[1:] + ccr[:-1]) / 2) * 100)


def fpr95(scores: np.ndarray, unknown: np.ndarray | None = None) -> float:
    """FPR95 in percent: the fraction of unknown queries accepted (u <= tau) at the smallest tau that accepts at least 95% of the known
    queries, i.e. tau = the ceil(0.95 n_known)-th smallest known score (72nd of 75); ties with tau count as accepted. Lower is better."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.r_[np.zeros(N_KNOWN, bool), np.ones(N_UNKNOWN, bool)] if unknown is None else np.asarray(unknown, bool)
    known = np.sort(s[~y])
    tau = known[int(np.ceil(0.95 * known.size - 1e-12)) - 1]
    return float((s[y] <= tau).mean() * 100)


def episode_oscr(S: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """S (n_episodes, 150) open-set scores, correct (n_episodes, 75) bool per known query -> per-episode OSCR."""
    return np.array([oscr(s, c) for s, c in zip(np.asarray(S), np.asarray(correct))])


def episode_fpr95(S: np.ndarray) -> np.ndarray:
    return np.array([fpr95(s) for s in np.asarray(S)])


def episode_auroc(S: np.ndarray) -> np.ndarray:
    """S (n_episodes, 150) -> per-episode AUROC (n_episodes,)."""
    return np.array([auroc(s) for s in np.asarray(S)])


def accuracy(pred_known: np.ndarray, target: np.ndarray | None = None) -> float:
    """Percent correct over the known queries; default target = class-major 5 x 15."""
    p = np.asarray(pred_known)
    t = np.repeat(np.arange(5), 15) if target is None else np.asarray(target)
    return float((p == t).mean() * 100)


def ci95(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(1.96 * v.std() / np.sqrt(len(v)))


def summarise(per_episode_auroc: np.ndarray, per_episode_acc: np.ndarray) -> dict:
    a, c = np.asarray(per_episode_auroc), np.asarray(per_episode_acc)
    return {"auroc": float(a.mean()), "auroc_ci95": ci95(a), "acc": float(c.mean()), "acc_ci95": ci95(c), "n_episodes": int(len(a))}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Row a minus row b on the same episodes."""
    from scipy.stats import wilcoxon
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    p = float(wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    return {"mean": float(d.mean()), "ci95": ci95(d), "wilcoxon_p": p, "frac_gain": float((d > 0).mean()), "frac_tie": float((d == 0).mean()),
            "n_episodes": int(len(d))}


def mean_std(xs) -> dict:
    """Mean and sample std (ddof 1) over seeds, as in the recorded seed summaries."""
    import statistics
    xs = [float(x) for x in xs]
    return {"mean": statistics.mean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "per_seed": xs}
