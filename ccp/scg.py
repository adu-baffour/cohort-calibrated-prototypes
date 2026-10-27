"""The stratified cohort gap SCG, a validation diagnostic (paper Section 3.5, Eq. 8). The registration documents and the recorded result
files use an older name and older keys for it (docs/naming.md).

For one configuration (host, dataset, shot), with x(q) = max_k c_k(q) (mirror views) and m(q) = m_K(q), K = 1000:
  - pool the known and unknown queries of all episodes and divide them into 10 bins at the deciles of x (np.quantile, linear
    interpolation); a value on a bin edge goes to the upper bin, the maximum to the last bin;
  - per bin j (Eq. 8): d_j = (mean m of unknown - mean m of known) / s_j, s_j the pooled within-bin standard deviation
    s_j^2 = ((n_kn - 1) s_kn^2 + (n_unk - 1) s_unk^2) / (n_kn + n_unk - 2), sample variances (ddof 1);
  - bins with fewer than two known or two unknown queries, or with s_j = 0, are excluded;
  - SCG = the unweighted mean of d_j over the valid bins.
The raw gap (unstandardized mean difference) is reported alongside.
"""
from __future__ import annotations

import numpy as np

N_BINS = 10


def scg_bins(x: np.ndarray, m: np.ndarray, y: np.ndarray) -> list[dict]:
    """Per decile bin of x (y = 1 unknown, 0 known): sizes, mean m per group, the raw gap unknown - known, the pooled within-bin SD and the
    standardized gap d_j (Eq. 8); NaN where a bin is excluded."""
    x, m, y = (np.asarray(a).ravel() for a in (x, m, y))
    edges = np.quantile(x, np.linspace(0, 1, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_BINS - 1)
    out = []
    for i in range(N_BINS):
        k, u = m[(b == i) & (y == 0)], m[(b == i) & (y == 1)]
        gap = float(u.mean() - k.mean()) if k.size and u.size else float("nan")
        sd = (float(np.sqrt(((k.size - 1) * k.var(ddof=1) + (u.size - 1) * u.var(ddof=1)) / (k.size + u.size - 2)))
              if k.size >= 2 and u.size >= 2 else float("nan"))
        out.append({"bin": i + 1, "x_lo": float(edges[i]), "x_hi": float(edges[i + 1]), "x_mean": float(x[b == i].mean()) if (b == i).any() else float("nan"),
                    "n_known": int(k.size), "n_unknown": int(u.size),
                    "m_known": float(k.mean()) if k.size else float("nan"), "m_unknown": float(u.mean()) if u.size else float("nan"),
                    "gap": gap, "sd_pooled": sd, "std_gap": gap / sd if np.isfinite(sd) and sd > 0 else float("nan")})
    return out


def scg_gap(bins: list[dict], standardized: bool = True) -> float:
    """SCG: the mean over valid bins of the standardized gap (registered) or of the raw gap (reported)."""
    g = [b["std_gap" if standardized else "gap"] for b in bins]
    return float(np.mean([x for x in g if np.isfinite(x)]))


def scg_per_shot(x: np.ndarray, m: np.ndarray) -> dict:
    """Standardized and raw SCG of one set of episodes in the FSOR layout (last 75 queries unknown)."""
    y = np.tile(np.r_[np.zeros(75), np.ones(75)], (x.shape[0], 1))
    bins = scg_bins(x, m, y)
    return {"bins": bins, "std_gap": scg_gap(bins), "raw_gap": scg_gap(bins, standardized=False)}


def scg_reading(gap_mini: float, gap_tiered: float) -> str:
    """The reading registered for the variants round, on the standardized gaps (mean over bins, then over shots): tieredImageNet at most
    1/3 of mini-ImageNet's -> little signal on tieredImageNet; otherwise the signal exists."""
    if not gap_mini > 0:
        return "undefined (mini gap is not positive)"
    if gap_tiered <= gap_mini / 3:
        return "the base cohort carries little signal on tiered"
    return "the signal exists; the weighting or fusion is the limitation"
