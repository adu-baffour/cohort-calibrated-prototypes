"""Fusion of a trained host's own unknown score D with a cohort score u (paper Section 3.4, Eq. 7).

Fixed-weight fusion (inductive; reported):
  H_fix = D + w u,   w = sigma_A(D) / sigma_A(u)
  sigma_A = standard deviation (ddof 0) of ALL query scores of ALL valA episodes of that host, seed and shot, concatenated (1,000 x 150
  values). One frozen constant per host, seed and shot, estimated once on valA (configs/hosts/*.yaml fusion.frozen_weights).
  u = CCP-A for H_fix; u = COSF for the plain-cosine control K_fix = D + w2 COSF (w2 = sigma_A(D) / sigma_A(COSF)).
  The same rule fuses each training-free baseline with the reference score (Section 4.4). All scores are "higher = more likely unknown".

Episode-wise standardization (transductive; excluded from the reported test results):
  z(D) + z(u), z standardizing with the mean and std of the episode's 150 query scores, so a query's fused score depends on the other
  queries of its episode. It reaches validation AUROC within 0.02 points of fixed-weight fusion (Section 3.4). Kept only to re-render
  and check the recorded validation and test records that contain it ("H" and "K" in the registration documents).
"""
from __future__ import annotations

import numpy as np

TRANSDUCTIVE_LABEL = "uses the other queries of the episode (transductive)"


def fit_weight(D_valA: np.ndarray, U_valA: np.ndarray) -> dict:
    """w = sigma_A(D) / sigma_A(U) over every valA query score (ddof 0). Arrays (n_episodes, 150), float64."""
    D, U = np.asarray(D_valA, dtype=np.float64), np.asarray(U_valA, dtype=np.float64)
    if D.shape != U.shape:
        raise ValueError(f"D {D.shape} and U {U.shape} differ")
    return {"w": float(D.std() / U.std()), "std_D": float(D.std()), "std_U": float(U.std()), "n_scores": int(D.size)}


def h_fix(D: np.ndarray, U: np.ndarray, w: float) -> np.ndarray:
    """Eq. 7: D + w U (w a frozen constant)."""
    return D + w * U


def standardize_episode(v: np.ndarray) -> np.ndarray:
    """Per-episode standardization over the episode's queries (axis 1; transductive)."""
    return (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-12)


def episode_standardization(D: np.ndarray, U: np.ndarray) -> np.ndarray:
    """z(D) + z(U) per episode (transductive)."""
    return standardize_episode(D) + standardize_episode(U)


def fused_rows(D: np.ndarray, A: np.ndarray, C: np.ndarray, w: float, w2: float) -> dict:
    """Every recorded row of a trained host, under the paper's names (ccp.naming.EPISODE_STANDARDIZATION gives the recorded keys)."""
    return {"D": D, "H_fix": D + w * A, "K_fix": D + w2 * C, "episode-wise standardization": episode_standardization(D, A),
            "episode-wise standardization, COSF control": episode_standardization(D, C), "CCP-A": A, "COSF": C}


def frozen_weights(host_cfg: dict, key: str) -> tuple[float, float]:
    """(w, w2) from a host config's frozen weight table (configs/hosts/*.yaml, copied from the recorded JSON with provenance)."""
    t = host_cfg["fusion"]["frozen_weights"][key]
    return float(t["w"]), float(t["w2"])
