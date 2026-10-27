"""Baseline NNN, nearest-neighbor normalization (Chowdhury et al., EMNLP 2024; Section 4.4).

s_k = c_k - alpha m_k(p_k), m_k the mean of the k largest cosines of p_k to the bank; alpha in {0.25, 0.5, 0.75, 1} x k in {100, 1000}.
(alpha = 1, k = 1000 is CCP-A's prototype-only ablation.)
"""
from ccp.scores.base import register_score


def _make(alpha, k):
    def f(ep):
        return ep.cos(True) - alpha * ep.m_p(k)[None, :]
    f.__doc__ = f"NNN, alpha = {alpha}, k = {k}."
    return f


for _a in (0.25, 0.5, 0.75, 1.0):
    for _k in (100, 1000):
        register_score(f"NNN/alpha={_a}/k={_k}")(_make(_a, _k))
