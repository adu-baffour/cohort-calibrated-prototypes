"""Baseline kNN (Sun, Ming, Zhu, Li, ICML 2022): cosine to the k-th nearest bank image, as unknown evidence (Section 4.4).

s_(k)(q) = k-th largest cos(z(q), b), b in the bank, k in {10, 100, 1000}. sign +: e = -s_(k) (far from the base images = unknown, the
direction of Sun et al.); sign -: e = +s_(k). Both signs are considered on validation. Fused with the reference score as the query
cohort term.
"""
from ccp.scores.base import register_evidence


def _make(k, sign):
    def f(ep):
        s = ep.tf.kth_query(k)[ep.r_q]
        return -s if sign == "+" else s
    f.__doc__ = f"kNN evidence, k = {k}, sign {sign}."
    return f


for _k in (10, 100, 1000):
    for _s in ("+", "-"):
        register_evidence(f"kNN/k={_k}/sign={_s}")(_make(_k, _s))
