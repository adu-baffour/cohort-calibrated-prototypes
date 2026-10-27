"""Shared-cohort correction, a registered method variant of the variants round (Section 5.6).

o_k = |N(q) & N(p_k)| / K, N(v) = indices of the K = 1000 nearest bank images of v; s_k = c_k - 1/2 [(1 - gamma o_k) m_K(q) + m_K(p_k)];
label argmax_k s_k, unknown score -max_k s_k. gamma = 0 is CCP-A (bitwise). Grid gamma in {0.25, 0.5, 1.0}; gamma = 0.25 selected on valA.
"""
from ccp.cohort import REGISTERED_K, cohort_overlap, shared_cohort_matrix
from ccp.scores.base import register_score

GAMMAS = (0.25, 0.5, 1.0)


def shared_cohort(ep, gamma: float):
    mq, _, iq = ep.topk_q(REGISTERED_K)
    mp, _, ip = ep.topk_p(REGISTERED_K)
    return shared_cohort_matrix(ep.cos(True), mq, mp, cohort_overlap(iq, ip, ep.tf.bank.B.shape[0]), gamma)


def _variant(gamma):
    def f(ep):
        return shared_cohort(ep, gamma)
    f.__doc__ = f"Shared-cohort correction with gamma = {gamma}."
    return f


for _g in GAMMAS:
    register_score(f"shared-cohort correction/gamma={_g}")(_variant(_g))
