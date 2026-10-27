"""Full AS-norm, a registered method variant of the variants round (Section 5.6): CCP-A additionally divided by the top-K standard
deviations. Same K = 1000 nearest bank cosines as CCP-A, mean and std (ddof 0):

s_k = 1/2 [(c_k - mu_K(q)) / sd_K(q) + (c_k - mu_K(p_k)) / sd_K(p_k)], sd floored at 1e-6; label argmax_k s_k, unknown score -max_k s_k.
mu_K is m_K (Eq. 3). Source: adaptive S-norm (Matejka et al., Interspeech 2017) with the top-K cohort.
"""
from ccp.cohort import REGISTERED_K, full_as_norm_matrix
from ccp.scores.base import register_score


@register_score("full AS-norm")
def full_as_norm(ep):
    mq, sq, _ = ep.topk_q(REGISTERED_K)
    mp, sp, _ = ep.topk_p(REGISTERED_K)
    return full_as_norm_matrix(ep.cos(True), mq, sq, mp, sp)
