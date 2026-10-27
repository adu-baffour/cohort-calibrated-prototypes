"""CCP-S (Section 3.3): S-norm over all base images in closed form (mean and standard deviation of each vector's cosines to the bank)."""
from ccp.cohort import ccp_s_matrix
from ccp.scores.base import register_score


@register_score("CCP-S")
def ccp_s(ep):
    mq, sq = ep.s_q(True)
    mp, sp = ep.s_p(True)
    return ccp_s_matrix(ep.cos(True), mq, sq, mp, sp)
