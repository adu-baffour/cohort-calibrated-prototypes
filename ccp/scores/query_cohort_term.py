"""Baseline: the query cohort term m_K(q), K = 1000, alone as unknown evidence (Section 4.4; greater affinity favors rejection).

Combined with the reference score by fixed-weight fusion (Eq. 7): COSF + w e on frozen hosts (COSF label), host score + w e on trained
hosts (host label), w the pooled standard-deviation ratio on valA. It is the query statistic of CCP-A with a different weight and without
the prototype correction.
"""
from ccp.cohort import REGISTERED_K
from ccp.scores.base import register_evidence


@register_evidence("query cohort term")
def query_cohort_term(ep):
    return ep.m_q(REGISTERED_K)
