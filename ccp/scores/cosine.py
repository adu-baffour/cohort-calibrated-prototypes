"""References without cohort normalization (Section 3.2): COS (single view) and COSF (mirror views, Eqs. 1-2)."""
from ccp.scores.base import register_score


@register_score("COS")
def cos(ep):
    """c_k(q) with single-view embeddings for queries and prototypes."""
    return ep.cos(flip=False)


@register_score("COSF")
def cosf(ep):
    """c_k(q) = z(q) . p_k with mirror-view embeddings (Eqs. 1-2): the reference score of the frozen hosts."""
    return ep.cos(flip=True)
