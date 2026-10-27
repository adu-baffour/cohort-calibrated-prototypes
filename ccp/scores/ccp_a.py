"""CCP-A (Eq. 4) and the validation ablations of Table 7(a). Every ablation changes exactly one registered choice."""
from ccp.cohort import REGISTERED_K, ccp_a_matrix
from ccp.scores.base import register_score


@register_score("CCP-A")
def ccp_a(ep):
    """Registered CCP-A: both cohort terms, top-1000 image bank, mean only, mirror views."""
    return ccp_a_matrix(ep.cos(True), ep.m_q(REGISTERED_K), ep.m_p(REGISTERED_K))


def _k_variant(k):
    def f(ep):
        return ccp_a_matrix(ep.cos(True), ep.m_q(k), ep.m_p(k))
    f.__doc__ = f"CCP-A with cohort size K = {k} (cohort size; K = all: the whole-bank mean)."
    return f


for _k in (100, 300, 3000, "all"):
    register_score(f"CCP-A/K={_k}")(_k_variant(_k))


@register_score("CCP-A/query only")
def query_only(ep):
    """Query correction only, c_k - m_K(q), coefficient 1. Leaves the label unchanged (the term is constant over k)."""
    return ep.cos(True) - ep.m_q(REGISTERED_K)[:, None]


@register_score("CCP-A/prototype only")
def prototype_only(ep):
    """Prototype correction only, c_k - m_K(p_k), coefficient 1."""
    return ep.cos(True) - ep.m_p(REGISTERED_K)[None, :]


def class_mean_topk(n_classes: int) -> int:
    """Top-k over the class-mean bank matched to the registered fraction 1000 / 38,400 (mini-ImageNet 2 of 64, tieredImageNet 9 of 351)."""
    return max(1, round(1000 / 38400 * n_classes))


@register_score("CCP-A/class means, matched fraction")
def class_means_matched(ep):
    """Bank replaced by the unit base-class means, the nearest fraction matched to K = 1000 of 38,400 (cohort composition)."""
    k = class_mean_topk(ep.tf.bank.class_means.shape[0])
    return ccp_a_matrix(ep.cos(True), ep.m_q(k, source="class_means"), ep.m_p(k, source="class_means"))


@register_score("CCP-A/class means, all")
def class_means_all(ep):
    """Bank replaced by the unit base-class means, mean over all of them (cohort composition)."""
    return ccp_a_matrix(ep.cos(True), ep.m_q("all", source="class_means"), ep.m_p("all", source="class_means"))


@register_score("CCP-A/single view")
def single_view(ep):
    """No mirror views for queries, prototypes and bank."""
    return ccp_a_matrix(ep.cos(False), ep.m_q(REGISTERED_K, flip=False), ep.m_p(REGISTERED_K, flip=False))
