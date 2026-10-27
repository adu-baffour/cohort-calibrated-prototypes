"""Baseline ZN (Fei et al., ICCV 2021; Section 4.4): standardize each embedding across feature dimensions, then cosine scoring.

n(v) = (v - mean(v)) / std(v) over the d dimensions (std with ddof 0) of the raw mirror-view sum g(x) + g(flip x);
prototype = n(mean of the standardized supports); s_k = cos(n(g(q)), p''_k).
"""
import torch.nn.functional as F

from ccp.scores.base import register_score


def zn(v):
    return (v - v.mean(1, keepdim=True)) / v.std(1, unbiased=False, keepdim=True)


@register_score("ZN")
def zn_score(ep):
    q = F.normalize(zn(ep.tf.qf[ep.r_q]), dim=1)
    p = zn(ep.tf.qf[ep.r_sup]).view(ep.way, ep.shot, -1).mean(1)
    return q @ F.normalize(p, dim=1).T
