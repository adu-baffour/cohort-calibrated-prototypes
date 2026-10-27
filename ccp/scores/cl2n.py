"""Baseline CL2N (SimpleShot, Wang et al., arXiv 1911.04623; Section 4.4): subtract the bank mean, renormalize, rebuild prototypes.

f(v) = n(v - b_bar) on the unit mirror-view embeddings; prototype = n(mean of f over the S supports); s_k = cos(f(z(q)), p'_k).
"""
import torch.nn.functional as F

from ccp.scores.base import register_score


@register_score("CL2N")
def cl2n(ep):
    b = ep.tf.bank.bbar[None, :]
    fq = F.normalize(ep.tf.uf[ep.r_q] - b, dim=1)
    fs = F.normalize(ep.tf.uf[ep.r_sup] - b, dim=1).view(ep.way, ep.shot, -1).mean(1)
    return fq @ F.normalize(fs, dim=1).T
