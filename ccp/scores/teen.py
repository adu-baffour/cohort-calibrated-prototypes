"""Baseline TEEN (Wang et al., NeurIPS 2023; Section 4.4): move prototypes toward base-class means with softmax similarity weights.

p^_k = n(alpha p_k + (1 - alpha) sum_b w_kb mu_b), w_kb = softmax over base classes of tau cos(p_k, mu_b), mu_b the unit base-class
means; s_k = cos(z(q), p^_k). The authors' defaults tau = 16, alpha = 0.5; no tuning.
"""
import torch
import torch.nn.functional as F

from ccp.scores.base import register_score

TAU, ALPHA = 16.0, 0.5


@register_score("TEEN")
def teen(ep):
    mu = ep.tf.bank.class_means
    w = torch.softmax(TAU * ep.pf @ mu.T, dim=1)
    p = F.normalize(ALPHA * ep.pf + (1 - ALPHA) * w @ mu, dim=1)
    return ep.tf.uf[ep.r_q] @ p.T
