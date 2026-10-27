"""Baseline QB-Norm (Bogolin et al., CVPR 2022; Section 4.4): querybank = the base bank, gallery = the five prototypes.

Z_k = log sum_b exp(t cos(b, p_k)), temperature t in {10, 20, 50}.
  QB-Norm IS : s_k = t c_k - Z_k + log |B| (inverted softmax for every query)
  QB-Norm DIS: hub set A = {k : k = argmax_j cos(b, p_j) for some b in B}; a query whose nearest prototype argmax_k c_k is in A gets
               the IS scores, any other query t c_k (dynamic inverted softmax).
Both produce identical results in every evaluated configuration (Section 4.4).
"""
import math

import torch

from ccp.scores.base import register_score


def _is(ep, t):
    cb = ep.tf.bank.B @ ep.pf.T                       # (|B|, way)
    Z = torch.logsumexp(t * cb, dim=0)                # (way,)
    return t * ep.cos(True) - Z[None, :] + math.log(cb.shape[0]), cb


def _make(t, dynamic):
    def f(ep):
        s_is, cb = _is(ep, t)
        if not dynamic:
            return s_is
        hubs = torch.zeros(ep.way, dtype=torch.bool, device=cb.device)
        hubs[cb.argmax(1).unique()] = True
        use = hubs[ep.cos(True).argmax(1)]
        return torch.where(use[:, None], s_is, t * ep.cos(True))
    f.__doc__ = f"QB-Norm {'DIS' if dynamic else 'IS'}, temperature = {t}."
    return f


for _t in (10, 20, 50):
    register_score(f"QB-Norm IS/temperature={_t}")(_make(_t, False))
    register_score(f"QB-Norm DIS/temperature={_t}")(_make(_t, True))
