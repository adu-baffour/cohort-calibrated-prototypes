"""Baseline RMD, relative Mahalanobis distance (Ren et al., arXiv 2106.09022; Section 4.4).

Prototypes as class centers and one shared covariance from the bank, shrunk toward a scaled identity:
Sigma_delta = (1 - delta) C + delta (tr C / d) I, delta in {0.1, 0.5, 1.0}. MD_k = (z - p_k)^T Sigma^-1 (z - p_k),
MD_0 = (z - b_bar)^T Sigma^-1 (z - b_bar) (background: the bank mean); s_k = -(MD_k - MD_0).
"""
import torch

from ccp.scores.base import register_score


def _make(delta):
    def f(ep):
        P = ep.tf.bank.shrunk_precision(delta)
        q = ep.tf.uf[ep.r_q]
        d = q[:, None, :] - ep.pf[None, :, :]
        md = torch.einsum("nkd,de,nke->nk", d, P, d)
        d0 = q - ep.tf.bank.bbar[None, :]
        md0 = torch.einsum("nd,de,ne->n", d0, P, d0)
        return -(md - md0[:, None])
    f.__doc__ = f"RMD, delta = {delta}."
    return f


for _d in (0.1, 0.5, 1.0):
    register_score(f"RMD/delta={_d}")(_make(_d))
