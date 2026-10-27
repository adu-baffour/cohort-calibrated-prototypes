"""Baseline episode CSLS (Conneau et al., ICLR 2018; Section 4.4): a TRANSDUCTIVE reference only (it uses the other queries of the
episode, so it is not inductive).

Neighborhoods inside the episode, K = 10: r(q) = mean of the 10 largest cosines of z(q) to the other queries and the prototypes;
r(p_k) = mean of the 10 largest cosines of p_k to the queries and the other prototypes; s_k = 2 c_k - r(q) - r(p_k).
"""
import torch

from ccp.scores.base import register_score

K = 10
TRANSDUCTIVE = True


@register_score("episode CSLS")
def episode_csls(ep):
    Q, P = ep.tf.uf[ep.r_q], ep.pf
    qq = Q @ Q.T
    qq.fill_diagonal_(float("-inf"))
    qp = Q @ P.T
    pp = P @ P.T
    pp.fill_diagonal_(float("-inf"))
    rq = torch.cat([qq, qp], 1).topk(K, dim=1).values.mean(1)
    rp = torch.cat([qp.T, pp], 1).topk(K, dim=1).values.mean(1)
    return 2 * qp - rq[:, None] - rp[None, :]
