"""FRN head (Wertheimer, Tang, Hariharan, "Few-Shot Classification with Feature Map Reconstruction Networks", CVPR 2021), re-implemented
from models/FRN.py and trainers/frn_train.py of https://github.com/Tsingularity/FRN (MIT licence; see THIRD_PARTY_LICENSES.md).

Ridge-regression feature-map reconstruction: for each class c with support feature matrix S_c (k*r x d) and the query feature matrix Q (r x d),
Q_bar = rho * Q (S_c^T S_c + lam I)^{-1} S_c^T S_c (Woodbury form), lam = (k r / d) * exp(alpha) + 1e-6, rho = exp(beta); distance = mean over
the r locations of ||Q_bar - Q||^2; logits = -distance * scale (learnable, init 1); alpha, beta learnable, init 0. ResNet-12 feature maps are
divided by sqrt(640). Training adds 0.03 * sum_{i<j} ||S_i_hat S_j_hat^T||_F^2 (auxrank). Used only by scripts/train/train_frn.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FRNHead(nn.Module):
    def __init__(self, d: int = 640, resolution: int = 25):
        super().__init__()
        self.d, self.resolution = d, resolution
        self.feature_divisor = math.sqrt(640)
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.r = nn.Parameter(torch.zeros(2))  # [alpha, beta]

    def feature_map(self, fmap):
        n = fmap.shape[0]
        return (fmap.float() / self.feature_divisor).view(n, self.d, -1).permute(0, 2, 1).contiguous()

    @staticmethod
    def recon_dist(query, support, alpha, beta):
        """query (nq*r, d); support (way, k*r, d) -> (nq*r, way) squared reconstruction errors (Woodbury form)."""
        lam = support.size(1) / support.size(2) * alpha.exp() + 1e-6
        rho = beta.exp()
        st = support.permute(0, 2, 1)
        sts = st.matmul(support)
        m_inv = torch.linalg.inv(sts + torch.eye(sts.size(-1), device=sts.device, dtype=sts.dtype).unsqueeze(0) * lam)
        q_bar = query.matmul(m_inv.matmul(sts)).mul(rho)
        return (q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)

    def forward(self, fmap, way: int, shot: int):
        f = self.feature_map(fmap)
        r, d = self.resolution, self.d
        support = f[: way * shot].view(way, shot * r, d)
        query = f[way * shot:].reshape(-1, d)
        neg = self.recon_dist(query, support, self.r[0], self.r[1]).neg().view(-1, r, way).mean(1)
        return neg * self.scale, support


def auxrank(support):
    way = support.size(0)
    support = support / support.norm(2).unsqueeze(-1)
    i, j = torch.tril_indices(way, way, offset=-1, device=support.device)
    dists = support.index_select(0, i).matmul(support.index_select(0, j).permute(0, 2, 1))
    return dists.pow(2).sum(-1).sum(-1).sum().mul(0.03)
