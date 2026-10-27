"""Cohort bank, cohort statistic and the cohort-calibrated score matrices (paper Sections 3.2-3.3, 4.3, 5.6; Eqs. 3-6).

Bank. B = {z(b) : b a base image}, embedded with the host's own encoder (Eq. 1): all 38,400 base images on mini-ImageNet and CIFAR-FS,
    the RandomState(20260926) subset of 38,400 of the 448,695 tieredImageNet base images (BGR input), all 5,903 CUB base images for the
    foundation-model hosts; ResNet-12 transfer to CUB keeps the mini-ImageNet bank (Section 4.3).
Eq. 3. m_K(v) = mean of the K largest cosines v . b, b in B; K = 1000 in every test round. Computed in chunks of 2,048 rows of v
    (registered chunking; results are bitwise reproducible only with the same chunking). K = all: m(v) = v . mean(B).
Eq. 4. CCP-A s_k(q) = c_k(q) - 1/2 [m_K(q) + m_K(p_k)]; label argmax_k s_k; unknown score -max_k s_k.
Eq. 5. u(q) = 1/2 m_K(q) - max_k [c_k(q) - 1/2 m_K(p_k)]: the same unknown score with the query term taken outside the maximum.
Eq. 6. CCP-QW u_QW(q) = lambda m_K(q) - max_k [c_k(q) - beta m_K(p_k)] with CCP-A's label (lambda = beta = 1/2 is CCP-A).
CCP-S (Section 3.3, no equation number). S-norm over ALL base images in closed form: m(v) = v . b_bar, s(v) = sqrt(v^T C v) + 1e-6 with
    b_bar and C (ddof 0) the mean and covariance of the unit base embeddings; s_k(q) = 1/2 [(c_k - m(p_k)) / s(p_k) + (c_k - m(q)) / s(q)].
The registered method variants of the variants round (Section 5.6): full AS-norm and the shared-cohort correction.

Bank files (.npz, release assets, sha256 pinned in configs/hosts/*.yaml) keep the raw GAP pair (plain and mirror view) in the precision of
the host's registered path (fp32, or fp16 for tieredImageNet frozen FEAT / TANE pre-trained and SnaTCHer-F), the closed-form CCP-S
statistics and the unit class means; the unit bank is formed in fp32 on load (docs/protocol.md, "Numerical precision").
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ccp.embedding import unit_flip

REGISTERED_K = 1000
CHUNK = 2048  # rows per bank matmul (registered chunking)


def _torch():
    import torch
    import torch.nn.functional as F
    return torch, F


def m_topk(V, bank, k: int = REGISTERED_K, chunk: int = CHUNK):
    """Eq. 3: mean of the k largest cosines of each row of V (unit) to `bank` (unit), chunked as registered."""
    torch, _ = _torch()
    out = []
    for a in range(0, V.shape[0], chunk):
        out.append((V[a:a + chunk] @ bank.T).topk(k, dim=1).values.mean(1))
    return torch.cat(out)


def topk_stats(V, bank, k: int = REGISTERED_K, chunk: int = CHUNK):
    """Mean (computed exactly as m_topk), std (ddof 0) and bank indices (int32) of the k largest cosines of each row of V to `bank`, from
    one top-k call per chunk (full AS-norm and the shared-cohort correction)."""
    torch, _ = _torch()
    mean, std, idx = [], [], []
    for a in range(0, V.shape[0], chunk):
        t = (V[a:a + chunk] @ bank.T).topk(k, dim=1)
        mean.append(t.values.mean(1))
        std.append(t.values.std(1, unbiased=False))
        idx.append(t.indices.to(torch.int32))
    return torch.cat(mean), torch.cat(std), torch.cat(idx)


def m_all(V, bank):
    """K = all: the mean over the whole bank, V . mean(bank) (the 'whole-bank mean' ablation)."""
    return V @ bank.mean(0)


def closed_form_stats(B_unit_all):
    """b_bar (d,) and C (d, d), ddof 0, of the unit base embeddings (CCP-S); covariance accumulated in float64 as registered."""
    bbar = B_unit_all.mean(0)
    X = (B_unit_all - bbar).double()
    C = (X.T @ X / B_unit_all.shape[0]).float()
    return bbar, C


def s_norm_stats(V, bbar, C):
    """CCP-S statistics m(v) = v . b_bar, s(v) = sqrt(v^T C v) + 1e-6."""
    torch, _ = _torch()
    return V @ bbar, torch.sqrt(torch.einsum("nd,de,ne->n", V, C, V).clamp_min(0)) + 1e-6


def ccp_a_matrix(c, m_q, m_p):
    """Eq. 4: c (nq, way) cosines, m_q (nq,), m_p (way,) -> s (nq, way) = c - 1/2 (m_p + m_q)."""
    return c - 0.5 * (m_p[None, :] + m_q[:, None])


def unknown_score(M):
    """Eq. 5 for any score matrix M (nq, way): u = -max_k M_k (higher = more likely unknown)."""
    return -M.max(1).values


def ccp_qw_matrix(c, m_q, m_p, lam: float, beta: float):
    """Eq. 6, CCP-QW: c - 1/2 (2 beta m(p_k) + 2 lam m(q)); -max_k of it equals lam m(q) - max_k [c_k - beta m(p_k)] (the query term is
    constant over k). Written so that lam = beta = 1/2 gives the operations of ccp_a_matrix (bitwise equal). The CCP-QW label is CCP-A's,
    not the argmax of this matrix."""
    return c - 0.5 * ((2 * beta) * m_p[None, :] + (2 * lam) * m_q[:, None])


def full_as_norm_matrix(c, mu_q, sd_q, mu_p, sd_p, floor: float = 1e-6):
    """Full AS-norm (registered variant): 1/2 [(c - mu_K(q)) / sd_K(q) + (c - mu_K(p_k)) / sd_K(p_k)], sd floored at `floor`
    (sd = 1 gives CCP-A)."""
    return 0.5 * ((c - mu_q[:, None]) / sd_q.clamp_min(floor)[:, None] + (c - mu_p[None, :]) / sd_p.clamp_min(floor)[None, :])


def cohort_overlap(idx_q, idx_p, n_bank: int):
    """o (nq, way): |N(q) & N(p_k)| / K for index sets idx_q (nq, K) and idx_p (way, K) of the same bank (shared-cohort correction)."""
    torch, _ = _torch()
    mask = torch.zeros(idx_p.shape[0], n_bank, dtype=torch.bool, device=idx_p.device)
    mask.scatter_(1, idx_p.long(), True)
    hits = mask[:, idx_q.long()].sum(2)            # (way, nq)
    return hits.T.to(torch.float32) / idx_q.shape[1]


def shared_cohort_matrix(c, m_q, m_p, overlap, gamma: float):
    """Shared-cohort correction (registered variant): c - 1/2 [(1 - gamma o_k) m(q) + m(p_k)]; gamma = 0 gives the operations of
    ccp_a_matrix (bitwise equal)."""
    return c - 0.5 * (m_p[None, :] + (1 - gamma * overlap) * m_q[:, None])


def ccp_s_matrix(c, mq, sq, mp, sp):
    """CCP-S: 1/2 [(c - m(p)) / s(p) + (c - m(q)) / s(q)]."""
    return 0.5 * ((c - mp[None, :]) / sp[None, :] + (c - mq[:, None]) / sq[:, None])


@dataclass
class Bank:
    """A host's cohort bank. raw_* are (n_bank, d) arrays in their stored precision; bbar / C are the CCP-S statistics over all embedded
    base images; class_means are unit mirror-view base-class means over all embedded base images."""
    raw_noflip: np.ndarray
    raw_flip: np.ndarray
    bbar: np.ndarray
    C: np.ndarray
    class_means: np.ndarray
    bank_index: np.ndarray
    meta: dict = field(default_factory=dict)
    bbar_noflip: np.ndarray | None = None
    C_noflip: np.ndarray | None = None

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> str:
        from ccp.utils.io import save_npz
        from ccp.utils.paths import sha256_file
        arrays = {"raw_noflip": self.raw_noflip, "raw_flip": self.raw_flip, "bbar": self.bbar, "C": self.C, "class_means": self.class_means,
                  "bank_index": self.bank_index, "meta": np.frombuffer(json.dumps(self.meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)}
        if self.bbar_noflip is not None:
            arrays.update({"bbar_noflip": self.bbar_noflip, "C_noflip": self.C_noflip})
        save_npz(path, **arrays)
        return sha256_file(path)

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str | None = None) -> "Bank":
        from ccp.utils.paths import check_sha256
        if expected_sha256:
            check_sha256(path, expected_sha256, f"bank {path}")
        z = np.load(path)
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        return cls(z["raw_noflip"], z["raw_flip"], z["bbar"], z["C"], z["class_means"], z["bank_index"], meta,
                   z["bbar_noflip"] if "bbar_noflip" in z.files else None, z["C_noflip"] if "C_noflip" in z.files else None)

    # ---------------------------------------------------------------- device views
    def to_device(self, device: str = "cuda") -> "DeviceBank":
        return DeviceBank(self, device)


class DeviceBank:
    """Unit bank and statistics on a device, fp32."""

    def __init__(self, bank: Bank, device: str):
        torch, _ = _torch()
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
        self.device = device
        self.meta = bank.meta
        n0, nf = t(bank.raw_noflip), t(bank.raw_flip)
        self.B = unit_flip(n0, nf)                  # unit mirror-view bank (CCP-A)
        self.B0 = unit_flip(n0)                     # unit single-view bank (single-view ablation)
        self.bbar, self.C = t(bank.bbar).float(), t(bank.C).float()
        self.class_means = t(bank.class_means).float()
        self.bbar0 = t(bank.bbar_noflip).float() if bank.bbar_noflip is not None else None
        self.C0 = t(bank.C_noflip).float() if bank.C_noflip is not None else None

    def shrunk_precision(self, delta: float):
        """(Sigma_delta)^-1 with Sigma_delta = (1 - delta) C + delta (tr C / d) I (baseline RMD), computed in float64, cached."""
        import torch
        if not hasattr(self, "_prec"):
            self._prec = {}
        if delta not in self._prec:
            C = self.C.double()
            d = C.shape[0]
            S = (1 - delta) * C + delta * (torch.trace(C) / d) * torch.eye(d, dtype=C.dtype, device=C.device)
            self._prec[delta] = torch.linalg.inv(S).float()
        return self._prec[delta]

    def m(self, V, k: int | str = REGISTERED_K, flip: bool = True, source: str = "images"):
        """Cohort statistic of the rows of V against the image bank (source='images') or the class-mean bank ('class_means')."""
        bank = self.class_means if source == "class_means" else (self.B if flip else self.B0)
        if k == "all":
            return m_all(V, bank)
        return m_topk(V, bank, int(k))


def build_bank(raw_noflip_all, raw_flip_all, labels_all, bank_rows, meta: dict, store_dtype: str, with_noflip_stats: bool = False) -> Bank:
    """Bank from the raw GAP pair of ALL embedded base images (torch tensors on GPU, any float dtype).
    bank_rows: rows of the embedded set that form the CCP-A bank (all rows for mini-ImageNet). CCP-S statistics and class means use all rows."""
    torch, F = _torch()
    Ball = unit_flip(raw_noflip_all, raw_flip_all)
    bbar, C = closed_form_stats(Ball)
    lab = torch.as_tensor(np.asarray(labels_all), device=Ball.device)
    # class sums as a one-hot matrix product: deterministic on GPU (index_add_ uses atomic adds and is not)
    onehot = F.one_hot(lab, int(lab.max()) + 1).to(Ball.dtype)
    cm = F.normalize(onehot.T @ Ball, dim=1)
    rows = torch.as_tensor(np.asarray(bank_rows), device=Ball.device)
    dt = {"fp16": torch.float16, "fp32": torch.float32}[store_dtype]
    extra = {}
    if with_noflip_stats:
        b0, c0 = closed_form_stats(unit_flip(raw_noflip_all))
        extra = {"bbar_noflip": b0.cpu().numpy(), "C_noflip": c0.cpu().numpy()}
    meta = {**meta, "store_dtype": store_dtype, "n_bank": int(len(bank_rows)), "n_all_embedded": int(Ball.shape[0]), "d": int(Ball.shape[1])}
    return Bank(raw_noflip_all[rows].to(dt).cpu().numpy(), raw_flip_all[rows].to(dt).cpu().numpy(), bbar.cpu().numpy(), C.cpu().numpy(),
                cm.cpu().numpy(), np.asarray(meta.get("bank_global_index", bank_rows)), meta, **extra)
