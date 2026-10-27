"""Score registry and the per-episode context.

A score is a function f(ep: EpisodeContext) -> M, a (150, way) torch tensor of per-class similarities (higher = more similar). The
unknown score of a query is -max_k M[q, k] (higher = more likely unknown; Eq. 5) and its predicted label is argmax_k M[q, k]. An evidence
score (query cohort term, kNN) is f(ep) -> (150,) unknown evidence without a label. Scores are registered under the paper's names
(ccp.naming maps them to the recorded keys).

Inductive property (Section 3.1): every quantity an EpisodeContext exposes for query q is a function of q's own embedding, the support set,
the bank and frozen constants. Per-query cohort statistics are computed row-wise over the whole target split once (TargetFeatures), so a
query's score cannot depend on which other queries share its episode (tests/test_inductive.py). The one exception is the episode CSLS
baseline, a transductive reference.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ccp.cohort import REGISTERED_K, DeviceBank, m_all, m_topk, s_norm_stats, topk_stats
from ccp.embedding import prototypes, unit_flip

WAY, NQ = 5, 15
_REGISTRY: dict[str, Callable] = {}
_EVIDENCE: dict[str, Callable] = {}


def register_score(name: str):
    def deco(fn):
        if name in _REGISTRY or name in _EVIDENCE:
            raise KeyError(f"score {name!r} registered twice")
        _REGISTRY[name] = fn
        fn.score_name = name
        return fn
    return deco


def register_evidence(name: str):
    """An evidence score: f(ep) -> (n_queries,) tensor, higher = more likely unknown, no label (query cohort term, kNN)."""
    def deco(fn):
        if name in _EVIDENCE or name in _REGISTRY:
            raise KeyError(f"score {name!r} registered twice")
        _EVIDENCE[name] = fn
        fn.score_name = name
        return fn
    return deco


def get_evidence(name: str) -> Callable:
    if name not in _EVIDENCE:
        raise KeyError(f"unknown evidence score {name!r}; registered: {sorted(_EVIDENCE)}")
    return _EVIDENCE[name]


def registered_evidence() -> list[str]:
    return sorted(_EVIDENCE)


def get_score(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown score {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_scores() -> list[str]:
    return sorted(_REGISTRY)


class TargetFeatures:
    """Embeddings of every image of the target split for one host (raw GAP pair, rows = split order) and the host's bank.
    Per-query cohort statistics are computed for ALL rows at once in the registered chunking (bitwise reproducible)."""

    def __init__(self, raw_noflip, raw_flip, bank: DeviceBank):
        self.bank = bank
        self.q0 = raw_noflip.float().to(bank.device)
        self.qf = self.q0 + raw_flip.float().to(bank.device)
        self.u0 = unit_flip(self.q0)
        self.uf = unit_flip(self.qf)
        self._cache: dict = {}

    def m_query(self, k=REGISTERED_K, flip: bool = True, source: str = "images"):
        key = ("m", k, flip, source)
        if key not in self._cache:
            self._cache[key] = self.bank.m(self.uf if flip else self.u0, k, flip, source)
        return self._cache[key]

    def topk_query(self, k: int = REGISTERED_K):
        """(mean, std, bank indices) of the k largest mirror-view cosines of every target row (full AS-norm, shared-cohort correction)."""
        key = ("topk", k)
        if key not in self._cache:
            self._cache[key] = topk_stats(self.uf, self.bank.B, k)
        return self._cache[key]

    def kth_query(self, k: int):
        """k-th largest cosine of every target row (mirror view) to the image bank, chunked as m_topk (kNN baseline)."""
        key = ("kth", k)
        if key not in self._cache:
            import torch
            out = []
            for a in range(0, self.uf.shape[0], 2048):
                out.append((self.uf[a:a + 2048] @ self.bank.B.T).topk(k, dim=1).values[:, -1])
            self._cache[key] = torch.cat(out)
        return self._cache[key]

    def s_query(self, flip: bool = True):
        key = ("s", flip)
        if key not in self._cache:
            if flip:
                self._cache[key] = s_norm_stats(self.uf, self.bank.bbar, self.bank.C)
            else:
                if self.bank.C0 is None:
                    raise ValueError("bank has no single-view CCP-S statistics")
                self._cache[key] = s_norm_stats(self.u0, self.bank.bbar0, self.bank.C0)
        return self._cache[key]


class EpisodeContext:
    """One episode. rows: target-split rows in the order support (class-major, way x shot), then the 150 queries."""

    def __init__(self, tf: TargetFeatures, rows: np.ndarray, shot: int, way: int = WAY):
        import torch
        self.tf, self.shot, self.way = tf, shot, way
        r = torch.as_tensor(np.asarray(rows), device=tf.bank.device)
        if (r < 0).any():
            raise IndexError("episode index maps to no target image")
        ns = way * shot
        self.r_sup, self.r_q = r[:ns], r[ns:]
        self.p0 = prototypes(tf.q0[self.r_sup], way, shot)   # single-view prototypes
        self.pf = prototypes(tf.qf[self.r_sup], way, shot)   # Eq. 2
        self._c: dict = {}

    def cos(self, flip: bool = True):
        """c_k(q) = z(q) . p_k (Section 3.2); flip=False: the single-view COS."""
        if flip not in self._c:
            self._c[flip] = (self.tf.uf[self.r_q] @ self.pf.T) if flip else (self.tf.u0[self.r_q] @ self.p0.T)
        return self._c[flip]

    def m_q(self, k=REGISTERED_K, flip: bool = True, source: str = "images"):
        """m_K(q) of the episode's queries (Eq. 3)."""
        return self.tf.m_query(k, flip, source)[self.r_q]

    def m_p(self, k=REGISTERED_K, flip: bool = True, source: str = "images"):
        """m_K(p_k) of the episode's prototypes (Eq. 3)."""
        P = self.pf if flip else self.p0
        bank = self.tf.bank
        if source == "class_means":
            b = bank.class_means
        else:
            b = bank.B if flip else bank.B0
        return m_all(P, b) if k == "all" else m_topk(P, b, int(k))

    def topk_q(self, k: int = REGISTERED_K):
        mu, sd, idx = self.tf.topk_query(k)
        return mu[self.r_q], sd[self.r_q], idx[self.r_q]

    def topk_p(self, k: int = REGISTERED_K):
        return topk_stats(self.pf, self.tf.bank.B, k)

    def s_q(self, flip: bool = True):
        m, s = self.tf.s_query(flip)
        return m[self.r_q], s[self.r_q]

    def s_p(self, flip: bool = True):
        b = self.tf.bank
        return s_norm_stats(self.pf, b.bbar, b.C) if flip else s_norm_stats(self.p0, b.bbar0, b.C0)


def score_episode(ep: EpisodeContext, names: list[str]) -> dict:
    """{name: (unknown score (150,) numpy, accuracy on the known queries (%), M (150, way))}."""
    import torch
    target = torch.arange(ep.way, device=ep.tf.bank.device).repeat_interleave(NQ)
    out = {}
    for n in names:
        M = get_score(n)(ep)
        s = -M.max(1).values
        acc = (M.argmax(1)[: ep.way * NQ] == target).float().mean().item() * 100
        out[n] = (s.cpu().numpy(), acc, M)
    return out
