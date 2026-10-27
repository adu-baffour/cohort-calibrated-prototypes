"""SnaTCHer-F (Jeong et al., CVPR 2021) on FEAT's released meta-trained ResNet-12 (the README-linked feat-1-shot.pth / feat-5-shot.pth,
NOT the unlinked feat-1-shot-improved.pth). One model per shot, no seeds, no training.

Model: params = encoder.* (FEAT Res12, loaded strictly into our ResNet12) + slf_attn.* (SnaTCHer's own MultiHeadAttention class from
third_party/SnaTCHer @ 335c424, model/models/SnaTCHerF.py, 1 head, d = 640). Both in eval mode, fp32, no autocast.
Per episode (SnaTCHer's evaluate_test, fsl_trainer_SnaTCHerF.py:52-165, in class-major order with the 150 queries batched; each query
is still scored independently):
  bproto = mean of the K known supports per class (5, d); proto = slf_attn(bproto)
  logits = -||q - proto||^2 / 64 (150, 5); FEAT prediction = argmax; accuracy on the 75 known queries
  S(q)   = sum_k ||slf_attn(bproto with row argmax(logits_q) replaced by q)_k - proto_k||^2 / 64   (higher = more likely unknown)
  Dist(q) = -max_k logits (FEAT's distance score)
Fidelity against SnaTCHer's own evaluate_test: scripts/fidelity/snatcher_fidelity.py (max |diff| 7.6e-6 on the same encoder batch).
CCP features: the same encoder's layer-4 GAP (registered convention), flip views, fp16 stored precision; no slf_attn.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from ccp.hosts.base import BATCH, Host, checked_ckpt, normalize_u8
from ccp.hosts.resnet12 import load_prefixed
from ccp.utils.paths import third_party_root


def mha_class():
    root = str(third_party_root() / "SnaTCHer")
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from model.models.SnaTCHerF import MultiHeadAttention
    except ImportError as e:
        raise ImportError("SnaTCHer code not found: run `python third_party/clone.py SnaTCHer` (docs/checkpoints.md)") from e
    return MultiHeadAttention


class SnaTCHerF(Host):
    kind = "snatcher_f"
    trained = True

    def __init__(self, cfg, shot=None, device="cuda"):
        import torch
        super().__init__(cfg, shot, device)
        if shot not in (1, 5):
            raise ValueError("SnaTCHer-F needs shot 1 or 5 (one released model per shot)")
        sd = torch.load(checked_ckpt(cfg["checkpoints"][f"{shot}shot"]), map_location="cpu", weights_only=False)["params"]
        self.enc = load_prefixed(sd, "encoder.").to(device).eval()
        attn = mha_class()(argparse.Namespace(), 1, 640, 640, 640, dropout=0.5)
        attn.load_state_dict({k[len("slf_attn."):]: v for k, v in sd.items() if k.startswith("slf_attn.")}, strict=True)
        self.attn = attn.to(device).eval()

    def encoder_forward(self, x_u8):
        return self.enc(normalize_u8(x_u8))

    def pooled(self, arr: np.ndarray):
        """fp32 average-pooled 640-d embeddings (FEAT Res12 output) of every row, batches of 500, no autocast."""
        import torch
        out = []
        with torch.no_grad():
            for a in range(0, arr.shape[0], BATCH):
                x = torch.from_numpy(np.ascontiguousarray(arr[a:a + BATCH])).to(self.device)
                out.append(self.enc(normalize_u8(x)).mean((2, 3)))
        return torch.cat(out)

    def episode(self, Es, Eq):
        """Es (5, K, d) known supports class-major; Eq (150, d). Returns logits (150, 5), S (150,), Dist (150,), accuracy (%)."""
        import torch
        with torch.no_grad():
            bproto = Es.mean(1)
            proto = self.attn(bproto[None], bproto[None], bproto[None])[0][0]
            logits = -(Eq[:, None, :] - proto[None]).pow(2).sum(2) / 64.0
            pred = logits.argmax(1)
            P = bproto[None].repeat(Eq.shape[0], 1, 1)
            P[torch.arange(Eq.shape[0], device=Eq.device), pred] = Eq
            PP = self.attn(P, P, P)[0]
            S = (PP - proto[None]).pow(2).sum(-1).sum(-1) / 64.0
            tgt = torch.arange(5, device=Eq.device).repeat_interleave(15)
            acc = float((pred[:75] == tgt).float().mean() * 100)
        return logits, S, -logits.max(1).values, acc

    def own_scores(self, eps, rows_of, E):
        """S, Dist (n, 150), 'correct' (n, 75) per known query and FEAT accuracy (n,) on Episodes `eps`; rows_of(global idx) -> rows of E."""
        import torch
        S, Dist, acc, C = [], [], [], []
        tgt = torch.arange(5, device=self.device).repeat_interleave(15)
        for e in range(eps.n):
            r = torch.as_tensor(rows_of(eps.indices(e)), device=self.device)
            l_, s_, d_, a_ = self.episode(E[r[:5 * self.shot]].view(5, self.shot, -1), E[r[5 * self.shot:]])
            C.append((l_.argmax(1)[:75] == tgt).cpu().numpy())
            S.append(s_.cpu().numpy())
            Dist.append(d_.cpu().numpy())
            acc.append(a_)
        return {"S": np.array(S), "Dist": np.array(Dist), "correct": np.array(C)}, np.array(acc)
