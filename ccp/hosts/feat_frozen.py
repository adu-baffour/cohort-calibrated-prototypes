"""Frozen FEAT ResNet-12 pre-training checkpoint (Res12-pre.pth; mini-ImageNet and tieredImageNet), used as a pure encoder.
State dict: params['encoder.*'] (FEAT's Res12), loaded strictly after 'layerN.0.' -> 'layerN.' mapping. No meta-training, no slf_attn."""
from __future__ import annotations

from ccp.hosts._backbone import ResNetHost
from ccp.hosts.base import checked_ckpt
from ccp.hosts.resnet12 import load_prefixed


class FeatFrozen(ResNetHost):
    kind = "feat_frozen"

    def __init__(self, cfg, shot=None, device="cuda"):
        import torch
        super().__init__(cfg, shot, device)
        sd = torch.load(checked_ckpt(cfg["checkpoint"]), map_location="cpu", weights_only=False)["params"]
        self._finish(load_prefixed(sd, "encoder."))
