"""Released TANE / DNPG pre-trained ResNet-12 (miniImageNet_pre.pth, tieredImageNet_pre.pth), used as a pure encoder.
State dict: params['feature.*'], loaded strictly after 'layerN.0.' -> 'layerN.' mapping.

FLAG (docs/protocol.md, "Known limitations"): the stored epoch of these released checkpoints (mini 66, tiered 81) is consistent with
selection on the TEST split by the original authors (their released pre-training code selects `max_meta` on the test loader). Every
comparison on these backbones is paired on the same backbone, but absolute numbers may be optimistic.
"""
from __future__ import annotations

from ccp.hosts._backbone import ResNetHost
from ccp.hosts.base import checked_ckpt
from ccp.hosts.resnet12 import load_prefixed

TEST_SELECTION_FLAG = "released checkpoint; stored epoch consistent with test-split selection by the original authors"


class TanePre(ResNetHost):
    kind = "tane_pre"

    def __init__(self, cfg, shot=None, device="cuda"):
        import torch
        super().__init__(cfg, shot, device)
        sd = torch.load(checked_ckpt(cfg["checkpoint"]), map_location="cpu", weights_only=False)["params"]
        self._finish(load_prefixed(sd, "feature."))
        self.flag = TEST_SELECTION_FLAG
