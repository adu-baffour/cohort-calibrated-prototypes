"""FRN (Wertheimer et al., CVPR 2021) fine-tuned from the frozen FEAT mini checkpoint, seeds 0-2 (scripts/train/train_frn.py).
Only the backbone ('bb' state dict of best.pth, selected on closed-set valA 5-shot) is used as the CCP encoder; the FRN head is not."""
from __future__ import annotations

from ccp.hosts._backbone import ResNetHost
from ccp.hosts.base import checked_ckpt
from ccp.hosts.resnet12 import ResNet12


class FRN(ResNetHost):
    kind = "frn"

    def __init__(self, cfg, shot=None, device="cuda"):
        import torch
        super().__init__(cfg, shot, device)
        net = ResNet12()
        net.load_state_dict(torch.load(checked_ckpt(cfg["checkpoint"]), map_location="cpu", weights_only=False)["bb"])
        self._finish(net)
