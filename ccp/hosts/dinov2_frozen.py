"""Frozen DINOv2 image encoders (Oquab et al., TMLR 2024): ViT-S/14, ViT-B/14, ViT-L/14 without registers.

Code: third_party/dinov2 @ 7764ea0f912e53c92e82eb78a2a1631e92725fc8 (`python third_party/clone.py dinov2`), built with
torch.hub.load(<local clone>, 'dinov2_vit{s,b,l}14', source='local', pretrained=False); the weights are the backbone files of
dl.fbaipublicfiles.com/dinov2 (dinov2_vit{s,b,l}14_pretrain.pth), downloaded once, sha256 pinned in the host config, checked before loading
and loaded strictly. xFormers is not installed, so DINOv2 uses its plain-PyTorch attention / MLP code paths.
Feature: forward_features(x)['x_norm_clstoken'], the final-norm CLS token (384 / 768 / 1024-d). Input 224 x 224 (16 x 16 patches; the
position embeddings of the 518 x 518 pre-training size are interpolated by DINOv2's own interpolate_pos_encoding).
Normalisation: ImageNet mean 0.485 0.456 0.406, std 0.229 0.224 0.225.
"""
from __future__ import annotations

from ccp.hosts.base import checked_ckpt
from ccp.hosts.vit_input import FoundationHost
from ccp.utils.paths import third_party_root

PINNED = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARCHS = ("vits14", "vitb14", "vitl14")


def build_dinov2(arch: str):
    import torch
    root = third_party_root() / "dinov2"
    if not (root / "hubconf.py").exists():
        raise ImportError("DINOv2 code not found: run `python third_party/clone.py dinov2` (pinned commit 7764ea0)")
    if arch not in ARCHS:
        raise ValueError(arch)
    return torch.hub.load(str(root), f"dinov2_{arch}", source="local", pretrained=False)


class Dinov2Frozen(FoundationHost):
    kind = "dinov2_frozen"
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD

    def __init__(self, cfg, shot=None, device="cuda", model=None):
        import torch
        super().__init__(cfg, shot, device)
        if model is None:
            path = checked_ckpt(cfg["checkpoint"])             # the pinned local file, sha256 checked first; nothing is downloaded
            model = build_dinov2(cfg["arch"])                  # torch.hub, source='local', pretrained=False
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.net = model.to(device).eval()

    def features(self, x):
        return self.net.forward_features(x)["x_norm_clstoken"]
