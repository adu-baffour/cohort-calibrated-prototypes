"""CIFAR-FS backbone of the final round: the RFS ResNet-12 pre-trained with cross-entropy on the 64 CIFAR-FS base classes
(scripts/train/train_rfs_cifarfs.py, recipe in docs/preregistration/final_round.md), used as a frozen encoder. 32 x 32 RGB input,
RFS normalisation (dataset/cifar.py: mean 0.5071 0.4867 0.4408, std 0.2675 0.2565 0.2761). Checkpoint: {'model': state_dict, ...},
loaded strictly after its sha256 matches the host config."""
from __future__ import annotations

from ccp.hosts.base import Host, checked_ckpt

RFS_MEAN = (0.5071, 0.4867, 0.4408)
RFS_STD = (0.2675, 0.2565, 0.2761)


def normalize_cifar(x_u8):
    """uint8 (n, 32, 32, 3) CUDA tensor -> float (n, 3, 32, 32) with the RFS CIFAR constants (ToTensor + Normalize)."""
    import torch
    m = torch.tensor(RFS_MEAN, device=x_u8.device).view(1, 3, 1, 1)
    s = torch.tensor(RFS_STD, device=x_u8.device).view(1, 3, 1, 1)
    return (x_u8.permute(0, 3, 1, 2).float() / 255.0 - m) / s


class RFSFrozen(Host):
    kind = "rfs_frozen"

    def __init__(self, cfg, shot=None, device="cuda"):
        import torch

        from ccp.hosts.rfs_resnet12 import resnet12_cifarfs
        super().__init__(cfg, shot, device)
        sd = torch.load(checked_ckpt(cfg["checkpoint"]), map_location="cpu", weights_only=False)["model"]
        net = resnet12_cifarfs(**cfg.get("model_args", {}))
        net.load_state_dict(sd, strict=True)
        self.net = net.to(device).eval()

    def encoder_forward(self, x_u8):
        from ccp.hosts.rfs_resnet12 import layer4_map
        return layer4_map(self.net, normalize_cifar(x_u8))
