"""Input rule and feature convention of the frozen foundation-model hosts (docs/preregistration/foundation_round.md, Section 2).

One rule for every dataset: the registered stored uint8 arrays (mini eval84, tiered 84 x 84 converted BGR -> RGB by ccp.pipeline.host_images,
CIFAR-FS eval32, CUB eval84; analysis view CUB eval224), on the GPU:
    x / 255 -> F.interpolate(size 224, mode='bicubic', align_corners=False, antialias=True) -> clamp(0, 1) -> (x - mean) / std
with the model's own constants. Arrays already 224 x 224 (CUB eval224) are not resampled. torch's bicubic differs slightly from PIL's; this
torch path is the definition. The interpolation and normalisation run in fp32 (F.interpolate is not an autocast op); the model forward runs
under the registered bf16 autocast of ccp.hosts.base.gap().

Feature convention: encoder_forward returns the global feature as an (n, D, 1, 1) map, so the registered gap() applies unchanged (bf16
autocast forward, the 1 x 1 map rounded to fp16, fp32 mean, i.e. the feature itself rounded to fp16 and stored in fp32), batches of 500.
"""
from __future__ import annotations

from ccp.hosts.base import Host

SIZE = 224


def to_model_input(x_u8, mean, std, size: int = SIZE):
    """uint8 (n, h, w, 3) RGB CUDA tensor -> float (n, 3, size, size), the registered resize / clamp / normalize."""
    import torch
    import torch.nn.functional as F
    x = x_u8.permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(0.0, 1.0)
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


class FoundationHost(Host):
    """Frozen image encoder; subclasses set MEAN / STD and self.features (float (n, 3, 224, 224) -> (n, D))."""

    MEAN: tuple = ()
    STD: tuple = ()

    def __init__(self, cfg, shot=None, device="cuda"):
        super().__init__(cfg, shot, device)
        if self.channel_order != "RGB":
            raise ValueError(f"{self.name}: foundation hosts read RGB (channel_order: RGB); got {self.channel_order}")

    def features(self, x):
        raise NotImplementedError

    def encoder_forward(self, x_u8):
        f = self.features(to_model_input(x_u8, self.MEAN, self.STD))
        return f[:, :, None, None]
