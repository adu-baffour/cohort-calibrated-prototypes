"""Embeddings and prototypes (paper Section 3.2, Eqs. 1-2).

g(x) is the raw feature of image x: global average pooling of the final feature map (ResNet-12), the projected image embedding (CLIP) or
the final class token (DINOv2), computed under the registered convention of ccp.hosts.base.gap (bfloat16 autocast forward, feature map
rounded to fp16, mean over positions in fp32, fixed batches of 500 images).

  Eq. 1  z(x) = n(g(x) + g(flip x)),   n(v) = v / ||v||_2          (fp32; the sum order is plain view + mirror view)
  Eq. 2  p_k = n((1/S) sum_i [g(x_ki) + g(flip x_ki)])            (the support features are averaged BEFORE normalization;
                                                                    for S = 1, p_k = z(x_k1))
"""
from __future__ import annotations


def unit_flip(raw_noflip, raw_flip=None):
    """Eq. 1: z = n(g + g_flip) in fp32 (raw_flip None: single view, z = n(g))."""
    import torch.nn.functional as F
    z = raw_noflip.float() + (raw_flip.float() if raw_flip is not None else 0)
    return F.normalize(z, dim=1)


def prototypes(raw_support_sum, way: int, shot: int):
    """Eq. 2: p_k = n(mean over the S supports of class k of the unnormalized flip-view sums), supports in class-major order.
    raw_support_sum: (way * shot, d) fp32 tensor of g(x) + g(flip x) (or g(x) alone for the single-view prototypes)."""
    import torch.nn.functional as F
    return F.normalize(raw_support_sum.view(way, shot, -1).mean(1), dim=1)
