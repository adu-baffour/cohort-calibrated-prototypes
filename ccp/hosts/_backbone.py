"""Shared helper for hosts whose encoder is a plain ResNet-12 run in channels_last memory format (frozen FEAT, FRN, TANE)."""
from __future__ import annotations

from ccp.hosts.base import Host, normalize_u8


class ResNetHost(Host):
    net = None

    def _finish(self, net):
        import torch
        net = net.to(self.device).eval()
        if self.channels_last:
            net = net.to(memory_format=torch.channels_last)
        self.net = net

    def encoder_forward(self, x_u8):
        import torch
        x = normalize_u8(x_u8)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.net(x)
