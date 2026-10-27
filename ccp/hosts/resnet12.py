"""ResNet-12 as in FRN (models/backbones/ResNet.py, drop_rate 0) and FEAT / TANE (same layout and parameter names after mapping):
four residual blocks with 64 / 160 / 320 / 640 channels, three bias-free 3x3 convs + BatchNorm per block, LeakyReLU(0.1),
1x1-conv + BN shortcut, 2x2 max-pool after every block. 84 x 84 input -> 640 x 5 x 5. Used in eval mode only here."""
from __future__ import annotations

import re

import torch
import torch.nn as nn


def conv3x3(cin, cout):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int):
        super().__init__()
        self.conv1, self.bn1 = conv3x3(inplanes, planes), nn.BatchNorm2d(planes)
        self.conv2, self.bn2 = conv3x3(planes, planes), nn.BatchNorm2d(planes)
        self.conv3, self.bn3 = conv3x3(planes, planes), nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(2)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + self.downsample(x))
        return self.maxpool(out)


class ResNet12(nn.Module):
    out_channels = 640

    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock(3, 64)
        self.layer2 = BasicBlock(64, 160)
        self.layer3 = BasicBlock(160, 320)
        self.layer4 = BasicBlock(320, 640)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def load_prefixed(sd: dict, prefix: str) -> "ResNet12":
    """ResNet12 from a state dict whose backbone keys start with `prefix` (FEAT 'encoder.', TANE 'feature.'): strips the prefix and
    'module.', maps 'layerN.0.' -> 'layerN.', loads strictly."""
    m = {re.sub(r"^layer(\d)\.0\.", r"layer\1.", k[len(prefix):].replace("module.", "")): v for k, v in sd.items() if k.startswith(prefix)}
    net = ResNet12()
    net.load_state_dict(m, strict=True)
    return net
