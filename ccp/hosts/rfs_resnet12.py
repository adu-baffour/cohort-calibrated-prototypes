"""The ResNet-12 of RFS (Tian et al., ECCV 2020), loaded from RFS's own code: third_party/RFS @ f8c837ba93c62dd0ac68a2f4019c619aa86b8421
models/resnet.py (fetched by `python third_party/clone.py RFS`; the RFS repository has no licence file, so its code is fetched, never copied
here). CIFAR-FS settings of RFS models/util.py create_model: resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=64).

The file is loaded under an isolated module name (RFS and DNPG both ship a top-level `models` package). Facts that matter for training and
are therefore part of the recipe: layers 1-2 apply dropout(0.1), layers 3-4 DropBlock(size 2) whose keep rate ramps to 0.9 over 40,000
forward calls of each block, counted in train AND eval mode; the DropBlock masks use a hard-coded .cuda() (training needs a GPU).
For CCP the encoder output is the layer-4 map (640 x 2 x 2 at 32 x 32 input); RFS's AdaptiveAvgPool2d(1) of that map is the same GAP.
"""
from __future__ import annotations

import importlib.util
import sys

from ccp.utils.paths import third_party_root

PINNED = "f8c837ba93c62dd0ac68a2f4019c619aa86b8421"
CIFARFS_ARGS = {"avg_pool": True, "drop_rate": 0.1, "dropblock_size": 2, "num_classes": 64}


def rfs_resnet_module():
    name = "ccp_third_party_rfs_resnet"
    if name in sys.modules:
        return sys.modules[name]
    path = third_party_root() / "RFS" / "models" / "resnet.py"
    if not path.exists():
        raise ImportError("RFS code not found: run `python third_party/clone.py RFS` (pinned commit f8c837b)")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resnet12_cifarfs(**overrides):
    """RFS resnet12 with the CIFAR-FS arguments of create_model."""
    return rfs_resnet_module().resnet12(**{**CIFARFS_ARGS, **overrides})


def layer4_map(net, x):
    """The layer-4 feature map (the input of RFS's average pool)."""
    return net.layer4(net.layer3(net.layer2(net.layer1(x))))
