"""Frozen CLIP ViT-B/16 image encoder (Radford et al., ICML 2021), OpenAI weights. Image encoder only: no text encoder, no class names.

Code: CLIP's own clip/model.py from third_party/CLIP @ d05afc436d78f1c48dc0dbf8e5980a9d471f35f6 (`python third_party/clone.py CLIP`), loaded
under an isolated module name; only torch / numpy are needed (the tokenizer and its ftfy / regex dependencies are never imported).
Weights: ViT-B-16.pt (the TorchScript archive of CLIP's _MODELS URL), sha256 pinned in the host config and checked before loading; only that
local file is read, nothing is downloaded. The loader reproduces clip.load(<local path>, jit=False) step by step (torch.jit.load of the archive
on the CPU -> its state dict -> clip/model.py build_model()) instead of calling clip.load, because importing the clip package imports its
tokenizer, whose ftfy / regex dependencies are not in the pinned environment. Then .float() (fp32 weights; clip.load would keep fp16 on
CUDA) and only the visual tower is kept. Feature: visual(x), the projected 512-d image embedding that CLIP uses for similarity (= encode_image).
Normalisation: CLIP's mean 0.48145466 0.4578275 0.40821073, std 0.26862954 0.26130258 0.27577711.
"""
from __future__ import annotations

import importlib.util
import sys

from ccp.hosts.base import checked_ckpt
from ccp.hosts.vit_input import FoundationHost
from ccp.utils.paths import third_party_root

PINNED = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_model_module():
    name = "ccp_third_party_clip_model"
    if name in sys.modules:
        return sys.modules[name]
    path = third_party_root() / "CLIP" / "clip" / "model.py"
    if not path.exists():
        raise ImportError("CLIP code not found: run `python third_party/clone.py CLIP` (pinned commit d05afc4)")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_visual(ckpt_path):
    """CLIP visual tower from the TorchScript archive (as clip.load(jit=False)), fp32, eval mode."""
    import torch
    try:
        sd = torch.jit.load(str(ckpt_path), map_location="cpu").eval().state_dict()
    except RuntimeError:
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = clip_model_module().build_model(sd).float().eval()
    return model.visual


class ClipFrozen(FoundationHost):
    kind = "clip_frozen"
    MEAN, STD = CLIP_MEAN, CLIP_STD

    def __init__(self, cfg, shot=None, device="cuda", visual=None):
        super().__init__(cfg, shot, device)
        v = visual if visual is not None else load_visual(checked_ckpt(cfg["checkpoint"]))
        self.net = v.to(device).eval()

    def features(self, x):
        return self.net(x)
