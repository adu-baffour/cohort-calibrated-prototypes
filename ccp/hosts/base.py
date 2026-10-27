"""Host adapters: a host = a frozen encoder (+ for trained FSOR hosts, its own open-set score) with its registered preprocessing.

Feature convention (registered; identical for every host): layer-4 feature map of the host's ResNet-12 under a bf16 autocast forward,
rounded to fp16, global average pooling in fp32 (gap()). Views: plain and horizontally flipped (gap_pair()). Images are embedded in
fixed batches of 500 from the first row of the embedded array (cuDNN results depend on batch composition, so this is part of the
definition for bitwise reproducibility; docs/protocol.md, "Numerical precision").

Every checkpoint is loaded only after its sha256 matches the host config (configs/hosts/*.yaml).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ccp.data.mini import FEAT_MEAN, FEAT_STD
from ccp.utils.paths import check_sha256, load_yaml, resolve_ckpt, REPO

BATCH = 500      # images per encoder forward (registered)
CHUNK = 20000    # images moved to the GPU at once (a multiple of BATCH, so it does not change any batch)


def load_host_config(name: str) -> dict:
    p = Path(name)
    if not p.suffix:
        p = REPO / "configs" / "hosts" / f"{name}.yaml"
    return load_yaml(p)


def checked_ckpt(entry: dict) -> Path:
    """Resolve a checkpoint entry {path, sha256} and verify its sha256."""
    p = resolve_ckpt(entry["path"])
    check_sha256(p, entry["sha256"], f"checkpoint {entry['path']}")
    return p


def normalize_u8(x_u8):
    """uint8 (n, 84, 84, 3) CUDA tensor in the host's channel order -> float (n, 3, 84, 84), FEAT / TANE / DNPG constants."""
    import torch
    m = torch.tensor(FEAT_MEAN, device=x_u8.device).view(1, 3, 1, 1)
    s = torch.tensor(FEAT_STD, device=x_u8.device).view(1, 3, 1, 1)
    return (x_u8.permute(0, 3, 1, 2).float() / 255.0 - m) / s


def gap(forward, x_u8, batch: int = BATCH):
    """Registered GAP: bf16 autocast forward, local map rounded to fp16, mean over positions in fp32. Returns (n, 640) fp32."""
    import torch
    out = []
    with torch.no_grad():
        for a in range(0, x_u8.shape[0], batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fm = forward(x_u8[a:a + batch])
            out.append(fm.float().flatten(2).half().float().mean(2))
    return torch.cat(out)


class Host:
    """Base adapter. Subclasses set self.encoder_forward (uint8 CUDA batch -> layer-4 map) and may add own_scores()."""

    kind = "base"
    trained = False  # True for hosts with their own open-set score D (SnaTCHer-F, DNPG)

    def __init__(self, cfg: dict, shot: int | None = None, device: str = "cuda"):
        self.cfg, self.shot, self.device = cfg, shot, device
        self.name = cfg["name"]
        self.dataset = cfg["dataset"]
        self.view = cfg["view"]
        self.channel_order = cfg["channel_order"]
        self.feature_dtype = cfg["feature_dtype"]
        self.channels_last = bool(cfg.get("channels_last", False))

    # ------------------------------------------------------------------ features
    def encoder_forward(self, x_u8):
        raise NotImplementedError

    def gap_pair(self, x_u8):
        """(plain, flipped) raw GAP of a uint8 CUDA batch, in the host's registered stored precision (fp32 on GPU or fp16 on CPU)."""
        b = int(self.cfg.get("batch", BATCH))   # registered 500; a foundation host may use 250 only after an out-of-memory error (A4)
        z0, zf = gap(self.encoder_forward, x_u8, b), gap(self.encoder_forward, x_u8.flip(2), b)
        if self.feature_dtype == "fp16":
            return z0.half().cpu(), zf.half().cpu()
        return z0, zf

    def embed(self, arr: np.ndarray):
        """Raw GAP pair of every row of `arr` (n, 84, 84, 3) uint8, already in the host's channel order."""
        import torch
        z0, zf = [], []
        for a in range(0, arr.shape[0], CHUNK):
            x = torch.from_numpy(np.array(arr[a:a + CHUNK])).to(self.device)  # copy: memory-mapped arrays are read-only
            p0, pf = self.gap_pair(x)
            z0.append(p0)
            zf.append(pf)
            del x
        return torch.cat(z0), torch.cat(zf)

    # ------------------------------------------------------------------ bank
    def bank_entry(self) -> dict:
        b = self.cfg["bank"]
        return b[f"{self.shot}shot"] if f"{self.shot}shot" in b else b

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "dataset": self.dataset, "view": self.view, "channel_order": self.channel_order,
                "feature_dtype": self.feature_dtype, "channels_last": self.channels_last, "shot": self.shot}


def build_host(name_or_cfg, shot: int | None = None, device: str = "cuda", **kw) -> Host:
    cfg = load_host_config(name_or_cfg) if isinstance(name_or_cfg, str) else name_or_cfg
    from ccp.hosts import clip_frozen, dinov2_frozen, dnpg, feat_frozen, frn, rfs_frozen, snatcher_f, tane_pre
    kinds = {"feat_frozen": feat_frozen.FeatFrozen, "frn": frn.FRN, "tane_pre": tane_pre.TanePre, "snatcher_f": snatcher_f.SnaTCHerF,
             "dnpg": dnpg.DNPG, "rfs_frozen": rfs_frozen.RFSFrozen, "clip_frozen": clip_frozen.ClipFrozen,
             "dinov2_frozen": dinov2_frozen.Dinov2Frozen}
    return kinds[cfg["kind"]](cfg, shot=shot, device=device, **kw)
