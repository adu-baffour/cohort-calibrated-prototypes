"""F1: CLIP loader fidelity (foundation round; base images only, no validation or test data).

    python scripts/fidelity/clip_loader_fidelity.py

(a) the clip_frozen host path (ccp/hosts/clip_frozen.py: the archive's state dict -> build_model -> .float() -> visual), and (b) the official
package from the pinned clone: clip.load(<pinned local ViT-B-16.pt>, device='cuda', jit=False), then .float(), model.encode_image (the
projected 512-d embedding). Both through the registered input rule (ccp/hosts/vit_input.py) and the registered gap() convention (bf16
autocast forward, the 1 x 1 map rounded to fp16, fp32 mean, batches of 500), on the first 500 mini base images (eval84), plain and flipped
views. Expected: bitwise-identical features. Writes results/foundation_round/clip_loader_fidelity.{json,md}.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "foundation_round"


def compare(n: int = 500) -> dict:
    import torch
    import torch.nn.functional as F

    from ccp.hosts import build_host, load_host_config
    from ccp.hosts.base import checked_ckpt, gap
    from ccp.hosts.clip_frozen import CLIP_MEAN, CLIP_STD
    from ccp.hosts.vit_input import to_model_input
    from ccp.pipeline import host_images
    from ccp.utils.paths import third_party_root
    cfg = load_host_config("mini_clip_b16")
    host = build_host(cfg)                                                     # (a)
    sys.path.insert(0, str(third_party_root() / "CLIP"))
    import clip                                                                # the official package from the pinned clone
    model, _ = clip.load(str(checked_ckpt(cfg["checkpoint"])), device="cuda", jit=False)
    model = model.float().eval()                                               # (b)

    def fwd(x_u8):
        return model.encode_image(to_model_input(x_u8, CLIP_MEAN, CLIP_STD))[:, :, None, None]
    x = torch.from_numpy(np.array(host_images(host, "train")[:n])).cuda()
    out = {}
    with torch.no_grad():
        a0, af = host.gap_pair(x)
        b0, bf = gap(fwd, x), gap(fwd, x.flip(2))
    for view, a, b in (("plain", a0, b0), ("flip", af, bf)):
        a, b = a.float(), b.float()
        out[view] = {"bitwise_identical": bool(torch.equal(a, b)), "max_abs_diff": float((a - b).abs().max()),
                     "min_cosine": float(F.cosine_similarity(a, b, dim=1).min()), "dim": int(a.shape[1])}
    out["images"] = f"the first {n} mini base images (eval84)"
    out["clip_package"] = {"ftfy": _ver("ftfy"), "regex": _ver("regex"), "clip_commit": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"}
    return out


def _ver(p):
    import importlib.metadata as m
    return m.version(p)


def main():
    r = compare()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    r.update({"when": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "git_head": head})
    (OUT / "clip_loader_fidelity.json").write_text(json.dumps(r, indent=1), encoding="utf-8", newline="\n")
    L = ["# CLIP loader fidelity (F1; base images only)", "",
         "(a) clip_frozen host path vs (b) the official `clip.load(<pinned local ViT-B-16.pt>, device='cuda', jit=False)` from the pinned clone, then "
         "`.float()` and `encode_image`; both through the registered input rule and gap() convention, on " + r["images"] + f" (code at {head[:7]}).", "",
         "| View | bitwise identical | max abs difference | min cosine |", "|---|---|---|---|"]
    for v in ("plain", "flip"):
        L.append(f"| {v} | {r[v]['bitwise_identical']} | {r[v]['max_abs_diff']:.3e} | {r[v]['min_cosine']:.9f} |")
    L += ["", f"Packages: ftfy {r['clip_package']['ftfy']}, regex {r['clip_package']['regex']}, CLIP @ d05afc4."]
    (OUT / "clip_loader_fidelity.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
