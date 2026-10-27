"""FRN fine-tuning from the frozen FEAT mini checkpoint (documented; NOT run by the release verification). Produces frn/mini/seed{s}/best.pth.

    python scripts/train/train_frn.py --seed 0 [--out runs/frn] [--workers 24]

Recipe (FRN experiments/mini-ImageNet/FRN/ResNet-12_finetune/train.sh): SGD nesterov, momentum 0.9, lr 1e-3, weight decay 5e-4, 150 epochs,
x0.1 at epochs 70 and 120; 20-way 5-shot 15-query episodes; one epoch = one pass of FRN's meta_batchsampler without replacement; loss CE +
auxrank; train transform RandomResizedCrop(84) + ColorJitter(0.4, 0.4, 0.4) + RandomHorizontalFlip on the ORIGINAL JPEGs.
Recorded deviations: init = FEAT Res12-pre.pth (FRN's own pre-training stage skipped; FRN weights unavailable); head starts at scale 1,
alpha = beta = 0; normalisation = FEAT's constants for train and eval (FRN uses ImageNet's); bf16 autocast; model selection every 5 epochs on
the closed-set valA 5-shot manifest (manifests/mini/valA_5way_5shot.json: 2,000 episodes, 15 queries; FRN validates with 1,000 episodes of
16 queries) using the FRN head on the eval84 view; seeds 0, 1, 2 (torch.manual_seed(seed) and the sampler's RandomState(seed)).
Measured on one RTX 4090: 59.8 / 60.2 / 60.3 min for seeds 0 / 1 / 2 (best.pth at epochs 115 / 120 / 145, closed-set valA 5-shot 81.59 /
81.73 / 81.67), peak memory 10.2 GB (research log research/status/fsor_partC.json, not released: it also holds excluded experiments).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ccp.data import mini  # noqa: E402
from ccp.data.images import open_split  # noqa: E402
from ccp.episodes import ClosedEpisodes  # noqa: E402
from ccp.hosts.base import checked_ckpt, load_host_config, normalize_u8  # noqa: E402
from ccp.hosts.resnet12 import load_prefixed  # noqa: E402
from ccp.utils.paths import data_root  # noqa: E402
from scripts.train.frn_head import FRNHead, auxrank  # noqa: E402


class TrainAugDataset:
    """Original JPEG -> RandomResizedCrop(84) + ColorJitter(0.4, 0.4, 0.4) + HFlip -> tensor normalised with FEAT constants."""

    def __init__(self, paths):
        import torchvision.transforms as T
        self.paths = paths
        self.t = T.Compose([T.RandomResizedCrop(84), T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), T.RandomHorizontalFlip(),
                            T.ToTensor(), T.Normalize(mini.FEAT_MEAN, mini.FEAT_STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.t(Image.open(self.paths[i]).convert("RGB"))


class FrnEpochSampler:
    """FRN meta_batchsampler: draws without replacement until fewer than `way` classes have >= shot + query images left."""

    def __init__(self, c2i: dict, way: int, shot: int, query: int, seed: int):
        self.c2i, self.way, self.shot, self.query = c2i, way, shot, query
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        pools = {c: list(self.rng.permutation(v)) for c, v in self.c2i.items()}
        k = self.shot + self.query
        while True:
            ok = [c for c, v in pools.items() if len(v) >= k]
            if len(ok) < self.way:
                return
            cls = self.rng.choice(ok, self.way, replace=False)
            sup, qry = [], []
            for c in cls:
                take = [pools[c].pop() for _ in range(k)]
                sup += take[: self.shot]
                qry += take[self.shot:]
            yield sup + qry


def closed_eval(bb, head, val_arr, eps):
    import torch
    target = torch.arange(eps.way, device="cuda").repeat_interleave(eps.n_query)
    accs = []
    with torch.no_grad():
        for e in range(eps.n):
            x = torch.from_numpy(np.array(val_arr[eps.batch_indices(e) - 38400])).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fm = bb(normalize_u8(x).contiguous(memory_format=torch.channels_last))
            pred = head(fm.float(), eps.way, eps.shot)[0].argmax(1)
            accs.append((pred == target).float().mean().item() * 100)
    return float(np.mean(accs)), float(1.96 * np.std(accs) / math.sqrt(len(accs)))


def main():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "frn")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--image-root", type=Path, default=None, help="original JPEGs (default $CCP_DATA/mini/raw/miniImageNet)")
    a = ap.parse_args()
    run = a.out / f"seed{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    idx = mini.index(a.image_root or data_root() / "mini" / "raw" / "miniImageNet")
    torch.manual_seed(a.seed)
    sd = torch.load(checked_ckpt(load_host_config("frozen_feat_mini")["checkpoint"]), map_location="cpu", weights_only=False)["params"]
    bb = load_prefixed(sd, "encoder.").cuda().to(memory_format=torch.channels_last)
    head = FRNHead().cuda()
    opt = torch.optim.SGD(list(bb.parameters()) + list(head.parameters()), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 120], gamma=0.1)
    c2i = {}
    for i in np.where(idx["split"] == "train")[0]:
        c2i.setdefault(int(idx["labels"][i]), []).append(int(i))
    sampler = FrnEpochSampler({c: np.asarray(v) for c, v in c2i.items()}, 20, 5, 15, seed=a.seed)
    loader = DataLoader(TrainAugDataset(idx["paths"][:38400]), batch_sampler=sampler, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val = open_split("mini", "val", "eval84").arr
    VA5 = ClosedEpisodes("mini", "valA_5way_5shot")
    target = torch.arange(20, device="cuda").repeat_interleave(15)
    best, log, t0 = {"acc5": -1}, [], time.time()
    for epoch in range(150):
        bb.train(); head.train()
        for x in loader:
            x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fm = bb(x)
            logits, support = head(fm[:400].float(), 20, 5)
            loss = F.cross_entropy(logits, target) + auxrank(support)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        rec = {"epoch": epoch + 1, "loss": float(loss), "minutes": (time.time() - t0) / 60}
        if (epoch + 1) % 5 == 0 or epoch + 1 == 150:
            bb.eval(); head.eval()
            rec["valA5"], _ = closed_eval(bb, head, val, VA5)
            if rec["valA5"] > best["acc5"]:
                best = {"acc5": rec["valA5"], "epoch": epoch + 1}
                torch.save({"bb": bb.state_dict(), "head": head.state_dict(), "epoch": epoch + 1, "valA": best}, run / "best.pth")
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"bb": bb.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch + 1,
                    "best": best, "log": log}, run / "last.pth")


if __name__ == "__main__":
    main()
