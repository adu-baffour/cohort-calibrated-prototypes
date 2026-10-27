"""CIFAR-FS backbone of the final round: RFS supervised pre-training (Tian et al., ECCV 2020), no distillation, recipe as published.

    python scripts/train/train_rfs_cifarfs.py --seed 0          # the registered run (seed 1 only if seed 0 crashes or diverges, once)

Recipe (docs/preregistration/final_round.md, Section A; RFS @ f8c837b train_supervised.py, models/util.py, dataset/transform_cfg.py; the CIFAR-FS
schedule from the paper's Sec. 4.1 and the RFS author's recipe in WangYueFt/rfs issue #5):
  model      RFS's own resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=64), third_party/RFS @ f8c837b models/resnet.py
  data       the 64 CIFAR-FS base classes, 38,400 images 32 x 32 (CCP_DATA/cifarfs/prepared/eval32_train.npy)
  augment    transform D: RandomCrop(32, padding=4), ColorJitter(0.4, 0.4, 0.4), RandomHorizontalFlip, ToTensor, Normalize(RFS CIFAR)
  optimiser  SGD lr 0.05, momentum 0.9, weight decay 5e-4, batch 64 (shuffle, drop_last), cross-entropy, fp32 (no AMP)
  schedule   90 epochs; lr = 0.05 * 0.1 ** (number of {45, 60, 75} strictly below the epoch number), epochs numbered 1..90
  per epoch  RFS's validate() pass: an eval-mode forward over all 38,400 training images (test transform, batch 32, in order). It changes no
             weight, but it advances every block's DropBlock counter exactly as in RFS, so it is part of the recipe.
  selection  none: the last epoch is the model (RFS saves {model}_last.pth).
Deviations stated in the pre-registration: seed set (RFS sets none), data-loader workers 0 (RFS 8), cudnn.benchmark off. A NaN loss stops the run (divergence -> one rerun with seed 1, same recipe).
Outputs: $CCP_ASSETS/checkpoints/rfs/cifarfs/seed{S}/last.pth ({'model', 'epoch', 'recipe'}), train_log.json (per epoch: lr, loss, accuracy,
train-image eval accuracy, wall time).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccp.data.images import labels as split_labels, open_split  # noqa: E402
from ccp.hosts.rfs_frozen import RFS_MEAN, RFS_STD  # noqa: E402
from ccp.utils.paths import assets_root, sha256_file  # noqa: E402

RECIPE = {"epochs": 90, "lr": 0.05, "lr_decay_epochs": [45, 60, 75], "lr_decay_rate": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
          "batch_size": 64, "eval_batch_size": 32, "drop_rate": 0.1, "dropblock_size": 2, "num_classes": 64}


def lr_at(epoch: int) -> float:
    """RFS util.adjust_learning_rate: steps = sum(epoch > decay epochs)."""
    steps = int(np.sum(epoch > np.asarray(RECIPE["lr_decay_epochs"])))
    return RECIPE["lr"] * RECIPE["lr_decay_rate"] ** steps


class Base:
    """The 64 base classes; transform D (train) or the test transform."""

    def __init__(self, arr, lab, train: bool):
        import torchvision.transforms as T
        from PIL import Image
        norm = T.Normalize(mean=list(RFS_MEAN), std=list(RFS_STD))
        if train:
            self.t = T.Compose([lambda x: Image.fromarray(x), T.RandomCrop(32, padding=4), T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                T.RandomHorizontalFlip(), lambda x: np.asarray(x), T.ToTensor(), norm])
        else:
            self.t = T.Compose([lambda x: Image.fromarray(x), T.ToTensor(), norm])
        self.arr, self.lab = arr, lab

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        return self.t(np.asarray(self.arr[i])), int(self.lab[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, choices=(0, 1))
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from ccp.hosts.rfs_resnet12 import resnet12_cifarfs
    out = assets_root() / "checkpoints" / "rfs" / "cifarfs" / f"seed{a.seed}"
    if (out / "last.pth").exists():
        raise SystemExit(f"{out / 'last.pth'} exists: the registered run is done once")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    arr = np.asarray(open_split("cifarfs", "train", "eval32").arr)
    lab = split_labels("cifarfs", "train")
    assert arr.shape == (38400, 32, 32, 3) and sorted(set(lab.tolist())) == list(range(64))
    g = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(Base(arr, lab, True), batch_size=RECIPE["batch_size"], shuffle=True, drop_last=True, num_workers=0, generator=g)
    eval_loader = DataLoader(Base(arr, lab, False), batch_size=RECIPE["eval_batch_size"], shuffle=False, drop_last=False, num_workers=0)
    model = resnet12_cifarfs(drop_rate=RECIPE["drop_rate"], dropblock_size=RECIPE["dropblock_size"], num_classes=RECIPE["num_classes"]).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=RECIPE["lr"], momentum=RECIPE["momentum"], weight_decay=RECIPE["weight_decay"])
    crit = nn.CrossEntropyLoss().cuda()
    log = []
    for epoch in range(1, RECIPE["epochs"] + 1):
        t0 = time.time()
        for pg in opt.param_groups:
            pg["lr"] = lr_at(epoch)
        model.train()
        tot, n, hit = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.float().cuda(), y.cuda()
            o = model(x)
            loss = crit(o, y)
            if not math.isfinite(loss.item()):
                json.dump({"diverged_at_epoch": epoch, "log": log}, open(out / "train_log.json", "w"), indent=1)
                raise SystemExit(f"NaN / inf loss at epoch {epoch}: divergence (pre-registered: one rerun with seed 1)")
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
            hit += int((o.argmax(1) == y).sum())
        model.eval()                       # RFS validate(): eval forward over the training images (advances the DropBlock counters)
        ehit = 0
        with torch.no_grad():
            for x, y in eval_loader:
                ehit += int((model(x.float().cuda()).argmax(1) == y.cuda()).sum())
        log.append({"epoch": epoch, "lr": lr_at(epoch), "loss": tot / n, "train_acc": hit / n * 100, "train_images_eval_acc": ehit / len(lab) * 100,
                    "seconds": time.time() - t0, "dropblock_counter_layer4": int(model.layer4[0].num_batches_tracked)})
        print(json.dumps(log[-1]), flush=True)
        json.dump({"seed": a.seed, "recipe": RECIPE, "log": log}, open(out / "train_log.json", "w"), indent=1)
    torch.save({"model": model.state_dict(), "epoch": RECIPE["epochs"], "recipe": RECIPE, "seed": a.seed}, out / "last.pth")
    print(f"saved {out / 'last.pth'} sha256 {sha256_file(out / 'last.pth')}", flush=True)


if __name__ == "__main__":
    main()
