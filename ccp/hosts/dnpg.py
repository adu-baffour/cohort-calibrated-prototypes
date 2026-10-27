"""DNPG (Zhang et al., ACM MM 2024) trained by us from the released TANE/DNPG pre-trained backbone (docs/checkpoints.md):
mini-ImageNet seeds 0-2 and tieredImageNet seed 0 (validation only). Code: third_party/DNPG @ 38b8fd1 plus our patches
(third_party/patches/dnpg.patch):
  - validation-based selection: DNPG's per-epoch evaluation loader ('test' partition) reads our validation classes (a hard link of the
    val file; the launcher refuses anything else)                                  -> the "test->val hard-link guard"
  - class-count patch: Phase 2 RPL / GCPL centers use the dataset's class count (64 mini, 351 tiered) instead of FC100's hard-coded 60
  - save-path patches (hard-coded Linux paths)
  - tiered word-embedding placeholders: zero arrays for few-shot-wordemb-*.npz, which DNPG loads but never uses (proof:
    results/recorded/validation/dnpg_placeholder_check.json)
  - tiered 5-shot gradient accumulation: micro-batch 1 episode, 2 micro-batches per optimiser step (env DNPG_ACCUM_EPISODES=1;
    BatchNorm sees one episode at a time). Only tiered 5-shot seed 0 used it.

Own score D (DNPG's code path, trainer/FSEval.py): FeatureNet.open_forward(test=True) -> softmax over [5 calibrated prototypes, max over
negative prototypes]; D = probability of the negative column (higher = more likely unknown); accuracy = argmax of the first 5 columns on
the known queries. Test-time preprocessing as DNPG: each support image repeated n_aug_support_samples = 5 times with RandomCrop(84,
padding 8) + RandomHorizontalFlip, torch RNG seeded per episode with 1,000,000 * shot + episode; queries ToTensor + Normalize.
CCP features: DNPG's (meta-tuned) backbone layer-4 GAP, registered convention, fp32, flip views.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ccp.hosts.base import Host, checked_ckpt, normalize_u8
from ccp.utils.paths import REPO, third_party_root
from ccp.utils.seeding import dnpg_episode_seed


def _import_dnpg():
    root = third_party_root() / "DNPG"
    if not (root / "train.py").exists():
        raise ImportError("DNPG code not found: run `python third_party/clone.py DNPG` (applies third_party/patches/dnpg.patch)")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # DNPG checkpoints pickle numpy scalars (torch >= 2.6 default is weights_only)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    shims = str(REPO / "third_party" / "shims")
    if shims not in sys.path:
        sys.path.append(shims)  # END of sys.path: a real installed package always wins
    cwd = os.getcwd()
    os.chdir(root)
    try:
        import train as dnpg_train
        from architectures.NetworkPre import FeatureNet
    finally:
        os.chdir(cwd)
    return dnpg_train, FeatureNet


def build_model(ckpt: str, phase2: str, shot: int, tiered: bool):
    """FeatureNet exactly as DNPG's train.py builds it (its own argparse with the training command's arguments), meta weights strict."""
    import torch
    dnpg_train, FeatureNet = _import_dnpg()
    ds, n_cls = ("tieredImageNet", 351) if tiered else ("miniImageNet", 64)
    a = dnpg_train.parser.parse_args(["--dataset", ds, "--data_root", str(REPO / "data" / "dnpg_unused"), "--n_ways", "5", "--n_shots", str(shot),
                                      "--gpus", "0", "--pretrained_model_path", str(phase2), "--learning_rate", "0.07", "--tunefeat", "0.0001",
                                      "--tune_part", "4", "--cosine", "--train_weight_base", "1", "--num_workers", "0"])
    a.n_train_runs = a.n_train_runs * a.n_train_para
    a.n_gpu = len(a.gpus.split(","))
    a.train_weight_base = a.train_weight_base == 1
    a.base_seman_calib = a.base_seman_calib == 1
    a.base_size = n_cls
    full = torch.load(phase2, map_location="cpu", weights_only=False)
    net = FeatureNet(a, a.restype, n_cls, (full, None))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(sd["cls_params"], strict=True)
    return net, a, {"epoch": sd.get("epoch")}


class DNPG(Host):
    kind = "dnpg"
    trained = True

    def __init__(self, cfg, shot=None, device="cuda", checkpoint: str = "max_acc"):
        super().__init__(cfg, shot, device)
        if shot not in (1, 5):
            raise ValueError("DNPG needs shot 1 or 5 (one meta-trained model per shot)")
        ck = checked_ckpt(cfg["checkpoints"][f"{checkpoint}_{shot}shot"])
        p2 = checked_ckpt(cfg["checkpoints"]["phase2"])
        net, self.args, self.meta = build_model(str(ck), str(p2), shot, tiered=cfg["dataset"] == "tiered")
        self.net = net.to(device).eval()
        self.checkpoint = checkpoint

    def encoder_forward(self, x_u8):
        f = self.net.feature
        return f.layer4(f.layer3(f.layer2(f.layer1(normalize_u8(x_u8)))))

    def own_scores(self, eps, get_img):
        """D (n, 150), 'correct' (n, 75) per known query and DNPG accuracy (n,) on Episodes `eps`; get_img(global index) -> uint8 (84, 84, 3)
        in the host's channel order."""
        import torch
        import torchvision.transforms as T
        from PIL import Image
        from ccp.data.mini import FEAT_MEAN, FEAT_STD
        norm = T.Normalize(FEAT_MEAN, FEAT_STD)
        aug = T.Compose([T.RandomCrop(84, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), norm])
        plain = T.Compose([T.ToTensor(), norm])
        K, shot = self.args.n_aug_support_samples, self.shot
        D = np.zeros((eps.n, 150), dtype=np.float64)
        acc = np.zeros(eps.n)
        C = np.zeros((eps.n, 75), dtype=bool)
        qy = torch.arange(5).repeat_interleave(15)
        with torch.no_grad():
            for e in range(eps.n):
                torch.manual_seed(dnpg_episode_seed(shot, e))
                sup = eps.support[e]
                sx = torch.stack([aug(Image.fromarray(get_img(i))) for c in range(5) for _ in range(K) for i in sup[c]])
                qx = torch.stack([plain(Image.fromarray(get_img(i))) for i in eps.query[e].reshape(-1)])
                ux = torch.stack([plain(Image.fromarray(get_img(i))) for i in eps.unk_query[e].reshape(-1)])
                sy = torch.arange(5).repeat_interleave(K * shot)
                uy = torch.full((75,), 5)
                imgs = tuple(x.unsqueeze(0).to(self.device) for x in (sx, qx, sx, ux))
                labs = tuple(y.unsqueeze(0).to(self.device) for y in (sy, qy, sy, uy))
                conj = (torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 5, dtype=torch.long))
                _, _, (qp, up) = self.net(imgs, labs, conj, test=True)
                probs = torch.cat([qp[0], up[0]]).cpu().numpy()
                D[e] = probs[:, -1]
                C[e] = probs[:75, :5].argmax(1) == qy.numpy()
                acc[e] = float(C[e].mean() * 100)
        return {"D": D, "correct": C}, acc
