"""Technical check of the CIFAR-FS backbone (final round, validation only; requested before the bundle approval). Changes no registered
choice and no file under ccp/, scripts/ or configs/; writes results/final_round/cifarfs_backbone_check.{md,json}.

    python scripts/fidelity/cifarfs_backbone_check.py

1. Feature equivalence on 1,000 CIFAR-FS val images (RandomState(20261005) sample of the 9,600, sorted): (a) the host pipeline of
   cifarfs_rfs_seed0 (registered GAP: bf16 autocast forward, fp16-rounded layer-4 map, fp32 mean; plain view), (b) RFS's own model forward in
   fp32, eval mode, RFS test transform (ToTensor + Normalize), the avg-pooled 640-d feature (forward(is_feat=True)[0][-1]). Cosine per
   image: min and mean. Eval-mode checks: every module has training == False, two eval forwards are bitwise equal (DropBlock returns its input
   and F.dropout is the identity when not training), and a train-mode forward differs (they would be active).
2. Context only (no threshold): RFS's evaluation (eval/meta_eval.py meta_test: L2-normalised features, LogisticRegression(penalty='l2',
   random_state=0, C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial'); support images tiled 5 x through train transform D as in
   RFS's MetaCIFAR100 with n_aug_support_samples = 5; queries through the test transform) on the 5 known classes of the CIFAR-FS valA FSOR
   episodes (supports + 75 known queries), 1- and 5-shot, on the 640-d features and on the logits (RFS's default use_logit=True), and without
   support augmentation. Final-epoch training accuracy from the training log.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "final_round"
SEED = 20261005
N_AUG = 5


def main():
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.linear_model import LogisticRegression

    from ccp.data.images import open_split
    from ccp.episodes import Episodes
    from ccp.hosts import build_host, load_host_config
    from ccp.hosts.rfs_frozen import RFS_MEAN, RFS_STD

    torch.manual_seed(SEED)
    cfg = load_host_config("cifarfs_rfs_seed0")
    host = build_host(cfg)                         # loads the checkpoint after its sha256 check
    net = host.net                                 # RFS's own resnet12 (third_party/RFS @ f8c837b), eval mode
    val = open_split("cifarfs", "val", "eval32")
    rows = np.sort(np.random.RandomState(SEED).choice(len(val), 1000, replace=False))
    arr = np.ascontiguousarray(np.asarray(val.arr)[rows])
    norm = T.Normalize(mean=list(RFS_MEAN), std=list(RFS_STD))
    test_t = T.Compose([lambda x: Image.fromarray(x), T.ToTensor(), norm])
    train_t = T.Compose([lambda x: Image.fromarray(x), T.RandomCrop(32, padding=4), T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                         T.RandomHorizontalFlip(), lambda x: np.asarray(x), T.ToTensor(), norm])

    # ---- 1. feature equivalence
    z0, _ = host.embed(arr)                                            # (a) registered pipeline, plain view
    a = z0.float().cuda()
    x = torch.stack([test_t(im) for im in arr]).cuda()
    net.eval()
    with torch.no_grad():
        (feats, _) = net(x, is_feat=True)
        b = feats[-1].float()                                          # (b) fp32, eval, avg-pooled 640-d
        b2 = net(x, is_feat=True)[0][-1].float()
    cos = F.cosine_similarity(a, b, dim=1).cpu().numpy()
    training_flags = sorted({m.training for m in net.modules()})
    eval_repeat_equal = bool(torch.equal(b, b2))
    net.train()
    with torch.no_grad():
        bt = net(x[:200], is_feat=True)[0][-1].float()
    net.eval()
    train_mode_differs = bool(not torch.equal(bt, b[:200]))
    eq = {"n_images": 1000, "rows": "RandomState(20261005).choice(9600, 1000, replace=False), sorted",
          "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()), "cosine_p01": float(np.quantile(cos, 0.01)),
          "max_abs_diff_per_dim": float((a - b).abs().max()), "relative_norm_diff_mean": float(((a - b).norm(dim=1) / b.norm(dim=1)).mean()),
          "eval_mode": {"module_training_flags": training_flags, "two_eval_forwards_bitwise_equal": eval_repeat_equal,
                        "train_mode_forward_differs": train_mode_differs,
                        "dropblock_in_eval": "DropBlock.forward returns x when not self.training (RFS models/resnet.py)",
                        "dropout_in_eval": "F.dropout(..., training=self.training) is the identity in eval mode"}}

    # ---- 2. RFS's evaluation on the known classes of the valA episodes (context only)
    def rfs_feats(imgs, aug):
        t = train_t if aug else test_t
        xx = torch.stack([t(im) for im in imgs]).cuda()
        with torch.no_grad():
            (fs, logit) = net(xx, is_feat=True)
        return fs[-1].float(), logit.float()

    def lr_acc(sf, sy, qf, qy):
        sf, qf = F.normalize(sf, dim=1).cpu().numpy(), F.normalize(qf, dim=1).cpu().numpy()   # meta_eval normalize (L2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(penalty="l2", random_state=0, C=1.0, solver="lbfgs", max_iter=1000, multi_class="multinomial")
            clf.fit(sf, sy)
        return float((clf.predict(qf) == qy).mean() * 100)

    off = val.offset
    full = np.asarray(val.arr)
    ctx = {}
    for sh in (1, 5):
        eps = Episodes("cifarfs", f"valA_fsor_5way_{sh}shot")
        res = {"LR_features_aug5": [], "LR_logits_aug5": [], "LR_features_noaug": []}
        for e in range(eps.n):
            sup = eps.support[e].reshape(-1) - off                       # class-major, 5 x shot
            qry = eps.query[e].reshape(-1) - off                         # 75 known queries
            sy, qy = np.repeat(np.arange(5), sh), np.repeat(np.arange(5), 15)
            sup_t = np.tile(full[sup], (N_AUG, 1, 1, 1))                 # RFS MetaCIFAR100: np.tile(support, (n_aug, 1, 1, 1))
            sy_t = np.tile(sy, N_AUG)
            sf_a, sl_a = rfs_feats(sup_t, aug=True)
            qf, ql = rfs_feats(full[qry], aug=False)
            sf_p, _ = rfs_feats(full[sup], aug=False)
            res["LR_features_aug5"].append(lr_acc(sf_a, sy_t, qf, qy))
            res["LR_logits_aug5"].append(lr_acc(sl_a, sy_t, ql, qy))
            res["LR_features_noaug"].append(lr_acc(sf_p, sy, qf, qy))
        ctx[str(sh)] = {k: {"mean": float(np.mean(v)), "ci95": float(1.96 * np.std(v) / np.sqrt(len(v)))} for k, v in res.items()}
    sel = json.loads((OUT / "cifarfs_selection_valA.json").read_text(encoding="utf-8"))["sanity_backbone"]
    log = json.loads((OUT / "train_log_cifarfs_rfs_seed0.json").read_text(encoding="utf-8"))["log"][-1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    out = {"feature_equivalence": eq, "rfs_evaluation_valA_known": ctx,
           "ours_nearest_centroid_valA": {sh: sel[sh]["COSF_closed_set_acc_valA"] for sh in ("1", "5")},
           "rfs_published_test": {"1": "71.5 +- 0.8", "5": "86.0 +- 0.5", "source": "Tian et al., ECCV 2020, Table 2, Ours-simple"},
           "final_epoch": {"epoch": log["epoch"], "train_acc_augmented_train_mode": log["train_acc"], "train_images_eval_acc": log["train_images_eval_acc"],
                           "loss": log["loss"]},
           "when": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "git_head": head}
    (OUT / "cifarfs_backbone_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    f = lambda d: f"{d['mean']:.2f} ± {d['ci95']:.2f}"  # noqa: E731
    L = ["# CIFAR-FS backbone: technical check (validation only)", "",
         "Requested before the final-round bundle approval; changes no registered choice and no code or configuration "
         f"(`scripts/fidelity/cifarfs_backbone_check.py`, code at {head[:7]}).", "",
         "## 1. Feature equivalence (1,000 CIFAR-FS val images)", "",
         "(a) host pipeline of cifarfs_rfs_seed0: registered GAP (bf16 autocast forward, fp16-rounded layer-4 map, fp32 mean), plain view. "
         "(b) RFS's own model, fp32, eval mode, RFS test transform, avg-pooled 640-d feature.", "",
         "| Cosine (a, b) min | mean | 1st percentile | max abs difference per dimension | mean relative norm difference |", "|---|---|---|---|---|",
         f"| {eq['cosine_min']:.6f} | {eq['cosine_mean']:.6f} | {eq['cosine_p01']:.6f} | {eq['max_abs_diff_per_dim']:.2e} | {eq['relative_norm_diff_mean']:.2e} |", "",
         f"Eval mode: module training flags {training_flags}; two eval forwards bitwise equal: {eval_repeat_equal}; a train-mode forward differs: "
         f"{train_mode_differs}. DropBlock returns its input and dropout is the identity when not training, so both are inactive in eval mode.", "",
         "## 2. RFS's evaluation on the known classes of CIFAR-FS valA (context only, no threshold)", "",
         "Per episode: the 5 known classes (supports and their 75 queries); L2-normalised features, logistic regression as in RFS "
         "eval/meta_eval.py; supports tiled 5 x through RFS transform D (n_aug_support_samples = 5), queries through the test transform. "
         "Mean ± 95% CI over 1,000 episodes.", "",
         "| Classifier | 1-shot | 5-shot |", "|---|---|---|",
         f"| RFS LR on 640-d features, 5 augmented support copies | {f(ctx['1']['LR_features_aug5'])} | {f(ctx['5']['LR_features_aug5'])} |",
         f"| RFS LR on logits (RFS default use_logit), 5 augmented copies | {f(ctx['1']['LR_logits_aug5'])} | {f(ctx['5']['LR_logits_aug5'])} |",
         f"| RFS LR on 640-d features, no support augmentation | {f(ctx['1']['LR_features_noaug'])} | {f(ctx['5']['LR_features_noaug'])} |",
         f"| ours: nearest-centroid cosine (COSF), flip views, same episodes | {sel['1']['COSF_closed_set_acc_valA']:.2f} | {sel['5']['COSF_closed_set_acc_valA']:.2f} |",
         "| RFS published, CIFAR-FS **test**, Ours-simple (Table 2) | 71.5 ± 0.8 | 86.0 ± 0.5 |", "",
         f"Final epoch ({log['epoch']}): training accuracy {log['train_acc']:.2f}% (train mode, augmented batches), {log['train_images_eval_acc']:.2f}% "
         f"on the training images in eval mode (RFS's per-epoch validate pass), loss {log['loss']:.4f}.", ""]
    (OUT / "cifarfs_backbone_check.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
