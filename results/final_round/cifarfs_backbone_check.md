# CIFAR-FS backbone: technical check (validation only)

Requested before the final-round bundle approval; changes no registered choice and no code or configuration (`checks/cifarfs_backbone_check.py`, code at 33dfa48).

## 1. Feature equivalence (1,000 CIFAR-FS val images)

(a) host pipeline of cifarfs_rfs_seed0: registered GAP (bf16 autocast forward, fp16-rounded layer-4 map, fp32 mean), plain view. (b) RFS's own model, fp32, eval mode, RFS test transform, avg-pooled 640-d feature.

| Cosine (a, b) min | mean | 1st percentile | max abs difference per dimension | mean relative norm difference |
|---|---|---|---|---|
| 0.999879 | 0.999984 | 0.999936 | 5.23e-02 | 5.61e-03 |

Eval mode: module training flags [False]; two eval forwards bitwise equal: True; a train-mode forward differs: True. DropBlock returns its input and dropout is the identity when not training, so both are inactive in eval mode.

## 2. RFS's evaluation on the known classes of CIFAR-FS valA (context only, no threshold)

Per episode: the 5 known classes (supports and their 75 queries); L2-normalised features, logistic regression as in RFS eval/meta_eval.py; supports tiled 5 x through RFS transform D (n_aug_support_samples = 5), queries through the test transform. Mean ± 95% CI over 1,000 episodes.

| Classifier | 1-shot | 5-shot |
|---|---|---|
| RFS LR on 640-d features, 5 augmented support copies | 62.77 ± 0.74 | 79.36 ± 0.53 |
| RFS LR on logits (RFS default use_logit), 5 augmented copies | 63.38 ± 0.76 | 78.25 ± 0.56 |
| RFS LR on 640-d features, no support augmentation | 62.48 ± 0.73 | 79.26 ± 0.53 |
| ours: nearest-centroid cosine (COSF), flip views, same episodes | 63.20 | 79.78 |
| RFS published, CIFAR-FS **test**, Ours-simple (Table 2) | 71.5 ± 0.8 | 86.0 ± 0.5 |

Final epoch (90): training accuracy 97.74% (train mode, augmented batches), 99.61% on the training images in eval mode (RFS's per-epoch validate pass), loss 0.0908.

