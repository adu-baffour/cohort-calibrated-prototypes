# T2 plug-in on published models (mini-ImageNet validation, information only)

What could run: the released pre-trained ResNet-12 shared by TANE and DNPG (third_party/TANE/checkpoints/miniImageNet_pre.pth, sha256 2bd34c4594b77faa3710bca3f363f54219754951e24bc353be0c1b41f3d22bbb; state-dict groups ['cls_classifier', 'feature', 'rot_classifier']; 96 backbone entries loaded strictly into our ResNet-12; base classifier [64, 640]), used as an embedder. Load check: its own 64-way classifier gives 88.36% top-1 on the 38,400 base training images (whole-image 84 x 84 view; inference only).

What could not run: TANE ATT-G / SEMAN-G and DNPG with their own openness scores. The meta-trained TANE checkpoints were never released, and DNPG's phase-2 and meta-learning checkpoints return 404 (research/fsor/PLUGIN_SCOPING.md). So no row below is the authors' model.

View: whole-image resize to 84 x 84 (their RFS-pickle geometry, reproduced from the original JPEGs, LANCZOS), their normalisation (= FEAT's). Cohort for CCP-A / CCP-S: this backbone's embeddings of the 38,400 mini-ImageNet base images. Episodes: our hashed valA_fsor / valB_fsor files.

| Score | val-A 1-shot AUROC | val-B 1-shot | val-A 5-shot | val-B 5-shot | acc val-A 1 | acc val-A 5 |
|---|---|---|---|---|---|---|
| their own openness (negative-class softmax) | not runnable | | | | | |
| COS | 72.28 ± 0.60 | 72.48 ± 0.57 | 81.06 ± 0.45 | 80.86 ± 0.45 | 67.51 | 83.94 |
| COSF | 72.53 ± 0.60 | 72.66 ± 0.57 | 81.32 ± 0.44 | 81.04 ± 0.45 | 67.96 | 84.25 |
| CCP-A | 75.03 ± 0.55 | 74.91 ± 0.54 | 84.75 ± 0.36 | 84.48 ± 0.37 | 69.09 | 84.38 |
| CCP-S | 75.16 ± 0.56 | 75.20 ± 0.55 | 85.23 ± 0.38 | 85.04 ± 0.38 | 69.45 | 84.39 |

Published test-class rows for the methods built on this backbone (research/fsor/sota.csv): ATT-G 68.11 / 72.41, 83.12 / 79.85; SEMAN-G 68.24 / 72.85, 83.48 / 82.07; DNPG 69.10 / 74.18, 83.74 / 83.64 (acc / AUROC, 1-shot then 5-shot).

GPU 0.7 min.
