## FSOR FINAL TEST, part MINI (single evaluation of the registered bundle part, approved by the human)

Bundle token eaa3aab9d353da1e; 600 test episodes per shot, 5-way, 5 known + 5 unknown classes, 15 queries per class; accuracy and AUROC (unknown = positive) with 95% CI over episodes.

| Dataset | Backbone | Row | Shot | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|---|---|---|
| mini-ImageNet | frozen_FEAT | COS | 1 | 63.28 ± 0.84 | 68.27 ± 0.79 |
| mini-ImageNet | frozen_FEAT | COSF | 1 | 63.33 ± 0.85 | 68.42 ± 0.79 |
| mini-ImageNet | frozen_FEAT | CCP-A | 1 | 64.95 ± 0.83 | 70.90 ± 0.71 |
| mini-ImageNet | frozen_FEAT | CCP-S | 1 | 65.84 ± 0.82 | 71.27 ± 0.76 |
| mini-ImageNet | frozen_FEAT | COS | 5 | 81.79 ± 0.55 | 74.64 ± 0.76 |
| mini-ImageNet | frozen_FEAT | COSF | 5 | 82.02 ± 0.54 | 74.64 ± 0.76 |
| mini-ImageNet | frozen_FEAT | CCP-A | 5 | 82.37 ± 0.53 | 78.88 ± 0.63 |
| mini-ImageNet | frozen_FEAT | CCP-S | 5 | 82.53 ± 0.54 | 79.00 ± 0.68 |
| mini-ImageNet | FRN_seed0 | COS | 1 | 65.18 ± 0.85 | 69.40 ± 0.86 |
| mini-ImageNet | FRN_seed0 | COSF | 1 | 65.22 ± 0.84 | 69.47 ± 0.86 |
| mini-ImageNet | FRN_seed0 | CCP-A | 1 | 65.27 ± 0.86 | 70.59 ± 0.79 |
| mini-ImageNet | FRN_seed0 | CCP-S | 1 | 66.16 ± 0.84 | 70.86 ± 0.83 |
| mini-ImageNet | FRN_seed0 | COS | 5 | 80.36 ± 0.57 | 75.37 ± 0.79 |
| mini-ImageNet | FRN_seed0 | COSF | 5 | 80.36 ± 0.57 | 75.41 ± 0.79 |
| mini-ImageNet | FRN_seed0 | CCP-A | 5 | 80.44 ± 0.56 | 77.86 ± 0.67 |
| mini-ImageNet | FRN_seed0 | CCP-S | 5 | 80.68 ± 0.56 | 77.85 ± 0.72 |
| mini-ImageNet | FRN_seed1 | COS | 1 | 65.01 ± 0.85 | 69.59 ± 0.84 |
| mini-ImageNet | FRN_seed1 | COSF | 1 | 65.13 ± 0.85 | 69.68 ± 0.84 |
| mini-ImageNet | FRN_seed1 | CCP-A | 1 | 64.99 ± 0.84 | 70.54 ± 0.78 |
| mini-ImageNet | FRN_seed1 | CCP-S | 1 | 65.88 ± 0.83 | 70.81 ± 0.82 |
| mini-ImageNet | FRN_seed1 | COS | 5 | 80.13 ± 0.58 | 75.81 ± 0.76 |
| mini-ImageNet | FRN_seed1 | COSF | 5 | 80.24 ± 0.57 | 75.84 ± 0.76 |
| mini-ImageNet | FRN_seed1 | CCP-A | 5 | 80.20 ± 0.57 | 77.82 ± 0.66 |
| mini-ImageNet | FRN_seed1 | CCP-S | 5 | 80.55 ± 0.56 | 77.70 ± 0.73 |
| mini-ImageNet | FRN_seed2 | COS | 1 | 65.33 ± 0.86 | 69.69 ± 0.85 |
| mini-ImageNet | FRN_seed2 | COSF | 1 | 65.37 ± 0.85 | 69.80 ± 0.85 |
| mini-ImageNet | FRN_seed2 | CCP-A | 1 | 65.38 ± 0.85 | 70.77 ± 0.78 |
| mini-ImageNet | FRN_seed2 | CCP-S | 1 | 66.35 ± 0.85 | 71.12 ± 0.82 |
| mini-ImageNet | FRN_seed2 | COS | 5 | 80.30 ± 0.58 | 75.66 ± 0.78 |
| mini-ImageNet | FRN_seed2 | COSF | 5 | 80.46 ± 0.58 | 75.70 ± 0.78 |
| mini-ImageNet | FRN_seed2 | CCP-A | 5 | 80.35 ± 0.56 | 77.96 ± 0.66 |
| mini-ImageNet | FRN_seed2 | CCP-S | 5 | 80.85 ± 0.55 | 77.94 ± 0.72 |
| mini-ImageNet | TANE_DNPG_pretrain | COS | 1 | 65.41 ± 0.81 | 70.21 ± 0.78 |
| mini-ImageNet | TANE_DNPG_pretrain | COSF | 1 | 65.71 ± 0.81 | 70.37 ± 0.78 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-A | 1 | 67.10 ± 0.80 | 72.92 ± 0.69 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-S | 1 | 67.51 ± 0.81 | 73.24 ± 0.70 |
| mini-ImageNet | TANE_DNPG_pretrain | COS | 5 | 82.77 ± 0.57 | 76.74 ± 0.75 |
| mini-ImageNet | TANE_DNPG_pretrain | COSF | 5 | 83.04 ± 0.57 | 76.92 ± 0.75 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-A | 5 | 83.25 ± 0.56 | 81.35 ± 0.58 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-S | 5 | 83.22 ± 0.57 | 81.81 ± 0.60 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed0 | CCP-A | 1 | 67.00 ± 0.84 | 71.87 ± 0.75 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed0 | CCP-A | 5 | 82.80 ± 0.53 | 79.41 ± 0.64 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed1 | CCP-A | 1 | 66.80 ± 0.83 | 71.83 ± 0.75 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed1 | CCP-A | 5 | 82.71 ± 0.53 | 79.37 ± 0.64 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed2 | CCP-A | 1 | 67.10 ± 0.85 | 71.96 ± 0.75 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed2 | CCP-A | 5 | 82.88 ± 0.53 | 79.46 ± 0.64 |

| Summary over seeds (mean ± std) | Shot | Accuracy | AUROC |
|---|---|---|---|
| **PRIMARY**: FUSION frozen FEAT + FRN seed s (CCP-A), s = 0, 1, 2 | 1 | 66.97 ± 0.15 | 71.89 ± 0.07 |
| **PRIMARY**: FUSION frozen FEAT + FRN seed s (CCP-A), s = 0, 1, 2 | 5 | 82.80 ± 0.09 | 79.41 ± 0.04 |
| **SECONDARY**: frozen FEAT + CCP-A (single backbone, training-free; 95% CI) | 1 | 64.95 ± 0.83 | 70.90 ± 0.71 |
| **SECONDARY**: frozen FEAT + CCP-A (single backbone, training-free; 95% CI) | 5 | 82.37 ± 0.53 | 78.88 ± 0.63 |
| FRN seeds 0-2, COS (non-headline) | 1 | 65.17 ± 0.16 | 69.56 ± 0.15 |
| FRN seeds 0-2, COSF (non-headline) | 1 | 65.24 ± 0.12 | 69.65 ± 0.16 |
| FRN seeds 0-2, CCP-A (non-headline) | 1 | 65.21 ± 0.20 | 70.63 ± 0.12 |
| FRN seeds 0-2, CCP-S (non-headline) | 1 | 66.13 ± 0.23 | 70.93 ± 0.16 |
| FRN seeds 0-2, COS (non-headline) | 5 | 80.26 ± 0.12 | 75.62 ± 0.22 |
| FRN seeds 0-2, COSF (non-headline) | 5 | 80.35 ± 0.11 | 75.65 ± 0.22 |
| FRN seeds 0-2, CCP-A (non-headline) | 5 | 80.33 ± 0.12 | 77.88 ± 0.08 |
| FRN seeds 0-2, CCP-S (non-headline) | 5 | 80.69 ± 0.15 | 77.83 ± 0.12 |

Decisions after this test: none (pre-registered).

## FSOR FINAL TEST, part TIERED (single evaluation of the registered bundle part, approved by the human)

Bundle token fd048685b138b1f9; 600 test episodes per shot, 5-way, 5 known + 5 unknown classes, 15 queries per class; accuracy and AUROC (unknown = positive) with 95% CI over episodes.

| Dataset | Backbone | Row | Shot | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|---|---|---|
| tieredImageNet | frozen_FEAT_tiered (BGR) | COS | 1 | 69.68 ± 0.92 | 73.99 ± 0.79 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | COSF | 1 | 70.18 ± 0.93 | 74.14 ± 0.79 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-A | 1 | 70.83 ± 0.91 | 74.56 ± 0.81 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-S | 1 | 71.27 ± 0.92 | 75.15 ± 0.81 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | COS | 5 | 85.27 ± 0.60 | 82.32 ± 0.59 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | COSF | 5 | 85.66 ± 0.60 | 82.44 ± 0.59 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-A | 5 | 85.72 ± 0.60 | 82.79 ± 0.61 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-S | 5 | 85.66 ± 0.61 | 82.49 ± 0.64 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COS | 1 | 69.40 ± 0.89 | 74.06 ± 0.78 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COSF | 1 | 69.75 ± 0.91 | 74.27 ± 0.78 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-A | 1 | 70.03 ± 0.90 | 74.34 ± 0.79 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-S | 1 | 70.35 ± 0.89 | 74.64 ± 0.80 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COS | 5 | 84.96 ± 0.60 | 81.85 ± 0.60 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COSF | 5 | 85.20 ± 0.60 | 82.06 ± 0.60 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-A | 5 | 85.15 ± 0.59 | 82.32 ± 0.61 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-S | 5 | 84.94 ± 0.60 | 81.85 ± 0.65 |
| tieredImageNet | FUSION frozen_FEAT_tiered+TANE_tiered_pretrain (BGR; non-headline) | CCP-A | 1 | 72.36 ± 0.91 | 75.63 ± 0.80 |
| tieredImageNet | FUSION frozen_FEAT_tiered+TANE_tiered_pretrain (BGR; non-headline) | CCP-A | 5 | 86.86 ± 0.58 | 83.52 ± 0.60 |

| Headline | Shot | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|---|
| **HEADLINE**: frozen_FEAT_tiered + CCP-A (BGR input, single backbone, training-free) | 1 | 70.83 ± 0.91 | 74.56 ± 0.81 |
| **HEADLINE**: frozen_FEAT_tiered + CCP-A (BGR input, single backbone, training-free) | 5 | 85.72 ± 0.60 | 82.79 ± 0.61 |

Decisions after this test: none (pre-registered).

## Notes on the FSOR test blocks (added 2026-09-28; no recorded number is changed)

- Query statistics in the fusion rows. The MINI PRIMARY FUSION rows (frozen FEAT + FRN seed s, CCP-A) and the TIERED FUSION row
  (frozen FEAT tiered + TANE tiered, CCP-A, non-headline) score each query with z(score 1) + z(score 2), where each z uses the mean and
  standard deviation of that episode's 150 query scores. The fused score of a query therefore depends on the other queries of the
  episode (query-batch statistics); these rows are not strictly inductive. The single-backbone rows (COS, COSF, CCP-A, CCP-S, including
  the MINI SECONDARY and the TIERED HEADLINE) use no query statistics. See research/fsor/PROTOCOL.md, "Query statistics in fused scores".

## FSOR FINAL TEST, part MINI-DNPG (single evaluation of the registered bundle part, approved by the human)

Bundle token 01fcac4666048f3e; 600 test episodes per shot, 5-way, 5 known + 5 unknown classes, 15 queries per class; DNPG max_acc.pth seeds 0-2 (released TANE/DNPG backbone; selected on validation classes); H_fix = D + w * CCP-A, K_fix = D + w2 * COSF with w / w2 frozen on valA; accuracy = DNPG classifier for every row; AUROC (unknown = positive) with 95% CI over episodes. Per-episode values: research/fsor/final_mini_dnpg_episodes.npz (sha256 c42ea5f8a947c01e).

### 1shot, test

| Row | seed 0 | seed 1 | seed 2 | mean ± std over seeds | accuracy mean ± std |
|---|---|---|---|---|---|
| D (DNPG own score) | 73.95 ± 0.76 | 73.58 ± 0.77 | 73.64 ± 0.77 | 73.72 ± 0.20 | 68.91 ± 0.15 |
| **H_fix (HEADLINE)** | 74.45 ± 0.75 | 74.22 ± 0.75 | 74.24 ± 0.75 | 74.30 ± 0.13 | 68.91 ± 0.15 |
| K_fix | 73.47 ± 0.79 | 73.22 ± 0.80 | 73.25 ± 0.80 | 73.31 ± 0.14 | 68.91 ± 0.15 |
| H (per-episode z, query statistics) | 74.44 ± 0.75 | 74.22 ± 0.75 | 74.23 ± 0.75 | 74.30 ± 0.13 | 68.91 ± 0.15 |
| K | 73.48 ± 0.79 | 73.22 ± 0.80 | 73.24 ± 0.80 | 73.31 ± 0.14 | 68.91 ± 0.15 |
| CCP-A | 73.63 ± 0.73 | 73.52 ± 0.73 | 73.38 ± 0.73 | 73.51 ± 0.12 | 68.91 ± 0.15 |
| COSF | 71.63 ± 0.82 | 71.57 ± 0.82 | 71.50 ± 0.82 | 71.57 ± 0.07 | 68.91 ± 0.15 |
| H_fix − D (paired) | +0.50 ± 0.14 | +0.64 ± 0.14 | +0.60 ± 0.15 | +0.58 ± 0.08 | |
| H_fix − K_fix (paired) | +0.97 ± 0.15 | +1.01 ± 0.15 | +0.99 ± 0.16 | +0.99 ± 0.02 | |

### 5shot, test

| Row | seed 0 | seed 1 | seed 2 | mean ± std over seeds | accuracy mean ± std |
|---|---|---|---|---|---|
| D (DNPG own score) | 83.47 ± 0.56 | 83.01 ± 0.58 | 82.89 ± 0.58 | 83.12 ± 0.30 | 83.88 ± 0.03 |
| **H_fix (HEADLINE)** | 84.00 ± 0.55 | 83.86 ± 0.56 | 83.68 ± 0.56 | 83.85 ± 0.16 | 83.88 ± 0.03 |
| K_fix | 82.00 ± 0.64 | 81.98 ± 0.64 | 81.74 ± 0.65 | 81.91 ± 0.15 | 83.88 ± 0.03 |
| H (per-episode z, query statistics) | 83.98 ± 0.55 | 83.84 ± 0.56 | 83.66 ± 0.56 | 83.83 ± 0.16 | 83.88 ± 0.03 |
| K | 82.04 ± 0.64 | 81.99 ± 0.64 | 81.76 ± 0.64 | 81.93 ± 0.15 | 83.88 ± 0.03 |
| CCP-A | 82.18 ± 0.58 | 82.27 ± 0.58 | 82.17 ± 0.58 | 82.21 ± 0.06 | 83.88 ± 0.03 |
| COSF | 77.81 ± 0.77 | 78.04 ± 0.76 | 77.88 ± 0.77 | 77.91 ± 0.12 | 83.88 ± 0.03 |
| H_fix − D (paired) | +0.54 ± 0.13 | +0.85 ± 0.13 | +0.79 ± 0.12 | +0.73 ± 0.16 | |
| H_fix − K_fix (paired) | +2.00 ± 0.16 | +1.88 ± 0.15 | +1.95 ± 0.16 | +1.94 ± 0.06 | |

Decisions after this test: none (pre-registered).

## FSOR FINAL TEST, part MINI-SNATCHER (single evaluation of the registered bundle part, approved by the human)

Bundle token 19da50a9bf457980; 600 test episodes per shot, 5-way, 5 known + 5 unknown classes, 15 queries per class; FEAT's released meta-trained ResNet-12 (one model per shot); SnaTCHer-F score S; H_fix = S + w * CCP-A, K_fix = S + w2 * COSF with w / w2 frozen on valA; accuracy = FEAT classifier for S / H_fix / K_fix / Dist, own argmax for CCP-A / COSF alone; AUROC (unknown = positive) with 95% CI over episodes. Per-episode values: research/fsor/final_mini_snatcher_episodes.npz (sha256 a5323f41e155a7d0).

### 1shot, test

| Row | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|
| S (SnaTCHer-F score) | 67.28 ± 0.81 | 69.88 ± 0.86 |
| **H_fix = S + w · CCP-A (HEADLINE)** | 67.28 ± 0.81 | 71.55 ± 0.79 |
| K_fix = S + w2 · COSF | 67.28 ± 0.81 | 70.14 ± 0.85 |
| CCP-A alone | 67.14 ± 0.83 | 72.08 ± 0.72 |
| COSF alone | 66.27 ± 0.83 | 69.59 ± 0.84 |
| Dist (FEAT, −max logit) | 67.28 ± 0.81 | 69.10 ± 0.86 |
| H_fix − S (paired) | | +1.67 ± 0.21 |
| H_fix − K_fix (paired) | | +1.41 ± 0.14 |

### 5shot, test

| Row | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|
| S (SnaTCHer-F score) | 82.78 ± 0.55 | 77.28 ± 0.79 |
| **H_fix = S + w · CCP-A (HEADLINE)** | 82.78 ± 0.55 | 79.38 ± 0.70 |
| K_fix = S + w2 · COSF | 82.78 ± 0.55 | 77.49 ± 0.77 |
| CCP-A alone | 82.99 ± 0.53 | 80.07 ± 0.62 |
| COSF alone | 83.06 ± 0.53 | 76.70 ± 0.76 |
| Dist (FEAT, −max logit) | 82.78 ± 0.55 | 75.80 ± 0.81 |
| H_fix − S (paired) | | +2.10 ± 0.17 |
| H_fix − K_fix (paired) | | +1.89 ± 0.12 |

Decisions after this test: none (pre-registered).
