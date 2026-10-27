# FSOR bundle dry run on validation files (pre-flight P7, 2026-09-25; not a test result)

Output of `scripts/final_test.py fsor mini --dry-run-val` and `fsor tiered --dry-run-val` after the P1-P6 changes. MINI reproduces the D2 dry run exactly (312 values, max difference 0.0); TIERED reproduces the P1 BGR tvA reference exactly (64 values). The TIERED token printed below is the pre-approvable one; the approvable block's token is in CAMPAIGN_LOG.md.


## FSOR DRY RUN ON VALIDATION FILES, part MINI (not a test result; no approval needed)

Bundle token eaa3aab9d353da1e; validation episodes (mini valA_fsor / tiered tvA_fsor), 5-way, 5 known + 5 unknown classes, 15 queries per class; accuracy and AUROC (unknown = positive) with 95% CI over episodes.

| Dataset | Backbone | Row | Shot | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|---|---|---|
| mini-ImageNet | frozen_FEAT | COS | 1 | 69.19 ± 0.67 | 74.06 ± 0.56 |
| mini-ImageNet | frozen_FEAT | COSF | 1 | 69.53 ± 0.68 | 74.31 ± 0.56 |
| mini-ImageNet | frozen_FEAT | CCP-A | 1 | 70.53 ± 0.67 | 76.14 ± 0.52 |
| mini-ImageNet | frozen_FEAT | CCP-S | 1 | 71.20 ± 0.67 | 76.53 ± 0.53 |
| mini-ImageNet | frozen_FEAT | COS | 5 | 84.97 ± 0.41 | 82.82 ± 0.40 |
| mini-ImageNet | frozen_FEAT | COSF | 5 | 85.28 ± 0.41 | 82.99 ± 0.40 |
| mini-ImageNet | frozen_FEAT | CCP-A | 5 | 85.42 ± 0.40 | 85.46 ± 0.35 |
| mini-ImageNet | frozen_FEAT | CCP-S | 5 | 85.57 ± 0.40 | 86.16 ± 0.36 |
| mini-ImageNet | FRN_seed0 | COS | 1 | 70.30 ± 0.69 | 75.22 ± 0.59 |
| mini-ImageNet | FRN_seed0 | COSF | 1 | 70.50 ± 0.68 | 75.41 ± 0.59 |
| mini-ImageNet | FRN_seed0 | CCP-A | 1 | 70.22 ± 0.69 | 75.85 ± 0.55 |
| mini-ImageNet | FRN_seed0 | CCP-S | 1 | 70.99 ± 0.68 | 76.19 ± 0.57 |
| mini-ImageNet | FRN_seed0 | COS | 5 | 83.26 ± 0.44 | 83.81 ± 0.41 |
| mini-ImageNet | FRN_seed0 | COSF | 5 | 83.41 ± 0.44 | 83.95 ± 0.41 |
| mini-ImageNet | FRN_seed0 | CCP-A | 5 | 83.16 ± 0.44 | 84.62 ± 0.38 |
| mini-ImageNet | FRN_seed0 | CCP-S | 5 | 83.56 ± 0.44 | 85.41 ± 0.38 |
| mini-ImageNet | FRN_seed1 | COS | 1 | 70.41 ± 0.68 | 75.38 ± 0.58 |
| mini-ImageNet | FRN_seed1 | COSF | 1 | 70.69 ± 0.68 | 75.56 ± 0.58 |
| mini-ImageNet | FRN_seed1 | CCP-A | 1 | 70.27 ± 0.68 | 75.89 ± 0.55 |
| mini-ImageNet | FRN_seed1 | CCP-S | 1 | 71.08 ± 0.68 | 76.19 ± 0.57 |
| mini-ImageNet | FRN_seed1 | COS | 5 | 83.29 ± 0.44 | 83.98 ± 0.41 |
| mini-ImageNet | FRN_seed1 | COSF | 5 | 83.45 ± 0.44 | 84.12 ± 0.40 |
| mini-ImageNet | FRN_seed1 | CCP-A | 5 | 83.19 ± 0.43 | 84.66 ± 0.38 |
| mini-ImageNet | FRN_seed1 | CCP-S | 5 | 83.54 ± 0.43 | 85.36 ± 0.38 |
| mini-ImageNet | FRN_seed2 | COS | 1 | 70.54 ± 0.68 | 75.19 ± 0.59 |
| mini-ImageNet | FRN_seed2 | COSF | 1 | 70.75 ± 0.68 | 75.38 ± 0.59 |
| mini-ImageNet | FRN_seed2 | CCP-A | 1 | 70.46 ± 0.67 | 75.89 ± 0.56 |
| mini-ImageNet | FRN_seed2 | CCP-S | 1 | 71.24 ± 0.67 | 76.23 ± 0.57 |
| mini-ImageNet | FRN_seed2 | COS | 5 | 83.40 ± 0.44 | 83.81 ± 0.41 |
| mini-ImageNet | FRN_seed2 | COSF | 5 | 83.54 ± 0.44 | 83.96 ± 0.41 |
| mini-ImageNet | FRN_seed2 | CCP-A | 5 | 83.38 ± 0.43 | 84.74 ± 0.38 |
| mini-ImageNet | FRN_seed2 | CCP-S | 5 | 83.78 ± 0.43 | 85.46 ± 0.38 |
| mini-ImageNet | TANE_DNPG_pretrain | COS | 1 | 67.51 ± 0.67 | 72.28 ± 0.60 |
| mini-ImageNet | TANE_DNPG_pretrain | COSF | 1 | 67.96 ± 0.67 | 72.53 ± 0.60 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-A | 1 | 69.09 ± 0.66 | 75.03 ± 0.55 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-S | 1 | 69.45 ± 0.66 | 75.16 ± 0.56 |
| mini-ImageNet | TANE_DNPG_pretrain | COS | 5 | 83.94 ± 0.41 | 81.06 ± 0.45 |
| mini-ImageNet | TANE_DNPG_pretrain | COSF | 5 | 84.25 ± 0.41 | 81.32 ± 0.44 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-A | 5 | 84.38 ± 0.41 | 84.75 ± 0.36 |
| mini-ImageNet | TANE_DNPG_pretrain | CCP-S | 5 | 84.39 ± 0.41 | 85.23 ± 0.38 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed0 | CCP-A | 1 | 71.94 ± 0.67 | 77.22 ± 0.53 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed0 | CCP-A | 5 | 85.38 ± 0.41 | 86.11 ± 0.36 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed1 | CCP-A | 1 | 72.05 ± 0.67 | 77.25 ± 0.53 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed1 | CCP-A | 5 | 85.39 ± 0.40 | 86.12 ± 0.35 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed2 | CCP-A | 1 | 72.10 ± 0.66 | 77.24 ± 0.53 |
| mini-ImageNet | FUSION frozen_FEAT+FRN_seed2 | CCP-A | 5 | 85.49 ± 0.40 | 86.16 ± 0.36 |

| Summary over seeds (mean ± std) | Shot | Accuracy | AUROC |
|---|---|---|---|
| **PRIMARY**: FUSION frozen FEAT + FRN seed s (CCP-A), s = 0, 1, 2 | 1 | 72.03 ± 0.08 | 77.23 ± 0.01 |
| **PRIMARY**: FUSION frozen FEAT + FRN seed s (CCP-A), s = 0, 1, 2 | 5 | 85.42 ± 0.06 | 86.13 ± 0.03 |
| **SECONDARY**: frozen FEAT + CCP-A (single backbone, training-free; 95% CI) | 1 | 70.53 ± 0.67 | 76.14 ± 0.52 |
| **SECONDARY**: frozen FEAT + CCP-A (single backbone, training-free; 95% CI) | 5 | 85.42 ± 0.40 | 85.46 ± 0.35 |
| FRN seeds 0-2, COS (non-headline) | 1 | 70.42 ± 0.12 | 75.27 ± 0.10 |
| FRN seeds 0-2, COSF (non-headline) | 1 | 70.65 ± 0.13 | 75.45 ± 0.10 |
| FRN seeds 0-2, CCP-A (non-headline) | 1 | 70.32 ± 0.13 | 75.87 ± 0.02 |
| FRN seeds 0-2, CCP-S (non-headline) | 1 | 71.10 ± 0.13 | 76.20 ± 0.02 |
| FRN seeds 0-2, COS (non-headline) | 5 | 83.31 ± 0.08 | 83.87 ± 0.10 |
| FRN seeds 0-2, COSF (non-headline) | 5 | 83.47 ± 0.07 | 84.01 ± 0.10 |
| FRN seeds 0-2, CCP-A (non-headline) | 5 | 83.24 ± 0.12 | 84.67 ± 0.06 |
| FRN seeds 0-2, CCP-S (non-headline) | 5 | 83.63 ± 0.13 | 85.41 ± 0.05 |

Decisions after this test: none (pre-registered).


## FSOR DRY RUN ON VALIDATION FILES, part TIERED (not a test result; no approval needed)

Bundle token 73552c1348850aaa; validation episodes (mini valA_fsor / tiered tvA_fsor), 5-way, 5 known + 5 unknown classes, 15 queries per class; accuracy and AUROC (unknown = positive) with 95% CI over episodes.

| Dataset | Backbone | Row | Shot | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|---|---|---|
| tieredImageNet | frozen_FEAT_tiered (BGR) | COS | 1 | 67.04 ± 0.71 | 73.04 ± 0.56 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | COSF | 1 | 67.42 ± 0.71 | 73.22 ± 0.56 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-A | 1 | 68.53 ± 0.71 | 74.57 ± 0.56 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-S | 1 | 68.69 ± 0.70 | 74.76 ± 0.56 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | COS | 5 | 83.57 ± 0.48 | 81.18 ± 0.45 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | COSF | 5 | 83.82 ± 0.47 | 81.34 ± 0.45 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-A | 5 | 84.00 ± 0.47 | 82.75 ± 0.45 |
| tieredImageNet | frozen_FEAT_tiered (BGR) | CCP-S | 5 | 83.85 ± 0.47 | 82.56 ± 0.48 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COS | 1 | 64.96 ± 0.72 | 70.95 ± 0.59 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COSF | 1 | 65.27 ± 0.73 | 71.08 ± 0.59 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-A | 1 | 66.42 ± 0.72 | 72.68 ± 0.57 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-S | 1 | 66.38 ± 0.72 | 72.28 ± 0.58 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COS | 5 | 81.70 ± 0.49 | 78.35 ± 0.50 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | COSF | 5 | 81.99 ± 0.49 | 78.60 ± 0.50 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-A | 5 | 82.26 ± 0.48 | 80.80 ± 0.48 |
| tieredImageNet | TANE_tiered_pretrain (BGR) | CCP-S | 5 | 81.97 ± 0.49 | 80.23 ± 0.50 |

Decisions after this test: none (pre-registered).
