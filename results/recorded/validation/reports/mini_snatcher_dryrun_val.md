# MINI-SNATCHER dry run on validation files (2026-09-30; not a test result)

Output of `scripts/final_test.py fsor mini-snatcher --dry-run-val` (valA_fsor and valB_fsor through the test code path). Every row (48 AUROC / accuracy values) and both paired differences reproduce results/fsor/snatcher_val.md exactly (max |difference| 0). Evidence: research/status/fsor_final_dryrun_mini-snatcher_episodes.npz (sha256 7e9d7d0d516562ba...). The token printed below (9d7f11a6bea708fc) is the pre-evidence block; the final block token is in CAMPAIGN_LOG.md.


## FSOR DRY RUN ON VALIDATION FILES, part MINI-SNATCHER (not a test result; no approval needed)

Bundle token 9d7f11a6bea708fc; validation episodes (mini valA_fsor / tiered tvA_fsor), 5-way, 5 known + 5 unknown classes, 15 queries per class; FEAT's released meta-trained ResNet-12 (one model per shot); SnaTCHer-F score S; H_fix = S + w * CCP-A, K_fix = S + w2 * COSF with w / w2 frozen on valA; accuracy = FEAT classifier for S / H_fix / K_fix / Dist, own argmax for CCP-A / COSF alone; AUROC (unknown = positive) with 95% CI over episodes. Per-episode values: research/status/fsor_final_dryrun_mini-snatcher_episodes.npz (sha256 7e9d7d0d516562ba).

### 1shot, valA

| Row | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|
| S (SnaTCHer-F score) | 71.29 ± 0.66 | 74.89 ± 0.59 |
| **H_fix = S + w · CCP-A (HEADLINE)** | 71.29 ± 0.66 | 76.64 ± 0.55 |
| K_fix = S + w2 · COSF | 71.29 ± 0.66 | 75.62 ± 0.58 |
| CCP-A alone | 71.66 ± 0.67 | 76.78 ± 0.53 |
| COSF alone | 71.11 ± 0.68 | 75.26 ± 0.58 |
| Dist (FEAT, −max logit) | 71.29 ± 0.66 | 73.69 ± 0.60 |
| H_fix − S (paired) | | +1.75 ± 0.15 |
| H_fix − K_fix (paired) | | +1.02 ± 0.09 |

### 1shot, valB

| Row | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|
| S (SnaTCHer-F score) | 70.63 ± 0.66 | 74.99 ± 0.57 |
| **H_fix = S + w · CCP-A (HEADLINE)** | 70.63 ± 0.66 | 76.66 ± 0.54 |
| K_fix = S + w2 · COSF | 70.63 ± 0.66 | 75.70 ± 0.56 |
| CCP-A alone | 71.16 ± 0.68 | 76.64 ± 0.53 |
| COSF alone | 70.53 ± 0.68 | 75.27 ± 0.56 |
| Dist (FEAT, −max logit) | 70.63 ± 0.66 | 73.75 ± 0.59 |
| H_fix − S (paired) | | +1.67 ± 0.15 |
| H_fix − K_fix (paired) | | +0.96 ± 0.08 |

### 5shot, valA

| Row | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|
| S (SnaTCHer-F score) | 85.16 ± 0.41 | 84.56 ± 0.42 |
| **H_fix = S + w · CCP-A (HEADLINE)** | 85.16 ± 0.41 | 86.51 ± 0.36 |
| K_fix = S + w2 · COSF | 85.16 ± 0.41 | 85.28 ± 0.39 |
| CCP-A alone | 85.80 ± 0.40 | 86.31 ± 0.34 |
| COSF alone | 85.88 ± 0.40 | 84.62 ± 0.38 |
| Dist (FEAT, −max logit) | 85.16 ± 0.41 | 82.52 ± 0.47 |
| H_fix − S (paired) | | +1.94 ± 0.12 |
| H_fix − K_fix (paired) | | +1.22 ± 0.06 |

### 5shot, valB

| Row | Accuracy ± 95% CI | AUROC ± 95% CI |
|---|---|---|
| S (SnaTCHer-F score) | 85.20 ± 0.40 | 84.55 ± 0.42 |
| **H_fix = S + w · CCP-A (HEADLINE)** | 85.20 ± 0.40 | 86.32 ± 0.37 |
| K_fix = S + w2 · COSF | 85.20 ± 0.40 | 85.09 ± 0.40 |
| CCP-A alone | 85.77 ± 0.38 | 85.98 ± 0.34 |
| COSF alone | 85.89 ± 0.38 | 84.27 ± 0.39 |
| Dist (FEAT, −max logit) | 85.20 ± 0.40 | 82.63 ± 0.48 |
| H_fix − S (paired) | | +1.77 ± 0.12 |
| H_fix − K_fix (paired) | | +1.23 ± 0.06 |

Decisions after this test: none (pre-registered).
