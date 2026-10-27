# MINI-DNPG dry run on validation files (Part 2e, 2026-09-28; not a test result)

Output of `scripts/final_test.py fsor mini-dnpg --dry-run-val` (valA_fsor and valB_fsor through the test code path; images decoded on the fly). All 168 AUROC / accuracy values reproduce results/fsor/dnpg_hfix.md within 3.6e-5 and the D / CCP-A / COSF / H / K rows of results/fsor/dnpg_seeds012.md within 3.6e-5 (tolerance 0.01). Per-episode evidence: research/status/fsor_final_dryrun_mini-dnpg_episodes.npz (sha256 aff436be8683d4b2...). The bundle token printed below (c1a20167ff6598f1) is the pre-evidence block; the final block token is 01fcac4666048f3e.

Enable training base class weights
Enable training base class weights
Enable training base class weights
Enable training base class weights
Enable training base class weights
Enable training base class weights

## FSOR DRY RUN ON VALIDATION FILES, part MINI-DNPG (not a test result; no approval needed)

Bundle token c1a20167ff6598f1; validation episodes (mini valA_fsor / tiered tvA_fsor), 5-way, 5 known + 5 unknown classes, 15 queries per class; DNPG max_acc.pth seeds 0-2 (released TANE/DNPG backbone; selected on validation classes); H_fix = D + w * CCP-A, K_fix = D + w2 * COSF with w / w2 frozen on valA; accuracy = DNPG classifier for every row; AUROC (unknown = positive) with 95% CI over episodes. Per-episode values: research/status/fsor_final_dryrun_mini-dnpg_episodes.npz (sha256 aff436be8683d4b2).

### 1shot, valA

| Row | seed 0 | seed 1 | seed 2 | mean ± std over seeds | accuracy mean ± std |
|---|---|---|---|---|---|
| D (DNPG own score) | 76.14 ± 0.59 | 75.89 ± 0.59 | 76.14 ± 0.58 | 76.06 ± 0.14 | 71.10 ± 0.06 |
| **H_fix (HEADLINE)** | 77.26 ± 0.57 | 77.18 ± 0.57 | 77.32 ± 0.56 | 77.25 ± 0.07 | 71.10 ± 0.06 |
| K_fix | 76.47 ± 0.59 | 76.34 ± 0.60 | 76.54 ± 0.59 | 76.45 ± 0.10 | 71.10 ± 0.06 |
| H (per-episode z, query statistics) | 77.25 ± 0.57 | 77.16 ± 0.57 | 77.31 ± 0.56 | 77.24 ± 0.07 | 71.10 ± 0.06 |
| K | 76.47 ± 0.59 | 76.34 ± 0.60 | 76.54 ± 0.59 | 76.45 ± 0.10 | 71.10 ± 0.06 |
| CCP-A | 76.66 ± 0.54 | 76.68 ± 0.54 | 76.67 ± 0.54 | 76.67 ± 0.01 | 71.10 ± 0.06 |
| COSF | 75.14 ± 0.60 | 75.20 ± 0.60 | 75.23 ± 0.60 | 75.19 ± 0.04 | 71.10 ± 0.06 |
| H_fix − D (paired) | +1.12 ± 0.12 | +1.28 ± 0.12 | +1.18 ± 0.12 | +1.19 ± 0.08 | |
| H_fix − K_fix (paired) | +0.79 ± 0.10 | +0.83 ± 0.11 | +0.78 ± 0.11 | +0.80 ± 0.03 | |

### 1shot, valB

| Row | seed 0 | seed 1 | seed 2 | mean ± std over seeds | accuracy mean ± std |
|---|---|---|---|---|---|
| D (DNPG own score) | 76.29 ± 0.57 | 76.06 ± 0.57 | 76.27 ± 0.57 | 76.21 ± 0.13 | 70.38 ± 0.02 |
| **H_fix (HEADLINE)** | 77.31 ± 0.55 | 77.23 ± 0.55 | 77.32 ± 0.55 | 77.28 ± 0.05 | 70.38 ± 0.02 |
| K_fix | 76.64 ± 0.57 | 76.51 ± 0.57 | 76.66 ± 0.57 | 76.60 ± 0.08 | 70.38 ± 0.02 |
| H (per-episode z, query statistics) | 77.30 ± 0.55 | 77.22 ± 0.55 | 77.32 ± 0.55 | 77.28 ± 0.05 | 70.38 ± 0.02 |
| K | 76.62 ± 0.57 | 76.50 ± 0.57 | 76.65 ± 0.57 | 76.59 ± 0.08 | 70.38 ± 0.02 |
| CCP-A | 76.54 ± 0.54 | 76.56 ± 0.54 | 76.53 ± 0.54 | 76.54 ± 0.01 | 70.38 ± 0.02 |
| COSF | 75.26 ± 0.57 | 75.31 ± 0.57 | 75.29 ± 0.57 | 75.29 ± 0.02 | 70.38 ± 0.02 |
| H_fix − D (paired) | +1.01 ± 0.11 | +1.17 ± 0.12 | +1.05 ± 0.12 | +1.08 ± 0.08 | |
| H_fix − K_fix (paired) | +0.67 ± 0.10 | +0.71 ± 0.10 | +0.66 ± 0.10 | +0.68 ± 0.03 | |

### 5shot, valA

| Row | seed 0 | seed 1 | seed 2 | mean ± std over seeds | accuracy mean ± std |
|---|---|---|---|---|---|
| D (DNPG own score) | 86.19 ± 0.37 | 85.95 ± 0.36 | 86.24 ± 0.37 | 86.13 ± 0.16 | 85.27 ± 0.06 |
| **H_fix (HEADLINE)** | 87.47 ± 0.34 | 87.39 ± 0.34 | 87.50 ± 0.34 | 87.46 ± 0.05 | 85.27 ± 0.06 |
| K_fix | 86.52 ± 0.38 | 86.43 ± 0.37 | 86.55 ± 0.38 | 86.50 ± 0.06 | 85.27 ± 0.06 |
| H (per-episode z, query statistics) | 87.47 ± 0.34 | 87.39 ± 0.34 | 87.49 ± 0.34 | 87.45 ± 0.05 | 85.27 ± 0.06 |
| K | 86.50 ± 0.38 | 86.41 ± 0.37 | 86.52 ± 0.38 | 86.48 ± 0.06 | 85.27 ± 0.06 |
| CCP-A | 86.43 ± 0.34 | 86.27 ± 0.34 | 86.40 ± 0.34 | 86.37 ± 0.08 | 85.27 ± 0.06 |
| COSF | 84.00 ± 0.42 | 83.84 ± 0.42 | 84.00 ± 0.42 | 83.95 ± 0.09 | 85.27 ± 0.06 |
| H_fix − D (paired) | +1.28 ± 0.09 | +1.45 ± 0.09 | +1.26 ± 0.09 | +1.33 ± 0.10 | |
| H_fix − K_fix (paired) | +0.95 ± 0.08 | +0.96 ± 0.08 | +0.95 ± 0.08 | +0.96 ± 0.00 | |

### 5shot, valB

| Row | seed 0 | seed 1 | seed 2 | mean ± std over seeds | accuracy mean ± std |
|---|---|---|---|---|---|
| D (DNPG own score) | 85.94 ± 0.37 | 85.69 ± 0.36 | 86.01 ± 0.37 | 85.88 ± 0.17 | 85.45 ± 0.05 |
| **H_fix (HEADLINE)** | 87.26 ± 0.34 | 87.16 ± 0.34 | 87.27 ± 0.34 | 87.23 ± 0.06 | 85.45 ± 0.05 |
| K_fix | 86.36 ± 0.38 | 86.26 ± 0.37 | 86.39 ± 0.37 | 86.34 ± 0.07 | 85.45 ± 0.05 |
| H (per-episode z, query statistics) | 87.25 ± 0.34 | 87.15 ± 0.34 | 87.26 ± 0.34 | 87.22 ± 0.06 | 85.45 ± 0.05 |
| K | 86.34 ± 0.38 | 86.24 ± 0.37 | 86.36 ± 0.37 | 86.31 ± 0.06 | 85.45 ± 0.05 |
| CCP-A | 86.22 ± 0.34 | 86.07 ± 0.35 | 86.19 ± 0.34 | 86.16 ± 0.08 | 85.45 ± 0.05 |
| COSF | 83.93 ± 0.42 | 83.75 ± 0.42 | 83.93 ± 0.42 | 83.87 ± 0.10 | 85.45 ± 0.05 |
| H_fix − D (paired) | +1.32 ± 0.09 | +1.47 ± 0.09 | +1.26 ± 0.09 | +1.35 ± 0.11 | |
| H_fix − K_fix (paired) | +0.90 ± 0.08 | +0.90 ± 0.08 | +0.88 ± 0.08 | +0.89 ± 0.01 | |

Decisions after this test: none (pre-registered).
