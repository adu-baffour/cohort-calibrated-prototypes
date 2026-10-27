# MINI-DNPG Part 1: inductive fusion H_fix on validation (DNPG seeds 0-2, max_acc.pth)

Pre-registration: research/dnpg/DNPG_SETUP.md section 10 (commit 0c2a8db, before any valB computation for these rows). H_fix = D + w · CCP-A, K_fix = D + w2 · COSF; w = pooled std(D) / pooled std(CCP-A), w2 = pooled std(D) / pooled std(COSF), ddof 0 over all 1,000 x 150 valA query scores, one constant per seed and shot. H, K = per-episode z (query statistics). D = DNPG own score (selected on validation classes; optimistic vs training-free rows). Accuracy = DNPG classifier for every row. 1,000 episodes per file.

Reproduction of the H-COMB per-episode numbers (D, CCP-A, COSF, H, K AUROC and D accuracy, all seeds, shots, valA / valB): max |difference| 8.89e-06.

## Weights (valA)

| Seed | Shot | std D | std CCP-A | std COSF | w | w2 |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.2758 | 0.1050 | 0.1147 | 2.6263 | 2.4038 |
| 0 | 5 | 0.3084 | 0.1067 | 0.1160 | 2.8890 | 2.6579 |
| 1 | 1 | 0.2787 | 0.1056 | 0.1153 | 2.6400 | 2.4174 |
| 1 | 5 | 0.3038 | 0.1043 | 0.1131 | 2.9121 | 2.6851 |
| 2 | 1 | 0.2815 | 0.1069 | 0.1170 | 2.6329 | 2.4064 |
| 2 | 5 | 0.3058 | 0.1066 | 0.1158 | 2.8675 | 2.6397 |

## valB (rule set), 1-shot (AUROC ± 95% CI; accuracy = DNPG classifier)

| Row | seed 0 | seed 1 | seed 2 | mean seeds 1-2 |
|---|---|---|---|---|
| D | 76.29 ± 0.57 | 76.06 ± 0.57 | 76.27 ± 0.57 | 76.17 |
| H_fix | 77.31 ± 0.55 | 77.23 ± 0.55 | 77.32 ± 0.55 | 77.27 |
| K_fix | 76.64 ± 0.57 | 76.51 ± 0.57 | 76.66 ± 0.57 | 76.59 |
| H (per-episode z, query statistics) | 77.30 ± 0.55 | 77.22 ± 0.55 | 77.32 ± 0.55 | 77.27 |
| K (per-episode z, query statistics) | 76.62 ± 0.57 | 76.50 ± 0.57 | 76.65 ± 0.57 | 76.58 |
| CCP-A | 76.54 ± 0.54 | 76.56 ± 0.54 | 76.53 ± 0.54 | 76.54 |
| COSF | 75.26 ± 0.57 | 75.31 ± 0.57 | 75.29 ± 0.57 | 75.30 |
| accuracy (all rows) | 70.38 ± 0.65 | 70.40 ± 0.65 | 70.37 ± 0.65 | |

## valB (rule set), 5-shot (AUROC ± 95% CI; accuracy = DNPG classifier)

| Row | seed 0 | seed 1 | seed 2 | mean seeds 1-2 |
|---|---|---|---|---|
| D | 85.94 ± 0.37 | 85.69 ± 0.36 | 86.01 ± 0.37 | 85.85 |
| H_fix | 87.26 ± 0.34 | 87.16 ± 0.34 | 87.27 ± 0.34 | 87.21 |
| K_fix | 86.36 ± 0.38 | 86.26 ± 0.37 | 86.39 ± 0.37 | 86.32 |
| H (per-episode z, query statistics) | 87.25 ± 0.34 | 87.15 ± 0.34 | 87.26 ± 0.34 | 87.20 |
| K (per-episode z, query statistics) | 86.34 ± 0.38 | 86.24 ± 0.37 | 86.36 ± 0.37 | 86.30 |
| CCP-A | 86.22 ± 0.34 | 86.07 ± 0.35 | 86.19 ± 0.34 | 86.13 |
| COSF | 83.93 ± 0.42 | 83.75 ± 0.42 | 83.93 ± 0.42 | 83.84 |
| accuracy (all rows) | 85.45 ± 0.38 | 85.40 ± 0.38 | 85.50 ± 0.38 | |

## valA (information; w / w2 fitted here), 1-shot (AUROC ± 95% CI; accuracy = DNPG classifier)

| Row | seed 0 | seed 1 | seed 2 | mean seeds 1-2 |
|---|---|---|---|---|
| D | 76.14 ± 0.59 | 75.89 ± 0.59 | 76.14 ± 0.58 | 76.02 |
| H_fix | 77.26 ± 0.57 | 77.18 ± 0.57 | 77.32 ± 0.56 | 77.25 |
| K_fix | 76.47 ± 0.59 | 76.34 ± 0.60 | 76.54 ± 0.59 | 76.44 |
| H (per-episode z, query statistics) | 77.25 ± 0.57 | 77.16 ± 0.57 | 77.31 ± 0.56 | 77.24 |
| K (per-episode z, query statistics) | 76.47 ± 0.59 | 76.34 ± 0.60 | 76.54 ± 0.59 | 76.44 |
| CCP-A | 76.66 ± 0.54 | 76.68 ± 0.54 | 76.67 ± 0.54 | 76.68 |
| COSF | 75.14 ± 0.60 | 75.20 ± 0.60 | 75.23 ± 0.60 | 75.21 |
| accuracy (all rows) | 71.05 ± 0.66 | 71.08 ± 0.66 | 71.17 ± 0.66 | |

## valA (information; w / w2 fitted here), 5-shot (AUROC ± 95% CI; accuracy = DNPG classifier)

| Row | seed 0 | seed 1 | seed 2 | mean seeds 1-2 |
|---|---|---|---|---|
| D | 86.19 ± 0.37 | 85.95 ± 0.36 | 86.24 ± 0.37 | 86.09 |
| H_fix | 87.47 ± 0.34 | 87.39 ± 0.34 | 87.50 ± 0.34 | 87.45 |
| K_fix | 86.52 ± 0.38 | 86.43 ± 0.37 | 86.55 ± 0.38 | 86.49 |
| H (per-episode z, query statistics) | 87.47 ± 0.34 | 87.39 ± 0.34 | 87.49 ± 0.34 | 87.44 |
| K (per-episode z, query statistics) | 86.50 ± 0.38 | 86.41 ± 0.37 | 86.52 ± 0.38 | 86.47 |
| CCP-A | 86.43 ± 0.34 | 86.27 ± 0.34 | 86.40 ± 0.34 | 86.34 |
| COSF | 84.00 ± 0.42 | 83.84 ± 0.42 | 84.00 ± 0.42 | 83.92 |
| accuracy (all rows) | 85.28 ± 0.41 | 85.20 ± 0.41 | 85.32 ± 0.41 | |

## Rule (valB)

| | 1-shot | 5-shot | Needed | Holds |
|---|---|---|---|---|
| E1 mean seeds 0-2 (H − H_fix) | -0.01 (-0.01, -0.01, -0.01) | -0.01 (-0.01, -0.01, -0.01) | ≤ 0.20 | yes |
| E2 mean seeds 1-2 (H_fix − D) | +1.11 (+1.17, +1.05) | +1.37 (+1.47, +1.26) | ≥ +0.50, each seed > 0 | yes |
| E3 mean seeds 1-2 (H_fix − K_fix) | +0.69 (+0.71, +0.66) | +0.89 (+0.90, +0.88) | ≥ +0.30 | yes |

**PASS**: H_fix becomes the MINI-DNPG headline; H stays as a row labelled 'per-episode z (query statistics)'.
