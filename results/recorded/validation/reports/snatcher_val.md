# SnaTCHer-F host on validation (mini valA / valB, tiered tvA / tvB)

Pre-registration: research/snatcher/SNATCHER_SETUP.md section 2 (commit be8d61a, before any valB / tvB scoring). FEAT's released meta-trained ResNet-12 per shot (one model per shot, no seeds); tiered input BGR. 1,000 episodes per file (5 known + 5 unknown classes, 15 queries each). Cells: AUROC ± 95% CI / accuracy ± 95% CI. Accuracy: FEAT's classifier for S, H_fix, K_fix, Dist; own argmax for CCP-A and COSF alone. w / w2 frozen from set A (research/status/snatcher_weights.json).

## Weights (set A, frozen before set B)

| Dataset | Shot | w | w2 | frozen at |
|---|---|---|---|---|
| mini | 1 | 39.1849 | 36.4960 | 2026-09-30 09:09:03 |
| mini | 5 | 47.2292 | 44.8809 | 2026-09-30 09:09:19 |
| tiered | 1 | 42.7782 | 41.5850 | 2026-09-30 09:10:25 |
| tiered | 5 | 48.4866 | 47.4243 | 2026-09-30 09:11:27 |

## mini, 1-shot

| Row | valA | valB |
|---|---|---|
| S (SnaTCHer) | 74.89 ± 0.59 / 71.29 ± 0.66 | 74.99 ± 0.57 / 70.63 ± 0.66 |
| **H_fix = S + w · CCP-A** | 76.64 ± 0.55 / 71.29 ± 0.66 | 76.66 ± 0.54 / 70.63 ± 0.66 |
| K_fix = S + w2 · COSF | 75.62 ± 0.58 / 71.29 ± 0.66 | 75.70 ± 0.56 / 70.63 ± 0.66 |
| CCP-A | 76.78 ± 0.53 / 71.66 ± 0.67 | 76.64 ± 0.53 / 71.16 ± 0.68 |
| COSF | 75.26 ± 0.58 / 71.11 ± 0.68 | 75.27 ± 0.56 / 70.53 ± 0.68 |
| Dist (FEAT, −max logit) | 73.69 ± 0.60 / 71.29 ± 0.66 | 73.75 ± 0.59 / 70.63 ± 0.66 |
| H_fix − S (paired) | +1.75 ± 0.15 | +1.67 ± 0.15 |
| H_fix − K_fix (paired) | +1.02 ± 0.09 | +0.96 ± 0.08 |

## mini, 5-shot

| Row | valA | valB |
|---|---|---|
| S (SnaTCHer) | 84.56 ± 0.42 / 85.16 ± 0.41 | 84.55 ± 0.42 / 85.20 ± 0.40 |
| **H_fix = S + w · CCP-A** | 86.51 ± 0.36 / 85.16 ± 0.41 | 86.32 ± 0.37 / 85.20 ± 0.40 |
| K_fix = S + w2 · COSF | 85.28 ± 0.39 / 85.16 ± 0.41 | 85.09 ± 0.40 / 85.20 ± 0.40 |
| CCP-A | 86.31 ± 0.34 / 85.80 ± 0.40 | 85.98 ± 0.34 / 85.77 ± 0.38 |
| COSF | 84.62 ± 0.38 / 85.88 ± 0.40 | 84.27 ± 0.39 / 85.89 ± 0.38 |
| Dist (FEAT, −max logit) | 82.52 ± 0.47 / 85.16 ± 0.41 | 82.63 ± 0.48 / 85.20 ± 0.40 |
| H_fix − S (paired) | +1.94 ± 0.12 | +1.77 ± 0.12 |
| H_fix − K_fix (paired) | +1.22 ± 0.06 | +1.23 ± 0.06 |

## tiered, 1-shot

| Row | tvA | tvB |
|---|---|---|
| S (SnaTCHer) | 73.17 ± 0.60 / 67.98 ± 0.72 | 69.17 ± 0.65 / 64.57 ± 0.72 |
| **H_fix = S + w · CCP-A** | 74.44 ± 0.58 / 67.98 ± 0.72 | 70.03 ± 0.65 / 64.57 ± 0.72 |
| K_fix = S + w2 · COSF | 73.85 ± 0.58 / 67.98 ± 0.72 | 69.83 ± 0.64 / 64.57 ± 0.72 |
| CCP-A | 74.65 ± 0.58 / 68.98 ± 0.71 | 69.98 ± 0.65 / 65.55 ± 0.72 |
| COSF | 73.53 ± 0.58 / 67.92 ± 0.73 | 69.69 ± 0.63 / 64.54 ± 0.72 |
| Dist (FEAT, −max logit) | 69.00 ± 0.61 / 67.98 ± 0.72 | 65.57 ± 0.65 / 64.57 ± 0.72 |
| H_fix − S (paired) | +1.26 ± 0.13 | +0.85 ± 0.14 |
| H_fix − K_fix (paired) | +0.59 ± 0.08 | +0.20 ± 0.08 |

## tiered, 5-shot

| Row | tvA | tvB |
|---|---|---|
| S (SnaTCHer) | 81.21 ± 0.47 / 82.88 ± 0.49 | 77.22 ± 0.56 / 80.90 ± 0.52 |
| **H_fix = S + w · CCP-A** | 82.76 ± 0.46 / 82.88 ± 0.49 | 78.25 ± 0.56 / 80.90 ± 0.52 |
| K_fix = S + w2 · COSF | 82.14 ± 0.45 / 82.88 ± 0.49 | 78.02 ± 0.54 / 80.90 ± 0.52 |
| CCP-A | 82.94 ± 0.45 / 84.06 ± 0.47 | 78.03 ± 0.56 / 82.15 ± 0.51 |
| COSF | 81.47 ± 0.44 / 83.77 ± 0.48 | 77.38 ± 0.52 / 81.88 ± 0.50 |
| Dist (FEAT, −max logit) | 75.59 ± 0.53 / 82.88 ± 0.49 | 72.37 ± 0.58 / 80.90 ± 0.52 |
| H_fix − S (paired) | +1.55 ± 0.10 | +1.03 ± 0.11 |
| H_fix − K_fix (paired) | +0.62 ± 0.07 | +0.23 ± 0.06 |

## Pre-registered rule (values 1-shot / 5-shot)

| Dataset | S1 H_fix − S on B (≥ +0.50) | S2 H_fix − K_fix on B (≥ +0.30) | S3 H_fix − S on A (> 0) | Verdict |
|---|---|---|---|---|
| mini | +1.67 / +1.77 (holds) | +0.96 / +1.23 (holds) | +1.75 / +1.94 (holds) | **GO** |
| tiered | +0.85 / +1.03 (holds) | +0.20 / +0.23 (FAILS) | +1.26 / +1.55 (holds) | **NO-GO** |
