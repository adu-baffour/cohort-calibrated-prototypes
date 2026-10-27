# Foundation-model round on TEST (single pass of bundle part yaml-fsor-foundation-round)

Bundle token 204f66d342b118be; done 2026-10-06 12:51:26 (26.1 GPU-minutes), code at 9ab5fc4. NaN found: False. Decisions after this test: none. Pretraining overlap: docs/preregistration/foundation_round.md, Section 3 (absolute numbers are not comparable with ResNet-12 FSOR methods; the claim is the paired CCP-A vs COSF comparison on identical features).

## H1: CCP-A − COSF AUROC > 0 (32 cells; Holm over one-sided Wilcoxon p)

Holm-significant cells (adjusted p < 0.05, positive mean): 16 of 32; Bonferroni-simultaneous 95% CI excludes zero: 15 of 32.

| Cell (dataset, backbone, shot) | gain ± 95% CI | Bonferroni CI | % improved | p (one-sided) | p (Holm) | headroom-normalised (A5) |
|---|---|---|---|---|---|---|
| mini|clip_b16|1 | +7.83 ± 0.51 | [+7.01, +8.64] | 93% | 1.7e-91 | 4.7e-90 | +0.428 ± 0.022 |
| mini|clip_b16|5 | +5.81 ± 0.36 | [+5.24, +6.38] | 96% | 2.0e-97 | 6.1e-96 | +0.580 ± 0.023 |
| tiered|clip_b16|1 | +1.85 ± 0.58 | [+0.91, +2.79] | 63% | 4.2e-12 | 8.8e-11 | +0.020 ± 0.044 |
| tiered|clip_b16|5 | +0.86 ± 0.49 | [+0.07, +1.65] | 58% | 1.1e-05 | 1.9e-04 | -0.104 ± 0.064 |
| cifarfs|clip_b16|1 | +8.03 ± 0.42 | [+7.35, +8.71] | 95% | 6.2e-97 | 1.9e-95 | +0.322 ± 0.015 |
| cifarfs|clip_b16|5 | +6.02 ± 0.30 | [+5.54, +6.50] | 96% | 1.1e-97 | 3.5e-96 | +0.412 ± 0.017 |
| cub|clip_b16|1 | +5.90 ± 0.37 | [+5.30, +6.51] | 94% | 3.8e-93 | 1.1e-91 | +0.336 ± 0.018 |
| cub|clip_b16|5 | +4.59 ± 0.27 | [+4.16, +5.03] | 95% | 1.1e-93 | 3.1e-92 | +0.470 ± 0.021 |
| mini|dinov2_s14|1 | -0.60 ± 0.12 | [-0.79, -0.41] | 32% | 1.0e+00 | 1.0e+00 | -0.282 ± 0.073 |
| mini|dinov2_s14|5 | -0.05 ± 0.05 | [-0.12, +0.03] | 40% | 1.0e+00 | 1.0e+00 | -0.428 ± 0.162 |
| tiered|dinov2_s14|1 | -1.34 ± 0.26 | [-1.76, -0.92] | 35% | 1.0e+00 | 1.0e+00 | -0.199 ± 0.038 |
| tiered|dinov2_s14|5 | -0.73 ± 0.13 | [-0.95, -0.52] | 31% | 1.0e+00 | 1.0e+00 | -0.408 ± 0.101 |
| cifarfs|dinov2_s14|1 | -0.60 ± 0.22 | [-0.96, -0.24] | 40% | 1.0e+00 | 1.0e+00 | -0.150 ± 0.039 |
| cifarfs|dinov2_s14|5 | -0.28 ± 0.10 | [-0.45, -0.11] | 41% | 1.0e+00 | 1.0e+00 | -0.393 ± 0.141 |
| cub|dinov2_s14|1 | +1.67 ± 0.19 | [+1.37, +1.97] | 78% | 6.4e-58 | 1.6e-56 | +0.190 ± 0.038 |
| cub|dinov2_s14|5 | +0.94 ± 0.11 | [+0.77, +1.12] | 82% | 4.6e-62 | 1.2e-60 | +0.270 ± 0.051 |
| mini|dinov2_b14|1 | -0.67 ± 0.12 | [-0.85, -0.48] | 32% | 1.0e+00 | 1.0e+00 | -0.508 ± 0.178 |
| mini|dinov2_b14|5 | -0.17 ± 0.05 | [-0.24, -0.10] | 31% | 1.0e+00 | 1.0e+00 | -0.577 ± 0.163 |
| tiered|dinov2_b14|1 | -1.15 ± 0.21 | [-1.49, -0.82] | 32% | 1.0e+00 | 1.0e+00 | -0.294 ± 0.056 |
| tiered|dinov2_b14|5 | -0.50 ± 0.09 | [-0.64, -0.36] | 27% | 1.0e+00 | 1.0e+00 | -0.797 ± 0.301 |
| cifarfs|dinov2_b14|1 | -0.54 ± 0.22 | [-0.90, -0.18] | 42% | 1.0e+00 | 1.0e+00 | -0.262 ± 0.073 |
| cifarfs|dinov2_b14|5 | -0.27 ± 0.07 | [-0.39, -0.16] | 41% | 1.0e+00 | 1.0e+00 | -0.674 ± 0.258 |
| cub|dinov2_b14|1 | +0.61 ± 0.13 | [+0.40, +0.83] | 64% | 1.3e-23 | 3.0e-22 | +0.039 ± 0.091 |
| cub|dinov2_b14|5 | +0.29 ± 0.07 | [+0.17, +0.41] | 60% | 6.0e-20 | 1.3e-18 | +0.061 ± 0.108 |
| mini|dinov2_l14|1 | -0.88 ± 0.13 | [-1.09, -0.67] | 25% | 1.0e+00 | 1.0e+00 | -0.727 ± 0.172 |
| mini|dinov2_l14|5 | -0.21 ± 0.05 | [-0.29, -0.14] | 32% | 1.0e+00 | 1.0e+00 | -1.134 ± 0.414 |
| tiered|dinov2_l14|1 | -0.56 ± 0.17 | [-0.84, -0.28] | 38% | 1.0e+00 | 1.0e+00 | -0.217 ± 0.055 |
| tiered|dinov2_l14|5 | -0.28 ± 0.06 | [-0.39, -0.18] | 34% | 1.0e+00 | 1.0e+00 | -0.452 ± 0.132 |
| cifarfs|dinov2_l14|1 | +0.89 ± 0.29 | [+0.42, +1.35] | 59% | 3.5e-09 | 7.1e-08 | -0.015 ± 0.045 |
| cifarfs|dinov2_l14|5 | +0.46 ± 0.09 | [+0.32, +0.61] | 68% | 1.3e-23 | 3.0e-22 | -0.233 ± 0.449 |
| cub|dinov2_l14|1 | +0.23 ± 0.13 | [+0.02, +0.44] | 49% | 2.2e-06 | 4.1e-05 | -0.230 ± 0.133 |
| cub|dinov2_l14|5 | +0.07 ± 0.08 | [-0.06, +0.20] | 48% | 3.0e-05 | 5.2e-04 | -0.387 ± 0.335 |

## H2: test D1 vs test gain: Spearman rho +0.908, one-sided permutation p 0.0000 (n = 32)
## H2b: valA D1 (frozen in the bundle) vs test gain: Spearman rho +0.782, one-sided permutation p 0.0000
## H2-s (within dataset x shot, 8 strata of 4 backbones; A1): mean within-stratum rho +0.675, one-sided p 0.0003
## H2b-s: valA D1 vs test gain within strata: mean rho +0.700, one-sided p 0.0003

## H3 (exploratory, descriptive): DINOv2 S -> B -> L

| dataset | shot | COSF AUROC S / B / L | raw gain S / B / L | headroom-normalised, cell level S / B / L | headroom-normalised, paired S / B / L |
|---|---|---|---|---|---|
| mini | 1 | 94.46 / 95.96 / 96.53 | -0.60 / -0.67 / -0.88 | -0.109 / -0.165 / -0.254 | -0.282 / -0.508 / -0.727 |
| mini | 5 | 98.34 / 98.95 / 99.19 | -0.05 / -0.17 / -0.21 | -0.028 / -0.161 / -0.263 | -0.428 / -0.577 / -1.134 |
| tiered | 1 | 88.32 / 91.95 / 93.21 | -1.34 / -1.15 / -0.56 | -0.114 / -0.143 / -0.083 | -0.199 / -0.294 / -0.217 |
| tiered | 5 | 94.65 / 96.59 / 97.46 | -0.73 / -0.50 / -0.28 | -0.137 / -0.147 / -0.112 | -0.408 / -0.797 / -0.452 |
| cifarfs | 1 | 91.01 / 92.74 / 90.74 | -0.60 / -0.54 / +0.89 | -0.067 / -0.075 / +0.096 | -0.150 / -0.262 / -0.015 |
| cifarfs | 5 | 97.42 / 98.18 / 97.40 | -0.28 / -0.27 / +0.46 | -0.107 / -0.150 / +0.179 | -0.393 / -0.674 / -0.233 |
| cub | 1 | 92.55 / 96.29 / 97.14 | +1.67 / +0.61 / +0.23 | +0.224 / +0.166 / +0.081 | +0.190 / +0.039 / -0.230 |
| cub | 5 | 96.55 / 98.11 / 98.24 | +0.94 / +0.29 / +0.07 | +0.274 / +0.152 / +0.040 | +0.270 / +0.061 / -0.387 |
