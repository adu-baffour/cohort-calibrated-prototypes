# Tiered DNPG, seed 0 (validation only: tvA / tvB)

Pre-registration: research/dnpg/DNPG_SETUP.md section 11 (commit 8cfd64b, unchanged). Released TANE tiered backbone (stored epoch 81, consistent with test-selected max_meta); DNPG Phase 2 + meta-learning, BGR npz, per-epoch selection on the 97 tiered val classes (selected on validation classes). 1,000 episodes per file (5 known + 5 unknown classes, 15 queries each). H_fix = D + w · CCP-A, K_fix = D + w2 · COSF, w / w2 frozen from tvA before tvB was scored (research/status/dnpg_tiered_weights.json). Accuracy = DNPG classifier.

## Weights (tvA, frozen)

| Seed | Checkpoint | Shot | w | w2 | frozen at |
|---|---|---|---|---|---|
| 0 | max_acc | 1 | 2.8520 | 2.6993 | 2026-09-29 14:16:48 |
| 0 | max_acc | 5 | 3.0350 | 2.9511 | 2026-09-29 14:23:12 |
| 0 | last | 1 | 2.9212 | 2.7731 | 2026-09-29 14:19:30 |
| 0 | last | 5 | 3.0166 | 2.9366 | 2026-09-29 14:27:51 |

## max_acc (rule checkpoint), 1-shot

| Seed (epoch) | Row | tvA AUROC | tvB AUROC | accuracy tvA / tvB |
|---|---|---|---|---|
| 0 (13) | D (DNPG own; selected on validation classes) | 73.67 ± 0.60 | 69.09 ± 0.67 | 68.36 / 64.16 |
| 0 (13) | **H_fix** | 74.49 ± 0.58 | 69.73 ± 0.66 | 68.36 / 64.16 |
| 0 (13) | K_fix | 73.80 ± 0.59 | 69.20 ± 0.66 | 68.36 / 64.16 |
| 0 (13) | H (per-episode z, query statistics) | 74.48 ± 0.58 | 69.72 ± 0.66 | 68.36 / 64.16 |
| 0 (13) | K (per-episode z, query statistics) | 73.78 ± 0.59 | 69.20 ± 0.66 | 68.36 / 64.16 |
| 0 (13) | CCP-A | 74.62 ± 0.57 | 69.87 ± 0.65 | 68.36 / 64.16 |
| 0 (13) | COSF | 73.42 ± 0.58 | 68.93 ± 0.65 | 68.36 / 64.16 |
| 0 | H_fix − D (paired) | +0.82 ± 0.10 | +0.64 ± 0.11 | |
| 0 | H_fix − K_fix (paired) | +0.69 ± 0.09 | +0.53 ± 0.10 | |

## max_acc (rule checkpoint), 5-shot

| Seed (epoch) | Row | tvA AUROC | tvB AUROC | accuracy tvA / tvB |
|---|---|---|---|---|
| 0 (30) | D (DNPG own; selected on validation classes) | 81.87 ± 0.50 | 77.76 ± 0.58 | 83.75 / 81.10 |
| 0 (30) | **H_fix** | 82.85 ± 0.47 | 78.13 ± 0.58 | 83.75 / 81.10 |
| 0 (30) | K_fix | 82.13 ± 0.48 | 77.70 ± 0.58 | 83.75 / 81.10 |
| 0 (30) | H (per-episode z, query statistics) | 82.84 ± 0.47 | 78.13 ± 0.58 | 83.75 / 81.10 |
| 0 (30) | K (per-episode z, query statistics) | 82.12 ± 0.48 | 77.70 ± 0.58 | 83.75 / 81.10 |
| 0 (30) | CCP-A | 83.02 ± 0.46 | 78.15 ± 0.58 | 83.75 / 81.10 |
| 0 (30) | COSF | 81.80 ± 0.47 | 77.25 ± 0.58 | 83.75 / 81.10 |
| 0 | H_fix − D (paired) | +0.97 ± 0.10 | +0.37 ± 0.11 | |
| 0 | H_fix − K_fix (paired) | +0.72 ± 0.08 | +0.42 ± 0.10 | |

## last (last epoch, information), 1-shot

| Seed (epoch) | Row | tvA AUROC | tvB AUROC | accuracy tvA / tvB |
|---|---|---|---|---|
| 0 (40) | D (DNPG own; selected on validation classes) | 73.83 ± 0.59 | 69.04 ± 0.67 | 68.17 / 63.81 |
| 0 (40) | **H_fix** | 74.66 ± 0.57 | 69.63 ± 0.66 | 68.17 / 63.81 |
| 0 (40) | K_fix | 74.00 ± 0.58 | 69.24 ± 0.66 | 68.17 / 63.81 |
| 0 (40) | H (per-episode z, query statistics) | 74.66 ± 0.57 | 69.62 ± 0.66 | 68.17 / 63.81 |
| 0 (40) | K (per-episode z, query statistics) | 73.99 ± 0.58 | 69.24 ± 0.66 | 68.17 / 63.81 |
| 0 (40) | CCP-A | 74.76 ± 0.56 | 69.72 ± 0.65 | 68.17 / 63.81 |
| 0 (40) | COSF | 73.68 ± 0.58 | 69.05 ± 0.65 | 68.17 / 63.81 |
| 0 | H_fix − D (paired) | +0.83 ± 0.10 | +0.60 ± 0.12 | |
| 0 | H_fix − K_fix (paired) | +0.66 ± 0.10 | +0.39 ± 0.11 | |

## last (last epoch, information), 5-shot

| Seed (epoch) | Row | tvA AUROC | tvB AUROC | accuracy tvA / tvB |
|---|---|---|---|---|
| 0 (40) | D (DNPG own; selected on validation classes) | 81.90 ± 0.50 | 77.64 ± 0.58 | 83.62 / 81.08 |
| 0 (40) | **H_fix** | 82.87 ± 0.47 | 78.05 ± 0.58 | 83.62 / 81.08 |
| 0 (40) | K_fix | 82.17 ± 0.48 | 77.67 ± 0.58 | 83.62 / 81.08 |
| 0 (40) | H (per-episode z, query statistics) | 82.86 ± 0.47 | 78.05 ± 0.58 | 83.62 / 81.08 |
| 0 (40) | K (per-episode z, query statistics) | 82.17 ± 0.48 | 77.67 ± 0.58 | 83.62 / 81.08 |
| 0 (40) | CCP-A | 83.05 ± 0.46 | 78.10 ± 0.58 | 83.62 / 81.08 |
| 0 (40) | COSF | 81.89 ± 0.47 | 77.29 ± 0.58 | 83.62 / 81.08 |
| 0 | H_fix − D (paired) | +0.97 ± 0.10 | +0.41 ± 0.11 | |
| 0 | H_fix − K_fix (paired) | +0.69 ± 0.08 | +0.38 ± 0.11 | |

## Pre-registered rule (seed 0, max_acc.pth)

| | 1-shot | 5-shot | Needed | Holds |
|---|---|---|---|---|
| T1 H_fix − D, tvB | +0.64 | +0.37 | ≥ +0.50 | no |
| T2 H_fix − K_fix, tvB | +0.53 | +0.42 | ≥ +0.30 | yes |
| T3 H_fix − D, tvA | +0.82 | +0.97 | > 0 | yes |

**NO-GO**: no tiered DNPG (seeds 1-2 not trained).
