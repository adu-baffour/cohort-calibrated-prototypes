# DNPG seed 0 as a plug-in host (R3; mini-ImageNet validation only)

DNPG (github iCGY96/DNPG @ 38b8fd1) Phase 2 + meta-learning, seed 0, from the TANE/DNPG shared pre-trained ResNet-12 (sha256 2bd34c45; released backbone, stored epoch 66, consistent with test-selected max_meta). Authors' recipe with the deviations in research/dnpg/DNPG_SETUP.md (per-epoch selection on 600 episodes of our 16 val classes). Pre-registration: DNPG_SETUP.md section 8 (committed bf5ab7e before any evaluation). Harness scripts/dnpg_eval.py; every row of a block comes from the same checkpoint. 1,000 episodes per file, 5 known + 5 unknown classes, 15 queries per class; AUROC / closed-set accuracy, 95% CI over episodes.

DNPG own rows: **selected on validation classes (optimistic vs training-free rows)**.

## max_acc.pth (headline checkpoint), 1-shot (epoch 25)

| Row | valA AUROC | valA acc | valB AUROC | valB acc |
|---|---|---|---|---|
| DNPG own score (selected on validation classes (optimistic vs training-free rows)) | 76.14 ± 0.59 | 71.05 ± 0.66 | 76.29 ± 0.57 | 70.38 ± 0.65 |
| DNPG features + COS | 74.79 ± 0.59 | 70.14 ± 0.66 | 74.94 ± 0.57 | 69.41 ± 0.66 |
| DNPG features + COSF | 75.14 ± 0.60 | 70.64 ± 0.66 | 75.26 ± 0.57 | 69.92 ± 0.66 |
| DNPG features + CCP-A | 76.66 ± 0.54 | 71.03 ± 0.66 | 76.54 ± 0.54 | 70.37 ± 0.66 |
| DNPG features + CCP-S | 76.60 ± 0.56 | 71.43 ± 0.66 | 76.68 ± 0.55 | 70.63 ± 0.65 |
| DNPG own + CCP-A, z-sum (information only; accuracy = DNPG classifier) | 77.25 ± 0.57 | 71.05 ± 0.66 | 77.30 ± 0.55 | 70.38 ± 0.65 |

## max_acc.pth (headline checkpoint), 5-shot (epoch 34)

| Row | valA AUROC | valA acc | valB AUROC | valB acc |
|---|---|---|---|---|
| DNPG own score (selected on validation classes (optimistic vs training-free rows)) | 86.19 ± 0.37 | 85.28 ± 0.41 | 85.94 ± 0.37 | 85.45 ± 0.38 |
| DNPG features + COS | 83.64 ± 0.42 | 85.19 ± 0.41 | 83.62 ± 0.42 | 85.37 ± 0.38 |
| DNPG features + COSF | 84.00 ± 0.42 | 85.63 ± 0.40 | 83.93 ± 0.42 | 85.83 ± 0.38 |
| DNPG features + CCP-A | 86.43 ± 0.34 | 85.35 ± 0.40 | 86.22 ± 0.34 | 85.51 ± 0.38 |
| DNPG features + CCP-S | 86.67 ± 0.36 | 85.56 ± 0.41 | 86.49 ± 0.36 | 85.88 ± 0.38 |
| DNPG own + CCP-A, z-sum (information only; accuracy = DNPG classifier) | 87.47 ± 0.34 | 85.28 ± 0.41 | 87.25 ± 0.34 | 85.45 ± 0.38 |

## last epoch, 40checkpoint.pth (no selection; robustness), 1-shot (epoch 40)

| Row | valA AUROC | valA acc | valB AUROC | valB acc |
|---|---|---|---|---|
| DNPG own score (selected on validation classes (optimistic vs training-free rows)) | 76.33 ± 0.59 | 71.22 ± 0.66 | 76.45 ± 0.57 | 70.44 ± 0.66 |
| DNPG features + COS | 74.86 ± 0.59 | 70.20 ± 0.67 | 75.00 ± 0.57 | 69.36 ± 0.66 |
| DNPG features + COSF | 75.26 ± 0.60 | 70.66 ± 0.67 | 75.35 ± 0.57 | 69.99 ± 0.66 |
| DNPG features + CCP-A | 76.70 ± 0.54 | 71.00 ± 0.66 | 76.57 ± 0.53 | 70.35 ± 0.67 |
| DNPG features + CCP-S | 76.61 ± 0.56 | 71.39 ± 0.66 | 76.68 ± 0.55 | 70.66 ± 0.66 |
| DNPG own + CCP-A, z-sum (information only; accuracy = DNPG classifier) | 77.32 ± 0.56 | 71.22 ± 0.66 | 77.34 ± 0.55 | 70.44 ± 0.66 |

## last epoch, 40checkpoint.pth (no selection; robustness), 5-shot (epoch 40)

| Row | valA AUROC | valA acc | valB AUROC | valB acc |
|---|---|---|---|---|
| DNPG own score (selected on validation classes (optimistic vs training-free rows)) | 86.26 ± 0.37 | 85.32 ± 0.40 | 86.00 ± 0.37 | 85.48 ± 0.38 |
| DNPG features + COS | 83.65 ± 0.43 | 85.17 ± 0.41 | 83.64 ± 0.43 | 85.35 ± 0.39 |
| DNPG features + COSF | 84.02 ± 0.42 | 85.61 ± 0.40 | 83.96 ± 0.42 | 85.83 ± 0.39 |
| DNPG features + CCP-A | 86.46 ± 0.34 | 85.35 ± 0.40 | 86.27 ± 0.34 | 85.57 ± 0.38 |
| DNPG features + CCP-S | 86.63 ± 0.36 | 85.58 ± 0.41 | 86.46 ± 0.36 | 85.86 ± 0.38 |
| DNPG own + CCP-A, z-sum (information only; accuracy = DNPG classifier) | 87.50 ± 0.34 | 85.32 ± 0.40 | 87.28 ± 0.34 | 85.48 ± 0.38 |

## Pre-registered decision (max_acc.pth, paired on the same checkpoint)

| CCP-A − DNPG own AUROC | 1-shot | 5-shot |
|---|---|---|
| valA (need ≥ +0.5 both) | +0.52 | +0.23 |
| valB (need > 0 both) | +0.25 | +0.28 |
| last epoch, valA / valB (information) | +0.37 / +0.12 | +0.19 / +0.26 |

**NO-GO**: the 5-shot valA gain (+0.24) is below +0.5. Recorded as **"CCP does not improve a trained FSOR head (DNPG)"**. No further DNPG training; the clean chain and the MINI-DNPG bundle (pre-registered for GO) are not started.

## Selection optimism (DNPG own score)

Selection episodes = DNPG's own 600 per-epoch episodes (OpenMini 'test' = our val classes, seeded by index), re-scored with DNPG's run_test_fsl; support crops use a fresh torch seed, so the re-scored value differs slightly from the value stored at selection. Gap = selection episodes − valA / valB.

| Checkpoint | Shot | Stored at selection (acc / AUROC) | Selection re-scored (acc / AUROC) | Gap vs valA (acc / AUROC) | Gap vs valB (acc / AUROC) |
|---|---|---|---|---|---|
| max_acc | 1 | 70.60 / 76.01 | 70.45 ± 0.90 / 76.10 ± 0.78 | -0.60 / -0.04 | +0.07 / -0.19 |
| max_acc | 5 | 85.32 / 86.11 | 85.24 ± 0.54 / 86.16 ± 0.49 | -0.04 / -0.03 | -0.21 / +0.22 |
| last | 1 | — | 70.53 ± 0.91 / 76.34 ± 0.79 | -0.69 / +0.01 | +0.09 / -0.12 |
| last | 5 | — | 85.23 ± 0.54 / 86.22 ± 0.49 | -0.09 / -0.05 | -0.25 / +0.21 |

The selection-minus-valA/valB gaps of max_acc are between −0.60 and +0.22, the same size as the last-epoch checkpoint's (−0.69 to +0.21) and well inside the ±0.5-0.9 episode CIs: at seed 0, per-epoch selection on 600 evaluation episodes did not measurably inflate DNPG's number. The last-epoch checkpoint is at or above max_acc on valA and valB at both shots.

## Context

DNPG's own score on validation vs the TANE/DNPG pre-trained backbone with cosine (T2 row, 72.28 / 81.06 valA AUROC): 76.14 / 86.19, +3.86 / +5.13; the paper's implied improvement holds on validation. CCP-A on the same pre-trained backbone (T2) was 75.03 / 84.75; on DNPG's meta-trained features 76.66 / 86.43.

GPU: Phase 2 45.8 min, meta 1-shot 87.7 min, meta 5-shot 135.7 min, evaluation 19.6 min: 4.8 h. Not counted: a 13 h stall of the first, parallel meta launch (0 iterations; CAMPAIGN_LOG 2026-09-26).
