# T1: tieredImageNet confirmation of the frozen CCP candidates (training-free)

Pre-registered in the docstring of scripts/tiered_confirm.py and scripts/ccp_confirm.py before any tiered data was seen. All numbers are validation-class numbers (tvA / tvB = class-disjoint halves of the 97 validation classes); the published rows are test-class numbers.

## Data and checkpoint

- Source: learn2learn Zenodo record 7978538 (CC-BY-4.0), PNG pickles; md5 verified: train_images_png e1a48fcfa1c8b3a0b402a19ba892adfb (ok), val_images_png 6162949244106b348e24264057f5b66e (ok), test_images_png 4d9b27d03460a3b2c525c249655ae501 (ok).
- Images: train 448,695 / 351 classes, val 124,261 / 97, test 206,209 / 160 (test images not decoded or read). Decoded train channel mean (0-255): 122.0, 116.1, 104.6 (FEAT constants 120.4, 115.6, 104.5). Disk use raw + decoded 25.0 GB.
- Checkpoint: FEAT pre-training-only tiered Res12-pre.pth, Drive id 1URMvC6vb0Xd-OGEfUe-p2yV9jRrp4yY-, sha256 90bedb8ae0ad34946d2517a50d2636c0df52373fe5cc206d3a403516f5b51cf4; key groups ['encoder', 'fc'], fc [351, 640], no set-to-set (slf_attn) weights. Preprocessing: FEAT's tiered eval (ToTensor + Normalize with the Res12 constants; no resize).
- Splits: tvA 48 classes, tvB 49 classes (RandomState 20260925). Episode hashes: tvA_fsor_5way_1shot.npz c011975378c5f031; tvB_fsor_5way_1shot.npz 9beef0f061386cec; tvA_closed_5way_1shot.npz 266b0a934e4b6404; test_fsor_5way_1shot.npz 0557ec1b60692c7a; tvA_fsor_5way_5shot.npz 4104fb9e43dcc31c; tvB_fsor_5way_5shot.npz 6ad68ff10b7d2018; tvA_closed_5way_5shot.npz 0ea47ce57804fceb; test_fsor_5way_5shot.npz 789ffefb1a0ce091.

## Sanity check

| | 1-shot | 5-shot |
|---|---|---|
| frozen checkpoint, Euclidean nearest centroid, 1,000 tvA closed-set episodes (val classes) | 57.61 ± 0.67 | 78.90 ± 0.51 |
| FEAT README, ProtoNet ResNet-12 (test classes) | 68.23 | 84.03 |
| FEAT README, FEAT ResNet-12 (test classes) | 70.80 | 84.79 |

The frozen pre-training checkpoint is about 10 points below FEAT's published tiered ProtoNet accuracy. Load and preprocessing check (base training images, inference only): the checkpoint's own 351-way classifier is at chance (0.36%) with the learn2learn label order, because FEAT numbers classes by first appearance in MetaOptNet's files; after matching our classes to its classifier rows by optimal assignment (94% of classes agree with their row-wise argmax) it gives 58.8% top-1 on 20,000 base images, so the weights, the colour order and the normalisation are consistent. The gap is NOT explained: candidate causes are validation vs test classes, the learn2learn images vs MetaOptNet's npz copy, and FEAT's rows being episodically trained models. The pass rule is relative to COS on the same features, but absolute tiered numbers here are not comparable with the published rows.

**Addendum 2026-09-25 (D1, research/fsor/TIERED_GAP.md): the statement above that the colour order is consistent is WRONG.** The released tiered checkpoints (FEAT and TANE) were trained on BGR images (Ren et al.'s npz is cv2-decoded); our PIL decode is true RGB. Fed BGR, the checkpoint's own classifier gives 81.3% on base images and the tvA sanity rises to 62.81 / 82.84. T1's verdict below is unchanged (it was decided once on these RGB features); with BGR input the CCP-A gains are +1.53 / +1.17 (1-shot) and +1.57 / +0.97 (5-shot), information only.

## Candidates (AUROC / closed-set accuracy)

| Candidate | tvA 1-shot | tvB 1-shot | tvA 5-shot | tvB 5-shot |
|---|---|---|---|---|
| COS | 69.02 ± 0.57 / 61.63 | 65.82 ± 0.62 / 58.31 | 76.56 ± 0.49 / 79.45 | 73.45 ± 0.54 / 77.61 |
| COSF | 69.10 ± 0.57 / 62.11 | 65.96 ± 0.62 / 58.75 | 76.73 ± 0.49 / 79.79 | 73.53 ± 0.54 / 78.03 |
| CCP-A | 70.59 ± 0.57 / 63.57 | 67.28 ± 0.62 / 60.21 | 78.60 ± 0.49 / 80.18 | 75.30 ± 0.54 / 78.42 |
| CCP-S | 70.46 ± 0.59 / 63.68 | 67.23 ± 0.62 / 60.55 | 77.86 ± 0.52 / 79.78 | 75.21 ± 0.55 / 78.05 |

Chosen on tvA 1-shot: **CCP-A**.

## Pass rule

| Shot | Set | gain over COS | needed | accuracy change | within −0.3 | met |
|---|---|---|---|---|---|---|
| 1 | tvA | +1.57 | +1.0 | +1.93 | yes | yes |
| 1 | tvB | +1.46 | +1.0 | +1.90 | yes | yes |
| 5 | tvA | +2.03 | +0.5 | +0.73 | yes | yes |
| 5 | tvB | +1.85 | +0.5 | +0.82 | yes | yes |

**1-shot: PASS**
**5-shot: PASS**

## Published tieredImageNet rows (test classes; research/fsor/sota.csv, verified)

| Method | Venue | 1-shot acc | 1-shot AUROC | 5-shot acc | 5-shot AUROC | Table |
|---|---|---|---|---|---|---|
| SEMAN-G (ours) | CVPR 2022 | 71.06 ± 0.92 | 74.27 ± 0.77 | 86.02 ± 0.58 | 82.59 ± 0.57 | Table 2, p. 7177 (CVF PDF p. 7) |
| ATT-G (ours) | CVPR 2022 | 70.58 ± 0.93 | 73.43 ± 0.78 | 85.38 ± 0.61 | 81.64 ± 0.63 | Table 2, p. 7177 (CVF PDF p. 7) |
| k-NN [28] | CVPR 2023 | — | 73.97 | — | 80.22 | Table 1, p. 24011 (CVF PDF p. 5; identical in arXiv v2 p. 5) |
| DNPG | ACM MM 2024 | 71.52±0.32 | 75.70±0.21 | 85.54±0.22 | 82.94±0.23 | Table 1, p. 6 |
| DNPG | ACM MM 2024 | — | — | 84.71±0.69 | 83.20±0.64 | Table 6, p. 11 (Appendix C) |
| OPP+HDE | ECAI 2025 | 73.47±0.89 | 76.55±0.73 | 86.89±0.66 | 84.12±0.69 | Table 1, p. 2790 (PDF p. 5) |

GPU: features 1.9 min, total run 2.1 min; preparation (CPU decode) 2.1 min.
