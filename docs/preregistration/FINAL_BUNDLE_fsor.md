# FSOR test bundle (registered 2026-09-25, T3; split into MINI and TIERED 2026-09-25, D2; NOT run)

Two independently approved parts, one test pass each, no decisions after either. Run only by
`scripts/final_test.py fsor mini` / `scripts/final_test.py fsor tiered` (code: `research/fsor/fsor_final.py`), after the human
writes THAT PART's token into research/FINAL_APPROVED. A part's token is the first 16 hex characters of the sha256 of its
`yaml-fsor-<part>` block below (printed by `final_test.py fsor <part> --dry-run-val`, which runs the identical pipeline on
validation files, needs no approval and reads no test image). A second pass of a part is refused
(research/FINAL_TEST_FSOR_<PART>_DONE.json). All checkpoints and episode files are sha256-pinned. Candidates are the frozen
definitions of scripts/ccp_confirm.py: COS, COSF (flip), CCP-A (Sm, top-1000 cohort, flip), CCP-S (closed-form S over the
whole cohort, flip); FUSION uses the candidate chosen in T1 (CCP-A).

Metrics (both parts): closed-set accuracy (75 known queries) and AUROC (unknown = positive), mean over 600 episodes per shot
with 95% CI over episodes; FRN-based rows (FRN seeds 0-2 and FUSION) are also reported as mean ± std over seeds. Classifiers:
argmax c_k for COS / COSF, argmax of the normalised score for CCP-A / CCP-S, and for FUSION argmax of the sum of the two
backbones' per-class score matrices, each z-scored over all entries of the episode (the FUSION AUROC score is unchanged: the
per-episode z-scored sum of the two per-query scores).

## MINI (approvable)

HEADLINE rows, declared before any test pass:
- PRIMARY: FUSION of frozen FEAT and FRN seed s (s = 0, 1, 2) using CCP-A, reported as mean ± std over the three seeds for
  closed-set accuracy (argmax of the fused score) and AUROC.
- SECONDARY: frozen FEAT + CCP-A (single backbone, training-free), accuracy and AUROC with 95% CI.
- Every other row (COS / COSF / CCP-S on every backbone, FRN single-backbone rows, the TANE/DNPG plug-in rows) is non-headline.

TANE/DNPG shared pre-trained ResNet-12 (T2 plug-in rows; the authors' final TANE/DNPG models cannot be run): checkpoint
third_party/TANE/checkpoints/miniImageNet_pre.pth (sha256 below; `feature.*` keys, `layerN.0.` -> `layerN.`, strict load; its own
64-way head gives 88.36% on base images). Preprocessing = the authors' test transform: whole image, PIL LANCZOS resize to 84 x 84,
RGB, ToTensor, Normalize(mean (120.40, 115.59, 104.54)/255, std (70.68, 68.28, 72.55)/255); cohort = its own embeddings of the
38,400 base images under the same view.

```yaml-fsor-mini
version: 2
registered: 2026-09-25
part: mini
approvable: true
episodes:
  1shot: {file: test_fsor_5way_1shot.npz, dir: data/episodes_mini, sha256: c49c0e5a288a42a510ba024b21950fbf2a04637731de9175c83e61b9c2250a8f}
  5shot: {file: test_fsor_5way_5shot.npz, dir: data/episodes_mini, sha256: 99f2457add3d7f9955a8b1f970395bdda77120d7e92153e2ec850ec4d4a67c64}
backbones:
- {name: frozen_FEAT, kind: feat_pretrain, checkpoint: data/mini_imagenet/feat/Res12-pre.pth, sha256: 48e681c65e0a8a28d24f586f963da1f8b9e8aff339201f7380365bdd0ecc5c88, view: eval}
- {name: FRN_seed0, kind: frn, checkpoint: research/runs/mini_frn_featinit_seed0/best.pth, sha256: 54d9569670ca0d38a183a57dc6bc209b169c270203a2850a5b4c51bbf69ba178, view: eval}
- {name: FRN_seed1, kind: frn, checkpoint: research/runs/mini_frn_featinit_seed1/best.pth, sha256: 37cae1a1da309c76a5f0ab963ddf1ecbddbff283753cfc585fe005dd2ffc2b36, view: eval}
- {name: FRN_seed2, kind: frn, checkpoint: research/runs/mini_frn_featinit_seed2/best.pth, sha256: 4eb7ceba7071397b5a29fbc6db7db6200c81985f25f6821dccb62df9d6fc99b0, view: eval}
- {name: TANE_DNPG_pretrain, kind: tane_pretrain, checkpoint: third_party/TANE/checkpoints/miniImageNet_pre.pth, sha256: 2bd34c4594b77faa3710bca3f363f54219754951e24bc353be0c1b41f3d22bbb, view: whole, preprocessing: "whole image, PIL LANCZOS resize 84x84, RGB, ToTensor, Normalize FEAT/TANE constants"}
candidates: [COS, COSF, CCP-A, CCP-S]
fusion:
  candidate: CCP-A
  pairs: [[frozen_FEAT, FRN_seed0], [frozen_FEAT, FRN_seed1], [frozen_FEAT, FRN_seed2]]
  accuracy: argmax of the sum of the two per-class score matrices, each z-scored over the episode
headline:
  primary: {row: FUSION, candidate: CCP-A, pairs: all, report: mean ± std over seeds s = 0, 1, 2 of accuracy and AUROC}
  secondary: {backbone: frozen_FEAT, candidate: CCP-A}
  other_rows: non-headline
metrics: [closed-set accuracy, AUROC (unknown = positive)]
ci: 95% over 600 episodes per shot; FRN-based rows also mean ± std over seeds
decisions_after_test: none
```

## TIERED (rebuilt with BGR input, pre-flight P1, 2026-09-25)

D1 (research/fsor/TIERED_GAP.md) showed the released tiered checkpoints were trained on BGR tensors (Ren et al.'s cv2-decoded npz),
while our decode is true RGB; the T3 block fed RGB and is superseded (never approved). This block feeds BGR (channel flip of our RGB
uint8 before ToTensor + Normalize with the FEAT/TANE constants, 84 x 84, no resize) to two checkpoints:
- frozen FEAT tiered Res12-pre (the T1 backbone). CCP-A stays the candidate chosen in T1 (no re-selection); CCP-S is a reported row.
- TANE's released tiered pre-trained ResNet-12 (`tieredImageNet_pre.pth`, Google Drive file 10l5BuxwBd9Z0YHuK5ySrNFSqv3w8XnnM,
  `feature.*` keys, strict load; rotation-SSL pre-training): plug-in rows COS, COSF, CCP-A, CCP-S, mirroring the TANE rows in MINI.
Cohorts: CCP-A = each backbone's embeddings of the RandomState(20260926) subset of 38,400 base training images; CCP-S = all 448,695.

Validation reference (BGR; AUROC / closed-set accuracy; 1,000 FSOR episodes per shot on tvA and on tvB; scripts/tiered_gap.py).
Information only; T1's verdict (PASS, on RGB features) stands. CCP-A still gains at least +1.0 (1-shot) / +0.5 (5-shot) AUROC
over COS on tvA and tvB for both checkpoints.

| Backbone (BGR) | Split | COS | COSF | CCP-A | CCP-S | CCP-A − COS (needed) |
|---|---|---|---|---|---|---|
| frozen_FEAT_tiered | tvA 1-shot | 73.04 / 67.04 | 73.22 / 67.42 | 74.57 / 68.53 | 74.76 / 68.69 | +1.53 (+1.0) yes |
| frozen_FEAT_tiered | tvB 1-shot | 68.71 / 63.47 | 68.85 / 63.77 | 69.88 / 64.84 | 70.08 / 65.22 | +1.17 (+1.0) yes |
| frozen_FEAT_tiered | tvA 5-shot | 81.18 / 83.57 | 81.34 / 83.82 | 82.75 / 84.00 | 82.56 / 83.85 | +1.56 (+0.5) yes |
| frozen_FEAT_tiered | tvB 5-shot | 77.06 / 81.57 | 77.19 / 81.85 | 78.03 / 81.93 | 78.26 / 81.75 | +0.98 (+0.5) yes |
| TANE_tiered_pretrain | tvA 1-shot | 70.95 / 64.96 | 71.08 / 65.27 | 72.68 / 66.42 | 72.28 / 66.38 | +1.73 (+1.0) yes |
| TANE_tiered_pretrain | tvB 1-shot | 66.72 / 60.88 | 66.86 / 61.25 | 68.47 / 62.54 | 68.29 / 62.63 | +1.76 (+1.0) yes |
| TANE_tiered_pretrain | tvA 5-shot | 78.35 / 81.70 | 78.60 / 81.99 | 80.80 / 82.26 | 80.23 / 81.97 | +2.45 (+0.5) yes |
| TANE_tiered_pretrain | tvB 5-shot | 74.06 / 78.78 | 74.26 / 79.08 | 76.31 / 79.34 | 76.19 / 79.04 | +2.25 (+0.5) yes |

HEADLINE, declared before the tiered pass (R2, 2026-09-25): frozen FEAT tiered + CCP-A, BGR input (single backbone,
training-free), accuracy and AUROC with 95% CI over 600 test episodes per shot. FUSION of frozen FEAT tiered and TANE tiered with
CCP-A (per-episode z-scored sum, fused accuracy as in MINI) is a reported, non-headline row. Every other row is non-headline.

```yaml-fsor-tiered
version: 4
registered: 2026-09-25
part: tiered
approvable: true
status: headline declared before the tiered pass (R2, 2026-09-25); pre-flight guards P3-P6 apply
episodes:
  1shot: {file: test_fsor_5way_1shot.npz, dir: data/episodes_tiered, sha256: 0557ec1b60692c7aed523de3c1be89386f897f6b96462b8393bdb0534f0e232c}
  5shot: {file: test_fsor_5way_5shot.npz, dir: data/episodes_tiered, sha256: 789ffefb1a0ce09137e0a654e7cee807e71a3ac1fea42196a2018e9ac2140daf}
channel_order: BGR
backbones:
- {name: frozen_FEAT_tiered, kind: feat_pretrain, checkpoint: data/tiered_imagenet/feat/Res12-pre.pth, sha256: 90bedb8ae0ad34946d2517a50d2636c0df52373fe5cc206d3a403516f5b51cf4}
- {name: TANE_tiered_pretrain, kind: tane_pretrain, checkpoint: data/tiered_imagenet/second/tieredImageNet_pre.pth, sha256: 93b39fc2ed47fff92bdf520a8c846be64375082c38d31effee94826eca7f4572}
candidates: [COS, COSF, CCP-A, CCP-S]
headline:
  backbone: frozen_FEAT_tiered
  candidate: CCP-A
  report: accuracy and AUROC with 95% CI over 600 episodes per shot (BGR input)
  other_rows: non-headline
fusion:
  candidate: CCP-A
  pairs: [[frozen_FEAT_tiered, TANE_tiered_pretrain]]
  accuracy: argmax of the sum of the two per-class score matrices, each z-scored over the episode
  role: reported, non-headline
ccpA_subset_seed: 20260926
ccpA_subset_size: 38400
metrics: [closed-set accuracy, AUROC (unknown = positive)]
ci: 95% over 600 episodes per shot
decisions_after_test: none
```


## MINI-DNPG (approvable; H-COMB GO 2026-09-28, inductive headline H_fix 2026-09-28)

DNPG (Phase 2 + meta-learning from the released TANE/DNPG mini backbone; 3 seeds) as a trained FSOR host, evaluated on the same 600 mini
test episodes as part MINI. HEADLINE: H_fix = D + w · CCP-A (fixed weight frozen on valA; inductive) against DNPG's own score D, same
checkpoint (max_acc.pth) and same episodes, mean over seeds 0-2. H (per-episode z, query statistics) is kept as a labelled row. Validation
reference: results/fsor/dnpg_seeds012.md and results/fsor/dnpg_hfix.md. Run only by `scripts/final_test.py fsor mini-dnpg` after the human
writes this block's token into research/FINAL_APPROVED; `--dry-run-val` runs the same code path on valA_fsor and valB_fsor.
Decisions after this test: none.

```yaml-fsor-mini-dnpg
version: 2
registered: 2026-09-28
part: mini-dnpg
approvable: true
status: "dry run passed, approvable (H-COMB GO results/fsor/dnpg_seeds012.md; Part 1 PASS results/fsor/dnpg_hfix.md)"
episodes:
  1shot: {file: test_fsor_5way_1shot.npz, dir: data/episodes_mini, sha256: c49c0e5a288a42a510ba024b21950fbf2a04637731de9175c83e61b9c2250a8f}
  5shot: {file: test_fsor_5way_5shot.npz, dir: data/episodes_mini, sha256: 99f2457add3d7f9955a8b1f970395bdda77120d7e92153e2ec850ec4d4a67c64}
  manifest: {file: data/episodes_mini/MANIFEST.json, sha256: 1c52325c40bda8b1c927285a76aa9973cee20cc8ced44e47e8ccf85d7e10d171}
  note: the same 600 test episodes per shot as part MINI, read from our manifest; DNPG's hard-linked 'test' loader is not used
host:
  code: github iCGY96/DNPG @ 38b8fd13656ae66c87488415223a8df24ad67c80 (patches research/dnpg/dnpg_patches.diff)
  backbone: {checkpoint: third_party/TANE/checkpoints/miniImageNet_pre.pth, sha256: 2bd34c4594b77faa3710bca3f363f54219754951e24bc353be0c1b41f3d22bbb}
  backbone_note: released pre-trained backbone (stored epoch 66, consistent with test-selected max_meta)
  selection: per-epoch on 600 episodes of the 16 mini val classes (selected on validation classes)
  preprocessing: whole image, PIL LANCZOS resize 84x84 decoded on the fly (scripts/ccp_plugin._whole), RGB, Normalize FEAT/TANE constants; D with DNPG test-time support x5 RandomCrop(84, pad 8) + flip, torch seed 1000000 * shot + episode index
  cohort: DNPG-feature embeddings of the 38,400 mini base images (layer-4 GAP, flip views)
seeds:
- seed: 0
  phase2: {checkpoint: research/dnpg/phase2/EntropyRot/miniImageNet_ResNet12_batch_128_lr_0.05/mini_last.pth, sha256: ecb712d1a868d4243556fbcb422fafb8520ced990d08e379389510006c095092}
  max_acc_1shot: {checkpoint: research/dnpg/meta/OpenMeta_miniImageNet/miniImageNet_600_shot_1/max_acc.pth, sha256: ebf32fdbe474903584669530a9bb5a46a0ccebd4d59a9abd97ba734211f06d09, epoch: 25}
  max_acc_5shot: {checkpoint: research/dnpg/meta/OpenMeta_miniImageNet/miniImageNet_600_shot_5/max_acc.pth, sha256: 7e5e7b2b472bf6cf2bce2397156fddcc92d78286cc04aa578d88ddf2759fa674, epoch: 34}
- seed: 1
  phase2: {checkpoint: research/dnpg/phase2_seed1/EntropyRot/miniImageNet_ResNet12_batch_128_lr_0.05/mini_last.pth, sha256: 077560ab2a38b8233f759bb176668aacaf9cc3e0d01c7b0d15ec1237ada9a951}
  max_acc_1shot: {checkpoint: research/dnpg/meta_seed1/OpenMeta_miniImageNet/miniImageNet_600_shot_1/max_acc.pth, sha256: 0c80ffdb6a7c0a3e4b6f0a57983373d23d69c75eaec19805049925c7b24ed85a, epoch: 29}
  max_acc_5shot: {checkpoint: research/dnpg/meta_seed1/OpenMeta_miniImageNet/miniImageNet_600_shot_5/max_acc.pth, sha256: 82e8fb64f9f2a41a768b5a55f5ed182ea68925e93e91c44812609a683820e0b0, epoch: 27}
- seed: 2
  phase2: {checkpoint: research/dnpg/phase2_seed2/EntropyRot/miniImageNet_ResNet12_batch_128_lr_0.05/mini_last.pth, sha256: e2a00f5c6076789bd83469f617e0b48daf4eb6d6148b613ecb6511df9e4716f0}
  max_acc_1shot: {checkpoint: research/dnpg/meta_seed2/OpenMeta_miniImageNet/miniImageNet_600_shot_1/max_acc.pth, sha256: 948fb2ed7c3df5a4f86d0334e6acfb08f0bef39ee8fe17711359174bce859594, epoch: 36}
  max_acc_5shot: {checkpoint: research/dnpg/meta_seed2/OpenMeta_miniImageNet/miniImageNet_600_shot_5/max_acc.pth, sha256: ad5f5878788c21de8794cb2e6d5108122563e06a8450cb4312011c3ef416efea, epoch: 35}
weights:
  definition: w = pooled std(D) / pooled std(CCP-A), w2 = pooled std(D) / pooled std(COSF), ddof 0 over all 1,000 x 150 valA query scores, max_acc.pth (research/dnpg/DNPG_SETUP.md section 10); frozen, never computed on test
  seed0_1shot: {w: 2.626274957560938, w2: 2.4038193100093403}
  seed0_5shot: {w: 2.888997703590982, w2: 2.6578522878784656}
  seed1_1shot: {w: 2.6399514523812333, w2: 2.4174174604993377}
  seed1_5shot: {w: 2.912096116907742, w2: 2.6850932050906575}
  seed2_1shot: {w: 2.6329064035074827, w2: 2.4063514307050413}
  seed2_5shot: {w: 2.867463077549045, w2: 2.6397040463604116}
rows:
  D: DNPG own openness score (negative-prototype probability); selected on validation classes
  H_fix: D + w * CCP-A (HEADLINE, inductive, paired against D on the same episodes and checkpoint)
  K_fix: D + w2 * COSF (ensembling control)
  H: z(D) + z(CCP-A), labelled 'per-episode z (query statistics)'
  CCP-A: DNPG features + CCP-A alone
  COSF: DNPG features + flip-averaged cosine
accuracy: DNPG's own classifier for every row
headline:
  row: H_fix vs D, same checkpoint (max_acc.pth) and same episodes
  report: mean over seeds 0-2 (± std over seeds) of AUROC and accuracy; per seed 95% CI over 600 episodes; paired H_fix - D and H_fix - K_fix per seed with 95% CI over episodes and as the mean over seeds
  other_rows: non-headline (K_fix, H, CCP-A, COSF)
per_episode_output: research/fsor/final_mini_dnpg_episodes.npz (accuracy and AUROC per episode for every row, seed and shot)
dry_run_evidence: {file: research/status/fsor_final_dryrun_mini-dnpg_episodes.npz, sha256: aff436be8683d4b2a3a20fc5ca541d96dae7d4995fad3124d2a264e2a534ad33, check: "every row of valA and valB (seeds 0-2, both shots) reproduces results/fsor/dnpg_seeds012.md and results/fsor/dnpg_hfix.md within 0.01"}
published_rows_test_class (mini-ImageNet, as reported; acc / AUROC 1-shot; acc / AUROC 5-shot):
  DNPG (ACM MM 2024, Table 1): 69.10 ± 0.29 / 74.18 ± 0.28; 83.74 ± 0.19 / 83.64 ± 0.19
  HDFA (arXiv 2506.08906, Table 4): 70.52 ± 0.85 / 76.53 ± 0.73; 82.91 ± 0.62 / 84.59 ± 0.55
  SEMAN-G (CVPR 2022, Table 2): 68.24 ± 0.82 / 72.85 ± 0.69; 83.48 ± 0.48 / 82.07 ± 0.58
  ATT-G (CVPR 2022, Table 2): 68.11 ± 0.81 / 72.41 ± 0.72; 83.12 ± 0.48 / 79.85 ± 0.57
metrics: [closed-set accuracy, AUROC (unknown = positive)]
ci: 95% over 600 episodes per shot; mean ± std over seeds
decisions_after_test: none
```


## MINI-SNATCHER (approvable; validation GO 2026-09-30)

SnaTCHer-F (CVPR 2021) on FEAT's released meta-trained ResNet-12 as a second trained FSOR host, on the same 600 mini test episodes as part
MINI. HEADLINE: H_fix = S + w · CCP-A against the SnaTCHer-F score S, same model and episodes, per shot (one released model per shot; no
seeds). Validation reference: results/fsor/snatcher_val.md (H_fix − S on valB +1.67 / +1.77; H_fix − K_fix +0.96 / +1.23). Run only by
`scripts/final_test.py fsor mini-snatcher` after the human writes this block's token into research/FINAL_APPROVED; `--dry-run-val` runs the
same code path on valA_fsor and valB_fsor. Decisions after this test: none.

```yaml-fsor-mini-snatcher
version: 1
registered: 2026-09-30
part: mini-snatcher
approvable: true
status: "dry run passed, approvable (validation GO: results/fsor/snatcher_val.md; pre-registration research/snatcher/SNATCHER_SETUP.md section 2)"
episodes:
  1shot: {file: test_fsor_5way_1shot.npz, dir: data/episodes_mini, sha256: c49c0e5a288a42a510ba024b21950fbf2a04637731de9175c83e61b9c2250a8f}
  5shot: {file: test_fsor_5way_5shot.npz, dir: data/episodes_mini, sha256: 99f2457add3d7f9955a8b1f970395bdda77120d7e92153e2ec850ec4d4a67c64}
  manifest: {file: data/episodes_mini/MANIFEST.json, sha256: 1c52325c40bda8b1c927285a76aa9973cee20cc8ced44e47e8ccf85d7e10d171}
  note: the same 600 test episodes per shot as part MINI, read from our manifest
host:
  code: "github MinkiJ/SnaTCHer @ 335c42469f0a7ad72c5c3480c8effc8c293823e0 (SnaTCHer-F score, adapter research/fsor/snatcher_rows.py); FEAT github Sha-Lab/FEAT @ 47bdc7c1"
  model: FEAT's released meta-trained ResNet-12 (encoder + slf_attn), one model per shot, no seeds; FEAT's code selects max_acc on its val split
  fidelity: "adapter vs SnaTCHer's own evaluate_test on 50 mini valA episodes per shot: max |diff| 7.6e-6 on the same encoder batch; end to end 5-shot up to 3.2e-3 (cuDNN batch numerics), accuracy identical"
  preprocessing: eval view Resize(92) short side + CenterCrop(84), RGB, Normalize FEAT constants (eval-view cache)
  cohort: CCP-A bank = the 38,400 mini base images embedded with the same encoder (layer-4 GAP, flip views)
checkpoints:
  1shot: {checkpoint: data/mini_imagenet/feat_meta/feat-1-shot.pth, sha256: 60b78143149fb891f1b4d187fe24f03fb431d12d4c853752bf7dac5f90460a41, source: "https://drive.google.com/file/d/1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ"}
  5shot: {checkpoint: data/mini_imagenet/feat_meta/feat-5-shot.pth, sha256: ee54213c754b2c25d07bb7b032549140d7be8d4e253f597cfa57a314696191ee, source: "https://drive.google.com/file/d/1tCU52WiK6JBydnALI_gF9nYiFCNvvT1A"}
weights:
  definition: "w = pooled std(S) / pooled std(CCP-A), w2 = pooled std(S) / pooled std(COSF), ddof 0 over all 1,000 x 150 valA query scores; frozen, never computed on test"
  1shot: {w: 39.18493949698625, w2: 36.495955512391575}
  5shot: {w: 47.22915457109874, w2: 44.88093755962801}
rows:
  S: SnaTCHer-F score
  H_fix: "S + w * CCP-A (HEADLINE, inductive, paired against S on the same episodes and model)"
  K_fix: "S + w2 * COSF (ensembling control)"
  CCP-A: CCP-A alone on the FEAT meta-trained encoder features
  COSF: flip-averaged cosine alone
  Dist: "FEAT distance score (-max logit)"
accuracy: FEAT's classifier for S, H_fix, K_fix, Dist; own argmax for CCP-A and COSF alone
headline:
  row: H_fix vs S, same model and same episodes, per shot
  report: "AUROC and accuracy with 95% CI over 600 episodes; paired H_fix - S and H_fix - K_fix with 95% CI over episodes"
  other_rows: non-headline (K_fix, CCP-A, COSF, Dist)
per_episode_output: research/fsor/final_mini_snatcher_episodes.npz
dry_run_evidence: {file: research/status/fsor_final_dryrun_mini-snatcher_episodes.npz, sha256: 7e9d7d0d516562baa1dd822e03091a64e66e2c3dc8259b03352a03055c660d24, check: "every row of valA and valB (both shots) reproduces results/fsor/snatcher_val.md within 0.01"}
published_rows_test_class (mini-ImageNet, as reported; acc / AUROC 1-shot; acc / AUROC 5-shot):
  SnaTCHer-F (CVPR 2021, Table 2 / supp. Table 2): 67.02 ± 0.85 / 68.27 ± 0.96; 82.02 ± 0.53 / 77.42 ± 0.73
metrics: [closed-set accuracy, AUROC (unknown = positive)]
ci: 95% over 600 episodes per shot
decisions_after_test: none
```
