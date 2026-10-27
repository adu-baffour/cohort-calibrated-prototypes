# Pre-registration: final evaluation round (written 2026-10-05, committed before any training or scoring of this round; amended once at approval, see "Amendment log")

After this round the method and the evaluation are frozen. It answers four reviewer questions with **one** further test pass:
(A) a third benchmark, CIFAR-FS; (B) cross-domain transfer, mini-ImageNet to CUB; (C) OSCR and FPR95 for every recorded row; (D) the D1
diagnostic on test episodes.

**No method choice is made.** CCP-A (registered, `configs/ccp/registered.yaml`) and V1 QW (λ = 1.0, β = 0.5, the pre-registered extension of
`docs/prereg_variants.md`) are fixed. **Every registered row goes to test whatever validation shows**; there are no GO / NO-GO gates, only
technical checks (hashes, exact reproduction, no NaN). Validation is used only to choose the CIFAR-FS baseline settings on valA (the
registered rule of `docs/preregistration/baselines.md`) and for previews. One bundle, one test pass; decisions after the test: none.

Code: `ccp/final_round.py` (one scorer for every row, the target-bank row, metrics per episode, D1), `ccp/metrics.py` (`oscr`, `fpr95`),
`ccp/pipeline.py` (cross-domain targets, `make_target_bank`), `ccp/hosts/rfs_frozen.py` + `ccp/hosts/rfs_resnet12.py` (RFS backbone),
`scripts/prepare_data/{cifarfs,cub}.py`, `scripts/train/train_rfs_cifarfs.py`, driver `scripts/final_round.py`; tests
`tests/test_metrics_open.py`, `tests/test_final_round.py`.

## 1. Data, sources and hashes

### CIFAR-FS (A)

- Images: the official CIFAR-100 python archive, `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` (169,001,437 bytes; MD5
  `eb9058c3a382ffc7106e4002c42a8d85` as published on the CIFAR page; sha256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`).
- Split: Bertinetto, Henriques, Torr, Vedaldi, *Meta-learning with differentiable closed-form solvers*, ICLR 2019: 64 / 16 / 20 classes.
  Canonical lists: the r2d2 split files as mirrored by learn2learn (`https://zenodo.org/records/7978538/files/cifar100.zip`, md5
  `fdb4405027b809aa4c059047f21ca3d1`, `cifar100/splits/bertinetto/{train,val,test}.txt`; read by learn2learn @ db5cee3
  `learn2learn/vision/datasets/cifarfs.py`), committed as `configs/datasets/cifarfs_split/*.txt`: train sha256
  `08d41acf21db7203341442e96b2226276217e379bbf809af34849506af1a8c5e`, val `e9ca1bb0404d6f804eb4df42923187eb07052a0c2f87820342b0da1f8c940278`,
  test `3720c9d858fa547dbf71ca80ede1f94a803279cfdedd3f7a9b9d3beb4ce5c020`. The same class sets are in torchmeta @ d55d89e
  (`assets/cifar100/cifar-fs/*.json`) and in MetaOptNet's `CIFAR_FS_val.pickle` (val checked; its test pickle was not opened).
  Val: otter, motorcycle, television, lamp, crocodile, shark, butterfly, beaver, beetle, tractor, flatfish, maple_tree, camel, crab, sea,
  cattle. Test: baby, bed, bicycle, chimpanzee, fox, leopard, man, pickup_truck, plain, poppy, rocket, rose, snail, sweet_pepper, table,
  telephone, wardrobe, whale, woman, worm.
- Layout (`scripts/prepare_data/cifarfs.py`): all 600 images per class (500 CIFAR-100 train + 100 CIFAR-100 test), as in CIFAR-FS; global
  order split, class-list order, CIFAR-100 train then test images in archive order; train 0-38,399, val 38,400-47,999, test 48,000-59,999;
  32 x 32 RGB, no resampling (view `eval32`). Prepared sha256 (2026-10-05): train `9147af46…a339`, val `19603a98…a293`, test `7515ad3f…ff7c`,
  index `309cb903…82b1` (full values in `configs/datasets/cifarfs.yaml`).

### CUB-200-2011 (B)

- Archive: `CUB_200_2011.tgz` (1,150,585,339 bytes; Caltech's MD5 `97eceeb196236b17998738112f37df78`; sha256
  `0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081`), from `https://data.caltech.edu/records/65de6-vp158`. The file used is the
  research repository's copy of that download (identical MD5 and size).
- Class split: the FRN rule documented in the research repository (`scripts/preprocess.py`, `research/fsor/CUB_SPLIT_CHECK.md`), 1-based class
  id i: i % 2 == 0 train (100 classes, 5,903 images), i % 4 == 1 val (50, 2,927), i % 4 == 3 test (50, 2,958).
- **Whole images, no bounding boxes** (as requested). Note: the research repository's FRN pipeline crops to the boxes; this round does not.
- Views: exactly mini-ImageNet's two test transforms (`ccp/data/mini.py`), so every mini host sees CUB in its own mini view: `eval84` (PIL
  Resize(92) short side bilinear + CenterCrop(84); frozen FEAT, SnaTCHer-F) and `whole84` (whole image, PIL LANCZOS 84 x 84; DNPG).
  Global order: split, class id, image id. Prepared sha256 in `configs/datasets/cub.yaml` (eval84 train `aafac6e3…3cdd`, val `d772e6a2…594c`,
  test `a9105e2d…f267`; whole84 train `69f24dd1…82ad`, val `9ec17dc0…cf5b`, test `78e79fee…0ffc4`; index `a4568cba…884a`).

Both datasets were prepared in phase 0 (decoding only). The loaders opened the train and val arrays for a shape check; no test image was
read, and no model has seen any of these images.

## 2. CIFAR-FS backbone (A)

One ResNet-12 pre-trained with cross-entropy on the 64 base classes with the published RFS recipe (Tian, Wang, Krishnan, Tenenbaum, Isola,
*Rethinking few-shot image classification: a good embedding is all you need?*, ECCV 2020), no distillation:

- Model: RFS's own `resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=64)` from `github.com/WangYueFt/rfs` @
  `f8c837ba93c62dd0ac68a2f4019c619aa86b8421` (`models/resnet.py`, `models/util.py` create_model for CIFAR-FS). The RFS repository has no licence
  file, so its code is fetched by `python third_party/clone.py RFS` and imported at run time; nothing of it is copied into this repository.
  Layers 1-2: dropout 0.1; layers 3-4: DropBlock (block 2) whose keep rate ramps to 0.9 over 40,000 forward calls of each block, counted in
  train and eval mode.
- Data and augmentation: RFS transform D (`dataset/transform_cfg.py`): RandomCrop(32, padding 4), ColorJitter(0.4, 0.4, 0.4),
  RandomHorizontalFlip, ToTensor, Normalize(mean 0.5071 0.4867 0.4408, std 0.2675 0.2565 0.2761); 32 x 32 input.
- Optimisation (`train_supervised.py`, paper Sec. 4.1): SGD, lr 0.05, momentum 0.9, weight decay 5e-4, batch 64 (shuffle, drop_last),
  plain cross-entropy, fp32. **90 epochs, lr x 0.1 after epochs 45, 60 and 75** (the paper: "decayed with a factor of 0.1 by three times …
  90 epochs for both CIFAR-FS and FC100"; the epochs 45 / 60 / 75 from the RFS author's recipe in WangYueFt/rfs issue #5; the code defaults
  100 epochs / 60, 80 are the miniImageNet ones). RFS's per-epoch `validate()` pass (an eval-mode forward over the 38,400 training images,
  batch 32) is reproduced because it advances the DropBlock counters. **Selection: none, the last epoch** (RFS saves `{model}_last.pth`; its
  "val" loader reads the training images).
- Deviations, all non-substantive: seed set to 0 (RFS sets none), data-loader workers 0 (RFS 8), cudnn.benchmark off.
- Train once, seed 0. No recipe change after any result. If training crashes or diverges (NaN / inf loss), rerun the same recipe once with
  seed 1 and report it; the host config then becomes `cifarfs_rfs_seed1` (a technical substitution, recorded in the bundle).
- Sanity only, not a gate: the 5-way closed-set accuracy of the backbone on CIFAR-FS valA (nearest-centroid cosine, COSF, on the 75 known
  queries of the FSOR episodes) is reported next to RFS's published test numbers (Table 2, "Ours-simple", logistic regression:
  71.5 ± 0.8 / 86.0 ± 0.5). The two are not the same classifier or split.
- Host: `configs/hosts/cifarfs_rfs_seed0.yaml` (frozen encoder, registered GAP convention of `ccp/hosts/base.py`, flip views, fp32). Bank:
  all 38,400 base images, flip views, K = 1000 (as on mini). The checkpoint and bank sha256 are pinned in phase 1 before any scoring.

## 3. Episodes and seeds

Registered FSOR protocol (5 known classes with K = 1 / 5 supports and 15 queries each, 5 unknown classes with 15 queries each),
`ccp.episodes.make_fsor_episodes`, seed = base + shot (the existing convention). Bases already in use: 20260950 (mini valA), 20260960
(valB), 20260970 (mini test), 20260980 (tvA), 20260990 (tvB), 20261000 (valC), 20261010 (tiered test); the research repository used
20260924-20260929 for in-domain CUB validation files. New bases:

| Set | Classes | Episodes per shot | Seed (1 / 5-shot) | Key exclusion |
|---|---|---|---|---|
| CIFAR-FS valA | 16 val | 1,000 | 20261021 / 20261025 | - |
| CIFAR-FS valB | 16 val | 1,000 | 20261031 / 20261035 | keys of valA of the same shot (as mini valB) |
| CIFAR-FS test | 20 test | 600 | 20261041 / 20261045 | - |
| CUB val | 50 val | 1,000 | 20261051 / 20261055 | - |
| CUB test | 50 test | 600 | 20261061 / 20261065 | - |

"valA / valB as halves of the validation episodes, as on mini" is implemented as on mini: two 1,000-episode files over the same validation
classes, B excluding A's (known, unknown) class keys. Manifests are written by `scripts/final_round.py make-manifests` (class labels only;
no image is read), hashed into `manifests/SHA256SUMS` and committed before any scoring.

## 4. Rows

Notation as in the paper. "host" is the host's own row: COSF for a frozen backbone in (A) and (B), COS in (C) (as recorded), S for
SnaTCHer-F, D for DNPG. Trained hosts keep their own classifier. All constants are frozen before this round unless stated.

**(A) CIFAR-FS** (host `cifarfs_rfs_seed0`): host (COSF), CCP-A, V1 (CCP-A label), and B1-B9 at the settings chosen on CIFAR-FS valA by the
registered baselines rule (per family, mean valA AUROC over both shots; ties to the earlier grid value; evidence baselines B1, B2 fused as
COSF + w e with w = pooled std ratio on valA, per shot). B9 is the transductive reference.

**(B) mini-ImageNet -> CUB** (hosts frozen FEAT mini, SnaTCHer-F mini, DNPG mini seeds 0-2; all banks = the mini base banks; nothing tuned on
CUB): host; CCP-A (frozen FEAT) / H_fix = D + w CCP-A with the frozen mini weights of `configs/hosts`; V1 (frozen FEAT) / V1fused = D + w_V1
u_V1 with the frozen weight of `results/variants/selection_valA.json`; the baselines chosen on mini for that host
(`results/baselines/selection_valA.json`, their mini valA weights; trained hosts: H[b] = D + w_b u_b). **Analysis row, a different
setting**: CCP-A@target / H_fix@target (same mini weight) with a **target-domain bank**: all 5,903 CUB training-class images embedded with the
same encoder in the same view (flip views, the host's stored precision, `ccp.pipeline.make_target_bank`), K = min(1000, 5,903) = 1000. A
second analysis row (amendment 1), CCP-A@target-K154 / H_fix@target-K154, uses the same bank with K matched to the registered fraction:
K_t = round(1000 × 5,903 / 38,400) = 154. Both rows are descriptive; neither is chosen over the other; both are scored on CUB val and test for
every host in (B). They use unlabelled target-domain images and are labelled as not the inductive cross-domain result. CUB validation
(1,000 episodes per shot) is a preview only.

**(C) recorded rows**: every recorded test host of the paper: frozen FEAT mini, FRN mini seeds 0-2, TANE mini, SnaTCHer-F mini, DNPG mini
seeds 0-2 (mini test); frozen FEAT tiered, TANE tiered (tiered test). Rows: host (COS / S / D), COSF, CCP-A, H_fix (trained hosts), V1
(frozen backbones; CCP-A label) / V1fused (trained hosts). Label convention of the recorded passes, kept for exact reproduction: DNPG reports
its own classifier's accuracy on every row (COSF and CCP-A alone included), SnaTCHer-F reports COSF and CCP-A with their own argmax labels;
OSCR uses the same labels. Fused rows use the host's labels.

## 5. Metrics

Per episode (150 queries; unknown = positive; higher u = more likely unknown), then mean and 95% CI (1.96 std / sqrt(n)) over episodes, and
paired per-episode differences against the host (DNPG: averaged over seeds 0-2 per episode, as in T1; (A) also against CCP-A), with
Wilcoxon p and the fraction of episodes improved:

- **AUROC** and **closed-set accuracy**: unchanged (`ccp.metrics.auroc`; accuracy over the 75 known queries).
- **OSCR** (`ccp.metrics.oscr`): a query is accepted as known at threshold τ when u ≤ τ. CCR(τ) = fraction of known queries accepted and
  correctly labelled; FPR(τ) = fraction of unknown queries accepted. τ runs over −∞ and every distinct score; OSCR = area under CCR against FPR
  (trapezoidal; tied scores give a straight segment, the convention of the average-rank AUROC, so with every label correct OSCR = AUROC), in %.
  The question's "u < τ" is written u ≤ τ over the observed scores; the two give the same curve points.
- **FPR95** (`ccp.metrics.fpr95`): the fraction of unknown queries accepted (u ≤ τ) at the smallest τ that accepts at least 95% of the known
  queries, i.e. τ = the ⌈0.95 · 75⌉ = 72nd smallest known score (ties with τ accepted), in %. Lower is better; for FPR95 "improved" means a
  lower value. (With a strict u < τ there is no smallest τ; the inclusive rule is its limit.)

Hand-computed test cases are in `tests/test_metrics_open.py`.

## 6. D1 (descriptive, no decision)

The registered statistic of `docs/prereg_variants.md` Section 7: per shot, the mean over 10 equal-count bins of the raw max cosine of the
standardised gap (mean m_K unknown − mean m_K known) / pooled within-bin SD; per dataset also the mean over shots. Computed for frozen FEAT on
mini, tiered and CUB (frozen FEAT mini with its mini bank on CUB images) and for the CIFAR-FS backbone, on validation (mini valA, tiered tvA,
CUB val, CIFAR-FS valA) and on test. On CUB (amendment 2, descriptive) the statistic is also computed with m_K(q) against the CUB
target-domain bank at K = 1000 and at K = 154, on val and test (the binning variable, the raw max cosine to the prototypes, does not depend
on the bank). One descriptive figure per set: the standardised gap (x) against the AUROC gain over COSF on the same episodes (y; CCP-A for the
base-bank points, CCP-A@target / CCP-A@target-K154 for the two target-bank points), one point per dataset, bank and shot, the two target-bank
variants with their own marker shapes; no line is fitted and no claim is made beyond description.

## 7. Published CIFAR-FS comparison (read, not run)

A published row is listed only if its protocol matches ours on every item (classes 64/16/20, 5-way + 5 unknown classes, 15 + 15 queries,
600 test episodes, 32 x 32 input, ResNet-12), each stated in the paper or fixed in its official code; otherwise it goes to a footnote with the
difference. Read so far (research of 2026-10-05):

- **Rows**: TANE (Huang, Ma, Han, Chang, CVPR 2022; Table 5; ATT-G and its baselines; 600 tasks, 5 + 5 classes, 15 + 15 queries, ResNet-12,
  32 x 32 images); GEL (Wang et al., CVPR 2023; Table 1: FEAT*, PEELER*, SnaTCHer-F*, RFDNet*, ATT-G, GEL; 600 test episodes fixed in the
  official code `helpers.py`, 32 x 32 stated in the supplement). Pre-training differs (TANE: CE + rotation; GEL: FEAT pre-training; ours: RFS CE)
  and is stated next to the table.
- **Footnotes**: DNPG (ACM MM 2024; 4,800 test tasks); HDFA (arXiv 2506.08906; number of test episodes and input size not stated; its
  SnaTCHer-F row differs from GEL's for the same method); SEMAN-G (uses word embeddings); OPP (Pattern Recognition 2024) and FSAN (Pattern
  Recognition 2024, class-name text) report CIFAR-FS but their numbers could not be read (paywalled); TNPNet (Neurocomputing 2025) is
  transductive and its numbers could not be read.
- Before the test pass this list is re-checked against the papers; changes are committed with a reason before the bundle is drafted.

## 8. Already-seen results (disclosed)

Everything of the paper, the baselines round and the variants round, including their test passes (V1 test: frozen FEAT tiered −0.50 / −0.68
against CCP-A). No CIFAR-FS or CUB number with any of these hosts has been computed. The research repository ran an in-domain CUB FSOR study
(FRN split, bounding-box crops, its own backbone; "FSOR step 0: FAIL", 2026-09-23), a different setting and not used here.

### Phase-0 incidents

- The CIFAR-FS and CUB test images were decoded on the CPU only, to write and hash the prepared arrays (Section 1). No model saw them, and
  no loader opened the test arrays.
- `scripts/final_round.py make-manifests` was run by mistake right after the phase-0 commit (to check its guard, which no longer applied once
  this file was committed) and wrote the 10 manifest files. They were deleted, `manifests/SHA256SUMS` was restored, and nothing was committed
  or scored. They are regenerated identically in phase 1 (same generator, seeds and class lists).
  Phase-1 check: the regenerated episodes are identical; the files differ only in the `git_head` header field (rebuilding the phase-1 CUB
  test 5-shot file with the earlier head in its header reproduces the earlier sha256 8f50d31e0ec88d60... exactly).

## 9. Phases and order (enforced by `scripts/final_round.py`; every step refuses on uncommitted code and records git HEAD)

0. This pre-registration, the code and the tests are committed. Stop for approval. **No training and no scoring before "approved".**
1. `make-manifests` (commit); train the CIFAR-FS backbone (`scripts/train/train_rfs_cifarfs.py --seed 0`), pin its checkpoint sha256
   (commit); `build-banks` (CIFAR-FS base bank, 9 CUB target banks), pin their sha256 (commit).
2. Validation: `select-cifarfs` (CIFAR-FS valA selection, commit), `report-cifarfs` (valB, commit), `cub-preview` (commit), `recorded-val`
   ((C) on mini valA / tiered tvA with OSCR / FPR95 and an exact reproduction check of the recorded validation dry runs, commit), `d1-val`
   (commit).
3. `draft-bundle`: one test bundle covering A-D (`yaml-fsor-final-round` in `docs/preregistration/FINAL_BUNDLE_final_round.md`) pinning this
   file, every validation record, every checkpoint, bank and test-manifest sha256, every frozen weight and choice, and the recorded test values
   of (C). `bundle --dry-run-val --set-approvable`: the bundle path on the validation sets; every number (AUROC, accuracy, OSCR, FPR95 of every
   row, every D1 value) must equal the committed validation outputs exactly. Report the token, the expected runtime and the command.
   `bundles/FINAL_APPROVED` is written only by the human.

Test pass (after approval only): `python scripts/final_round.py bundle`. Technical checks: manifest hashes; part (C) must reproduce every
recorded test AUROC and accuracy exactly; otherwise the pass is reported as a failed reproduction and the new metrics are not interpreted;
no NaN. Outputs `results/final_round/test.json`, `test_episodes.npz` (release asset), and `test.md` + `d1_test.png` rendered from them by
`test-report`. Every row is reported whatever it shows.

## 10. Compute estimate (one RTX 4090, one job at a time)

| Step | Estimate |
|---|---|
| CIFAR-FS backbone (90 epochs, 600 iterations + the train-image eval pass per epoch, CPU augmentation) | 1-2 h |
| Banks (CIFAR-FS base; 9 CUB target banks) | 5 min |
| Validation (select-cifarfs 8 min, report-cifarfs 3 min, cub-preview 12 min, recorded-val 30 min, d1-val 5 min) | about 1 h |
| Bundle dry run on validation | about 1 h |
| Test pass (A 3 min, B 10 min, C 30 min) | about 45 min |
| **Total** | **about 4-5 GPU-hours** |

## Amendment log

Amendment 1 (2026-10-05, requested by the human with the approval; nothing had been trained or scored, and no manifest of this round existed):

1. Target-bank K: the K = 1000 analysis row is kept and a second analysis row is added with K matched to the registered fraction,
   K_t = round(1000 × 5,903 / 38,400) = 154 (CCP-A@target-K154 / H_fix@target-K154). Both are descriptive, neither is chosen over the other,
   and both are scored on CUB val and test for every host in (B) (Section 4).
2. D1 with the target bank: on CUB, D1 is computed with the mini bank (as registered) and with the CUB target bank at K = 1000 and K = 154,
   on val and test; descriptive only; both added to the D1 figure with their own marker shapes (Section 6).
3. Phase-0 incidents recorded in Section 8.

The RFS checkpoint release stays undecided (docs/checkpoints.md notes that the CIFAR-FS result is reproducible from the training script,
the pinned RFS commit and seed 0). Everything else is unchanged.
