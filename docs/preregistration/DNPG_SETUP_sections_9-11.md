## 9. H-COMB pre-registration: DNPG seeds 1 and 2, validation only (advisor, 2026-09-27)

Written and committed before seed 1 starts. No test data is touched. The MINI and TIERED test passes are done and their bundle
blocks, research/FINAL_APPROVED and the STARTED / DONE files are not edited.

**Step 0 (test-pass per-episode metrics).** Neither research/FINAL_TEST_FSOR_MINI_DONE.json nor ..._TIERED_DONE.json holds per-episode
values (each row stores only auroc, auroc_ci95, acc, acc_ci95; `fsor_final.run_*` kept the per-episode scores in memory only), and no
sidecar file exists. The paired test analysis is skipped; no test episode is re-run or re-evaluated.

**Rows** (all from the same DNPG checkpoint's features and the same fixed valA_fsor / valB_fsor episodes as seed 0, harness
`scripts/dnpg_eval.py`):
- D = DNPG's own openness score (probability of the negative-prototype column), as in its code.
- H = z(D) + z(CCP-A): per-episode z-scoring over the episode's 150 queries, equal weights; exactly the seed-0 information row
  (`combined_*` in research/status/dnpg_eval.json).
- K = z(D) + z(COSF): the ensembling control; the same fusion with flip-averaged plain cosine (COSF) in place of CCP-A (`K_*`).
- CCP-A alone and COSF alone are also reported.
- Accuracy for every row = DNPG's own classifier (argmax of its first 5 probabilities on the known queries).
- Checkpoint for the rule: `max_acc.pth` (the headline checkpoint under the earlier ruling). The last-epoch checkpoint (`40checkpoint.pth`)
  is reported as information.

**Rule** (seeds 1 and 2 only; seed 0 formed the hypothesis and is reported separately; metric valA AUROC, 1-shot and 5-shot):
- R1: mean over seeds 1-2 of (H − D) ≥ +0.50 at both shots, AND H − D > 0 for each seed at both shots.
- R2: mean over seeds 1-2 of (H − K) ≥ +0.30 at both shots (the gain comes from the cohort, not from adding a second score).
- R3: on valB, mean over seeds 1-2 of (H − D) > 0 at both shots.
- GO only if R1, R2 and R3 all hold. Otherwise NO-GO: report and stop.
- K on seed 0 is computed as information only (seed 0 is re-evaluated with the same harness; its earlier results are kept in
  research/status/dnpg_eval_seed0_before_hcomb.json and must reproduce).

**What a seed changes** (from the code, `third_party/DNPG` @ 38b8fd1):
- `train.py:75` hard-codes `torch.manual_seed(0)`; there is no seed argument. Phase 2 (`pretrain/pretrain_openW/`) seeds nothing.
- Seeds 1 and 2 pass `--torch_seed N` to `scripts/dnpg_launch.py`, which seeds torch and numpy once at start and turns DNPG's
  `torch.manual_seed(0)` into `torch.manual_seed(N)`. Seed 0 ran without it (identical to the released code; its Phase 2 was unseeded).
- Torch's RNG drives: initialisation of the new modules (calibrator, negative-prototype generator), Phase 2 centre initialisation and
  batch shuffling, training augmentation (RandomCrop, ColorJitter, flip), dropout, and the test-time support crops.
- NOT changed by the seed: the meta-training episodes are drawn by `OpenMini('train', ...)` with `fix_seed=True` and `np.random.seed(item)`
  (`dataloader/dataloader.py:39`, `dataloader/mini_imagenet.py:82-83`), so every seed sees the same episode sequence; the per-epoch
  selection episodes and our valA / valB episodes are also the same for all seeds.

**Setup** (identical to seed 0 apart from the seed): released TANE/DNPG mini backbone (sha256 2bd34c45...), full chain per seed
(Phase 2 → meta 1-shot → meta 5-shot), authors' recipe with the deviations in this document, per-epoch selection on 600 episodes of our
16 val classes. Outputs: research/dnpg/{phase2,meta}_seedN (git-ignored), status research/status/dnpg_seedN.json, logs
research/runs/dnpg_*_seedN.log. Order: `scripts/dnpg_seeds.py` runs seed 1, then seed 2, then the evaluations of seeds 0, 1, 2
(max_acc and last epoch, both shots), strictly one GPU job at a time. Expected time, from seed 0: ~4.5 h per seed (Phase 2 ~0.75 h,
meta 1-shot ~1.5 h, meta 5-shot ~2.3 h), ~9 h training + ~1.5 h evaluation.

**Step 4, only if GO:** draft `yaml-fsor-mini-dnpg` (approvable: false; D, H (headline, paired against D), K, CCP-A, COSF over 3 seeds;
checkpoint sha256 of all seeds; the MINI test episodes and manifest hash), print its token without adding it to FINAL_APPROVED, and a
read-only tieredImageNet feasibility note (no training; ask before any download over 2 GB).

## 10. Inductive fusion H_fix: pre-registration (advisor, 2026-09-28)

Committed before anything is computed on valB for these rows. Reason: H = z(D) + z(CCP-A) standardises with the mean / std of each
episode's 150 query scores (query-batch statistics), so it is not strictly inductive (research/fsor/PROTOCOL.md, "Query statistics in
fused scores").

**Definition** (no query statistics; one constant per seed and shot, from valA only):
- H_fix = D + w · CCP-A,   w = pooled std(D) / pooled std(CCP-A)
- K_fix = D + w2 · COSF,   w2 = pooled std(D) / pooled std(COSF)
- "pooled std" = standard deviation (ddof 0) of all query scores of all 1,000 valA episodes of that seed and shot, concatenated
  (1,000 x 150 values). Scores as used everywhere: D = DNPG's negative-prototype probability, CCP-A = −max_k of its calibrated score,
  COSF = −max_k of the flip-averaged cosine (higher = more likely unknown). No offset is used (per-episode AUROC ignores offsets).
- Checkpoint: max_acc.pth of each seed (the headline checkpoint). Episodes: the same valA_fsor / valB_fsor files as seeds 0-2.
- Accuracy for every row = DNPG's own classifier.

**Rule** (valB AUROC; w, w2 from valA):
- E1: mean over seeds 0-2 of (H − H_fix) on valB ≤ 0.20, both shots (H_fix may be better; it must not be more than 0.20 worse).
- E2: mean over seeds 1-2 of (H_fix − D) on valB ≥ +0.50, both shots, and every seed (1, 2) > 0 at both shots.
- E3: mean over seeds 1-2 of (H_fix − K_fix) on valB ≥ +0.30, both shots.
- PASS (E1, E2, E3 all hold): H_fix becomes the MINI-DNPG headline; H stays as a row labelled "per-episode z (query statistics)".
- FAIL: stop after Part 1 and report.

**Data status at commit time.** The per-episode AUROCs of D, CCP-A and COSF on valB were computed in H-COMB (dnpg_seeds012.md), but the
per-query scores were not saved, so H_fix and K_fix cannot be derived from them. The raw per-query scores (valA and valB) are produced
after this commit by `scripts/dnpg_fix.py` (shared scoring code `research/fsor/dnpg_rows.py`, identical preprocessing and seeding to
`scripts/dnpg_eval.py`), which must reproduce the H-COMB per-episode numbers.

## Deviation record: tieredImageNet data and word-embedding placeholders (2026-09-28)

- Data: `scripts/dnpg_data_tiered.py` wrote `research/dnpg/data_tiered/tieredImageNet/few-shot-{train,val}.npz` from our decoded
  learn2learn arrays in **BGR** order (the released TANE tiered backbone was trained on BGR; research/fsor/TIERED_GAP.md), 448,695 train
  images / 351 classes and 124,261 val images / 97 classes (counts verified); `few-shot-test.npz` is a hard link to `few-shot-val.npz`.
  DNPG appends `/<dataset>` to `--data_root` (`dataloader/dataloader.py:26`, and the same in Phase 2), so `--data_root` is
  `research/dnpg/data_tiered` and the files sit in its `tieredImageNet/` subfolder (moved there after a first smoke run failed on the
  path; contents unchanged). Manifest: `research/dnpg/data_tiered/MANIFEST.json`. The launcher guard refuses tiered runs unless
  `few-shot-test.npz` is the hard link to val and no unlisted npz exists.
- Word-embedding files `few-shot-wordemb-{train,val,test}.npz` are required at load (`dataloader/tiered_imagenet.py:59-64`) but are not
  used: `Classifier.init_representation` ignores `seman_dict`, and `base_seman` / `support_seman` are set to `None` before use
  (`architectures/AttnClassifier.py:117-134, 170-171, 221-222`). Placeholders: float32 zeros (351 / 97 / 160, 300).
- Placeholder proof (`scripts/dnpg_placeholder_check.py`, status `research/status/dnpg_placeholder_check.json`): a smoke tiered Phase 2
  (1 epoch, 5 iterations; only to have a 351-class open-weight checkpoint), then 20 meta-training iterations (1-shot, `--torch_seed 0`) with
  zero (Z1), random (R) and zero again (Z2) placeholders, plus an all-NaN run. Logged per-iteration losses (Acc, Auroc, cls_ce, funit):
  identical in Z1, R, Z2 and NaN (21 / 21 each), so the pre-registered check holds. Weights after 20 steps differ even between the two
  identical zero runs (Z1 vs Z2 max |diff| 3.9e-6: GPU non-determinism); Z1 vs R 6.5e-4. With all-NaN placeholders every weight stays
  finite, so the placeholder values never enter the computation.

## 11. Tiered DNPG pre-registration (advisor, 2026-09-28)

Committed before any tiered DNPG training. Budget (human ruling): seed 0 only, launched by a separate instruction after the MINI-DNPG
test result. No tiered test image is read (DNPG's 'test' loader is the val hard link; our evaluation uses tvA / tvB files only).

**Chain, seed 0** (the code's fixed torch seed; no `--torch_seed`): released TANE tiered pre-trained ResNet-12
(`data/tiered_imagenet/second/tieredImageNet_pre.pth`, sha256 93b39fc2...4572; stored epoch 81, consistent with test-selected max_meta)
→ Phase 2 (50 epochs, batch 128, Adam 0.05, backbone frozen) → meta-learning 1-shot, then 5-shot (40 epochs x 300 iterations, lr 0.07 /
1e-4, tune_part 4, `--cosine`, train_weight_base 1, `--n_test_runs 600`), `--dataset tieredImageNet --data_root research/dnpg/data_tiered`,
BGR npz. Per-epoch selection runs on DNPG's 'test' partition = all 97 tiered val classes (selected on validation classes; tvA and tvB
classes included). Strictly sequential; estimated ~10 GPU-hours (research/dnpg/TIERED_FEASIBILITY.md).

**Rows** (same as mini), all on `max_acc.pth` of each shot and our fixed tiered validation episodes
(`data/episodes_tiered/tv{A,B}_fsor_5way_{1,5}shot.npz`, 1,000 each):
- D = DNPG own score (DNPG test-time preprocessing on the BGR images, support x5 RandomCrop + flip, torch seed 1,000,000 * shot + episode).
- H_fix = D + w · CCP-A; K_fix = D + w2 · COSF; w = pooled std(D) / pooled std(CCP-A), w2 = pooled std(D) / pooled std(COSF), ddof 0
  over all 1,000 x 150 **tvA** query scores, one constant per shot.
- H = z(D) + z(CCP-A) and K = z(D) + z(COSF) (per-episode z, query statistics), CCP-A and COSF alone, as labelled information rows.
- CCP-A / COSF on DNPG's layer-4 GAP features of the BGR images, flip views; CCP-A cohort = the RandomState(20260926) subset of 38,400
  base images (as in T1 and the TIERED bundle); CCP-S is not part of these rows.
- Accuracy = DNPG's own classifier for every row.

**Rule** (seed 0, AUROC):
- T1: H_fix − D on tvB ≥ +0.50, both shots.
- T2: H_fix − K_fix on tvB ≥ +0.30, both shots.
- T3: H_fix − D on tvA > 0, both shots.
- GO (T1, T2, T3 all hold) → seeds 1-2, then a TIERED-DNPG bundle part. Otherwise NO-GO for tiered DNPG.

The tiered evaluation harness (a BGR / tiered-cohort variant of research/fsor/dnpg_rows.py) is written before the seed-0 evaluation and
checked on the smoke checkpoints; it uses no tvB statistic in any weight.

## Deviation record: tiered 5-shot gradient accumulation (human ruling 2026-09-29)

The registered tiered chain (section 11) could not run its 5-shot meta-learning stage on the 24 GB GPU: with 2 episodes per training step
(`--n_train_para 2`) the step needs more than 24 GB (Windows spilled to system memory, ~20 s per iteration; with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` the first step failed at 23,958 MiB with "Failed to create GPU mapping"). The 1-shot
stage fits and was trained with the registered recipe. Ruling (option 1): **gradient accumulation** for tiered 5-shot only (seed 0 and,
if T1-T3 pass, seeds 1-2): each of the 2 episodes of a step is forwarded and back-propagated on its own with its loss divided by 2, then
one optimizer step (`third_party/DNPG/trainer/MetaTrainer.py`, active only with `DNPG_ACCUM_EPISODES=1`, set by `scripts/dnpg_run.py` for
tiered 5-shot; diff in research/dnpg/dnpg_patches.diff). Same data, learning rate, number of updates and loss (DNPG's losses are means over
equal-sized episodes). Difference: BatchNorm batch statistics are computed per episode instead of over both, and BatchNorm running statistics
are updated twice per step. Measured: peak 20,475 MiB, 0.83 s per training iteration, 0.18 s per evaluation episode (~6 min per epoch,
~4 h for 40 epochs). Mini runs and tiered 1-shot are unaffected (the variable is not set).

### Tiered 5-shot accumulation: exact configuration and cost (recorded 2026-09-29, after the NO-GO)

- Micro-batch: **1 episode** per forward / backward pass. For 5-shot this is 5 known classes x (5 support + 15 query) plus 5 open classes
  x (5 support + 15 query) = **200 images**; the registered step (`--n_train_para 2`) is 2 episodes = 400 images.
- Accumulation: **2 micro-batches per optimizer step** (loss of each divided by 2); 300 optimizer steps per epoch, 40 epochs, as registered.
- BatchNorm: **saw micro-batches** (batch statistics over one episode's 200 images, not the 400 of the full step) and updated its
  running statistics twice per optimizer step. Everything else (data order, lr, updates, loss) as registered.
- Used by: tiered seed 0, 5-shot meta-learning only (09-29 10:10-14:13). Tiered 1-shot, Phase 2 and all mini runs used the registered
  full batch.
- GPU-hours lost to the memory problem: **15.3 h** for the first 5-shot attempt (09-28 18:37 - 09-29 09:55, stopped at epoch 6 of 40 while
  crawling on the Windows system-memory fallback) plus **0.2 h** of memory tests = **15.5 h**. (Separately recorded: 13.1 h lost on
  09-25 / 09-26 to two parallel mini meta runs on the same fallback.)
- Prevention: `research/watchdog.py` (throughput below 50% of the running median for 10 minutes, GPU memory >= 97% for 10 minutes, or no
  first iteration in 30 minutes -> the job is stopped with exit code 3 and reported), hooked into scripts/dnpg_launch.py,
  research/trainer.py and scripts/mini_step1.py.
