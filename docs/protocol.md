# Evaluation protocol

Section numbers refer to the paper. Names follow the paper; [naming.md](naming.md) maps them to the names in the registration documents and
the recorded files.

## Episodes (Section 4.1)

Each episode has **5 known classes** (S support images and **15 queries** each; S = 1 or 5) and **5 unknown classes**, disjoint from the
known ones (**15 queries** each, no support): 75 known and 75 unknown queries. Episode files are fixed and hashed (`manifests/`, sha256 in
`manifests/SHA256SUMS`); every manifest header records its generator and seed, and `tests/test_manifests.py` regenerates the manifests from
them.

| Dataset | valA | valB | Test |
|---|---|---|---|
| mini-ImageNet (16 val / 20 test classes) | 1,000 episodes per shot | 1,000 per shot; no (known, unknown) class set shared with valA | **600 per shot** |
| tieredImageNet (97 val / 160 test classes) | 1,000 per shot on 48 val classes | 1,000 per shot on the other 49 val classes | **600 per shot** |
| CIFAR-FS (16 val / 20 test classes) | 1,000 per shot | 1,000 per shot, no class set shared with valA | **600 per shot** |
| CUB (50 val / 50 test classes) | 1,000 per shot (file `val_*`) | 1,000 per shot (foundation round) | **600 per shot** |

On tieredImageNet the valA / valB files are named `tvA` / `tvB`; its validation classes are split by `RandomState(20260925)`. The variants
round also used valC, 1,000 new validation episodes per shot (mini-ImageNet: all 16 validation classes, class sets of valA and valB excluded;
tieredImageNet: the valB classes, class sets of valB excluded).

## Metrics and statistics (Section 4.5)

Per episode, then the mean and the 95% confidence interval 1.96 sd / sqrt(n) over episodes (sd with ddof 0):

- **accuracy**: top-1 over the 75 known queries, before rejection;
- **AUROC** over all 150 queries with the unknown queries as positives (higher unknown score = more likely unknown), average ranks for ties;
- **OSCR**: the area under the CCR-FPR curve; a query is accepted when u(q) <= tau; CCR = known queries accepted and correctly classified,
  FPR = unknown queries accepted; with every known query correct, OSCR equals AUROC;
- **FPR95**: the fraction of unknown queries accepted at the smallest threshold that accepts at least 95% of the known queries, i.e. the
  72nd smallest of the 75 known scores (ties accepted).

Paired analyses use per-episode differences on identical episodes: mean, 95% CI, fraction of positive differences, Wilcoxon signed-rank
(`zero_method="wilcox"`; two-sided for the ResNet comparisons, one-sided for H1). DNPG's three seeds share the episodes: per-episode values are
averaged over seeds before a paired analysis. Seed summaries are mean and sample standard deviation (ddof 1). The foundation-model round adds
Holm over the 32 one-sided Wilcoxon tests (H1), Bonferroni-simultaneous CIs, and one-sided permutation Spearman tests with 100,000
permutations (H2: validation SCG vs test gain, seed 20261006; H3: test SCG vs test gain; within dataset-shot groups: seed 20261007). The
permutation p-value is (1 + #{rho_perm >= rho}) / (1 + n_perm), so its floor is 1 / 100,001, about 1e-5.

## The stratified cohort gap (Section 3.5, Eq. 8)

For one configuration (host, dataset, shot), with x(q) = max_k c_k(q) on mirror views and m(q) = m_K(q), K = 1000 (`ccp/scg.py`):

1. The known and unknown queries of all episodes are pooled.
2. **Bin edges.** The 11 edges are the deciles of x computed by `np.quantile(x, np.linspace(0, 1, 11))` with numpy's default linear
   interpolation. They are computed once and stored; no edge is rounded or recomputed.
3. **Assignment.** A query goes to bin `searchsorted(edges, x, side="right") - 1`, clipped to 0..9: a value **at or above an edge goes to
   the upper bin**, compared against the stored (computed) edge, and the maximum goes to the last bin. Because linear interpolation can
   leave an edge a few ulps away from the data value it was interpolated from (for example 30.000000000000004 for the 3rd decile of
   0, 1, ..., 100), "on the edge" always means equal to the stored edge. This is the code that produced every recorded SCG value; the
   paper's statement "values on bin edges are assigned to the upper bin" describes it, and no change to the paper is needed.
4. **Per bin j** (Eq. 8): d_j = (mean m of unknown - mean m of known) / s_j, with the pooled within-bin standard deviation
   s_j^2 = ((n_kn - 1) s_kn^2 + (n_unk - 1) s_unk^2) / (n_kn + n_unk - 2) and sample variances (ddof 1).
5. **Exclusions.** A bin with fewer than two known or fewer than two unknown queries, or with s_j = 0, is excluded.
6. **SCG** is the unweighted mean of d_j over the valid bins. The raw gap (the unstandardized mean difference) is reported alongside.

## valA and valB (Section 4.1)

Every choice was made on **valA**: the fusion weights w and w2, the baseline settings, the CCP-QW coefficients, and the valA part of every
acceptance criterion. **valB** confirmed registered rules and was never used to fit anything. The rules and their outcomes are in
`docs/preregistration/` and [results/validation_decisions.md](results/validation_decisions.md).

## Inductive scores (Section 3.1)

A score is **inductive** if a query's predicted label and unknown score depend only on that query, the support set, the base data of the host
and constants fixed before testing. CCP-A, CCP-S, CCP-QW, COS / COSF, every host's own score, fixed-weight fusion H_fix and the plain-cosine
control K_fix are inductive (`tests/test_inductive.py` permutes, removes and replaces the other queries of an episode). **Episode-wise
standardization** (z(D) + z(u) with the mean and standard deviation of the episode's 150 query scores; also the FUSION rows of rounds (i) and
(ii)) uses the other queries and is transductive; it is excluded from the reported test results. **Episode CSLS** is a transductive reference
baseline.

Implementation detail that keeps this true: per-query cohort statistics are computed row-wise over the whole target split once
(`ccp.pipeline.Target`), never over an episode's query set.

## Registered single-pass protocol (Section 4.6, Table 1)

Every test number comes from **one registered pass per round**. Each of the eight rounds was registered as a YAML bundle block
(`docs/preregistration/FINAL_BUNDLE_*.md`) and run on validation episodes first (dry run). The human approved it by writing its token (the
first 16 hex characters of the sha256 of the block) into the approval file (`results/recorded/test/FINAL_APPROVED` for rounds (i)-(iv),
`bundles/FINAL_APPROVED` for rounds (v)-(viii)); the round then ran once and refused a second run (STARTED / DONE markers).
`ccp.registration.check` recomputes every token from the registration documents. Dates and tokens: [registration_log.md](registration_log.md).

`scripts/evaluate.py` refuses test manifests unless `--allow-test` is given. Scoring a test manifest again with this code is possible (tier 2
of REPRODUCE.md), but it is a **new evaluation**, not the recorded result, and its output goes only to `results/generated/test_reevaluation/`.

## Channel order (Section 4.1, Table 3 note)

The released tieredImageNet checkpoints (FEAT Res12-pre, FEAT meta-trained, TANE pre-trained) were trained on **BGR** tensors (Ren et al.'s
cv2-decoded arrays). On 200 base-class 5-way episodes FEAT's tieredImageNet classifier scores 86.27 (RGB) vs 93.64 (BGR) at 1 shot and 94.96 vs
98.11 at 5 shots; the frozen FEAT tieredImageNet nearest-prototype accuracy on base classes is 77.90 (RGB) vs 87.61 (BGR) at 1 shot. The prepared
tieredImageNet arrays are therefore BGR. The mini-ImageNet checkpoints are RGB (FEAT's classifier: 92.37% RGB vs 71.67% BGR on base images).
Details: [results/implementation_checks.md](results/implementation_checks.md).

## Numerical precision (Section 4.7)

Numbers are bitwise reproducible only with the registered arithmetic: bfloat16-autocast encoder forward, feature map rounded to fp16, fp32
global average pooling, fixed encoder batches of 500 images from the first image of each embedded array, and chunks of 2,048 rows for bank
matrix products. The stored precision of the raw features differs between hosts in the registered passes: **fp32** for mini-ImageNet frozen
FEAT, FRN, TANE pre-trained and DNPG and for every foundation-model and CIFAR-FS host, **fp16** for the tieredImageNet frozen FEAT and TANE
pre-trained hosts and SnaTCHer-F. The banks keep that precision. Storing the unit bank itself in fp16 would move validation AUROC by up to
8e-5 and change one query's label in 1,000 validation episodes, so the banks store the raw pair and normalize in fp32 on load. The fp16 unit
bank of a 640-dimensional host would be 46.9 MB; the bank files are 199 MB (fp32 raw pair) or 101 MB (fp16 raw pair), each with the CCP-S
statistics and class means.

Several recorded *validation* results were computed from earlier fp16 feature caches (the premise study, the ablations, Figures 2 and 4).
Against those, the fp32 path differs by up to 5.7e-4 AUROC and a few labels in 1,000 episodes; `evaluate.py --feature-dtype fp16` reproduces
the fp16-cache numbers. Encoder outputs also depend on cuDNN batch composition: SnaTCHer's own code embeds 200 images per 5-shot episode
(including dummy unknown supports) and ours 175, which changes per-query values by up to 3.2e-3 but no accuracy.

## Known limitations (Section 5.7)

- **TANE pre-trained checkpoints.** The released TANE pre-trained checkpoints (`miniImageNet_pre.pth`, `tieredImageNet_pre.pth`), which also
  initialize our DNPG runs, store epochs 66 and 81. In the released pre-training code, checkpoints are written at those epochs only as
  `max_meta.pth`, selected on the **test** partition; `last.pth` is written at epochs 10, 20, ..., 90. The files also carry an `optimizer` key
  the released saving routine does not write, so the rule that produced them is not established; the stored epochs are consistent with
  test-split selection. Rows on these checkpoints carry a dagger; their CCP comparisons are paired on the same backbone.
- **Shared episode order across DNPG seeds.** DNPG draws its meta-training episodes with `np.random.seed(episode index)`, so seeds 0-2 see the
  same episode sequence; a seed changes initialization, augmentation, dropout and the test-time support crops only. The seed spread therefore
  understates episode-sampling variance.
- **tieredImageNet 5-shot gradient accumulation.** The tieredImageNet DNPG seed-0 5-shot meta-learning used gradient accumulation (micro-batch
  1 episode, 2 micro-batches per optimizer step; BatchNorm statistics per micro-batch) because the registered 2-episode step exceeds 24 GB.
  tieredImageNet DNPG is a validation-only result whose acceptance criterion was not met.
- **DNPG selection budget.** Our DNPG runs select `max_acc.pth` on 600 validation episodes per epoch (the released README: 3,000 test episodes).
- **Fusion weights** were fitted once on valA from features computed with a joint train+val embedding (research harness); the test-pass path
  embeds base and target images separately. The weights are frozen constants, so this changes nothing downstream.

## Test-split selection in released code

At the pinned commits, no data loader of DNPG (`38b8fd1`) or TANE (`fe089d4`) is constructed with the `'val'` partition. Their per-epoch
evaluation during meta-learning, the files it saves (`max_acc.pth`, `max_auroc.pth`) and, with the README's `--cosine` flag, the learning-rate
schedule (`ReduceLROnPlateau` on the running best accuracy) all use the `'test'` partition; the pre-training stages save `max_meta.pth` on the
test partition as well. The TANE paper's supplementary material states that the novel-validation split is used for model selection; the DNPG
paper does not state a selection split. With DNPG's default `test.py` settings the evaluation episodes are the first 1,000 of the 3,000
per-epoch selection episodes (both loaders seed episode `i` with `np.random.seed(i)`). Our DNPG runs point the `'test'` partition at the
validation classes (a hard link of the validation file), so every DNPG number here is selected on validation classes. FEAT's pre-training
(`Res12-pre.pth`) validates on its `val` split.

## Episode-wise standardization and fixed-weight fusion (Section 3.4)

Episode-wise standardization z(D) + z(CCP-A) uses the other queries of the episode. Its inductive replacement H_fix = D + w CCP-A (one frozen
w per host, seed and shot, from valA) was registered with the rule "mean over seeds 0-2 of (episode-wise standardization - H_fix) on valB
<= 0.20". Result: -0.008 (1 shot) and -0.010 (5 shots), so H_fix was not worse ("within 0.02 points", Section 3.4); on test (round (iii),
mean of seeds 0-2) 74.30 vs 74.30 (1 shot) and 83.83 vs 83.85 (5 shots). H_fix is the reported fusion everywhere.
