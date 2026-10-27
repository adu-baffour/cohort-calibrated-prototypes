# Pre-registration: method variants V1-V4 of CCP (written 2026-10-02, committed before any variant is scored on any split; amended once before approval, Section 10)

Purpose: test four changes to CCP on validation, with decision rules fixed in advance. CCP-A's registered definition and its recorded test
results are not changed by anything here. **No test data is used in this round.** A test bundle is drafted (for the human's approval) only if a
variant is promoted by the rules below. Code: `ccp/variants.py`, `ccp/scores/v2_ask.py`, `ccp/scores/v3_shc.py`, `ccp/cohort.py`
(`topk_stats`, `qw_matrix`, `ask_matrix`, `cohort_overlap`, `shc_matrix`), driver `scripts/variants.py`, tests `tests/test_variants.py`.

## 1. Hosts, cells and notation

Cells (8): {frozen FEAT mini, frozen FEAT tiered, SnaTCHer-F mini, DNPG mini (seeds 0-2)} x {1-shot, 5-shot}, the same hosts, banks,
checkpoints and frozen H_fix weights as the baselines round (`docs/preregistration/baselines.md`). A DNPG cell is the mean over seeds 0-2
(paired differences are averaged over seeds per episode, as in T1). Validation sets: mini valA / valB, tiered tvA / tvB (1,000 episodes per
shot each), and the new valC (Section 5).

Notation (registered CCP-A, `configs/ccp/registered.yaml`): c_k = cos(q, p_k) with flip views; m(v) = m_K(v), the mean of the K = 1000 largest
cosines of v to the host's unit flip-view bank (Eq. 3); N(v) = the indices of those K bank images. CCP-A: s_k = c_k − ½[m(q) + m(p_k)],
label argmax_k s_k, unknown score u = −max_k s_k. Trained hosts: H_fix = D + w·u, w = pooled std(D) / pooled std(u) over all valA query
scores (ddof 0), frozen per host, seed and shot (the values in `configs/hosts/*.yaml`). D = SnaTCHer-F's S or DNPG's D.

## 2. Variants

| Id | Name | Definition | Label | Grid (tie order) | Null setting (= CCP-A) |
|---|---|---|---|---|---|
| V1 | QW (decoupled query weight) | u = −max_k [c_k − β m(p_k)] + λ m(q) | CCP-A's | (λ, β): (0.5, 0.5) first, then λ ∈ {0.5, 0.75, 1.0, 1.25, 1.5} ascending × β ∈ {0, 0.5} ascending (10 points) | λ = β = 0.5 |
| V2 | ASK (top-K AS-norm) | s_k = ½[(c_k − μ_K(q)) / σ_K(q) + (c_k − μ_K(p_k)) / σ_K(p_k)]; μ_K, σ_K = mean and std (ddof 0) of the same K = 1000 nearest bank cosines; σ floored at 1e-6 | argmax_k s_k | none | none (σ ≡ 1 gives CCP-A; tested) |
| V3 | SHC (shared-cohort correction) | o_k = \|N(q) ∩ N(p_k)\| / K; s_k = c_k − ½[(1 − γ o_k) m(q) + m(p_k)] | argmax_k s_k | γ ∈ {0.25, 0.5, 1.0} ascending | γ = 0 (not in the grid) |
| V4 | FW (fusion weight; trained hosts only) | H = D + μ w u_CCP-A, μ ∈ {1, 0.5, 2, 4} in this tie order, then "CCP-A alone" (u_CCP-A without D); w = the frozen H_fix weight | the host's | per host, shared across the two shots | μ = 1 (= H_fix) |

For trained hosts, V1-V3 enter through fusion: H = D + w_v u_variant, with w_v recomputed by the registered std-ratio rule on valA for each
host, seed, shot and grid point; labels stay the host's. For the frozen backbones, a variant's row is its own unknown score (V1: with the CCP-A
label; V2, V3: with their own label).

Implementation facts, fixed here and tested (`tests/test_variants.py`):

- **The existing query-side ablation** (`CCP-A/query-side`, `ccp/scores/ccp_a.py`) is s_k = c_k − m_K(q): **weight 1 on m(q) and no
  prototype term**. In V1 notation it is λ = 1, β = 0. Its label is the COSF label; V1 keeps the CCP-A label instead, so V1 (1, 0) has the
  query-side AUROC but CCP-A's accuracy. The prototype-side ablation is c_k − m_K(p_k) (no query term).
- **B1 QTERM** of the baselines round (`ccp/scores/b1_qterm.py`, evidence e(q) = m_K(q); rows in `ccp/baselines.py`) is used in two ways:
  - frozen backbones: COSF + w m(q), with w = pooled std(COSF score) / pooled std(m(q)) on valA. This is V1 with β = 0 and λ = w. Its valA
    weights were 2.59 / 2.53 (frozen FEAT mini, 1 / 5-shot) and 2.89 / 2.77 (tiered), above the V1 grid.
  - trained hosts: the host's own score + w m(q), i.e. S + w m(q) (SnaTCHer-F) or D + w m(q) (DNPG), with w = pooled std(own) / pooled
    std(m(q)) on valA, and **no cosine term**. It is therefore not a V1 point. Its valA weights were 96.5 / 122.6 (SnaTCHer-F) and
    5.25-5.38 / 5.80-5.98 (DNPG seeds 0-2).
- V1 is computed as −max_k [c_k − ½(2β m(p_k) + 2λ m(q))], mathematically equal to the definition because m(q) is constant over k, and bitwise
  equal to CCP-A at λ = β = 0.5. V3 at γ = 0 is bitwise equal to CCP-A. V4 with μ = 1 is bitwise equal to H_fix.
- μ_K, σ_K and N(v) come from the same top-K call (per chunk of 2,048 target rows, the registered chunking) whose mean is the registered
  m_K. Every scored episode checks that this mean equals m_K in every bit, and that V1 at its null equals CCP-A; a mismatch stops the run.
- V2 and V3 are inductive (the query's statistics are computed row-wise over the whole target split; `tests/test_inductive.py` includes them).

## 3. Selection (validation A only: mini valA, tiered tvA)

- **V1 and V3**: one global setting, the grid point with the highest mean valA AUROC over all 8 cells (cell value = the AUROC of the variant's
  row; DNPG = mean over seeds). Ties within 1e-9 go to the earlier point of the tie order in Section 2; for V1, a tie with (0.5, 0.5) selects
  CCP-A.
- **V1 NO-GO**: if the selected V1 setting is (0.5, 0.5), i.e. CCP-A itself, V1 is NO-GO and is excluded from candidacy.
- **V2**: no selection. For trained hosts its fusion weight is fitted on valA as above.
- **V4**: per trained host (SnaTCHer-F mini; DNPG mini pooled over seeds), the option with the highest mean valA AUROC over that host's two
  shots (DNPG: and seeds). Ties go to the earlier option (μ = 1 first). If both hosts select μ = 1, V4 equals H_fix and its gain is 0.
- **Sensitivity table**: the per-cell optimum of every grid (V1, V3, V4 per cell) on valA is reported. It is never used for anything.
- Output, committed before any valB / tvB episode is scored with a variant: `results/variants/selection_valA.json` (choices, valA AUROC of
  every grid point per cell, fusion weights for the chosen V1-V3 settings, sensitivity table). `scripts/variants.py report` refuses to run
  until that file is committed and unmodified.
- On valB, valC and in any bundle only the selected settings are scored (no other grid point is evaluated there).

## 4. Already-seen validation results (disclosed; they shaped V1 and V4)

These were seen before this pre-registration. Frozen FEAT values come from `results/recorded/validation/ccp_ablations.json` (fp16 feature
cache; the registered fp32 path differs by at most 5.7e-4 AUROC, docs/protocol.md). AUROC, valA 1-shot / 5-shot | valB 1-shot / 5-shot:

| Host | Row | valA (tvA) | valB (tvB) |
|---|---|---|---|
| frozen FEAT mini | COSF | 74.31 / 82.99 | 74.35 / 82.64 |
| | CCP-A (registered) | 76.14 / 85.46 | 76.09 / 85.18 |
| | query-side only (V1 λ = 1, β = 0) | 76.13 / 86.41 | 76.14 / 86.22 |
| | prototype-side only | 74.98 / 83.35 | 74.88 / 83.03 |
| | K = 3000 | 76.67 / 86.09 | 76.73 / 85.81 |
| | CCP-S (mean + std over all base images) | 76.53 / 86.15 | 76.66 / 85.91 |
| | flip off | 75.83 / 85.24 | 75.78 / 84.95 |
| frozen FEAT tiered | COSF | 73.22 / 81.34 | 68.85 / 77.19 |
| | CCP-A (registered) | 74.57 / 82.75 | 69.88 / 78.03 |
| | query-side only | 74.23 / 83.17 | 69.53 / 78.68 |
| | prototype-side only | 74.03 / 81.25 | 69.55 / 76.55 |
| | K = 3000 | 74.68 / 82.60 | 70.03 / 78.16 |
| | CCP-S | 74.76 / 82.56 | 70.08 / 78.26 |
| | flip off | 74.31 / 82.55 | 69.70 / 77.90 |
| DNPG mini (mean of seeds 0-2) | D | 76.06 / 86.13 | 76.21 / 85.88 |
| | H_fix (CCP-A) | 77.25 / 87.46 | 77.28 / 87.23 |
| | H_fix (query-side only) | 77.25 / 87.58 | 77.27 / 87.33 |
| | H_fix (prototype-side only) | 76.78 / 86.87 | 76.87 / 86.68 |
| | H_fix (K = 3000) | 77.20 / 87.39 | 77.28 / 87.16 |
| | H_fix (CCP-S) | 76.94 / 87.26 | 77.06 / 87.03 |
| | fusion multiplier μ = 0.25 / 0.5 / 2 / 4 (D + μ w CCP-A) | 76.71, 77.01, 77.29, 77.15 / 86.88, 87.22, 87.41, 87.15 | 76.82, 77.09, 77.28, 77.10 / 86.65, 86.99, 87.18, 86.93 |

Also seen: the whole baselines round (`results/baselines/validation.md`, valA selection and valB, and its single **test** pass of 2026-10-01,
`results/baselines/test.md`): every inductive baseline is below CCP on valB except RMD for tiered 5-shot (+0.08) and SnaTCHer-F 5-shot (+0.13);
B1 QTERM lowers AUROC below the host everywhere. The recorded CCP-A / H_fix test results of the paper are known. The F6 sensitivity (K and μ)
and the Figure 2 mechanism panels are known.

**V1 and V4 were motivated by these valB results** (the query-side row beats CCP-A at 5-shot on mini and tiered valB; DNPG H_fix is flat in
μ around 1). valB is therefore not a clean confirmation set for them, which is why a fresh valC confirmation (G4) is required for promotion.
V2 (a top-K version of the CCP-S idea) and V3 (new) were not tuned on any result, but they are held to the same rules.

## 5. valC (fresh confirmation episodes)

- 1,000 FSOR episodes per shot per dataset from the validation split, written by the registered generator
  `ccp.episodes.make_fsor_episodes`.
- **Seeds** follow the existing convention seed = base + shot (valA 20260950 + shot, valB 20260960 + shot, tvA 20260980 + shot, tvB
  20260990 + shot), with base **20261000**: **20261001** (1-shot) and **20261005** (5-shot), for mini and tiered alike.
- **mini**: all 16 validation classes. The (known, unknown) class key of every episode differs from every key in valA and valB of the same shot.
- **tiered**: the **49 tvB classes only**: the generator gets the tvB subset of the validation class-to-index map, exactly as tvB was made.
  The keys of tvB of the same shot are excluded.
- Files `manifests/{mini,tiered}/valC_fsor_5way_{1,5}shot.json`, sha256 added to `manifests/SHA256SUMS`; each header records the seed, the
  classes, the excluded key sets and `git_head`. They are generated by `python scripts/variants.py make-valc` only after this
  pre-registration, the valA selection and the valB report are committed. They are committed before any valC episode is scored
  (`scripts/variants.py valc` refuses otherwise).
- Limitations (valC is independent of the earlier sets in **episode sampling** only):
  - **mini valC** reuses the same 16 validation classes and the same images as valA (selection) and valB (G1-G3). Only the class
    combinations and the drawn images per episode are new.
  - **tiered valC** reuses the 49 tvB classes and their images, which valB-stage gating (G1-G3) already used. It is class-disjoint from tvA,
    the selection set, as tvB is. So on tiered, valC adds fresh episode draws but no new classes, and it does not test transfer to the tvA
    classes.

## 6. Decision rules

All gains are paired per-episode AUROC differences of the variant's row against the reference REF on the same episodes: CCP-A (frozen FEAT
cells) or H_fix with the frozen weight (trained-host cells); DNPG cells average the differences over seeds 0-2 per episode. A cell's gain is the
mean over its 1,000 episodes. Applicable cells: 8 for V1-V3, the 4 trained-host cells for V4. Thresholds are inclusive (≥).

- **G1 (valB)**: mean of the cell gains over the applicable cells ≥ +0.30.
- **G2 (valB)**: no applicable cell has a gain below −0.20.
- **G3 (valB)**: in every frozen-FEAT cell, accuracy change (mean accuracy of the variant's label minus CCP-A's) ≥ −0.20 points. Trained hosts
  keep their classifier (unchanged by design); V1 keeps the CCP-A label (change 0 by construction).
- **G4 (valC)**: mean cell gain ≥ +0.15 over the applicable cells, and G2 holds on valC (no cell below −0.20).

Promotion:

1. Among the variants passing G1-G3 (V1 only if not NO-GO), the one with the highest valB mean gain is the **candidate** (ties to the earlier
   of V1, V2, V3, V4).
2. The candidate is **promoted** only if it also passes G4. If it fails G4, nothing is promoted; there is no fallback to the next variant.
3. Promotion makes it a pre-registered extension, reported next to CCP-A. CCP-A's recorded test results stay in the paper unchanged.
4. If no variant is promoted, there is no test pass for this round; it is reported as a validation result.

All four variants are scored on valC at their valA settings and reported; only the candidate's G4 decides.

## 7. Diagnostic D1 (not used for any decision)

Figure 2(b) of this repository (`scripts/figures/fig2_mechanism.py`, the paper's mechanism panel) for frozen FEAT on **tiered tvA**, and
the same panel for **mini valA** for comparison (valA only; registered fp32 path, not the fp16 figure cache): for every query of every episode,
x = max_k c_k and m = m_K(q); 10 equal-count bins of x over all queries of a set (known and unknown pooled); per bin the mean m of known and
of unknown queries.

**Statistic (standardised gap).** Per bin, d = (mean m unknown − mean m known) / s_p, where s_p = sqrt(((n_k − 1) s_k² + (n_u − 1) s_u²) /
(n_k + n_u − 2)) is the usual two-group pooled within-bin SD of m_K(q) (s² with ddof 1). The statistic is the mean of d over the 10 bins,
then per dataset the mean over the two shots. A bin with fewer than two known or two unknown queries, or with s_p = 0, is skipped. Reason:
m_K has a different spread on mini and tiered, so raw gaps are not comparable. The raw gap (mean m unknown − mean m known, averaged the same
way) is kept in every output but is not read.

**Registered reading** (standardised gaps): if tiered's ≤ ⅓ of mini's, "the base cohort carries little signal on tiered"; otherwise "the
signal exists; the weighting or fusion is the limitation". The reading is undefined if mini's standardised gap is not positive. Outputs
`results/variants/d1/d1.json` (both gaps per shot and dataset, the ratio and the reading), `d1.csv` (per bin: means, raw gap, pooled SD,
standardised gap) and `d1_panel.png`. D1 may run at any point after approval.

## 8. Reporting plan and order

1. `select` → `results/variants/selection_valA.json`, committed (V1 NO-GO stated there if it applies).
2. `report` → `results/variants/validation_valB.{json,md}`: per cell and variant the paired gain ± 95% CI, accuracy change, G1-G3, the
   candidate. Committed.
3. `make-valc` → valC manifests, committed; then `valc` → `results/variants/validation_valC.{json,md}`: G4 for every variant, the promotion
   decision. Committed.
4. `d1` → `results/variants/d1/`. Committed.
5. Only if a variant is promoted: one test bundle drafted for the human's approval, containing CCP-A (it must reproduce the recorded test
   numbers exactly), all four variants at their valA settings, and the promotion decision written in the block; dry-run on validation
   through the bundle path; token reported; `bundles/FINAL_APPROVED` is written only by the human.

**Provenance.** Every step (`select`, `report`, `make-valc`, `valc`, `d1`) refuses to run if any tracked file under `ccp/`, `scripts/` or
`configs/` differs from HEAD (staged or not; `tests/test_variants.py` checks this). Each step writes `git rev-parse HEAD` into its output
JSON (`git_head`); `make-valc` writes it into each manifest header.

Every outcome is reported, including NO-GO and failed gates. No threshold, grid, tie order or set is changed after a result.

## 9. Compute estimate

One RTX 4090, one job at a time. Each of `select`, `report` and `valc` scores 14 (host, shot) runs on 1,000 episodes per shot. The
baselines round needed 13.2 min (valA) and 11.8 min (valB) for comparable runs; the extra top-K pass over the target split adds a little
(the tiered validation split has 124,261 images). Estimate: about 15 min each for `select`, `report` and `valc`, and 3 min for `d1`:
**about 50 GPU-minutes in total**.

## 10. Amendment log

Amendment 1 (2026-10-02, requested by the human before approval; nothing had been scored on any split, and no valC manifest existed):

1. valC seeds: base + shot with base 20261000 (20261001 / 20261005), replacing the single seed 20261002 of the first version.
2. tiered valC: drawn from the 49 tvB classes only, excluding the tvB keys of the same shot (first version: all 97 validation classes,
   excluding the tvA and tvB keys). mini valC is unchanged (all 16 classes, valA and valB keys excluded). Limitations stated per dataset.
3. D1: the registered reading uses the standardised gap (raw gap divided by the pooled within-bin SD, mean over bins, then over shots);
   the raw gap is kept in the outputs only.
4. Provenance: every step refuses if tracked files under ccp/, scripts/ or configs/ differ from HEAD, and records git HEAD in its output.
5. Section 2: the trained-host definition of B1 QTERM (own score + w m_K(q), no cosine term) added next to the frozen-backbone one.

Everything else (variants, grids, tie orders, selection, G1-G4, promotion, reporting) is unchanged.
