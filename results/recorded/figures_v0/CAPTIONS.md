# Draft figure captions

## Figure 2
**Figure 2. How the cohort statistic changes the open-set score (validation; frozen FEAT, mini-ImageNet valB).**
(a) Each point is one query from the first 40 valB 1-shot episodes (6,000 queries; circles = known, triangles = unknown; contours =
2-D kernel density per group). x: raw maximum cosine to the five prototypes (flip-averaged); y: the query's cohort statistic m_K(q), the mean
of its top-1000 cosines to the 38,400 base images. Dotted line: pooled equal-error threshold of cosine alone; dash-dot line: pooled
equal-error threshold of the one-sided query score x − m_K(q) (the "query-side only" variant in (c)). Pooled over these queries (illustrative,
not the per-episode metric), AUROC is 73.2 for cosine and 75.7 for the one-sided query score. The registered symmetric CCP-A equals
the average of the two one-sided scores, ½[(c − m(q)) + (c − m(p))]; panel (a) shows only the query term, the prototype term varies by class.
Pearson correlation between m_K(q) and the raw max cosine over all 1,000 valB 1-shot episodes: 0.49 (known) and 0.28 (unknown).
(b) Mean m_K(q) (± 95% CI over queries) in ten equal-count bins of the raw max cosine (known and unknown queries pooled for the binning; all 1,000
valB episodes; 1-shot top, 5-shot bottom). 1-shot: unknown queries have a higher mean m_K(q) than known queries in 9 of 10 bins (non-overlapping CIs), lower in 0, overlapping in 1; and 5-shot: unknown queries have a higher mean m_K(q) than known queries in 10 of 10 bins (non-overlapping CIs), lower in 0, overlapping in 0. At similar raw cosine the unknown queries sit closer to more base images, so
subtracting m_K(q) lowers their score relative to known queries (1-shot, lowest bin: known 0.677 vs unknown 0.710;
highest bin: 0.749 vs 0.748).
(c) Per-episode AUROC and closed-set accuracy on valB (dot = mean, whiskers = 95% CI over 1,000 episodes) for cosine (COSF), query-side
calibration only (c − m(q)), prototype-side only (c − m(p)) and the symmetric CCP-A. AUROC 1-shot / 5-shot: COSF 74.35 / 82.64,
query-side 76.14 / 86.22, prototype-side 74.88 / 83.03,
CCP-A 76.09 / 85.18. Accuracy: COSF 68.55 / 85.24,
prototype-side 70.12 / 84.89, CCP-A 69.91 / 85.41;
the query term does not change the argmax, so query-side accuracy equals COSF's. Error bars are per-row 95% CIs over episodes. All rows use the
same episodes, so differences between rows are paired and much more precise. Validation only; not used for any selection.

## Figure 3
**Figure 3. Queries whose rank changes most under CCP-A (validation; frozen FEAT, mini-ImageNet valB, 1-shot).** Selection rule fixed before
viewing: each query's openness percentile within its episode (150 queries; higher = more unknown) under cosine (COSF) and under CCP-A; rows 1-3
are the unknown queries with the largest increase, row 4 the known query with the largest decrease, row 5 the known query with the largest
increase (a failure case); at most one query per episode. Each row shows the query (border: blue = known, orange = unknown), the support image
of its nearest class by raw cosine, its raw maximum cosine, its cohort statistic m_K(q) (mean of its top-1000 cosines to the 38,400 base
images) and both percentiles, and its five nearest base images with their cosines. Percentile changes: rows 1-3
36 → 77, 36 → 76, 44 → 84; row 4 78 → 15; row 5 18 → 62.
Validation only; not used for any selection.

## Figure 4
**Figure 4. Test gain versus the host's own AUROC (recorded test numbers; no re-evaluation).** Each point is one host on the 600 fixed test
episodes per shot. x: the host's own AUROC on test (cosine for backbones, the SnaTCHer-F score S, DNPG's own score D). y: the AUROC gain from the
cohort term, CCP-A − cosine for backbones (circles) and H_fix − host score for trained hosts (squares). Filled markers: mini-ImageNet; open
markers: tieredImageNet (frozen FEAT tiered and TANE tiered, BGR input). FRN and DNPG are means over seeds 0-2. Error bars (trained hosts only)
are paired 95% CIs over the 600 episodes (DNPG: of the seed-averaged per-episode difference). Every point lies above zero (smallest gain
+0.28; the lower ends of all error bars are above zero as well). The gains are smallest on tieredImageNet (frozen FEAT
+0.56 / +0.48, TANE +0.28 / +0.47; 1-shot / 5-shot) and for DNPG
(+0.58 ± 0.13 / +0.73 ± 0.12); SnaTCHer-F gains
+1.67 ± 0.21 / +2.10 ± 0.17. There is no monotone relation between the host's AUROC and
the gain: Spearman ρ = 0.01 (p = 0.99) over the ten mini-ImageNet points (five hosts × two shots); within a shot (five points each)
ρ = -0.30 (1-shot, p = 0.62) and ρ = -0.70 (5-shot, p = 0.19), neither significant. No line is fitted.
† The released TANE checkpoints' stored epochs are consistent with selection on the test split (see Section X); their comparisons are paired
on the same backbone.

## Figure 5
**Figure 5. Distribution of per-episode gains on test (recorded per-episode values; no re-evaluation).** Step histograms (1-point bins) of the
per-episode AUROC difference between the inductive fusion H_fix and the trained host's own score, over the 600 mini-ImageNet test episodes per
shot: DNPG (H_fix − D, averaged over seeds 0-2 per episode; solid, blue) and SnaTCHer-F (H_fix − S, one released model per shot; dashed,
orange). Solid grey line: 0; dotted lines: means. DNPG: mean +0.58 (1-shot) and +0.73 (5-shot), with
66% and 67% of episodes above 0. SnaTCHer-F: mean +1.67 and
+2.10, with 75% and 84% of episodes above 0.
Paired 95% CIs and Wilcoxon signed-rank tests: SnaTCHer-F H_fix − S +1.67 ± 0.21 (p = 4.0e-45)
and +2.10 ± 0.17 (p = 1.7e-75); DNPG H_fix − D +0.58 ± 0.14 (p = 1.1e-16) and +0.73 ± 0.12
(p = 1.1e-26) (results/fsor/mini_dnpg_test_paired.md, mini_snatcher_test_paired.md).

## Figure 6
**Figure 6 (supplement). Sensitivity to the cohort size and the fusion weight (validation only; not used for any selection).** (a) valB AUROC
(tieredImageNet: tvB) against the cohort size K (top-K base images, log scale; "all" = mean over the whole registered bank, after the axis break)
for frozen FEAT on mini-ImageNet (CCP-A), frozen FEAT on tieredImageNet (CCP-A, BGR input) and DNPG + CCP (H_fix(K) with its weight recomputed on
valA, mean of seeds 0-2); 1-shot solid and filled, 5-shot dashed and open; the registered K = 1000 is dotted. Frozen FEAT mini 1-shot:
74.49 (K = 100), 76.09 (K = 1000), 76.73 (K = 3000),
75.79 (all); 5-shot 83.12, 85.18, 85.81,
84.03. The single-backbone curves still rise at K = 3000 (K = 3000 is above K = 1000 for both backbones and
both shots), so the registered K = 1000 is not their maximum; the fused DNPG curve is flat from K = 300 to 3000 (range
0.12 in 1-shot and 0.13 in 5-shot). "All" is below K = 1000 for every host and shot; it is the lowest point for frozen FEAT tiered (5-shot) and DNPG (both shots),
while for frozen FEAT mini (both shots) and frozen FEAT tiered (1-shot) K = 100 is lower. (b) DNPG valB AUROC of H_fix = D + m·w·CCP-A against the multiplier m (w fitted on valA; mean of seeds 0-2),
with DNPG's own score D (dashed) and CCP-A alone (dash-dot) as references; the registered m = 1 is dotted. 1-shot: 76.82 (m = 0.25),
77.28 (m = 1), 77.10 (m = 4); D 76.21, CCP-A 76.54. 5-shot: 86.65, 87.23,
86.93; D 85.88, CCP-A 86.16. Paired AUROC difference from the registered m = 1 (per-episode
difference averaged over seeds 0-2, mean and 95% CI over the 1,000 valB episodes; recomputed from the saved validation scores in
research/status/dnpg_raw, whose levels agree with the plotted values): 1-shot m = 0.25: -0.47 ± 0.07; m = 0.5: -0.19 ± 0.04; m = 2: -0.01 ± 0.04; m = 4: -0.18 ± 0.08; 5-shot m = 0.25: -0.58 ± 0.05; m = 0.5: -0.24 ± 0.03; m = 2: -0.05 ± 0.03; m = 4: -0.30 ± 0.06. Error bars are per-row 95% CIs over
episodes (DNPG: mean of the per-seed CIs). All rows use the same episodes, so differences between rows are paired and much more precise.
