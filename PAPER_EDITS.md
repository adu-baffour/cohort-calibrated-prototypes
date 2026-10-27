# Edits to the paper

The edits found while matching the paper with the recorded results, and their status in the final version: `CCP_for_FSOR_v19.zip`
(sha256 `aa9196208a656854e582409e5d47fabd33dd3d33b22eaa1e42b8a33e0207d65b`, pinned in configs/paper.yaml; checked items pinned in
configs/paper_pins.json). No number in this repository was changed to match the paper. With v19, `python scripts/paper_numbers.py --check`
matches all 507 numbers of the text with no exception, and `python scripts/make_tables.py --check` finds Tables 1-8 identical.

## Required edits: all applied in v19

| # | Edit | Status in v19 |
|---|---|---|
| 1 | Section 5.5.3: CLIP bank concentration "0.81--0.90" → "0.80--0.90" (least concentrated CLIP bank, tieredImageNet: 0.8049; `results/derived/bank_geometry_exploratory.json`) | applied; the number check is strict again (no exception) |
| 2 | Figure 2: panels (a) and (b) at text width, `\includegraphics[width=\textwidth]{figs/fig2_mechanism.pdf}` | applied; `figs/fig2_mechanism.pdf` is byte-identical to `paper_assets/figures/fig2_mechanism.pdf` (sha256 `8cd8808c...`) |
| 3 | Section 4.2: SnaTCHer-F fidelity sentence with the end-to-end differences and the mean AUROC difference | applied (in two sentences, "the original authors' evaluation code"); every number is checked against `results/checks/snatcher_fidelity_end_to_end.json`, the 2.0.0 check written by `python scripts/fidelity/snatcher_fidelity.py --record` |
| 4 | Repository URL in the abstract and Section 4.7, and the data availability statement with the repository, the GitHub release v2.0.0 and the Zenodo DOI 10.5281/zenodo.23695548 | applied: `https://github.com/adu-baffour/cohort-calibrated-prototypes`, `https://github.com/adu-baffour/cohort-calibrated-prototypes/releases/tag/v2.0.0`, `https://doi.org/10.5281/zenodo.23695548` |

The SnaTCHer-F measurement behind item 3 (first 50 valA episodes per shot; it reproduces the recorded values of
`results/recorded/validation/snatcher.json` exactly):

| Shot | Comparison | Max. abs. difference, score S | Mean AUROC difference | Max. per-episode AUROC difference | Accuracy |
|---|---|---|---|---|---|
| 1 | end to end | 7.6×10⁻⁶ | 0 | 0 | identical |
| 5 | end to end | 3.2×10⁻³ | −3.6×10⁻⁴ points | 0.018 points | identical |
| 1, 5 | same encoder batch | 7.6×10⁻⁶ / 5.7×10⁻⁶ | 0 | 0 | identical |

## Optional edits

| # | Edit | Status in v19 |
|---|---|---|
| 5 | Table 6: OSCR and FPR95 changes from unrounded means (17 cells would change in the second decimal) | not applied: Table 6 keeps the stated rule "computed from means rounded to two decimals", which the repository reproduces |
| 6 | Section 5.5.3: say that the DINOv2 model-size trend is the registration's exploratory H3, not the paper's H3 | applied |

The 17 Table 6 cells of item 5, in case the rule is changed later (printed value from rounded means → value from unrounded means):
CLIP ViT-B/16: tieredImageNet 5-shot FPR95 −1.64 → −1.65; CIFAR-FS 1-shot OSCR +10.59 → +10.58; CIFAR-FS 5-shot OSCR +6.52 → +6.51;
CUB 1-shot OSCR +7.14 → +7.15. DINOv2 ViT-S/14: mini-ImageNet 1-shot FPR95 +4.39 → +4.40; mini-ImageNet 5-shot FPR95 −0.04 → −0.03;
tieredImageNet 1-shot FPR95 +6.80 → +6.81; tieredImageNet 5-shot FPR95 +3.68 → +3.69; CIFAR-FS 5-shot FPR95 +0.48 → +0.49; CUB 1-shot
FPR95 −7.19 → −7.18. DINOv2 ViT-B/14: CIFAR-FS 1-shot OSCR −0.51 → −0.52. DINOv2 ViT-L/14: mini-ImageNet 5-shot OSCR −0.26 → −0.25;
tieredImageNet 1-shot OSCR −0.60 → −0.59; tieredImageNet 1-shot FPR95 +3.96 → +3.97; CUB 1-shot OSCR +0.27 → +0.28; CUB 1-shot FPR95
−0.45 → −0.46; CUB 5-shot OSCR +0.04 → +0.05.

## Kept as printed

- Section 4.6, round (iii): "its 5-shot valA gain was +0.24 points". The recorded difference is 0.2347; +0.24 is the difference of the
  rounded means (86.43 − 86.19) and the value of the registration record. docs/paper_numbers.md notes it.

## End matter in v19

- CRediT statement, competing-interest declaration, funding statement and data availability statement: complete.
- The generative-AI declaration names the tool and its provider but no model version; the authors decide whether to add one.
- Figures 1 and 4 contain mini-ImageNet images; the repository does not redistribute them (README, "Figures with dataset images"). The
  paper is unaffected.
- The v19 sources compile as distributed (the cas-dc thumbnails are now included) to 20 pages, as does the zip's own `manuscript.pdf`.
