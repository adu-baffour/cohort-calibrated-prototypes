# Changelog

## 2.0.0 (2026-10-26)

First release of this repository. It replaces release 1.5.0 of the authors' earlier release repository (private), the code release of the
submitted version, and follows the final paper: every method, setting, table, figure and number is found under the paper's name.

**Recorded material, unchanged.** The recorded results of the eight registered rounds, the registration documents, bundles, approval file,
STARTED / DONE markers and episode manifests are byte-for-byte copies of that release (1.5.0, commit 96cae35), each listed with its sha256 and
source in results/PROVENANCE.md and checked by `scripts/provenance.py check` and the tests. They keep their original names; `ccp.recorded`
translates their keys to the paper's names on reading (docs/naming.md). The release assets are the 1.5.0 archives under new names, with the
same bytes (results/PROVENANCE.md, "Release assets").

**Names.** The paper's names throughout: SCG (was D1), H2 and H3 (registration H2b and H2), CCP-QW, full AS-norm, shared-cohort correction
and fusion-weight factor (V1-V4), the nine baselines by name (B1-B9), the RMD shrinkage δ and the QB-Norm temperature, criteria (i)-(iv)
(G1-G4), valA / valB on tieredImageNet (tvA / tvB), rounds (i)-(viii). `ccp/naming.py` is the single map; `scripts/release_scan.py` fails on
registration labels in the documentation.

**Code.** The scoring arithmetic is bit-identical to 1.5.0 (`tests/test_differential.py` against the 1.5.0 package with `CCP_OLD_PKG`).
Docstrings cite the paper's equations. The registered rounds live in `scripts/rounds/`; `scripts/rounds/render_recorded.py` re-renders every
recorded report from the recorded results and checks it against the recorded text. `ccp/assets.py` pins every release-asset file.

**Paper assets.** `scripts/make_tables.py` regenerates Tables 1-8 byte for byte, with the paper's bold / underline ranking rules.
`scripts/make_figures.py` makes Figures 2 and 3 and checks the plotted data of every figure against the recorded data; Figure 2 now has
panels (a) and (b) at text width (panel (c) is Table 7(a)). `scripts/paper_numbers.py` matches every number of the paper's text to its source
(docs/paper_numbers.md). PAPER_EDITS.md lists the edits the matching found.

**Detailed results.** docs/results/: per-seed results, OSCR and FPR95, baseline settings (former Supplementary Table S9), checkpoint and bank
checksums, implementation checks, validation decisions, ablations and variants, and the former Supplementary Figures S1-S3 as repository
figures. docs/registration_log.md lists the eight rounds and their tokens. The supplementary PDF of 1.5.0 is not continued.

**Reproduction.** `python scripts/reproduce.py all --tier 1` runs every check on a CPU. Tier 2 adds validation re-evaluation and, behind
`--allow-test`, a re-evaluation of the test episodes that writes only to results/generated/test_reevaluation/ and was not run for this release.

**Checks of this release.** results/checks/ holds validation checks run for 2.0.0 by the repository's own scripts, listed with their sha256 in results/PROVENANCE.md ("2.0.0 checks"): the end-to-end SnaTCHer-F fidelity with its AUROC differences.

**Dataset images.** Figures 1 and 4 contain mini-ImageNet images and are not part of the repository; `scripts/release_scan.py` fails on any
tracked file with dataset images.

**Third-party code.** DNPG moved from github.com/iCGY96/DNPG to github.com/gary23ai/DNPG; the pinned commit and patch are unchanged
(docs/checkpoints.md).
