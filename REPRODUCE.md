# Reproducing the paper

Three tiers. Tier 1 runs on a CPU from the recorded results and is what every reader can run; tiers 2 and 3 need the datasets,
checkpoints and a GPU. All runtimes were measured on one workstation (Windows 11, Python 3.10.11, PyTorch 2.14 with CUDA 13.0,
RTX 4090 24 GB, Intel Core i9, 128 GB RAM).

## Installation

Python 3.10 (the recorded results used 3.10.11); the versions of every package are pinned in `requirements.txt`.

```bash
git clone https://github.com/adu-baffour/cohort-calibrated-prototypes && cd cohort-calibrated-prototypes
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt && pip install -e . --no-deps
```

From PyPI this installs the CPU build of torch, which is all tier 1 needs; for a GPU (tiers 2 and 3), install the CUDA build of the
pinned torch and torchvision versions (pytorch.org). On Windows, clone into a short path (for example `C:\ccp`) or enable long-path
support: otherwise pip can fail inside sympy, whose test files exceed the 260-character path limit.

## Tests

| Mode | Command | Result (2026-10-29) |
|---|---|---|
| default (a fresh clone: no data, assets or third-party clones) | `pytest -q` | 362 passed, 19 skipped; 14 s |
| default, with the SnaTCHer, DNPG, CLIP and DINOv2 clones present | `pytest -q` | 366 passed, 15 skipped; 15 s |
| with data and release assets | `CCP_TEST_DATA=1 pytest -q` | 378 passed, 3 skipped; 2 min 2 s |

The 3 remaining skips need the RFS clone (`python third_party/clone.py RFS`: 1 test) or a GPU (`CCP_TEST_GPU=1`: 2 tests); without the
SnaTCHer, DNPG, CLIP and DINOv2 clones, the tests that import them are skipped too. DINOv2's tiny-model test warns that xFormers is not
installed (not needed). Tests marked `data` read recorded files, release assets and manifests only. The manifest tests regenerate every
manifest from its recorded seed and read class labels and manifests only (no image, feature or score).

Lint: `ruff check ccp scripts tests third_party/clone.py` (ruff 0.16.9).

## Tier 1: from the recorded results (CPU, about 2 minutes)

Needs the installation above and the release assets. Download the eight archives with `MANIFEST.txt` and `SHA256SUMS` from the GitHub
release v2.0.0 (https://github.com/adu-baffour/cohort-calibrated-prototypes/releases/tag/v2.0.0) or the Zenodo record
(https://doi.org/10.5281/zenodo.23695548), check them with `sha256sum -c SHA256SUMS`, and unpack all eight into one folder
(`tar -xf <archive> -C <folder>`; they hold `checkpoints/` and `recorded/`). Set `CCP_ASSETS` to that folder; without it, `CCP_ASSETS`
defaults to `<repo>/release_assets` (git-ignored). Tier 1 reads only the recorded archive; the checkpoints are for tier 2
(docs/checkpoints.md).

`python scripts/reproduce.py all --tier 1` runs every step below and ends with "tier 1: all OK" (measured: 106 s). One step:
`python scripts/reproduce.py <step> --tier 1`.

| Step | What it regenerates and checks | Command | Inputs |
|---|---|---|---|
| `provenance` | every recorded file, registration document, bundle, marker and manifest has its sha256 (results/PROVENANCE.md) | `python scripts/provenance.py check` | the repository |
| `registration` | the eight rounds' tokens: bundle block, DONE marker, approval file and test record agree | `ccp.registration.check()` | docs/preregistration/, results/, bundles/ |
| `recorded-reports` | every recorded report (25: rounds (i)-(viii)) re-renders from the recorded results to its recorded text, figures included | `python scripts/rounds/render_recorded.py all` | recorded results; per-episode test metrics (assets) |
| `derived` | results/derived/paired_derived.json equals a fresh computation | `python scripts/derive_paired.py --check` | per-episode test metrics (assets) |
| `tables` | Tables 1-8 equal paper_assets/tables/ byte for byte, ranking marks included | `python scripts/make_tables.py all --check` | recorded results |
| `figures` | the plotted data of Figures 2-4 equal the recorded data (results/recorded/figures_v0/; Figure 3: the data of the paper's figure); fonts embedded, text at least 7 pt; Figure 4 without its image panels | `python scripts/make_figures.py all --check --no-images` | recorded results; validation score cache (assets) |
| `results-docs` | docs/results/*.md, docs/registration_log.md and docs/naming.md are up to date | `python scripts/make_results_docs.py --check` | recorded results; configs |

To write the outputs instead of checking them: `python scripts/make_tables.py`, `python scripts/make_figures.py` (Figure 4's image
panels need `CCP_DATA`), `python scripts/make_results_docs.py`. Figure 1 is a diagram drawn by hand, not generated; Figures 1 and 4 show
mini-ImageNet images and are not redistributed.

## Tier 2: from the released checkpoints (GPU)

Needs the prepared data (`CCP_DATA`, docs/data.md), the checkpoints (`CCP_CKPT`, docs/checkpoints.md), the third-party code
(`python third_party/clone.py <name>`) and the cohort banks, rebuilt byte-identically to their pinned sha256 by
`python scripts/build_bank.py --all` (ResNet-12 hosts), `python scripts/rounds/final_round.py build-banks` and
`python scripts/rounds/foundation_round.py build-banks`. One GPU job at a time.

| What | Command | Expected | Runtime |
|---|---|---|---|
| validation equivalence, every host of rounds (i)-(iv) | `python scripts/reproduce.py val-equivalence --tier 2` | the recorded validation outputs: AUROC within 1e-4, accuracy identical | about 1 h |
| Figure 2 / 4 validation score cache | `python scripts/reproduce.py figure-cache --tier 2` | byte-identical to the release asset (sha256 in ccp/assets.py) | under 1 min |
| fidelity checks | `python scripts/fidelity/snatcher_fidelity.py`, `ccp_s_closed_form_check.py`, `channel_order_check.py`, `cifarfs_backbone_check.py`, `clip_loader_fidelity.py` | docs/results/implementation_checks.md; `snatcher_fidelity.py --record` writes the 2.0.0 check results/checks/snatcher_fidelity_end_to_end.json (25 s) | minutes each |
| round (v) validation | `python scripts/rounds/baselines.py select`, `report` | results/baselines/selection_valA.json, validation.json | |
| round (vi) validation | `python scripts/rounds/variants.py select`, `report`, `valc`, `scg` | results/variants/ (validation records) | |
| round (vii) validation | `python scripts/rounds/final_round.py select-cifarfs`, `report-cifarfs`, `cub-preview`, `recorded-val`, `scg-val` | results/final_round/ (validation records) | |
| round (viii) validation | `python scripts/rounds/foundation_round.py download-weights`, `prepare-cub224`, `sanity`, `sanity-acc`, `select-baselines`, `report-valB` | results/foundation_round/ (validation records) | |
| bundle dry runs (validation) | `python scripts/rounds/<round>.py dry-run` | the round's recorded dry run | |

Every round script writes to results/generated/rounds/<round>/ and never to the recorded files.

### Re-evaluation of the test episodes (not run by the authors for this release)

The paper's test numbers are the single registered passes recorded in results/. Re-scoring the test episodes with this code is a **new
evaluation**: it was not run for this release, and its results are not the paper's. The path exists for readers who want to check the code
on test data:

```bash
python scripts/reproduce.py test-reevaluation --tier 2 --allow-test --host frozen_feat_mini [--host ...]
```

It opens the test manifests only with `--allow-test`, prints a warning, writes only to results/generated/test_reevaluation/ (it refuses any
other output folder and never touches a recorded file), and marks every output with `"new_evaluation": true`. The same path is
`python scripts/evaluate.py --host <host> --episodes test_fsor_5way_1shot test_fsor_5way_5shot --allow-test`. It was tested on synthetic
manifests only (`tests/test_test_reevaluation.py`).

## Tier 3: from scratch (GPU-days)

`python scripts/reproduce.py training --tier 3` prints the commands.

| What | Command | Measured (one RTX 4090) |
|---|---|---|
| FRN fine-tuning, seeds 0-2 | `python scripts/train/train_frn.py --seed S` | 59.8 / 60.2 / 60.3 min per seed |
| DNPG mini-ImageNet, seeds 0-2 (Phase 2 and meta-training, 1- and 5-shot) | `python scripts/train/train_dnpg.py data --dataset mini`; `python scripts/train/train_dnpg.py run --dataset mini --seed S` | 4.47 h per seed |
| DNPG tieredImageNet, seed 0 | `python scripts/train/train_dnpg.py data --dataset tiered`; `... run --dataset tiered --seed 0` | 12.1 h |
| RFS CIFAR-FS, seed 0 | `python scripts/train/train_rfs_cifarfs.py --seed 0` (needs `python third_party/clone.py RFS`) | |

Training is documented, not re-run for this release. GPU non-determinism means that retrained weights differ slightly from the released
ones; evaluate a retrained checkpoint on validation (tier 2) after pinning its sha256 in its host configuration.

## Quickstart: DNPG with CCP on mini-ImageNet validation

```bash
export CCP_DATA=~/ccp-data CCP_ASSETS=~/ccp-assets CCP_CKPT=~/ccp-ckpt:$CCP_ASSETS/checkpoints
python third_party/clone.py DNPG
python scripts/build_bank.py --host dnpg_mini_seed0 --shot 1        # byte-identical to the pinned sha256
python scripts/evaluate.py --host dnpg_mini_seed0 --episodes valA_fsor_5way_1shot valB_fsor_5way_1shot --save-scores
# valB AUROC: host score 76.29, H_fix 77.31, K_fix 76.64, CCP-A 76.54 (equal to the recorded dry run of round (iii),
# results/recorded/validation/fsor_final_dryrun_val_mini-dnpg.json)
```

## Repository map

```
ccp/                 the package: embedding.py (Eqs. 1-2), cohort.py (Eqs. 3-6), fusion.py (Eq. 7), scg.py (Eq. 8), scores/ (one
                     module per score), baselines.py, stats.py, metrics.py, episodes.py, data/, hosts/, pipeline.py, naming.py,
                     recorded.py, registration.py, assets.py
configs/             hosts/ (checkpoints, banks, frozen fusion weights, sha256), datasets/ (incl. the class names of Figure 4),
                     ccp/registered.yaml
manifests/           fixed episode manifests (indices only; verbatim)
scripts/             reproduce.py, make_tables.py, make_figures.py, figures/, make_results_docs.py, derive_paired.py, evaluate.py,
                     build_bank.py, rounds/ (the registered rounds), fidelity/, prepare_data/, train/
results/             recorded results of the eight rounds and derived records (verbatim), checks/ (checks of this release); PROVENANCE.md
bundles/             approval file and STARTED / DONE markers of rounds (v)-(viii) (verbatim)
paper_assets/        Tables 1-8 and Figures 2-3 of the paper
docs/                protocol.md, naming.md, data.md, checkpoints.md, registration_log.md, results/, preregistration/
third_party/         clone.py (pinned commits), patches/dnpg.patch, fonts/ (Liberation Sans, OFL), shims/
tests/               pytest; data and GPU tests are opt-in (CCP_TEST_DATA=1, CCP_TEST_GPU=1)
```
