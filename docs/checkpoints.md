# Checkpoints, release assets and third-party weights

Every checkpoint and every cohort bank is loaded only after its sha256 matches `configs/hosts/<host>.yaml`; the full list of files and
hashes is [results/checkpoints.md](results/checkpoints.md) (generated from the host configurations). Checkpoint paths are relative to a
search root: set `CCP_CKPT` to the folder of the third-party files followed by `<CCP_ASSETS>/checkpoints` (`os.pathsep`-separated).

## Release assets (this work)

One archive per host plus the recorded caches, downloadable from either of two places with the same files:

- the GitHub release v2.0.0: https://github.com/adu-baffour/cohort-calibrated-prototypes/releases/tag/v2.0.0
- the Zenodo record: https://doi.org/10.5281/zenodo.23695548 (DOI 10.5281/zenodo.23695548)

Names, sizes and the sha256 of every archive and every member are in `MANIFEST.txt` and `SHA256SUMS` (published with the archives) and in
results/PROVENANCE.md, "Release assets"; check the downloads with `sha256sum -c SHA256SUMS`. Every archive is under 2 GB. Unpack every
archive into one folder, set `CCP_ASSETS` to it and add its `checkpoints/` folder to `CCP_CKPT`.

| Archive | Contents | Used for |
|---|---|---|
| `cohort-calibrated-prototypes-2.0.0-checkpoints-frn_mini_seed{0,1,2}.tar` | `checkpoints/frn/mini/seed<s>/best.pth` (the evaluated checkpoint, selected on validation) and `last.pth` | FRN rows (Table 2) |
| `cohort-calibrated-prototypes-2.0.0-checkpoints-dnpg_mini_seed{0,1,2}.tar` | `checkpoints/dnpg/mini/seed<s>/`: `phase2_mini_last.pth` (needed to build the model), `max_acc_{1,5}shot.pth` (the evaluated checkpoints, selected on validation classes), `last_{1,5}shot.pth` (information only) | DNPG rows (Tables 2, 4, 5) |
| `cohort-calibrated-prototypes-2.0.0-checkpoints-dnpg_tiered_seed0.tar` | `checkpoints/dnpg/tiered/seed0/` (same layout) | the tieredImageNet validation decision before round (iv) (not met; no test pass) |
| `cohort-calibrated-prototypes-2.0.0-recorded.tar` | `recorded/test/*.npz`: per-episode metrics of the recorded test passes; `recorded/validation/`: validation score caches (DNPG, SnaTCHer-F, the Figure 2 / 4 cache, the dry runs) | tier 1 (tables, figures, numbers); ccp/assets.py |

The archives are the 1.5.0 archives of the authors' earlier release repository (private) under new names: same members, same bytes (results/PROVENANCE.md lists both names and
checksums). The test archive holds per-episode metrics only (AUROC, accuracy, OSCR, FPR95 per row and episode); no image, feature, label
or score of a test image.

Training (one RTX 4090, measured): FRN 59.8 / 60.2 / 60.3 min for seeds 0 / 1 / 2; DNPG mini 4.47 h per seed (Phase 2 + meta-training,
1- and 5-shot); DNPG tieredImageNet seed 0 12.1 h. Commands: `python scripts/reproduce.py training --tier 3`.

### Not redistributed

- **RFS CIFAR-FS backbone** (`rfs/cifarfs/seed0/last.pth`, sha256 `69f7f2975bbac2c9ae299a3ec59bd0522f737b65a6d5012462741376889d31b9`,
  pinned in `configs/hosts/cifarfs_rfs_seed0.yaml`). The RFS code it was trained with has no licence file, so neither the code nor the
  checkpoint is redistributed. `python scripts/train/train_rfs_cifarfs.py --seed 0` with the pinned RFS commit (`python third_party/clone.py
  RFS`) reproduces it up to GPU non-determinism; the evaluated checkpoint is available from the authors on request.
- **Cohort banks**: rebuilt byte-identically by `python scripts/build_bank.py --host <host>` (ResNet-12 hosts), `python
  scripts/rounds/final_round.py build-banks` (RFS, CUB target banks) and `python scripts/rounds/foundation_round.py build-banks`
  (foundation models); hashes in [results/checkpoints.md](results/checkpoints.md).

## Third-party checkpoints (download from the source; not re-hosted)

| Place at `$CCP_CKPT/...` | Source | Used by |
|---|---|---|
| `third_party/feat/mini/Res12-pre.pth`, `third_party/feat/tiered/Res12-pre.pth` | FEAT README, pre-trained ResNet-12 (Google Drive) | frozen FEAT |
| `third_party/feat_meta/{mini,tiered}/feat-{1,5}-shot.pth` | FEAT README model zoo (Google Drive) | SnaTCHer-F |
| `third_party/tane/miniImageNet_pre.pth`, `third_party/tane/tieredImageNet_pre.pth` | TANE README (Google Drive) | TANE pre-trained; DNPG initialization |
| `third_party/tane/miniImageNet_category_vector.pickle` | TANE README (Google Drive) | DNPG training data only |
| `third_party/clip/ViT-B-16.pt` | OpenAI CLIP download URL (`python scripts/rounds/foundation_round.py download-weights`) | CLIP ViT-B/16 |
| `third_party/dinov2/dinov2_vit{s,b,l}14_pretrain.pth` | dl.fbaipublicfiles.com/dinov2 (same command) | DINOv2 ViT-S/14, B/14, L/14 |

Exact links and sha256: [results/checkpoints.md](results/checkpoints.md) and `configs/hosts/`. FEAT's `feat-1-shot-improved.pth` (in its
Drive folder, not linked in its README) is not used. The two TANE checkpoints' stored epochs (66 and 81) are consistent with selection on
the test split by their authors (paper, Section 4.2).

## Third-party code

`python third_party/clone.py <name>` clones FEAT, SnaTCHer, TANE, DNPG, RFS, CLIP and dinov2 at the pinned commits and applies our patch
(THIRD_PARTY_LICENSES.md). No third-party source file is copied into this repository.

**DNPG moved.** The research code cloned DNPG from `https://github.com/iCGY96/DNPG`; the repository now lives at
`https://github.com/gary23ai/DNPG` and the old URL redirects (HTTP 301; the GitHub API resolves both to gary23ai/DNPG). `clone.py` uses
the new URL. Verified on 2026-10-26: commit `38b8fd13656ae66c87488415223a8df24ad67c80` ("init", 2024-12-11) is the HEAD of `main` at the
new URL; `python third_party/clone.py DNPG` checks it out and `third_party/patches/dnpg.patch` applies cleanly (5 files, +36 / −12;
re-running is idempotent); the patched clone is identical, file for file, to the patched clone used by release 1.5.0 of the authors' earlier release repository (private).
