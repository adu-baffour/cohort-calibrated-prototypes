# Pre-registration: foundation-model backbones (CLIP, DINOv2) (written 2026-10-06, committed before any weight is downloaded and before any image passes through these models; amended once before phase 1, see "Amendment log")

**This round amends the statement "after this round the method and the evaluation are frozen" of `docs/preregistration/final_round.md`,
at the human's request** (a reviewer-risk assessment of 2026-10-05 found that the paper has no modern backbone). **The method is unchanged**:
CCP-A as registered (`configs/ccp/registered.yaml`: symmetric, K = 1000, flip views) and V1 QW (λ = 1.0, β = 0.5, `docs/prereg_variants.md`).
The round only adds frozen hosts. **No foundation-model number has been computed by anyone on this project** (Section 12 lists what the
research repository contains). There are no GO / NO-GO gates: every registered row goes to test whatever validation shows. One bundle,
one test pass; decisions after the test: none.

Code: `ccp/hosts/vit_input.py` (input rule, (n, D, 1, 1) feature convention), `ccp/hosts/clip_frozen.py`, `ccp/hosts/dinov2_frozen.py`
(registered in `build_host`), `ccp/foundation.py` (scorer that embeds each split once; registered statistics), `scripts/foundation_round.py`
(driver), `scripts/prepare_data/cub.py --views eval224`, 18 host configs `configs/hosts/{mini,tiered,cifarfs,cub}_{clip_b16,dinov2_s14,dinov2_b14,dinov2_l14}.yaml`
and `cub_{clip_b16,dinov2_b14}_eval224.yaml`; tests `tests/test_foundation.py`.

## 1. Backbones (frozen; image encoder only; no class names and no text encoder anywhere)

| Id | Model | Code (fetched by `third_party/clone.py`, never copied) | Weights | Feature |
|---|---|---|---|---|
| `clip_b16` (primary) | CLIP ViT-B/16, OpenAI (Radford et al., ICML 2021) | github.com/openai/CLIP @ `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (`clip/model.py` only, loaded under an isolated module name; the tokenizer is never imported) | `ViT-B-16.pt` from CLIP's `_MODELS` URL `…/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt` (350,837,078 bytes; the URL embeds the expected sha256, which is checked) | `visual(x)` = the projected 512-d image embedding CLIP uses for similarity (= `encode_image`) |
| `dinov2_s14` (ladder) | DINOv2 ViT-S/14, no registers (Oquab et al., TMLR 2024) | github.com/facebookresearch/dinov2 @ `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, `torch.hub.load(<local clone>, 'dinov2_vits14', source='local', pretrained=False)` | `dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` (88,283,115 bytes) | `forward_features(x)['x_norm_clstoken']`, 384-d |
| `dinov2_b14` (primary) | DINOv2 ViT-B/14, no registers | same | `…/dinov2_vitb14/dinov2_vitb14_pretrain.pth` (346,378,731 bytes) | final-norm CLS token, 768-d |
| `dinov2_l14` (ladder) | DINOv2 ViT-L/14, no registers | same | `…/dinov2_vitl14/dinov2_vitl14_pretrain.pth` (1,217,586,395 bytes) | final-norm CLS token, 1024-d |

- Total download 2,003,085,319 bytes (2.0 GB; approved by the human, "about 2.2 GB").
- CLIP: as in `clip.load(jit=False)`, the state dict of the TorchScript archive goes to `build_model()` (which converts to fp16); the model is then
  cast back to fp32 with `.float()` (clip.load keeps fp16 on CUDA; here fp32 weights are run under the registered bf16 autocast) and only the
  visual tower is kept.
- **Loaders read only the pinned local weight file, after its sha256 check, with no implicit download** (amendment A3; tested with every
  download function patched to fail, and with a hash mismatch that must refuse to load). DINOv2: the sha256 is checked first, then the hub model
  is built from the local clone with `pretrained=False` and `load_state_dict(strict=True)` from the pinned file. CLIP: the loader reproduces
  `clip.load(<local path>, jit=False)` step by step (the archive's state dict -> `build_model()`) instead of calling `clip.load`, because
  importing the `clip` package imports its tokenizer, whose `ftfy` / `regex` dependencies are not in the pinned environment; only the pinned
  local file is read.
- **CLIP loader fidelity (F1)**: on the first 500 mini base images (both views, registered input rule and gap()), the host path equals the
  official `clip.load(<pinned local file>, device='cuda', jit=False).float().encode_image` from the pinned clone **bit for bit**
  (`checks/clip_loader_fidelity.py`, `results/foundation_round/clip_loader_fidelity.md`; GPU + data test). ftfy 6.3.1 and regex 2026.9.29 were
  installed for this check only.
- DINOv2: weights loaded strictly from the pinned file (not through torch.hub's cache). xFormers is not installed, so DINOv2 uses its
  plain-PyTorch attention / MLP paths (its own documented fallback). The 518-pixel position embeddings are interpolated to 224 by DINOv2's own
  `interpolate_pos_encoding` (bicubic, interpolate_offset 0.1, no antialias).
- Licences (verified at the pinned commits): CLIP MIT ("Copyright (c) 2021 OpenAI"); DINOv2 Apache-2.0 for code and backbone weights (README
  "DINOv2 code and model weights are released under the Apache License 2.0"; changed from CC-BY-NC on 2023-08-31, commit 81b2b64). The newer
  X-Ray-DINO and Cell-DINO add-ons in the same repository carry non-commercial licences and are not used.
- Package versions: Python 3.10.11, torch 2.14.0+cu130, torchvision 0.29.0+cu130 (the release environment; nothing new is installed).
- **Every weight file's sha256 is pinned in phase 1, before any image passes through a model** (`download-weights`, then `pin`, committed).

## 2. Input: one rule for every dataset

- The registered stored arrays: mini `eval84`, tiered `bgr84` (84 x 84), CIFAR-FS `eval32`, CUB `eval84`. Every new host config says
  `channel_order: RGB` (the code's spelling); the tiered arrays are stored BGR and `ccp.pipeline.host_images` flips them to RGB (tested).
- On the GPU, in fp32: `x / 255 -> F.interpolate(size 224, mode='bicubic', align_corners=False, antialias=True) -> clamp(0, 1) -> (x - mean) / std`
  with the model's constants: CLIP mean 0.48145466 0.4578275 0.40821073, std 0.26862954 0.26130258 0.27577711; DINOv2 mean 0.485 0.456 0.406,
  std 0.229 0.224 0.225. torch's bicubic differs slightly from PIL's; **this torch path is the definition**. Arrays that are already
  224 x 224 (CUB eval224) are not resampled.
- **Analysis row only**: CUB view `eval224` (PIL Resize(256, bicubic) on the short side, long side int(256 · long / short) as torchvision,
  CenterCrop(224) with offsets int(round((side − 224) / 2)), from the original images; same global order as eval84; sha256 pinned in phase 1),
  for the two primary backbones (hosts `cub_clip_b16_eval224`, `cub_dinov2_b14_eval224`). It shows whether upsampling changes the CCP gain.
- Feature convention: `encoder_forward` returns the global feature as an (n, D, 1, 1) map, so the registered `gap()` of `ccp/hosts/base.py`
  applies unchanged (bf16 autocast forward, the 1 x 1 map rounded to fp16, fp32 mean; batches of 500; flip views); stored precision fp32.

## 3. Pretraining overlap (disclosure; no experiment)

**DINOv2** (arXiv 2304.07193 v2, TMLR 01/2024). Sec. 3: "Our selection of curated datasets is detailed in the appendix (Table 15) and contains
ImageNet-22k, the train split of ImageNet-1k, Google Landmarks and several fine-grained datasets." Table 15 (composition of LVD-142M):
ImageNet-22k included **as is** (14,197,086 images) and as a retrieval query (56.8M retrieved); ImageNet-1k **train** split as a retrieval
query (40,997,344 retrieved; App. A.4: k = 32 "to make this specific retrieval a core part of our LVD-142M dataset"); **CUB-200-2011 train
(5,994 images) as a retrieval query (1,000,000 retrieved images in the final set)**; also Oxford-IIIT Pet, Caltech 101, Food-101, Stanford Cars
and others. Table 18 also marks CIFAR-10 / CIFAR-100 as sources for retrieval, but they are absent from Table 15 (UNVERIFIED which is right).
De-duplication (Sec. 3, App. A.3) removed near-duplicates of the **test / validation** images of DINOv2's own evaluation benchmarks; it did
not remove ImageNet-1k **train** images, which is where mini- and tieredImageNet's images come from.

**CLIP** (arXiv 2103.00020). Sec. 2.2: WIT, "400 million (image, text) pairs collected form a variety of publicly available sources on the
Internet … whose text includes one of a set of 500,000 queries"; footnote 1: the queries include "all WordNet synsets not already in the
query list" (ImageNet's classes are WordNet synsets). Sec. 2.3: "We train CLIP from scratch without initializing the image encoder with
ImageNet weights." Sec. 5 (data overlap): no de-duplication before training; median overlap 2.2%, mean 3.2% over 35 evaluation sets;
"Birdsnap … has the second largest overlap at 12.1%"; CIFAR-100 duplicates "were false positives of small objects such as birds or planes".
No numeric overlap is given for ImageNet; CUB is not evaluated in the CLIP paper.

**Consequence, stated plainly.** These encoders have very likely seen the benchmark classes, and possibly the very images: mini- and
tieredImageNet are subsets of ImageNet, which DINOv2 used as is (ImageNet-22k) and as a retrieval core (ImageNet-1k train); DINOv2 retrieved
a million images around CUB's official training images, which span all 200 CUB classes, i.e. also the classes of our CUB validation and test
splits (the FRN split is by class, the official split is by image). Absolute numbers are therefore **not comparable** with ResNet-12 FSOR
methods trained on base classes only. **The only claim is the paired comparison CCP-A vs COSF on identical features**, and these rows go in a
separate table block in the paper.

## 4. Hosts and banks

- 4 backbones x 4 datasets = **16 hosts**, `configs/hosts/{mini,tiered,cifarfs,cub}_{clip_b16,dinov2_s14,dinov2_b14,dinov2_l14}.yaml`, plus the
  2 eval224 analysis hosts.
- Bank = that dataset's own base split, flip views, K = 1000: mini all 38,400; tiered the registered RandomState(20260926) subset of 38,400
  (`embed: subset_only`: only the subset is embedded; CCP-S is not a row); CIFAR-FS all 38,400; CUB the 100 training classes (5,903 images,
  the standard in-domain protocol, since nothing was trained on mini). eval224 hosts: the CUB training classes in the eval224 view.
- **Analysis row**: CUB scored with the **mini** base bank (bank mismatch on a general backbone), for every backbone (host `mini_<b>` on
  CUB eval84; rows COS, COSF, CCP-A, V1).
- Bank sha256 pinned in phase 1 (`build-banks`, then `pin`); one bank is rebuilt in memory and must be byte-identical.

## 5. Episodes (the committed manifests)

mini valA / valB / test; tiered tvA / tvB / test; CIFAR-FS valA / valB / test; CUB val (serves as valA) and test. **New**: CUB valB, 1,000
episodes per shot, seeds 20261071 / 20261075 (base 20261070 + shot), the (known, unknown) keys of CUB val of the same shot excluded, written by
`make-manifests` and committed before any scoring. As for mini valA / valB, CUB val and valB share the 50 validation classes.

## 6. Rows per host

COS, COSF (the host row), CCP-A, V1 (CCP-A labels), and B1-B9 at the settings chosen on each host's valA (tvA; CUB val) by the registered
baselines rule (`docs/preregistration/baselines.md`: per family the mean valA AUROC over both shots, ties to the earlier grid value; B1 / B2
evidence weights = pooled std ratio on valA, per shot), confirmed on valB. B9 remains the transductive reference. The analysis cells (CUB with
the mini bank; CUB eval224) have rows COS, COSF, CCP-A, V1 (no baselines).

## 7. Metrics

Accuracy, AUROC, OSCR and FPR95 (`ccp.metrics`, as registered in `docs/preregistration/final_round.md`, Section 5), mean and 95% CI over
episodes; paired per-episode differences against COSF with Wilcoxon p and the fraction of episodes improved.
**Multiplicity**: over the **32 primary cells** (4 backbones x 4 datasets x 2 shots), for the CCP-A − COSF AUROC gain: one-sided Wilcoxon
signed-rank p (H1: gain > 0), **Holm** step-down adjustment; a cell counts if its Holm-adjusted p < 0.05 and its mean is positive. Holm
adjusts p-values, not intervals; the corrected **interval** reported next to it is the Bonferroni-simultaneous 95% CI
(mean ± z_{1 − 0.05/64} · sd / √n), and the count of cells whose Bonferroni CI excludes zero is also reported.

**Headroom-normalised gain (amendment A5; descriptive only).** Added after the closed-set sanity accuracies of phase 1 were seen (valA COSF
5-way accuracy up to 99.1%) and **before any open-set number was computed**. For every cell and row, next to the raw AUROC gain: the per-episode
headroom-normalised gain h = (AUROC_row − AUROC_COSF) / (100 − AUROC_COSF), with its paired per-episode mean, 95% CI (1.96 std / √n) and the
fraction of episodes with h > 0; episodes with AUROC_COSF = 100 have no headroom (h undefined) and are excluded and counted
(`ccp.foundation.headroom_paired`). The cell-level version (mean AUROC_row − mean AUROC_COSF) / (100 − mean AUROC_COSF)
(`ccp.foundation.headroom_cell`) is shown with the raw gains in the H3 description. Nothing in H1, H2, H2b, H2-s, H2b-s, H3 or their statistics
changes.

## 8. Hypotheses (registered before any number exists; no GO / NO-GO)

- **H1**: CCP-A − COSF AUROC gain > 0. Report all 32 cells, the Holm count and the Bonferroni-CI count (Section 7).
- **H2 (mechanism)**: across the 32 test cells, the test D1 (registered standardised gap, per shot) and the test gain are positively
  rank-correlated: Spearman ρ, one-sided permutation p (100,000 permutations, RandomState(20261006)).
- **H2b (prediction)**: the valA D1 of each cell, frozen in the bundle before the test, predicts the test gain: Spearman ρ, one-sided
  permutation p (same procedure). Note: on CIFAR-FS in the final round, the **validation gain** of CCP-A over COSF (−0.53 / −0.07) disagreed
  in sign with the test gain (+0.47 / +1.52), while the D1 statistic was positive on both (validation 0.105 / 0.321, test 0.254 / 0.645).
- **H2-s and H2b-s (within-dataset robustness, amendment A1; reported alongside H2 and H2b, H2 stays primary)**: strata = the 8
  dataset x shot strata of 4 backbones each. Statistic: the mean over strata of the within-stratum Spearman ρ between D1 (test D1 for H2-s;
  valA D1, tiered tvA, for H2b-s) and the test gain. Null: independent permutations of the gains within each stratum, 100,000 Monte Carlo draws,
  RandomState(20261007); one-sided p = (1 + #{statistic under permutation >= observed}) / (1 + 100,000) (`ccp.foundation.stratified_spearman_perm`,
  tested on a hand-built example).
- **H3 (exploratory, description only)**: within DINOv2 S → B → L, per dataset and shot, does the gain fall as COSF AUROC rises? A table of
  the three COSF AUROCs and the three gains per (dataset, shot); no test, no claim beyond description.

## 9. Sanity checks (reported, not gates)

bf16 (registered) vs fp32 features on the first 500 mini base images, mean and min cosine, per backbone; 5-way closed-set COSF accuracy on
valA for each host (from the selection run); every weight file's sha256 (checked on every load); no NaN in any bank or result.

## 10. Phases (enforced by `scripts/foundation_round.py`; every step refuses on uncommitted code and records git HEAD)

0. This file, the host adapters, the configs, the driver and the tests. Commit. STOP for approval.
1. `download-weights` + `pin` (commit); `make-manifests` (commit); `prepare-cub224` (commit; sha256 of the eval224 arrays); `build-banks` +
   `pin` (commit); `sanity` (commit).
2. `select-baselines` on valA (writes the valA D1 of the 32 cells too; commit); `report-valB` (valB / tvB / CUB valB and the analysis cells on
   CUB val and valB; commit); `d1-val` (renders the frozen valA D1 predictions; commit).
3. `draft-bundle`: `docs/preregistration/FINAL_BUNDLE_foundation_round.md`, block `yaml-fsor-foundation-round`, pinning this file, every
   validation record, every weight, bank and test-manifest sha256, every frozen choice and the valA D1 predictions;
   `bundle --dry-run-val --set-approvable` must reproduce every committed validation number exactly. Report the token, the runtime and the
   command.

**Technical substitutions (amendment A4, registered now).** (i) If a host runs out of GPU memory at BATCH = 500, that host alone uses 250
(the `batch` field of its config; recorded in the bundle; the default 500 is unchanged for every other host). (ii) If the sanity step exposes a
demonstrable implementation bug, it may be fixed: the diff is committed with the reason and Tier 1 is rerun. (iii) **The input rule, the
feature choice, the banks, the rows and the hypotheses may not change after any number has been seen.**

After the human's approval only: one test pass (`bundle`) writing `results/foundation_round/{test.json, test_episodes.npz}`, then
`test-report` (test.md, d1_test.png, test_hypotheses.json from the recorded files only).

## 11. Compute estimate (one RTX 4090; refined in phase 1 from measured throughput)

Images passed through each encoder (flip views double them): banks about 2 x 121k; validation splits about 2 x 146k per pass (tiered val
124,261 images dominates); test splits about 2 x 233k (tiered test 206,209). Assumed throughput at 224 x 224, bf16: ViT-S 4,000 / s, CLIP
B/16 and DINOv2 B/14 1,500 / s, ViT-L 500 / s. Estimate: banks 25 min; `select-baselines` 60 min (embedding about 25 min, scoring 32
host-shots x about 1 min); `report-valB` 45 min; dry run about 100 min; test pass about 60 min. **Total about 5 GPU-hours.**

## 12. Already seen, and earlier foundation-model work on this project

Everything of the paper, the baselines, variants and final rounds, including their test passes (in particular: CIFAR-FS test CCP-A − COSF
+0.47 / +1.52; tiered gains small; V1 below the host on tiered).

Search of 2026-10-06 (this repository incl. git history; the research repository incl. git history; user caches): **no CLIP or DINOv2
feature, bank or number has ever been computed on this project, and no CLIP or DINOv2 weights are cached.** What exists:
- published numbers copied from other papers (e.g. UTL's CLIP RN50 / ViT-B/16 rows, OSLO's TIMM ViT-B/16 row) in
  `results/recorded/published/sota.csv` and in the research repository's literature notes (`research/fsor/SOTA*.md`, `AUDIT_NOVELTY.md`,
  `PROTOCOL.md`, `research/CAMPAIGN_LOG.md:489`), all marked "listed, not reproduced";
- literature search queries and a gap analysis (`Q1_Gap_Analysis_Image_Classification_Sept_2026.md`) that mention foundation models;
- unrelated cached weights from 2024 that no code of this project references: torchvision ImageNet ResNets and ViT-B/L/H in the torch hub
  cache and a DeiT-B checkpoint in the Hugging Face cache.

## 13. Deviations from the request, and why

- `channel_order: RGB` (upper case, the code's existing spelling) instead of `rgb`.
- "Holm-corrected CI": Holm corrects p-values; the corrected interval is the Bonferroni-simultaneous CI (Section 7). Both counts are reported.
- The CIFAR-FS remark in H2b is corrected: the numbers −0.53 / −0.07 were the validation CCP-A − COSF AUROC gains, not D1 values.
- Weights are 2.0 GB in total (CLIP ViT-B/16 is 0.35 GB, not 0.6 GB).
- The tiered banks embed only the 38,400-image subset (`subset_only`), as for the trained tiered hosts, not all 448,695 base images.
- Two extra host configs implement the eval224 analysis row (18 configs in total).
- CLIP is loaded by reproducing `clip.load(jit=False)` rather than calling it (Section 1, A3).

## 14. Known issues

- The bank metadata field `subset` carries the tiered subset seed (RandomState(20260926), 38,400) for every non-mini bank (CIFAR-FS, CUB and
  the foundation banks), although no subset is drawn for them. Nothing reads this field. It is left unchanged because the pinned sha256 of the
  CIFAR-FS bank (`configs/hosts/cifarfs_rfs_seed0.yaml`) covers these bytes (code comment in `ccp.pipeline.make_bank`; amendment A2).

## Amendment log

Amendment 1 (2026-10-06, requested by the human after reviewing phase 0; no number of any kind had been computed, no weight downloaded):

- A1: H2-s and H2b-s, the within dataset x shot robustness statistics (Section 8), with a unit test; H2 stays primary.
- A2: "Known issues" (Section 14) and a code comment in `make_bank` on the bank metadata field.
- A3: both loaders read only the pinned local weight file after the sha256 check, with no implicit download (Section 1); tests for a hash
  mismatch and for the absence of any download.
- A4: technical substitutions (Section 10): BATCH 250 for a host that runs out of memory, recorded in the bundle; implementation-bug fixes
  after the sanity step committed with their reason and a Tier 1 rerun; nothing else may change after a number is seen.

Amendment 2 (2026-10-06, requested by the human after reviewing phase 1; only closed-set COSF sanity accuracies, bf16-vs-fp32 cosines and
bank / weight hashes had been computed; no open-set number, no CCP row, no baseline):

- F1: CLIP loader fidelity check against the official package (bitwise identical; Section 1); ftfy / regex added to requirements.txt.
- A5: the descriptive headroom-normalised gain column (Section 7), added after the closed-set sanity accuracies were seen and before any
  open-set number was computed; no change to H1, H2, H2b, H2-s, H2b-s, H3 or their statistics.
