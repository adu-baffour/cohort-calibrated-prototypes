# SnaTCHer-F on FEAT's released meta-trained ResNet-12: second trained host (validation only)

Code: github MinkiJ/SnaTCHer @ 335c42469f0a7ad72c5c3480c8effc8c293823e0 (MIT), FEAT github Sha-Lab/FEAT @ 47bdc7c1672e00b027c67469d0291e7502918950 (MIT); clones in third_party/ (git-ignored). Feasibility: research/fsor/SECOND_HOST_FEASIBILITY.md. No training in this host: SnaTCHer trains nothing and FEAT's released models are used as they are (one model per shot; no seeds). FEAT's training code selects `max_acc.pth` on its val split (FEAT `model/trainer/base.py:50-62`); that the Drive files are those outputs is inferred from the code, not stated.

## 1. Checkpoints, published rows, fidelity and channel order (2026-09-30)

### Downloads (approved; each 56,316,902 B; git-ignored)

| Dataset | Shot | File | Source (FEAT README link) | sha256 |
|---|---|---|---|---|
| mini-ImageNet | 1 | data/mini_imagenet/feat_meta/feat-1-shot.pth | https://drive.google.com/file/d/1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ | 60b78143149fb891f1b4d187fe24f03fb431d12d4c853752bf7dac5f90460a41 |
| mini-ImageNet | 5 | data/mini_imagenet/feat_meta/feat-5-shot.pth | https://drive.google.com/file/d/1tCU52WiK6JBydnALI_gF9nYiFCNvvT1A | ee54213c754b2c25d07bb7b032549140d7be8d4e253f597cfa57a314696191ee |
| tieredImageNet | 1 | data/tiered_imagenet/feat_meta/feat-1-shot.pth | https://drive.google.com/file/d/1M93jdOjAn8IihICPKJg8Mb4B-eYDSZfE | 46e71b689d004b6e13fd7d00277d30acd0d79e227e9b288b64cf73cc04143263 |
| tieredImageNet | 5 | data/tiered_imagenet/feat_meta/feat-5-shot.pth | https://drive.google.com/file/d/1nM4HGGZmMpC57cpe7cqY3WdpL8WlyRay | 777cfb7550063b811a5da3753c403faccefd818f9c10d1873e6de08579e2f334 |

`feat-1-shot-improved.pth` (same Drive folder, not linked in the README) is not used. Each file holds `params` = 96 encoder.* + 7 slf_attn.* tensors.

### Published SnaTCHer-F rows (test classes, as reported; research/fsor/sota.csv, verified; CVPR 2021 Table 2 / supp. Tables 2-3)

| Dataset | 1-shot acc | 1-shot AUROC | 5-shot acc | 5-shot AUROC |
|---|---|---|---|---|
| mini-ImageNet | 67.02 ± 0.85 | 68.27 ± 0.96 | 82.02 ± 0.53 | 77.42 ± 0.73 |
| tieredImageNet | 70.52 ± 0.96 | 74.28 ± 0.80 | 84.74 ± 0.69 | 82.02 ± 0.64 |

### Adapter (research/fsor/snatcher_rows.py) and fidelity check

Adapter: our ResNet12 loaded strictly from the FEAT encoder keys, SnaTCHer's own MultiHeadAttention class for slf_attn, both eval, fp32; per episode bproto = mean of the K known supports, proto = slf_attn(bproto), logits = −||q − proto||² / 64, S(q) = SnaTCHer score (prototype of the argmax class replaced by q, slf_attn re-applied, Σ||· − proto||² / 64); the 150 queries are batched but each is scored independently; our class-major order.

Fidelity (scripts/snatcher.py fidelity; research/status/snatcher.json): the first 50 mini valA_fsor episodes per shot were run through SnaTCHer's own `FSLTrainer.evaluate_test` (unchanged; its SnaTCHerF model with its FEAT Res12; episodes laid out in its query-major 10-class order, unknown-class support slots filled with dummy images that its code drops; val split only; per-query values captured by wrapping its calc_auroc / count_acc) and through our adapter. Maximum absolute differences:

| Shot | Comparison | S (SnaTCHer score) | logits | Dist | accuracy (%) |
|---|---|---|---|---|---|
| 1 | end to end (our encoder batch of 5K + 150 images) | 7.6e-06 | 0.0e+00 | 0.0e+00 | 3.3e-06 |
| 1 | same encoder batch (their 10 x (K + 15) tensor) | 7.6e-06 | 0.0e+00 | 0.0e+00 | 3.3e-06 |
| 5 | end to end | 3.2e-03 | 1.8e-03 | 1.3e-03 | 3.3e-06 |
| 5 | same encoder batch | 5.7e-06 | 0.0e+00 | 0.0e+00 | 3.3e-06 |

The scoring code path matches SnaTCHer's within 7.6e-6 at both shots (logits and Dist identical) when both see the same encoder outputs. End to end at 5-shot the per-query values differ by up to 3.2e-3 (accuracy identical): the encoder batches differ in size (their 200 images, including the dummy unknown supports, vs our 175) and cuDNN's convolution results depend on batch composition; at 1-shot (160 vs 155 images) the outputs coincide. This is floating-point noise of the encoder, not a difference in the scoring.

### Tiered channel order (recorded before any tvA / tvB scoring)

FEAT classifier accuracy (slf_attn prototypes) on 200 closed-set 5-way episodes per shot from the 351 tiered BASE classes (RandomState(20260930 + K); our PIL decode fed as RGB or channel-flipped to BGR):

| Checkpoint | RGB | BGR |
|---|---|---|
| tiered feat-1-shot.pth | 86.27 ± 1.18 | 93.64 ± 0.86 |
| tiered feat-5-shot.pth | 94.96 ± 0.61 | 98.11 ± 0.33 |

**Tiered channel order: BGR** (both checkpoints), consistent with the D1 finding for the other released tiered checkpoints. Mini uses RGB (FEAT's mini checkpoints; the frozen FEAT mini rows were verified RGB in pre-flight P2).

