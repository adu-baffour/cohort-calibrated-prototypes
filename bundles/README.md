# Test-pass guard for new bundle parts

Each bundle part's test pass refuses to run unless the token of its YAML block is listed in `bundles/FINAL_APPROVED` (one token per line,
written by the human). It writes a STARTED marker before loading any test manifest and a DONE marker at the end, refuses a second pass, and
refuses while a STARTED marker exists without DONE.

| Part | Block | Command |
|---|---|---|
| training-free baselines | `yaml-fsor-baselines` in `docs/preregistration/FINAL_BUNDLE_baselines.md` | `scripts/baselines.py bundle` |
| method variants V1-V4 | `yaml-fsor-variants` in `docs/preregistration/FINAL_BUNDLE_variants.md` | `scripts/variants.py bundle` |
| final evaluation round (A CIFAR-FS, B mini -> CUB, C OSCR / FPR95, D D1) | `yaml-fsor-final-round` in `docs/preregistration/FINAL_BUNDLE_final_round.md` | `scripts/final_round.py bundle` |
| foundation-model round (frozen CLIP ViT-B/16, DINOv2 ViT-S/B/L-14) | `yaml-fsor-foundation-round` in `docs/preregistration/FINAL_BUNDLE_foundation_round.md` | `scripts/foundation_round.py bundle` |

Current state (all four passes are complete):

| File | Content | Written by |
|---|---|---|
| `FINAL_APPROVED` | `45175a9bbd7ef27d` (baselines), `eb11617bf34b95b9` (variants), `97b238d3482dd690` (final round), `204f66d342b118be` (foundation round) | the human |
| `FINAL_TEST_BASELINES_STARTED.json` | token `45175a9bbd7ef27d`, started 2026-10-01 14:55:36 | `scripts/baselines.py bundle` |
| `FINAL_TEST_BASELINES_DONE.json` | token `45175a9bbd7ef27d`, done 2026-10-01 15:04:08 | `scripts/baselines.py bundle` |
| `FINAL_TEST_VARIANTS_STARTED.json` | token `eb11617bf34b95b9`, started 2026-10-02 15:32:36 | `scripts/variants.py bundle` |
| `FINAL_TEST_VARIANTS_DONE.json` | token `eb11617bf34b95b9`, done 2026-10-02 15:41:00 | `scripts/variants.py bundle` |
| `FINAL_TEST_FINAL_ROUND_STARTED.json` | token `97b238d3482dd690`, started 2026-10-05 13:36:10 | `scripts/final_round.py bundle` |
| `FINAL_TEST_FINAL_ROUND_DONE.json` | token `97b238d3482dd690`, done 2026-10-05 13:53:39 | `scripts/final_round.py bundle` |
| `FINAL_TEST_FOUNDATION_ROUND_STARTED.json` | token `204f66d342b118be`, started 2026-10-06 12:25:18 | `scripts/foundation_round.py bundle` |
| `FINAL_TEST_FOUNDATION_ROUND_DONE.json` | token `204f66d342b118be`, done 2026-10-06 12:51:26 | `scripts/foundation_round.py bundle` |

Because the DONE markers exist, all four commands refuse to run again. The outputs of the single passes are `results/baselines/test.{json,md}`
with the release asset `recorded/test/baselines_test_episodes.npz`, and `results/variants/test.{json,md}` with the release asset
`recorded/test/variants_test_episodes.npz`, and `results/final_round/test.{json,md}` and `d1_test.png` with the release asset
`recorded/test/final_round_test_episodes.npz`, and `results/foundation_round/test.{json,md}`, `d1_test.png` and `test_hypotheses.json` with the
release asset `recorded/test/foundation_round_test_episodes.npz`. These files are never edited or deleted. A new bundle part needs its own pre-registered block, a new
token approved by the human in `FINAL_APPROVED`, and its own STARTED / DONE markers. The four paper passes (MINI, TIERED, MINI-DNPG,
MINI-SNATCHER) have their own approval file and markers in `results/recorded/test/`.
