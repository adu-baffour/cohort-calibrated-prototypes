# Method variants on TEST (single pass of bundle part yaml-fsor-variants)

Bundle token eb11617bf34b95b9 (docs/preregistration/FINAL_BUNDLE_variants.md), approved in bundles/FINAL_APPROVED; one pass, started 2026-10-02 15:32:36, done 2026-10-02 15:41:00 (8.4 GPU-minutes), code at 79f0b97. Pre-registration: docs/prereg_variants.md. Settings chosen on validation A and frozen; V1 QW was promoted on validation (G1-G3 on valB, G4 on valC) before this pass. 600 test episodes per shot. Rendered from results/variants/test.json and the per-episode file (sha256 86f9a2150a290dbfa27832434d955a61d5b2c0210a220549a557a75202bd5fcd); no test data was loaded to write this file.

Reference check: REF reproduces the recorded single-pass test AUROC and accuracy of every host and shot: yes (max |difference| 0.0e+00).

REF = CCP-A (frozen FEAT) or H_fix = D + w CCP-A (trained hosts). Cells: paired per-episode AUROC change vs REF, mean ± 95% CI over 600 episodes (% of episodes improved; DNPG: differences averaged over seeds 0-2 per episode); accuracy change in points (frozen FEAT; trained hosts keep their classifier). Decisions after this test: none.

Chosen on valA: {"V1 QW": "V1-QW/lambda=1.0/beta=0.5", "V2 ASK": "V2-ASK", "V3 SHC": "V3-SHC/gamma=0.25", "V4 FW": {"SnaTCHer-F mini": "V4-FW/mu=4", "DNPG mini": "V4-FW/mu=1"}}.

| Cell | REF AUROC | V1 QW | V2 ASK | V3 SHC | V4 FW |
|---|---|---|---|---|---|
| frozen FEAT mini 1-shot | 70.90 | +0.56 ± 0.22 (57%), acc +0.00 | +0.68 ± 0.11 (70%), acc +0.90 | +0.11 ± 0.07 (56%), acc +0.18 | n/a |
| frozen FEAT mini 5-shot | 78.88 | +1.96 ± 0.21 (78%), acc +0.00 | -0.74 ± 0.06 (14%), acc +0.06 | -0.39 ± 0.08 (38%), acc -0.42 | n/a |
| frozen FEAT tiered 1-shot | 74.56 | -0.50 ± 0.13 (37%), acc +0.00 | +0.15 ± 0.08 (54%), acc +0.26 | +0.10 ± 0.06 (54%), acc +0.00 | n/a |
| frozen FEAT tiered 5-shot | 82.79 | -0.68 ± 0.13 (33%), acc +0.00 | -0.60 ± 0.10 (28%), acc -0.07 | -0.10 ± 0.06 (46%), acc -0.45 | n/a |
| SnaTCHer-F mini 1-shot | 71.55 | +0.67 ± 0.11 (70%) | +0.37 ± 0.06 (71%) | -0.10 ± 0.03 (42%) | +0.50 ± 0.14 (58%) |
| SnaTCHer-F mini 5-shot | 79.38 | +1.14 ± 0.09 (85%) | -0.01 ± 0.03 (43%) | -0.34 ± 0.04 (24%) | +0.66 ± 0.12 (63%) |
| DNPG mini 1-shot | 74.30 | +0.23 ± 0.11 (57%) | +0.21 ± 0.06 (64%) | -0.10 ± 0.03 (42%) | +0.00 ± 0.00 (0%) |
| DNPG mini 5-shot | 83.85 | +0.66 ± 0.10 (70%) | -0.06 ± 0.03 (44%) | -0.34 ± 0.04 (24%) | +0.00 ± 0.00 (0%) |

| Family | applicable cells | mean change | min cell | max cell |
|---|---|---|---|---|
| V1 QW (promoted on validation) | 8 | +0.504 | -0.680 | +1.960 |
| V2 ASK | 8 | +0.000 | -0.743 | +0.683 |
| V3 SHC | 8 | -0.144 | -0.390 | +0.111 |
| V4 FW | 4 | +0.290 | +0.000 | +0.661 |

Descriptive only: the validation gates G1-G4 are not applied to test, and no decision is taken after this pass.
