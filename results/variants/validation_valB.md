# Method variants on validation B (tvB): gates G1-G3

Pre-registration: docs/prereg_variants.md. Settings chosen on validation A (results/variants/selection_valA.json) and frozen. REF = CCP-A (frozen FEAT) or H_fix = D + w CCP-A (trained hosts). Cells: paired per-episode AUROC change vs REF, mean ± 95% CI over 1,000 episodes (DNPG: differences averaged over seeds 0-2 per episode); accuracy change in points (frozen FEAT; trained hosts keep their classifier).

Chosen on valA: {"V1 QW": "V1-QW/lambda=1.0/beta=0.5", "V2 ASK": "V2-ASK", "V3 SHC": "V3-SHC/gamma=0.25", "V4 FW": {"SnaTCHer-F mini": "V4-FW/mu=4", "DNPG mini": "V4-FW/mu=1"}}.

V1 status: selected.

| Cell | V1 QW | V2 ASK | V3 SHC | V4 FW |
|---|---|---|---|---|
| frozen FEAT mini 1-shot | +0.25 ± 0.11, acc +0.00 | +0.12 ± 0.10, acc +0.18 | +0.59 ± 0.06, acc +0.48 | n/a |
| frozen FEAT mini 5-shot | +0.96 ± 0.09, acc +0.00 | -1.10 ± 0.06, acc -0.28 | +0.39 ± 0.05, acc -0.03 | n/a |
| frozen FEAT tiered 1-shot | +0.04 ± 0.09, acc +0.00 | -0.15 ± 0.06, acc +0.19 | +0.00 ± 0.05, acc -0.06 | n/a |
| frozen FEAT tiered 5-shot | +0.26 ± 0.09, acc +0.00 | -1.00 ± 0.09, acc -0.16 | -0.07 ± 0.05, acc -0.56 | n/a |
| SnaTCHer-F mini 1-shot | +0.43 ± 0.06 | +0.02 ± 0.05 | +0.10 ± 0.03 | +0.21 ± 0.09 |
| SnaTCHer-F mini 5-shot | +0.68 ± 0.04 | -0.13 ± 0.02 | -0.05 ± 0.02 | +0.09 ± 0.07 |
| DNPG mini 1-shot | +0.02 ± 0.07 | -0.21 ± 0.06 | +0.05 ± 0.03 | +0.00 ± 0.00 |
| DNPG mini 5-shot | +0.09 ± 0.05 | -0.11 ± 0.02 | -0.06 ± 0.02 | +0.00 ± 0.00 |

| Family | mean gain (G1 >= +0.30) | min cell (G2 >= -0.20) | min frozen acc change (G3 >= -0.20) | G1-G3 |
|---|---|---|---|---|
| V1 QW | +0.342 PASS | +0.021 PASS | +0.00 PASS | PASS |
| V2 ASK | -0.320 FAIL | -1.103 FAIL | -0.28 FAIL | FAIL |
| V3 SHC | +0.119 FAIL | -0.066 PASS | -0.56 FAIL | FAIL |
| V4 FW | +0.074 FAIL | +0.000 PASS | - PASS | FAIL |

Candidate (highest valB mean gain among G1-G3 passes; V1 excluded if NO-GO): V1 QW.
