# Method variants on validation C (fresh episodes): gate G4 and promotion

Pre-registration: docs/prereg_variants.md. Settings chosen on validation A (results/variants/selection_valA.json) and frozen. REF = CCP-A (frozen FEAT) or H_fix = D + w CCP-A (trained hosts). Cells: paired per-episode AUROC change vs REF, mean ± 95% CI over 1,000 episodes (DNPG: differences averaged over seeds 0-2 per episode); accuracy change in points (frozen FEAT; trained hosts keep their classifier).

Chosen on valA: {"V1 QW": "V1-QW/lambda=1.0/beta=0.5", "V2 ASK": "V2-ASK", "V3 SHC": "V3-SHC/gamma=0.25", "V4 FW": {"SnaTCHer-F mini": "V4-FW/mu=4", "DNPG mini": "V4-FW/mu=1"}}.

| Cell | V1 QW | V2 ASK | V3 SHC | V4 FW |
|---|---|---|---|---|
| frozen FEAT mini 1-shot | +0.16 ± 0.11, acc +0.00 | +0.15 ± 0.10, acc +0.29 | +0.57 ± 0.05, acc +0.46 | n/a |
| frozen FEAT mini 5-shot | +0.90 ± 0.09, acc +0.00 | -1.18 ± 0.06, acc -0.21 | +0.38 ± 0.05, acc +0.02 | n/a |
| frozen FEAT tiered 1-shot | -0.03 ± 0.09, acc +0.00 | -0.09 ± 0.06, acc +0.06 | +0.04 ± 0.05, acc -0.06 | n/a |
| frozen FEAT tiered 5-shot | +0.23 ± 0.08, acc +0.00 | -1.10 ± 0.09, acc -0.12 | -0.04 ± 0.05, acc -0.38 | n/a |
| SnaTCHer-F mini 1-shot | +0.35 ± 0.06 | +0.03 ± 0.05 | +0.10 ± 0.03 | +0.13 ± 0.09 |
| SnaTCHer-F mini 5-shot | +0.67 ± 0.04 | -0.16 ± 0.02 | -0.07 ± 0.02 | +0.17 ± 0.07 |
| DNPG mini 1-shot | +0.01 ± 0.07 | -0.22 ± 0.06 | +0.04 ± 0.03 | +0.00 ± 0.00 |
| DNPG mini 5-shot | +0.10 ± 0.05 | -0.17 ± 0.03 | -0.06 ± 0.02 | +0.00 ± 0.00 |

| Family | mean gain (>= +0.15) | min cell (>= -0.20) | G4 |
|---|---|---|---|
| V1 QW | +0.299 | -0.035 | PASS |
| V2 ASK | -0.343 | -1.184 | FAIL |
| V3 SHC | +0.120 | -0.066 | FAIL |
| V4 FW | +0.075 | +0.000 | FAIL |

Candidate from valB: V1 QW. Decision: V1 QW promoted (pre-registered extension; a test bundle is drafted for approval).
