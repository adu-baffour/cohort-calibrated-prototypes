# Pre-registration log (Supplementary S1)

Verbatim copies from the research repository (tag `pre-release-archive-2026-09-30`, commit `5f0c034`). Every rule below was committed
before the data it decides on was scored; no threshold was edited after a result. Check: `python scripts/reproduce.py S1` recomputes each
bundle token from its YAML block and matches it against the DONE files and `results/recorded/test/FINAL_APPROVED`.

| File | Source (research repository) | Content |
|---|---|---|
| `FINAL_BUNDLE_fsor.md` | `research/FINAL_BUNDLE.md`, from "# FSOR test bundle" to the end | the four test bundle parts (YAML blocks `yaml-fsor-mini`, `-tiered`, `-mini-dnpg`, `-mini-snatcher`) |
| `DNPG_SETUP_section_8_seed0.md` | `research/dnpg/DNPG_SETUP.md`, second section 8 ("Pre-registration before any seed-0 evaluation") | the seed-0 rule for CCP replacing D |
| `DNPG_SETUP_sections_9-11.md` | `research/dnpg/DNPG_SETUP.md`, from section 9 to the end | H-COMB (9), inductive H_fix (10), tiered DNPG (11), deviation records |
| `SNATCHER_SETUP_section_2.md` | `research/snatcher/SNATCHER_SETUP.md` section 2 | SnaTCHer-F rules S1-S3 |
| `SNATCHER_SETUP_section_1.md` | `research/snatcher/SNATCHER_SETUP.md` section 1 | checkpoints, fidelity check, tiered channel order (S3) |

## Tokens (human approvals, `results/recorded/test/FINAL_APPROVED`)

| Part | Token | Test pass |
|---|---|---|
| MINI | `eaa3aab9d353da1e` | done once |
| TIERED | `fd048685b138b1f9` | done once |
| MINI-DNPG | `01fcac4666048f3e` | done once |
| MINI-SNATCHER | `19da50a9bf457980` | done once |

## Outcomes, in order (all on validation unless marked test)

| Pre-registered decision | Rule | Outcome | Record |
|---|---|---|---|
| CCP replacing DNPG's own score D (seed 0, DNPG_SETUP.md "8. Pre-registration before any seed-0 evaluation") | CCP-A − D >= +0.5 AUROC on valA at both shots, same sign on valB | **NO-GO**: the 5-shot valA gain was +0.24 (< +0.5); recorded as "CCP does not improve a trained FSOR head (DNPG)" | `validation/reports/dnpg_seed0.md` |
| H-COMB, DNPG seeds 1-2 (section 9) | R1 mean (H − D) >= +0.50 on valA both shots and every seed > 0; R2 mean (H − K) >= +0.30; R3 mean (H − D) > 0 on valB | **GO**: R1 +1.22 / +1.34, R2 +0.80 / +0.97, R3 +1.10 / +1.36 (1-shot / 5-shot) | `validation/reports/dnpg_seeds012.md` |
| Inductive H_fix (section 10) | E1 mean over seeds 0-2 (H − H_fix) on valB <= 0.20; E2 mean seeds 1-2 (H_fix − D) >= +0.50 on valB, each seed > 0; E3 mean seeds 1-2 (H_fix − K_fix) >= +0.30 | **PASS**: E1 −0.01 / −0.01, E2 +1.11 / +1.37, E3 +0.69 / +0.89 (1-shot / 5-shot) | `validation/reports/dnpg_hfix.md` |
| MINI-DNPG test (bundle part) | none after the test (single pass) | recorded (test) | `test/RESULTS_fsor.md`, T1 |
| Tiered DNPG seed 0 (section 11) | T1 H_fix − D >= +0.50 on tvB; T2 H_fix − K_fix >= +0.30 on tvB; T3 H_fix − D > 0 on tvA; both shots | **NO-GO**: T1 fails at 5-shot (+0.64 / +0.37); T2 (+0.53 / +0.42) and T3 (+0.82 / +0.97) hold | `validation/reports/dnpg_tiered_seed0.md`, S6 |
| SnaTCHer-F (SNATCHER_SETUP.md section 2), per dataset | S1 H_fix − S >= +0.50 on B; S2 H_fix − K_fix >= +0.30 on B; S3 H_fix − S > 0 on A; both shots | **mini GO**; **tiered NO-GO**: S2 fails (+0.20 / +0.23); S1 and S3 hold | `validation/reports/snatcher_val.md`, S6 |
| MINI-SNATCHER test (bundle part) | none after the test | recorded (test) | `test/RESULTS_fsor.md`, T1 |

Earlier CCP decisions (premise, candidate choice on tvA, MINI and TIERED bundles) are recorded in `validation/reports/ccp_premise.md`,
`ccp_confirm_tiered.md` and the bundle file.
