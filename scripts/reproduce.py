"""Reproduce the paper's tables, figures and numbers, and check them (REPRODUCE.md has the full description).

    python scripts/reproduce.py list
    python scripts/reproduce.py all --tier 1                     # CPU, minutes: every check below; must end with "tier 1: all OK"
    python scripts/reproduce.py tables --tier 1                  # one step
    python scripts/reproduce.py val-equivalence --tier 2         # GPU: re-evaluate every host on validation, compare with the recorded outputs
    python scripts/reproduce.py figure-cache --tier 2            # GPU: rebuild the Figure 2 / 4 validation score cache, compare its sha256
    python scripts/reproduce.py test-reevaluation --tier 2 --allow-test --host frozen_feat_mini
                                                                 # GPU: RE-EVALUATION of test episodes, not run by the authors for this release
    python scripts/reproduce.py training --tier 3                # print the training commands (FRN, DNPG, RFS)

Tier 1 needs this repository and the release assets ($CCP_ASSETS: per-episode metric files and validation score caches); Figure 4 is
checked without its image panels (dataset images; `python scripts/make_figures.py Fig4` draws them from $CCP_DATA). The steps
that compare with the paper's own sources (tables, numbers, pins) need them extracted (python scripts/paper_pin.py extract); without them
those steps report "skipped" and the summary says so. Tier 2 also needs the datasets, third-party code (third_party/clone.py) and
checkpoints ($CCP_CKPT). One GPU job at a time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SKIPPED = 3     # exit code of a check whose inputs (the paper's sources) are not present

TIER1 = {   # step -> (commands, what it checks)
    "provenance": ([["scripts/provenance.py", "check"]], "every recorded file equals its sha256 in results/PROVENANCE.md"),
    "registration": ([["-c", "import sys; sys.path.insert(0, '.'); from ccp import registration; r = registration.check(); "
                             "bad = [x['round'] for x in r if not x['ok']]; print('registration:', f'{len(r)} rounds, tokens consistent' if not bad "
                             "else f'MISMATCH in {bad}'); sys.exit(1 if bad else 0)"]],
                     "the eight rounds' tokens (bundle block, DONE marker, approval, test record)"),
    "recorded-reports": ([["scripts/rounds/render_recorded.py", "all"]], "every recorded report re-renders to its recorded text"),
    "derived": ([["scripts/derive_paired.py", "--check"]], "results/derived/paired_derived.json equals a fresh computation"),
    "tables": ([["scripts/make_tables.py", "all", "--check"]], "Tables 1-8 equal paper_assets/tables/ and the paper's tables"),
    "figures": ([["scripts/make_figures.py", "all", "--check", "--no-images"]],
                "Figures 1-4 and the repository figures: plotted data equal the recorded data (Figure 4 without its image panels)"),
    "numbers": ([["scripts/paper_numbers.py", "--check"]], "every number of the paper's text matches its source (docs/paper_numbers.md)"),
    "results-docs": ([["scripts/make_results_docs.py", "--check"]], "docs/results/*.md and docs/registration_log.md are up to date"),
    "paper-pins": ([["scripts/paper_pin.py", "check"]], "the paper's checked files equal their pinned sha256 (configs/paper_pins.json)"),
}
TIER2 = {
    "val-equivalence": [["scripts/fidelity/validation_equivalence.py", "run"], ["scripts/fidelity/validation_equivalence.py", "compare"]],
    "figure-cache": [["scripts/figures/build_cache.py"]],
}
TEST_MANIFESTS = ["test_fsor_5way_1shot", "test_fsor_5way_5shot"]
TIER3 = ["python scripts/train/train_frn.py --seed {0,1,2}", "python scripts/train/train_dnpg.py data --dataset mini",
         "python scripts/train/train_dnpg.py run --dataset mini --seed {0,1,2}", "python scripts/train/train_rfs_cifarfs.py --seed 0"]


def run_cmd(args) -> int:
    shown = " ".join(a if " " not in a else repr(a) for a in args)
    print(f"+ python {shown}", flush=True)
    return subprocess.run([PY, *args], cwd=REPO).returncode


# a step that still checks something when the paper sources are absent: its status then says what was checked
PARTIAL = {"tables": "OK against paper_assets/; paper-source comparison skipped"}


def status(step: str, rcs: list[int]) -> str:
    """'OK', 'FAILED', 'skipped' (needs the paper sources) or the PARTIAL text, from the exit codes of the step's commands."""
    if all(rc == 0 for rc in rcs):
        return "OK"
    if all(rc in (0, SKIPPED) for rc in rcs):
        return PARTIAL.get(step, "skipped")
    return "FAILED"


def summary(res: dict[str, str]) -> str:
    failed = [s for s, r in res.items() if r == "FAILED"]
    skipped = [s for s, r in res.items() if r == "skipped"]
    if failed:
        return "tier 1: FAILED: " + ", ".join(failed)
    return "tier 1: all OK" + (f" (skipped: {', '.join(skipped)}: paper sources not extracted)" if skipped else "")


def tier1(step: str) -> str:
    cmds, what = TIER1[step]
    print(f"\n== {step}: {what}", flush=True)
    return status(step, [run_cmd(c) for c in cmds])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", help="all, list, or a step name")
    ap.add_argument("--tier", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--allow-test", action="store_true", help="tier 2 test-reevaluation: confirm that test episodes are re-evaluated")
    ap.add_argument("--host", action="append", default=[], help="tier 2 test-reevaluation: host configuration(s) (configs/hosts/)")
    a = ap.parse_args()
    if a.step == "list":
        print("tier 1:", ", ".join(TIER1))
        print("tier 2:", ", ".join(TIER2), "test-reevaluation (--allow-test --host <name>)")
        print("tier 3: training")
        return
    if a.tier == 1:
        steps = list(TIER1) if a.step == "all" else [a.step]
        t0 = time.time()
        res = {s: tier1(s) for s in steps}
        print(f"\n{', '.join(f'{s}: {r}' for s, r in res.items())}  ({time.time() - t0:.0f} s)")
        print(summary(res))
        sys.exit(1 if any(r == "FAILED" for r in res.values()) else 0)
    if a.tier == 2:
        if a.step == "test-reevaluation":
            if not a.allow_test or not a.host:
                sys.exit("test-reevaluation re-evaluates TEST episodes with the released code: a NEW evaluation, not run by the authors for this "
                         "release (the paper's numbers are the recorded single passes in results/). Pass --allow-test and at least one --host.")
            rc = 0
            for h in a.host:
                rc |= run_cmd(["scripts/evaluate.py", "--host", h, "--episodes", *TEST_MANIFESTS, "--allow-test", "--save-scores", "--build-bank"])
            sys.exit(rc)
        if a.step not in TIER2:
            sys.exit(f"no tier-2 step {a.step!r}; choose from {list(TIER2)} or test-reevaluation")
        sys.exit(max(run_cmd(c) for c in TIER2[a.step]))
    for cmd in TIER3:
        print(cmd)


if __name__ == "__main__":
    main()
