"""The round drivers (CPU parts): the registered bundle blocks, read and translated to the paper's names, agree with the recorded selection
records they were drafted from; every driver and auxiliary script loads (--help)."""
import subprocess
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT))
from scripts.rounds.common import bundle_block, record  # noqa: E402


def test_bundle_tokens_are_the_approved_ones():
    from ccp import naming
    tokens = {r[1]: r[2] for r in naming.ROUNDS}
    for rnd, name in (("baselines", "BASELINES"), ("variants", "VARIANTS"), ("final_round", "FINAL_ROUND"), ("foundation_round", "FOUNDATION_ROUND")):
        assert bundle_block(rnd)[1] == tokens[name]


def test_baselines_bundle_matches_the_selection_record():
    b, _ = bundle_block("baselines")
    sel = record("results/baselines/selection_valA.json")
    assert b["chosen"] == sel["chosen"]
    for h, x in b["hosts"].items():
        for sh in (1, 5):
            assert x["weights"][f"{sh}shot"] == sel["frozen"][h][str(sh)]["weights"]


def test_variants_bundle_matches_the_selection_record():
    b, _ = bundle_block("variants")
    sel = record("results/variants/selection_valA.json")
    assert b["chosen"] == sel["chosen"] and b["chosen"]["CCP-QW"] == "CCP-QW/lambda=1.0/beta=0.5"
    assert b["chosen"]["shared-cohort correction"] == "shared-cohort correction/gamma=0.25"
    for h, x in b["hosts"].items():
        if x["variant_weights"]:
            assert {k[0]: v for k, v in x["variant_weights"].items()} == sel["weights"][h]


def test_final_round_bundle_matches_the_cifarfs_selection():
    b, _ = bundle_block("final_round")
    sel = record("results/final_round/cifarfs_selection_valA.json")
    assert b["cifarfs_chosen"] == sel["chosen"] and b["cifarfs_weights"] == sel["weights"]


def test_foundation_bundle_matches_the_selection_record():
    b, _ = bundle_block("foundation_round")
    sel = record("results/foundation_round/selection_valA.json")
    n = 0
    for h, x in b["hosts"].items():
        if "chosen" in x:
            assert x["chosen"] == sel["chosen"][h] and x["evidence_weights"] == sel["weights"][h]
            n += 1
    assert n == 16
    assert {k: v for k, v in b["scg_valA_predictions"].items()} == {k: v["std_gap"] for k, v in sel["scg_valA"].items()}


@pytest.mark.parametrize("script", ["scripts/rounds/baselines.py", "scripts/rounds/variants.py", "scripts/rounds/final_round.py",
                                    "scripts/rounds/foundation_round.py", "scripts/evaluate.py", "scripts/build_bank.py",
                                    "scripts/fidelity/validation_equivalence.py"])
def test_scripts_load(script):
    args = ["--help"] if "validation_equivalence" not in script else []
    if not args:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ve", ROOT / script)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        return
    r = subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
