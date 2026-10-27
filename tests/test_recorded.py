"""Recorded outputs: provenance hashes, and (with the release assets) the tier-1 regeneration equals the recorded markdown."""
import hashlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_provenance_hashes():
    rows = (REPO / "results" / "recorded" / "PROVENANCE.md").read_text(encoding="utf-8").splitlines()
    n = 0
    for r in rows:
        m = re.match(r"^\| ([^|(][^|]*?) \| .* \| `([0-9a-f]{64})` \|", r)
        if m and not m.group(1).startswith("(source)"):
            p = REPO / "results" / "recorded" / m.group(1)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == m.group(2), m.group(1)
            n += 1
    assert n >= 50


def test_registration_tokens_of_all_eight_rounds():
    """Each round's token recomputed from its registration document equals its DONE marker, approval file and test record."""
    import sys
    sys.path.insert(0, str(REPO))
    from ccp import registration
    res = registration.check()
    assert [r["round"] for r in res] == ["(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)", "(vii)", "(viii)"]
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
    assert [r["started"][:10] for r in res] == sorted(r["started"][:10] for r in res)          # the rounds ran in this order


def test_published_opp_hde_tiered_verified():
    import csv
    rows = list(csv.DictReader(open(REPO / "results" / "recorded" / "published" / "published_sota.csv", encoding="utf-8")))
    r = [x for x in rows if x["method"] == "OPP+HDE" and x["benchmark"] == "tieredImageNet"]
    assert len(r) == 1 and r[0]["verification_status"].startswith("VERIFIED") and "row 788" in r[0]["notes"] and "FAIA251134" in r[0]["citation"]
    d = [x for x in rows if x["method"] == "DNPG" and x["benchmark"] == "mini-ImageNet"][0]
    assert (d["acc_1shot"], d["auroc_1shot"], d["acc_5shot"], d["auroc_5shot"]) == ("69.10±0.29", "74.18±0.28", "83.74±0.19", "83.64±0.19")


@pytest.mark.data
@pytest.mark.parametrize("round_name", ["fsor", "baselines", "variants", "final_round", "foundation_round"])
def test_recorded_reports_rerender_exactly(round_name):
    """Every recorded report (RESULTS_fsor.md and the paired tables of rounds (i)-(iv), test.md / png / test_hypotheses.json of rounds
    (v)-(viii), the validation reports) is re-rendered from the recorded files and the per-episode release assets only and equals the
    recorded bytes (no test data is loaded)."""
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.rounds import render_recorded
    assert render_recorded.check(round_name) == []
