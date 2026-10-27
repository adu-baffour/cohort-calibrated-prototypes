"""GPU + data: the release evaluation path reproduces the recorded bundle dry run exactly (frozen FEAT mini, valA, both shots)."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
@pytest.mark.data
def test_frozen_feat_mini_valA_matches_dry_run(tmp_path):
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.evaluate import evaluate
    rec = json.loads((REPO / "results" / "recorded" / "validation" / "fsor_final_dryrun_val_mini.json").read_text(encoding="utf-8"))["rows"]["frozen_FEAT"]
    out = evaluate("frozen_feat_mini", ["valA_fsor_5way_1shot", "valA_fsor_5way_5shot"], out_dir=tmp_path, build_bank_if_missing=True)
    for sh in (1, 5):
        rows = out[f"valA_fsor_5way_{sh}shot"]["rows"]
        for k in ("COS", "COSF", "CCP-A", "CCP-S"):
            assert abs(rows[k]["auroc"] - rec[f"{sh}shot"][k]["auroc"]) <= 1e-4
            assert rows[k]["acc"] == rec[f"{sh}shot"][k]["acc"]
