"""Fusion weights: formula, and fitting uses only the validation-A file it is given (never a test path)."""
import json

import numpy as np
import pytest

from ccp.fusion import fit_weight, fused_rows, h_fix


def test_fit_weight_formula():
    rng = np.random.RandomState(0)
    D, U = rng.rand(10, 150) * 3, rng.rand(10, 150)
    w = fit_weight(D, U)
    assert w["w"] == pytest.approx(D.std() / U.std()) and w["n_scores"] == 1500
    rows = fused_rows(D, U, U * 2, w["w"], 1.0)
    np.testing.assert_allclose(rows["H_fix"], h_fix(D, U, w["w"]))
    assert set(rows) == {"D", "H_fix", "K_fix", "episode-wise standardization", "episode-wise standardization, COSF control", "CCP-A", "COSF"}


def _write(tmp_path, manifest, dataset="mini"):
    rng = np.random.RandomState(1)
    np.savez(tmp_path / f"{manifest}.npz", **{"score_D": rng.rand(4, 150), "score_CCP-A": rng.rand(4, 150), "score_COSF": rng.rand(4, 150)})
    (tmp_path / f"{manifest}.json").write_text(json.dumps({"manifest": manifest, "host_info": {"dataset": dataset}}), encoding="utf-8")
    return tmp_path / f"{manifest}.npz"


def test_fit_uses_only_valA(tmp_path):
    from scripts.fit_fusion_weight import fit_from_file
    r = fit_from_file(_write(tmp_path, "valA_fsor_5way_1shot"))
    z = np.load(tmp_path / "valA_fsor_5way_1shot.npz")
    assert r["w"] == pytest.approx(z["score_D"].std() / z["score_CCP-A"].std())
    assert r["manifest"] == "valA_fsor_5way_1shot" and len(r["manifest_sha256"]) == 64


@pytest.mark.parametrize("name", ["valB_fsor_5way_1shot", "test_fsor_5way_1shot"])
def test_fit_refuses_other_sets(tmp_path, name):
    from scripts.fit_fusion_weight import fit_from_file
    with pytest.raises((ValueError, PermissionError)):
        fit_from_file(_write(tmp_path, name))


def test_frozen_weights_have_provenance():
    from ccp.hosts import load_host_config
    for h in ("dnpg_mini_seed0", "dnpg_mini_seed1", "dnpg_mini_seed2", "snatcher_f_mini", "snatcher_f_tiered", "dnpg_tiered_seed0"):
        f = load_host_config(h)["fusion"]
        assert f["provenance"] and all(f["frozen_weights"][f"{s}shot"]["w"] > 0 for s in (1, 5))
    assert load_host_config("dnpg_mini_seed0")["fusion"]["frozen_weights"]["1shot"]["w"] == 2.626274957560938  # FINAL_BUNDLE yaml-fsor-mini-dnpg
    assert load_host_config("snatcher_f_mini")["fusion"]["frozen_weights"]["5shot"]["w2"] == 44.88093755962801  # yaml-fsor-mini-snatcher
