"""Method variants V1-V4 (docs/prereg_variants.md) on synthetic data (CPU): exact reduction to CCP-A at each null setting, definitions against
direct computations, the selection / gate / promotion rules and the SCG (D1) binning."""
import numpy as np
import pytest
import torch

from ccp import fusion
from ccp import variants as V
from ccp.cohort import ccp_a_matrix, ccp_qw_matrix, cohort_overlap, full_as_norm_matrix, m_topk, shared_cohort_matrix, topk_stats
from ccp.scores import EpisodeContext, TargetFeatures, get_score
from conftest import make_episodes


@pytest.fixture(scope="module", params=[1, 5])
def ep(request, synth, bank):
    tf = TargetFeatures(synth["tgt"], synth["tgt_f"], bank.to_device("cpu"))
    eps = make_episodes(synth["tgt_lab"], request.param, 2, seed=21)
    rows = np.concatenate([eps["support"][0].reshape(-1), eps["query"][0].reshape(-1), eps["unk_query"][0].reshape(-1)])
    return EpisodeContext(tf, rows, request.param)


def test_topk_stats_mean_is_registered_m_and_std_indices_are_correct(ep):
    B = ep.tf.bank.B
    mu, sd, idx = topk_stats(ep.tf.uf, B, 1000)
    assert torch.equal(mu, m_topk(ep.tf.uf, B, 1000)) and torch.equal(mu, ep.tf.m_query(1000))
    sims = ep.tf.uf @ B.T
    top = sims.sort(1, descending=True).values[:, :1000]
    torch.testing.assert_close(sd, top.std(1, unbiased=False), rtol=0, atol=1e-6)
    torch.testing.assert_close(torch.gather(sims, 1, idx.long()).sort(1).values, top.sort(1).values, rtol=0, atol=0)
    mq, _, _ = ep.topk_q()
    mp, _, _ = ep.topk_p()
    assert torch.equal(mq, ep.m_q()) and torch.equal(mp, ep.m_p())


def test_v1_null_is_ccp_a_bitwise_and_definition(ep):
    c, mq, mp = ep.cos(True), ep.m_q(), ep.m_p()
    assert torch.equal(ccp_qw_matrix(c, mq, mp, 0.5, 0.5), ccp_a_matrix(c, mq, mp))
    for lam, beta in V.QW_GRID:
        u = -ccp_qw_matrix(c, mq, mp, lam, beta).max(1).values
        torch.testing.assert_close(u, -(c - beta * mp[None, :]).max(1).values + lam * mq, rtol=0, atol=1e-6)
    lam1 = -ccp_qw_matrix(c, mq, mp, 1.0, 0.0).max(1).values          # the query-side ablation (weight 1 on m(q), no prototype term)
    torch.testing.assert_close(lam1, -get_score("CCP-A/query only")(ep).max(1).values, rtol=0, atol=1e-6)


def test_episode_scores_labels_and_null_checks(ep):
    out = V.episode_scores(ep)
    assert set(out) == {"CCP-A", *V.SCORED}
    A = get_score("CCP-A")(ep)
    assert torch.equal(out["CCP-A"][0], -A.max(1).values) and torch.equal(out["CCP-A"][1], A.argmax(1))
    for v in V.CCP_QW:
        assert torch.equal(out[v][1], A.argmax(1)), v               # V1 keeps the CCP-A label
    assert torch.equal(out[V.CCP_QW_NULL][0], out["CCP-A"][0])
    for v in V.FULL_AS_NORM + V.SHARED_COHORT:
        M = get_score(v)(ep)
        assert torch.equal(out[v][0], -M.max(1).values) and torch.equal(out[v][1], M.argmax(1)), v


def test_v2_ask_definition_and_unit_std_reduction(ep):
    c = ep.cos(True)
    mq, sq, _ = ep.topk_q()
    mp, sp, _ = ep.topk_p()
    assert torch.equal(full_as_norm_matrix(c, mq, torch.ones_like(sq), mp, torch.ones_like(sp)), 0.5 * ((c - mq[:, None]) + (c - mp[None, :])))
    torch.testing.assert_close(full_as_norm_matrix(c, mq, torch.ones_like(sq), mp, torch.ones_like(sp)), ccp_a_matrix(c, mq, mp), rtol=0, atol=1e-6)
    direct = 0.5 * ((c - mq[:, None]) / sq[:, None] + (c - mp[None, :]) / sp[None, :])
    torch.testing.assert_close(get_score("full AS-norm")(ep), direct, rtol=0, atol=1e-5)
    tiny = full_as_norm_matrix(c, mq, torch.zeros_like(sq), mp, torch.zeros_like(sp))
    torch.testing.assert_close(tiny, 0.5 * ((c - mq[:, None]) / 1e-6 + (c - mp[None, :]) / 1e-6))


def test_v3_overlap_definition_and_null(ep):
    c = ep.cos(True)
    mq, _, iq = ep.topk_q()
    mp, _, ip = ep.topk_p()
    o = cohort_overlap(iq, ip, ep.tf.bank.B.shape[0])
    for q in (0, 37, 149):
        for k in range(5):
            n = len(set(iq[q].tolist()) & set(ip[k].tolist()))
            assert round(o[q, k].item() * 1000) == n and o[q, k].item() == np.float32(n) / np.float32(1000)
    assert torch.equal(shared_cohort_matrix(c, mq, mp, o, 0.0), ccp_a_matrix(c, mq, mp))
    for g in (0.25, 0.5, 1.0):
        torch.testing.assert_close(get_score(f"shared-cohort correction/gamma={g}")(ep), c - 0.5 * ((1 - g * o) * mq[:, None] + mp[None, :]), rtol=0, atol=1e-6)


def test_v4_rows_mu1_is_h_fix_and_alone():
    rng = np.random.RandomState(0)
    D, U = rng.randn(4, 150), rng.randn(4, 150)
    raw = {"own": D, "own_acc": np.full(4, 70.0), "CCP-A": (U, np.full(4, 60.0))}
    r = V.rows(raw, True, V.FUSION_WEIGHT_FACTOR, {}, 2.37)
    np.testing.assert_array_equal(r["fusion-weight factor/factor=1"][0], fusion.h_fix(D, U, 2.37))
    np.testing.assert_array_equal(r["reference"][0], r["fusion-weight factor/factor=1"][0])
    np.testing.assert_array_equal(r["fusion-weight factor/CCP-A alone"][0], U)
    np.testing.assert_array_equal(r["fusion-weight factor/factor=4"][0], D + 4 * 2.37 * U)
    assert all((r[v][1] == 70.0).all() for v in V.FUSION_WEIGHT_FACTOR)               # trained hosts keep their classifier


def test_frozen_rows_v1_uses_ccp_a_accuracy():
    raw = {"CCP-A": (np.zeros((2, 150)), np.array([60.0, 61.0]))}
    for v in V.SCORED:
        raw[v] = (np.ones((2, 150)), np.array([50.0, 50.0]))
    r = V.rows(raw, False, [V.CCP_QW[3], "full AS-norm", V.SHARED_COHORT[0], V.FUSION_WEIGHT_FACTOR[1]], {}, None)
    assert V.FUSION_WEIGHT_FACTOR[1] not in r
    np.testing.assert_array_equal(r[V.CCP_QW[3]][1], [60.0, 61.0])
    np.testing.assert_array_equal(r["full AS-norm"][1], [50.0, 50.0])


def test_grids_and_orders():
    assert len(V.CCP_QW) == 10 and V.qw_params(V.CCP_QW_NULL) == (0.5, 0.5) and V.CCP_QW[1] == "CCP-QW/lambda=0.5/beta=0.0"
    assert V.SHARED_COHORT == ["shared-cohort correction/gamma=0.25", "shared-cohort correction/gamma=0.5", "shared-cohort correction/gamma=1.0"] and V.FUSION_WEIGHT_FACTOR_NULL == "fusion-weight factor/factor=1"
    assert len(V.CELLS) == 8 and len(V.FACTOR_CELLS) == 4


def test_selection_ties_go_to_registered_order():
    assert V.choose({"a": 1.0, "b": 1.0 + 5e-10, "c": 0.9}, ["a", "b", "c"]) == "a"
    assert V.choose({"a": 1.0, "b": 1.0 + 2e-9}, ["a", "b"]) == "b"
    cell = {v: {c: 70.0 for c in V.CELLS} for v in V.CCP_QW}
    assert V.select_global(cell, V.CCP_QW)[0] == V.CCP_QW_NULL              # a tie with CCP-A selects CCP-A (V1 NO-GO)
    cell[V.CCP_QW[4]][V.CELLS[0]] = 70.8
    assert V.select_global(cell, V.CCP_QW)[0] == V.CCP_QW[4]
    c4 = {v: {c: 80.0 for c in V.FACTOR_CELLS} for v in V.FUSION_WEIGHT_FACTOR}
    c4["fusion-weight factor/factor=2"][("DNPG mini", 1)] = 80.5
    assert V.select_factor(c4) == {"SnaTCHer-F mini": "fusion-weight factor/factor=1", "DNPG mini": "fusion-weight factor/factor=2"}


def test_gates_thresholds_inclusive_and_promotion_without_fallback():
    cells = V.CELLS
    ok = V.criteria_valB({c: 0.30 for c in cells}, {c: -0.20 for c in cells}, cells)
    assert ok["(i)"] and ok["(ii)"] and ok["(iii)"] and ok["pass_i_iii"]
    low = V.criteria_valB({c: 0.29 for c in cells}, {c: 0.0 for c in cells}, cells)
    assert not low["(i)"]
    one_bad = {c: 0.6 for c in cells}
    one_bad[cells[3]] = -0.21
    assert not V.criteria_valB(one_bad, {c: 0.0 for c in cells}, cells)["(ii)"]
    acc_bad = {c: 0.0 for c in cells}
    acc_bad[("frozen FEAT tiered", 5)] = -0.25
    assert not V.criteria_valB({c: 0.5 for c in cells}, acc_bad, cells)["(iii)"]
    v4 = V.criteria_valB({c: 0.4 for c in V.FACTOR_CELLS}, {c: -9.0 for c in V.FACTOR_CELLS}, V.FACTOR_CELLS)
    assert v4["(iii)"] and v4["min_frozen_acc_change"] is None          # accuracy gate does not apply to trained hosts
    gates = {"CCP-QW": {"pass_i_iii": True, "mean_gain": 0.9}, "full AS-norm": {"pass_i_iii": True, "mean_gain": 0.5},
             "shared-cohort correction": {"pass_i_iii": False, "mean_gain": 1.2}, "fusion-weight factor": {"pass_i_iii": True, "mean_gain": 0.4}}
    assert V.candidate(gates, {"CCP-QW": True}) == "CCP-QW"
    assert V.candidate(gates, {"CCP-QW": False}) == "full AS-norm"
    g4 = {"CCP-QW": {"(iv)": False}, "full AS-norm": {"(iv)": True}}
    assert V.promotion("CCP-QW", g4) is None                          # no fallback to the next family
    assert V.promotion("full AS-norm", g4) == "full AS-norm" and V.promotion(None, g4) is None
    assert V.criterion_valC({c: 0.15 for c in cells}, cells)["(iv)"] and not V.criterion_valC({c: 0.14 for c in cells}, cells)["(iv)"]
    c2 = {c: 0.5 for c in cells}
    c2[cells[0]] = -0.3
    assert not V.criterion_valC(c2, cells)["(iv)"]


def test_steps_refuse_uncommitted_code_and_record_head(tmp_path):
    import subprocess

    from scripts.rounds import variants as SV
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for d in ("ccp", "scripts", "configs", "docs"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "f.txt").write_text("a")
    subprocess.run(g + ["add", "."], cwd=tmp_path, check=True)
    subprocess.run(g + ["commit", "-q", "-m", "x"], cwd=tmp_path, check=True)
    SV.require_clean_code(tmp_path)                                 # clean: passes
    assert len(SV.git_head(tmp_path)) == 40
    (tmp_path / "docs" / "f.txt").write_text("b")
    (tmp_path / "untracked.py").write_text("")
    SV.require_clean_code(tmp_path)                                 # docs/ and untracked files do not block
    for d in ("ccp", "scripts", "configs"):
        (tmp_path / d / "f.txt").write_text("b")
        with pytest.raises(SystemExit, match="uncommitted changes"):
            SV.require_clean_code(tmp_path)
        subprocess.run(g + ["add", d], cwd=tmp_path, check=True)    # staged but not committed still refuses
        with pytest.raises(SystemExit):
            SV.require_clean_code(tmp_path)
        subprocess.run(g + ["checkout", "-q", "HEAD", "--", d], cwd=tmp_path, check=True)
    SV.require_clean_code(tmp_path)
