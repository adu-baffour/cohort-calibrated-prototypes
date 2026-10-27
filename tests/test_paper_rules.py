"""The rules stated in the paper, each checked directly on synthetic inputs (CPU)."""
import numpy as np
import pytest
import torch

from conftest import make_episodes

from ccp import fusion, metrics, stats, variants
from ccp.cohort import ccp_a_matrix, ccp_qw_matrix, closed_form_stats, s_norm_stats, unknown_score
from ccp.embedding import unit_flip
from ccp.scg import scg_bins, scg_gap
from ccp.scores import EpisodeContext, TargetFeatures


@pytest.fixture(scope="module")
def eps(synth, bank):
    tf = TargetFeatures(synth["tgt"], synth["tgt_f"], bank.to_device("cpu"))
    out = []
    for shot in (1, 5):
        arr = make_episodes(synth["tgt_lab"], shot, 4, seed=shot)
        for e in range(4):
            rows = np.concatenate([arr["support"][e].reshape(-1), arr["query"][e].reshape(-1), arr["unk_query"][e].reshape(-1)])
            out.append(EpisodeContext(tf, rows, shot))
    return out


def test_eq4_eq5_identity(eps):
    """Eq. 5: -max_k s_k(q) = 1/2 m_K(q) - max_k [c_k(q) - 1/2 m_K(p_k)] (the query term is constant over k)."""
    for ep in eps:
        c, mq, mp = ep.cos(True).double(), ep.m_q().double(), ep.m_p().double()
        u4 = unknown_score(ccp_a_matrix(c, mq, mp))
        u5 = 0.5 * mq - (c - 0.5 * mp[None, :]).max(1).values
        torch.testing.assert_close(u4, u5, rtol=0, atol=1e-12)
        # Eq. 4, second line: the average of the query correction and the prototype correction
        torch.testing.assert_close(ccp_a_matrix(c, mq, mp), 0.5 * ((c - mq[:, None]) + (c - mp[None, :])), rtol=0, atol=1e-12)


def test_ccp_qw_labels_equal_ccp_a_labels(eps):
    """Eq. 6 keeps beta for the prototype term; with beta = 1/2 the predicted labels are CCP-A's for every lambda (Section 3.3)."""
    for ep in eps:
        c, mq, mp = ep.cos(True), ep.m_q(), ep.m_p()
        lab = ccp_a_matrix(c, mq, mp).argmax(1)
        for lam in (0.5, 0.75, 1.0, 1.25, 1.5):
            assert torch.equal(ccp_qw_matrix(c, mq, mp, lam, 0.5).argmax(1), lab)
        out = variants.episode_scores(ep)
        for name in variants.CCP_QW:
            assert torch.equal(out[name][1], out["CCP-A"][1])
        # and the CCP-QW unknown score is lambda m_K(q) - max_k [c_k - beta m_K(p_k)]
        u = out["CCP-QW/lambda=1.0/beta=0.5"][0].double()
        torch.testing.assert_close(u, mq.double() - (c.double() - 0.5 * mp.double()[None, :]).max(1).values, rtol=0, atol=1e-6)


def test_ccp_s_closed_form_equals_brute_force(synth):
    """CCP-S statistics: v . b_bar and sqrt(v^T C v) equal the mean and the standard deviation (ddof 0) of v's cosines to all base images."""
    B = unit_flip(synth["base"], synth["base_f"]).double()
    V = unit_flip(synth["tgt"], synth["tgt_f"]).double()
    bbar, C = closed_form_stats(B)
    m, s = s_norm_stats(V, bbar.double(), C.double())
    cos = V @ B.T
    torch.testing.assert_close(m, cos.mean(1), rtol=0, atol=1e-12)
    torch.testing.assert_close(s, cos.std(1, unbiased=False) + 1e-6, rtol=0, atol=1e-6)


def test_h_fix_weight_is_the_pooled_valA_sd_ratio_ddof0():
    """Eq. 7: w = sigma_A(D) / sigma_A(u), standard deviations (ddof 0) of ALL valA query scores of the host, seed and shot, pooled."""
    rng = np.random.RandomState(1)
    D = rng.rand(1000, 150) * 3 + rng.rand(1000, 1)            # episode offsets: pooling differs from a per-episode average
    U = rng.rand(1000, 150) * 0.2
    w = fusion.fit_weight(D, U)
    assert w["w"] == D.ravel().std(ddof=0) / U.ravel().std(ddof=0)
    assert w["w"] != D.ravel().std(ddof=1) / U.ravel().std(ddof=1)
    assert w["w"] != np.mean(D.std(1) / U.std(1))
    assert w["n_scores"] == 150_000
    np.testing.assert_array_equal(fusion.h_fix(D, U, w["w"]), D + w["w"] * U)


def test_scg_deciles_edges_to_upper_bin_and_exclusions():
    x = np.arange(100, dtype=float)                             # deciles at 0, 9.9, 19.8, ...
    edges = np.quantile(x, np.linspace(0, 1, 11))
    y = np.tile([0, 1], 50)
    m = np.arange(100, dtype=float) % 7
    bins = scg_bins(x, m, y)
    assert [b["n_known"] + b["n_unknown"] for b in bins] == [10] * 10
    # a value equal to a computed decile edge goes to the upper bin: x = 0..100 has the exact edge 10.0 (other edges of np.quantile
    # can carry rounding, e.g. 30.000000000000004, and the rule applies to the computed edge)
    xe = np.arange(101, dtype=float)
    assert np.quantile(xe, np.linspace(0, 1, 11))[1] == 10.0
    ye = (xe == 10.0).astype(int)
    be = scg_bins(xe, xe % 7, ye)
    assert be[1]["n_unknown"] == 1 and be[0]["n_unknown"] == 0 and be[0]["x_hi"] == 10.0 and be[1]["x_lo"] == 10.0
    assert be[9]["n_known"] + be[9]["n_unknown"] == 11                 # the maximum 100 is in the last bin
    # the maximum goes to the last bin
    assert bins[9]["x_hi"] == 99.0 and bins[9]["n_known"] + bins[9]["n_unknown"] == 10
    # pooled within-bin SD with sample variances
    b0 = bins[0]
    k, u = m[(x < edges[1]) & (y == 0)], m[(x < edges[1]) & (y == 1)]
    sp = np.sqrt(((k.size - 1) * k.var(ddof=1) + (u.size - 1) * u.var(ddof=1)) / (k.size + u.size - 2))
    assert b0["sd_pooled"] == pytest.approx(sp, rel=1e-15) and b0["std_gap"] == pytest.approx((u.mean() - k.mean()) / sp, rel=1e-15)
    # exclusions: fewer than two known (or unknown) queries, or zero pooled SD
    ys = y.copy()
    ys[:10] = 1
    ys[0] = 0                                                   # bin 1: one known query
    mz = m.copy()
    mz[10:20] = 3.0                                             # bin 2: zero SD
    b = scg_bins(x, mz, ys)
    assert np.isnan(b[0]["std_gap"]) and np.isnan(b[1]["std_gap"]) and all(np.isfinite(bb["std_gap"]) for bb in b[2:])
    assert scg_gap(b) == pytest.approx(np.mean([bb["std_gap"] for bb in b[2:]]))   # unweighted mean over the valid bins


def test_oscr_equals_auroc_when_every_known_query_is_correct():
    rng = np.random.RandomState(2)
    for _ in range(20):
        s = rng.randn(150)
        s[rng.rand(150) < 0.2] = 0.0                            # ties, including known-unknown ties
        assert metrics.oscr(s, np.ones(75, bool)) == pytest.approx(metrics.auroc(s), abs=1e-12)
    assert metrics.oscr(rng.randn(150), np.zeros(75, bool)) == 0.0


def test_fpr95_uses_the_72nd_smallest_of_75_known_scores():
    known = np.arange(75, dtype=float)                          # 72nd smallest known score = 71
    for tau_unknown, expect in ((70.5, 100.0), (71.0, 100.0), (71.5, 0.0)):
        s = np.r_[known, np.full(75, tau_unknown)]
        assert metrics.fpr95(s) == expect
    s = np.r_[known, np.linspace(60, 80, 75)]
    assert metrics.fpr95(s) == pytest.approx(100 * np.mean(np.linspace(60, 80, 75) <= 71.0))


def test_permutation_p_value_floor():
    """With n_perm permutations the smallest attainable one-sided p is 1 / (n_perm + 1) (Section 5.5.2: 1e-5 for 100,000)."""
    x = np.arange(32, dtype=float)
    r = stats.spearman_perm(x, x, n_perm=999, seed=0)
    assert r["rho"] == pytest.approx(1.0) and r["p_one_sided"] == 1 / 1000
    g = np.repeat(np.arange(8), 4)
    r = stats.within_group_spearman_perm(x, x, g, n_perm=99, seed=0)
    assert r["stat_mean_within_rho"] == pytest.approx(1.0) and r["p_one_sided"] >= 1 / 100
    assert stats.N_PERM == 100_000 and 1 / (stats.N_PERM + 1) == pytest.approx(1e-5, rel=1e-4)


def test_holm():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005, 0.01, 0.03, 0.04 -> 4 x 0.005, 3 x 0.01, 2 x 0.03, 1 x 0.04, made monotone
    assert stats.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert stats.holm([0.5, 0.9]) == [1.0, 1.0]
    assert stats.holm([0.001] * 3) == pytest.approx([0.003] * 3)
