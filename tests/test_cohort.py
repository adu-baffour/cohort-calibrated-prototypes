"""CCP-A / CCP-S properties on synthetic data (CPU)."""
import numpy as np
import torch

from ccp.cohort import ccp_a_matrix, closed_form_stats, m_all, m_topk, s_norm_stats, unit_flip
from ccp.scores import EpisodeContext, TargetFeatures, score_episode
from conftest import make_episodes


def _ctx(synth, bank, shot=1, e=0):
    tf = TargetFeatures(synth["tgt"], synth["tgt_f"], bank.to_device("cpu"))
    eps = make_episodes(synth["tgt_lab"], shot, 3, seed=1)
    rows = np.concatenate([eps["support"][e].reshape(-1), eps["query"][e].reshape(-1), eps["unk_query"][e].reshape(-1)])
    return tf, EpisodeContext(tf, rows, shot)


def test_ccp_a_equals_two_sided_average(synth, bank):
    _, ep = _ctx(synth, bank)
    c, mq, mp = ep.cos(True), ep.m_q(50), ep.m_p(50)
    torch.testing.assert_close(ccp_a_matrix(c, mq, mp), 0.5 * ((c - mq[:, None]) + (c - mp[None, :])), rtol=0, atol=1e-6)


def test_label_invariant_to_query_term(synth, bank):
    """The query term m_K(q) is constant over classes, so argmax_k s_k(q) = argmax_k [c_k - 1/2 m_K(p_k)] for any query statistic."""
    _, ep = _ctx(synth, bank, shot=5)
    c, mp = ep.cos(True), ep.m_p(50)
    a = ccp_a_matrix(c, ep.m_q(50), mp).argmax(1)
    b = ccp_a_matrix(c, torch.rand(c.shape[0]) * 10, mp).argmax(1)
    assert torch.equal(a, b)
    assert torch.equal(ep.cos(True).argmax(1), (ep.cos(True) - ep.m_q(50)[:, None]).argmax(1))  # query-side only keeps the COSF label


def test_ccp_s_closed_form_matches_brute_force(synth, bank):
    B = unit_flip(synth["base"], synth["base_f"])
    bbar, C = closed_form_stats(B)
    V = unit_flip(synth["tgt"], synth["tgt_f"])
    m_cf, s_cf = s_norm_stats(V, bbar, C)
    cos = V.double() @ B.double().T
    m_bf, s_bf = cos.mean(1), cos.std(1, unbiased=False) + 1e-6
    assert float((m_cf.double() - m_bf).abs().max()) < 1e-5
    assert float((s_cf.double() - s_bf).abs().max()) < 1e-5


def test_k_all_is_projection_on_bank_mean(synth, bank):
    B = unit_flip(synth["base"], synth["base_f"])
    V = unit_flip(synth["tgt"], synth["tgt_f"])
    torch.testing.assert_close(m_all(V, B), (V @ B.T).mean(1), rtol=0, atol=1e-6)
    torch.testing.assert_close(m_topk(V, B, B.shape[0]), (V @ B.T).mean(1), rtol=0, atol=1e-5)  # top-K with K = |bank|


def test_registered_scores_run_and_prototype_definition(synth, bank):
    tf, ep = _ctx(synth, bank, shot=5)
    out = score_episode(ep, ["COS", "COSF", "CCP-A", "CCP-S", "CCP-A/K=all", "CCP-A/query only", "CCP-A/prototype only",
                             "CCP-A/class means, matched fraction", "CCP-A/class means, all", "CCP-A/single view"])
    for s, acc, M in out.values():
        assert s.shape == (150,) and M.shape == (150, 5) and 0 <= acc <= 100
    # prototype = normalised mean of the UNNORMALISED flip-summed support vectors (Eq. 2)
    sup = torch.as_tensor(ep.r_sup)
    raw = (synth["tgt"][sup] + synth["tgt_f"][sup]).view(5, 5, -1).mean(1)
    torch.testing.assert_close(ep.pf, torch.nn.functional.normalize(raw, dim=1), rtol=0, atol=1e-6)


def test_bank_roundtrip_and_sha(tmp_path, bank):
    from ccp.cohort import Bank
    h1 = bank.save(tmp_path / "b.npz")
    h2 = bank.save(tmp_path / "c.npz")
    assert h1 == h2, "bank files must be byte-reproducible"
    b = Bank.load(tmp_path / "b.npz", h1)
    assert np.array_equal(b.raw_noflip, bank.raw_noflip) and b.meta == bank.meta
    import pytest
    with pytest.raises(ValueError):
        Bank.load(tmp_path / "b.npz", "0" * 64)
