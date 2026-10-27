"""The nine training-free baselines (Section 4.4) on synthetic data (CPU): shapes, definitions against direct computations, inductiveness, selection ties."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ccp import baselines as BL
from ccp.scores import EpisodeContext, TargetFeatures, get_evidence, get_score
from conftest import make_episodes


@pytest.fixture(scope="module")
def ctx(synth, bank):
    tf = TargetFeatures(synth["tgt"], synth["tgt_f"], bank.to_device("cpu"))
    eps = make_episodes(synth["tgt_lab"], 5, 2, seed=11)
    sup = eps["support"][0].reshape(-1)
    q = np.concatenate([eps["query"][0].reshape(-1), eps["unk_query"][0].reshape(-1)])
    return tf, sup, q


def ep_of(tf, sup, q):
    return EpisodeContext(tf, np.concatenate([sup, q]), 5)


def test_every_variant_runs(ctx):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    for v in BL.ALL_SETTINGS:
        out = get_evidence(v)(ep) if v in BL.EVIDENCE else get_score(v)(ep)
        assert out.shape == ((150,) if v in BL.EVIDENCE else (150, 5)) and torch.isfinite(out).all(), v


def test_b1_is_query_cohort_term(ctx):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    torch.testing.assert_close(get_evidence("query cohort term")(ep), ep.m_q(1000), rtol=0, atol=0)


def test_b2_kth_neighbour(ctx):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    sims = (tf.uf[ep.r_q] @ tf.bank.B.T).sort(1, descending=True).values
    for k in (10, 100, 1000):
        torch.testing.assert_close(get_evidence(f"kNN/k={k}/sign=+")(ep), -sims[:, k - 1], rtol=0, atol=1e-6)
        torch.testing.assert_close(get_evidence(f"kNN/k={k}/sign=-")(ep), sims[:, k - 1], rtol=0, atol=1e-6)


def test_b3_rmd_definition_and_euclidean_limit(ctx, synth):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    C, bbar = tf.bank.C.double(), tf.bank.bbar.double()
    for lam in (0.1, 0.5, 1.0):
        S = (1 - lam) * C + lam * torch.trace(C) / C.shape[0] * torch.eye(C.shape[0], dtype=C.dtype)
        P = torch.linalg.inv(S)
        z, p = tf.uf[ep.r_q].double(), ep.pf.double()
        md = torch.stack([((z - p[k]) @ P * (z - p[k])).sum(1) for k in range(5)], 1)
        md0 = ((z - bbar) @ P * (z - bbar)).sum(1)
        torch.testing.assert_close(get_score(f"RMD/delta={lam}")(ep).double(), -(md - md0[:, None]), rtol=1e-4, atol=1e-3)
    # lambda = 1: isotropic covariance, so the label is the nearest prototype in Euclidean distance
    lab = get_score("RMD/delta=1.0")(ep).argmax(1)
    assert torch.equal(lab, torch.cdist(tf.uf[ep.r_q], ep.pf).argmin(1))


def test_b4_cl2n(ctx):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    f = lambda v: F.normalize(v - tf.bank.bbar, dim=1)  # noqa: E731
    p = F.normalize(f(tf.uf[ep.r_sup]).view(5, 5, -1).mean(1), dim=1)
    torch.testing.assert_close(get_score("CL2N")(ep), f(tf.uf[ep.r_q]) @ p.T, rtol=0, atol=1e-6)


def test_b5_zn_is_affine_invariant_per_vector(synth, bank):
    """ZN standardises each raw vector: rescaling and shifting a raw query vector leaves its scores unchanged."""
    raw = synth["tgt"].clone(); raw_f = synth["tgt_f"].clone()
    eps = make_episodes(synth["tgt_lab"], 1, 1, seed=2)
    rows = np.concatenate([eps["support"][0].reshape(-1), eps["query"][0].reshape(-1), eps["unk_query"][0].reshape(-1)])
    tf1 = TargetFeatures(raw, raw_f, bank.to_device("cpu"))
    s1 = get_score("ZN")(EpisodeContext(tf1, rows, 1))
    qi = int(rows[5])
    raw2, raw2_f = raw.clone(), raw_f.clone()
    raw2[qi], raw2_f[qi] = 3.0 * raw[qi] + 0.7, 3.0 * raw_f[qi] + 0.7   # g + g_flip -> 3 (g + g_flip) + 1.4: same z-score
    s2 = get_score("ZN")(EpisodeContext(TargetFeatures(raw2, raw2_f, bank.to_device("cpu")), rows, 1))
    torch.testing.assert_close(s1[0], s2[0], rtol=0, atol=1e-5)


def test_b6_nnn(ctx):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    for a in (0.25, 1.0):
        for k in (100, 1000):
            ref = ep.cos(True) - a * (ep.pf @ tf.bank.B.T).topk(k, dim=1).values.mean(1)[None, :]
            torch.testing.assert_close(get_score(f"NNN/alpha={a}/k={k}")(ep), ref, rtol=0, atol=1e-6)


def test_b7_qb_is_and_dis(ctx):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    B = tf.bank.B
    for beta in (10, 20, 50):
        cb = B @ ep.pf.T
        Z = torch.log(torch.exp(beta * cb.double()).sum(0))
        ref = beta * ep.cos(True).double() - Z[None] + math.log(B.shape[0])
        torch.testing.assert_close(get_score(f"QB-Norm IS/temperature={beta}")(ep).double(), ref, rtol=1e-5, atol=1e-3)
        hubs = set(cb.argmax(1).tolist())
        dis = get_score(f"QB-Norm DIS/temperature={beta}")(ep)
        for i, k in enumerate(ep.cos(True).argmax(1).tolist()):
            want = get_score(f"QB-Norm IS/temperature={beta}")(ep)[i] if k in hubs else beta * ep.cos(True)[i]
            torch.testing.assert_close(dis[i], want, rtol=0, atol=1e-5)


def test_b8_teen(ctx):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    mu = tf.bank.class_means
    w = torch.softmax(16 * ep.pf @ mu.T, 1)
    p = F.normalize(0.5 * ep.pf + 0.5 * w @ mu, dim=1)
    torch.testing.assert_close(get_score("TEEN")(ep), tf.uf[ep.r_q] @ p.T, rtol=0, atol=1e-6)


def test_episode_csls_definition_and_transductive(ctx, synth):
    tf, sup, q = ctx
    ep = ep_of(tf, sup, q)
    Q, P = tf.uf[ep.r_q], ep.pf
    s = get_score("episode CSLS")(ep)
    i, k = 3, 2
    others = torch.cat([torch.cat([Q[:i], Q[i + 1:]]) @ Q[i], P @ Q[i]])
    rq = others.topk(10).values.mean()
    rp = torch.cat([Q @ P[k], torch.cat([P[:k], P[k + 1:]]) @ P[k]]).topk(10).values.mean()
    assert abs(float(s[i, k] - (2 * Q[i] @ P[k] - rq - rp))) < 1e-5
    # transductive: replacing the other queries changes query 0's score
    q2 = q.copy()
    others_idx = np.setdiff1d(np.arange(len(synth["tgt_lab"])), np.concatenate([sup, q]))
    q2[1:] = np.random.RandomState(0).choice(others_idx, len(q) - 1)
    s2 = get_score("episode CSLS")(ep_of(tf, sup, q2))
    assert not torch.allclose(s[0], s2[0])


@pytest.mark.parametrize("v", [v for v in BL.ALL_SETTINGS if v not in BL.TRANSDUCTIVE])
def test_inductive_baselines(ctx, v):
    """Every baseline except episode CSLS is inductive: a query's score is the same inside the episode and scored alone."""
    tf, sup, q = ctx
    fn = get_evidence(v) if v in BL.EVIDENCE else get_score(v)
    red = (lambda o: o) if v in BL.EVIDENCE else (lambda o: -o.max(1).values)
    full = red(fn(ep_of(tf, sup, q)))
    for i in (0, 77, 149):
        solo = red(fn(ep_of(tf, sup, q[i:i + 1])))
        assert abs(float(solo[0]) - float(full[i])) <= 1e-4 * max(1.0, abs(float(full[i])))


def test_selection_ties_go_to_the_smaller_value():
    vals = {v: 0.0 for v in BL.ALL_SETTINGS}
    vals["NNN/alpha=0.5/k=100"] = 1.0
    vals["NNN/alpha=0.75/k=1000"] = 1.0 + 1e-12   # a tie within 1e-9: the smaller alpha wins
    vals["kNN/k=100/sign=-"] = 2.0
    c = BL.choose(vals)
    assert c["NNN"] == "NNN/alpha=0.5/k=100" and c["kNN"] == "kNN/k=100/sign=-" and c["RMD"] == "RMD/delta=0.1"


def test_fusion_rows_use_the_host_label():
    rng = np.random.RandomState(0)
    raw = {"own": rng.rand(3, 150), "own_acc": np.array([60.0, 70.0, 80.0]), "CCP-A": (rng.rand(3, 150), np.zeros(3)),
           "CL2N": (rng.rand(3, 150), np.array([1.0, 2.0, 3.0]))}
    w = BL.fit_weights("dnpg", raw, ["CL2N"])
    r = BL.rows("dnpg", raw, ["CL2N"], w, 2.0)
    assert np.array_equal(r["CL2N"][1], raw["own_acc"]) and w["CL2N"] == pytest.approx(raw["own"].std() / raw["CL2N"][0].std())
