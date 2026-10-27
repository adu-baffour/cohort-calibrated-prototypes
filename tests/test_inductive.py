"""Inductive guarantee: a query's score depends only on that query, the support set, the bank and frozen constants."""
import numpy as np
import pytest

from ccp.scores import EpisodeContext, TargetFeatures, get_score
from conftest import make_episodes

SCORES = ["COS", "COSF", "CCP-A", "CCP-S", "CCP-A/K=all", "CCP-A/prototype only", "CCP-A/query only", "CCP-A/class means, matched fraction", "CCP-A/single view",
          "full AS-norm", "shared-cohort correction/gamma=1.0"]


@pytest.fixture(scope="module")
def setup(synth, bank):
    tf = TargetFeatures(synth["tgt"], synth["tgt_f"], bank.to_device("cpu"))
    eps = make_episodes(synth["tgt_lab"], 1, 2, seed=3)
    sup = eps["support"][0].reshape(-1)
    q = np.concatenate([eps["query"][0].reshape(-1), eps["unk_query"][0].reshape(-1)])
    return tf, sup, q


def scores(tf, sup, q):
    """{score: open-set score of every query in q}, for an episode with support `sup` and ANY number of queries."""
    ep = EpisodeContext(tf, np.concatenate([sup, q]), 1)
    return {k: (-get_score(k)(ep).max(1).values).numpy() for k in SCORES}


def test_permuting_other_queries(setup):
    tf, sup, q = setup
    base = scores(tf, sup, q)
    perm = np.random.RandomState(0).permutation(len(q))
    moved = scores(tf, sup, q[perm])
    for k in SCORES:
        np.testing.assert_array_equal(moved[k], base[k][perm])


def test_replacing_other_queries(setup, synth):
    tf, sup, q = setup
    base = scores(tf, sup, q)
    q2 = q.copy()
    others = np.setdiff1d(np.arange(len(synth["tgt_lab"])), np.concatenate([sup, q]))
    q2[1:] = np.random.RandomState(1).choice(others, len(q) - 1, replace=True)  # keep query 0, replace every other query
    new = scores(tf, sup, q2)
    for k in SCORES:
        assert new[k][0] == base[k][0]


@pytest.mark.parametrize("i", [0, 7, 80, 149])
def test_removing_other_queries(setup, i):
    """A query scored alone (every other query removed) gets its score inside the full episode. Equal up to float32 rounding only: a
    (1 x d) and a (150 x d) matrix product may use different BLAS kernels (1 ulp); permuting or replacing queries is bit-exact above."""
    tf, sup, q = setup
    base = scores(tf, sup, q)
    solo = scores(tf, sup, q[i:i + 1])
    for k in SCORES:
        assert abs(float(solo[k][0]) - float(base[k][i])) <= 1e-6


def test_zsum_fusion_is_not_inductive():
    """The labelled H / K rows use query-batch statistics: replacing other queries changes a query's fused score; H_fix does not."""
    from ccp.fusion import h_fix, episode_standardization
    rng = np.random.RandomState(0)
    D, A = rng.rand(1, 150), rng.rand(1, 150)
    D2, A2 = D.copy(), A.copy()
    D2[0, 1:], A2[0, 1:] = rng.rand(149), rng.rand(149)
    assert episode_standardization(D, A)[0, 0] != episode_standardization(D2, A2)[0, 0]
    assert h_fix(D, A, 2.5)[0, 0] == h_fix(D2, A2, 2.5)[0, 0]
