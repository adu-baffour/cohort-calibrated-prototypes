"""A tiny end-to-end run on a synthetic bank and synthetic episodes: bank file -> target features -> scores -> fusion -> metrics."""
import numpy as np

from ccp import fusion, metrics
from ccp.cohort import Bank
from ccp.scores import REGISTERED, EpisodeContext, TargetFeatures, score_episode
from conftest import make_episodes


def run(synth, bank_obj, shot, n=20):
    tf = TargetFeatures(synth["tgt"], synth["tgt_f"], bank_obj.to_device("cpu"))
    eps = make_episodes(synth["tgt_lab"], shot, n, seed=5)
    S = {k: np.zeros((n, 150)) for k in REGISTERED}
    A = {k: np.zeros(n) for k in REGISTERED}
    for e in range(n):
        rows = np.concatenate([eps["support"][e].reshape(-1), eps["query"][e].reshape(-1), eps["unk_query"][e].reshape(-1)])
        for k, (s, acc, _) in score_episode(EpisodeContext(tf, rows, shot), REGISTERED).items():
            S[k][e], A[k][e] = s, acc
    return S, A


def test_end_to_end(tmp_path, synth, bank):
    sha = bank.save(tmp_path / "bank.npz")
    b = Bank.load(tmp_path / "bank.npz", sha)
    for shot in (1, 5):
        S, A = run(synth, b, shot)
        res = {k: metrics.summarise(metrics.episode_auroc(S[k]), A[k]) for k in REGISTERED}
        for k in REGISTERED:
            assert 0 <= res[k]["auroc"] <= 100 and 0 <= res[k]["acc"] <= 100
        assert res["COSF"]["acc"] > 50  # the synthetic classes are separable
        # a trained host's own score D fused inductively with a frozen weight fitted on "set A" (first half of the episodes)
        rng = np.random.RandomState(shot)
        D = -S["COS"] + 0.3 * rng.randn(*S["COS"].shape)
        w = fusion.fit_weight(D[:10], S["CCP-A"][:10])["w"]
        rows = fusion.fused_rows(D[10:], S["CCP-A"][10:], S["COSF"][10:], w, w)
        p = metrics.paired(metrics.episode_auroc(rows["H_fix"]), metrics.episode_auroc(rows["D"]))
        assert np.isfinite(p["mean"]) and 0 <= p["frac_gain"] <= 1
    S2, _ = run(synth, b, 1)
    S3, _ = run(synth, b, 1)
    assert all(np.array_equal(S2[k], S3[k]) for k in REGISTERED)  # deterministic
