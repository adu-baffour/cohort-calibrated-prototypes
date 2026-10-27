"""Metrics against sklearn / scipy references."""
import numpy as np
import pytest

from ccp import metrics


@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_sklearn(seed):
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    s = rng.randn(150) + np.r_[np.zeros(75), np.ones(75)] * 0.7
    s[::7] = np.round(s[::7], 1)  # ties
    y = np.r_[np.zeros(75), np.ones(75)]
    assert metrics.auroc(s) == pytest.approx(roc_auc_score(y, s) * 100, abs=1e-10)
    perm = rng.permutation(150)
    assert metrics.auroc(s[perm], y[perm]) == pytest.approx(roc_auc_score(y[perm], s[perm]) * 100, abs=1e-10)


def test_paired_matches_scipy_wilcoxon():
    from scipy.stats import wilcoxon
    rng = np.random.RandomState(0)
    a, b = rng.rand(600) * 100, rng.rand(600) * 100
    b[:10] = a[:10]  # zero differences are dropped (zero_method="wilcox")
    p = metrics.paired(a, b)
    d = a - b
    assert p["wilcoxon_p"] == pytest.approx(wilcoxon(d, zero_method="wilcox").pvalue, rel=1e-12)
    assert p["mean"] == pytest.approx(d.mean())
    assert p["ci95"] == pytest.approx(1.96 * d.std() / np.sqrt(600))
    assert p["frac_gain"] == pytest.approx((d > 0).mean()) and p["frac_tie"] == pytest.approx(0.01 * 10 / 6)


def test_accuracy_and_ci():
    t = np.repeat(np.arange(5), 15)
    assert metrics.accuracy(t) == 100.0
    assert metrics.ci95(np.ones(10)) == 0.0
    s = metrics.summarise(np.array([70.0, 72.0]), np.array([60.0, 62.0]))
    assert s["auroc"] == 71.0 and s["acc"] == 61.0 and s["n_episodes"] == 2
