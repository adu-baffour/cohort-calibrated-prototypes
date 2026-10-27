"""SCG (Section 3.5, Eq. 8): decile binning of the raw maximum cosine, values on edges to the upper bin, the pooled within-bin SD
(ddof 1), the exclusion of bins with fewer than two known or two unknown queries or zero SD, the unweighted mean over valid bins."""
import numpy as np
import pytest

from ccp.scg import scg_bins, scg_gap, scg_reading


def test_scg_bins_match_figure_2b_and_reading():
    cond_means = pytest.importorskip("scripts.figures.fig2_mechanism").cond_means
    rng = np.random.RandomState(3)
    x, m = rng.rand(40, 150), rng.rand(40, 150)
    y = np.tile(np.r_[np.zeros(75), np.ones(75)], (40, 1))
    ours = scg_bins(x, m, y)
    ref = cond_means(x.ravel(), m.ravel(), y.ravel())
    kn = [r for r in ref if r["group"] == "known"] if "group" in ref[0] else ref[0::2]
    un = [r for r in ref if r["group"] == "unknown"] if "group" in ref[0] else ref[1::2]
    np.testing.assert_allclose([b["m_known"] for b in ours], [r["m_mean"] for r in kn], rtol=0, atol=1e-12)
    np.testing.assert_allclose([b["m_unknown"] for b in ours], [r["m_mean"] for r in un], rtol=0, atol=1e-12)
    assert scg_gap(ours, standardized=False) == pytest.approx(np.mean([b["m_unknown"] - b["m_known"] for b in ours]))
    xs, ms, ys = x.ravel(), m.ravel(), y.ravel()
    edges = np.quantile(xs, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, 9)
    std = []
    for i in range(10):
        k, u = ms[(b == i) & (ys == 0)], ms[(b == i) & (ys == 1)]
        sp = np.sqrt((((k - k.mean()) ** 2).sum() + ((u - u.mean()) ** 2).sum()) / (k.size + u.size - 2))   # two-group pooled SD
        assert ours[i]["sd_pooled"] == pytest.approx(sp, rel=1e-12)
        std.append((u.mean() - k.mean()) / sp)
    np.testing.assert_allclose([b_["std_gap"] for b_ in ours], std, rtol=1e-12)
    assert scg_gap(ours) == pytest.approx(np.mean(std))                # registered statistic: standardized
    # scale invariance: multiplying m by 10 changes the raw gap, not the standardized one
    ours10 = scg_bins(x, 10 * m, y)
    assert scg_gap(ours10) == pytest.approx(scg_gap(ours)) and scg_gap(ours10, False) == pytest.approx(10 * scg_gap(ours, False))
    one = scg_bins(np.arange(4.0), np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]))
    assert all(not np.isfinite(b_["std_gap"]) for b_ in one)          # a bin with < 2 known or < 2 unknown queries is undefined
    assert scg_reading(0.9, 0.3) == "the base cohort carries little signal on tiered"
    assert scg_reading(0.9, 0.31).startswith("the signal exists")
    assert scg_reading(-0.1, 0.1).startswith("undefined")
