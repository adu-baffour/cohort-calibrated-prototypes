"""OSCR and FPR95 (docs/preregistration/final_round.md) against hand-computed cases."""
import numpy as np
import pytest

from ccp import metrics


def test_oscr_all_correct_hand_case():
    # known 1, 2, 3 (all correct), unknown 2.5, 4. Curve (FPR, CCR): (0,0) (0,1/3) (0,2/3) (.5,2/3) (.5,1) (1,1) -> area 1/3 + 1/2 = 5/6
    s, y = np.array([1, 2, 3, 2.5, 4.0]), np.array([0, 0, 0, 1, 1], bool)
    assert metrics.oscr(s, [True, True, True], y) == pytest.approx(500 / 6)
    assert metrics.auroc(s, y) == pytest.approx(500 / 6)


def test_oscr_wrong_label_hand_case():
    # known 1 (correct), 2 (wrong), 3 (correct): CCR 1/3 until the unknown at 2.5, then 2/3 -> area 0.5 * 1/3 + 0.5 * 2/3 = 1/2
    s, y = np.array([1, 2, 3, 2.5, 4.0]), np.array([0, 0, 0, 1, 1], bool)
    assert metrics.oscr(s, [True, False, True], y) == pytest.approx(50.0)
    assert metrics.oscr(s, [False, False, False], y) == 0.0


def test_oscr_ties_are_a_straight_segment():
    # known 1, 2; unknown 2, 3 (tie at 2): points (0,0) (0,.5) (.5,1) (1,1) -> .375 + .5 = .875, the average-rank AUROC
    s, y = np.array([1, 2, 2, 3.0]), np.array([0, 0, 1, 1], bool)
    assert metrics.oscr(s, [True, True], y) == pytest.approx(87.5)
    assert metrics.auroc(s, y) == pytest.approx(87.5)


def test_oscr_equals_auroc_when_every_label_is_correct_and_bounded_by_accuracy():
    rng = np.random.RandomState(0)
    for _ in range(20):
        s = np.round(rng.randn(150), 1)                      # rounding creates ties
        assert metrics.oscr(s, np.ones(75, bool)) == pytest.approx(metrics.auroc(s))
        c = rng.rand(75) < 0.7
        o = metrics.oscr(s, c)
        assert 0 <= o <= c.mean() * 100 + 1e-9 and o <= metrics.auroc(s) + 1e-9


def test_fpr95_hand_cases():
    # 20 known 1..20: tau = 19th smallest = 19; unknown 18.5, 19 (tie: accepted), 19.5, 30 -> 2 / 4
    s = np.r_[np.arange(1, 21, dtype=float), [18.5, 19, 19.5, 30]]
    y = np.r_[np.zeros(20, bool), np.ones(4, bool)]
    assert metrics.fpr95(s, y) == pytest.approx(50.0)
    # FSOR layout: known 0..74 -> tau = 72nd smallest = 71; one unknown at 70.5, the rest at 100 -> 1 / 75
    s = np.r_[np.arange(75, dtype=float), [70.5], np.full(74, 100.0)]
    assert metrics.fpr95(s) == pytest.approx(100 / 75)
    # perfect separation -> 0; every unknown below every known -> 100
    assert metrics.fpr95(np.r_[np.zeros(75), np.ones(75)]) == 0.0
    assert metrics.fpr95(np.r_[np.ones(75), np.zeros(75)]) == 100.0


def test_episode_wrappers():
    rng = np.random.RandomState(1)
    S, C = rng.randn(3, 150), rng.rand(3, 75) < 0.8
    np.testing.assert_allclose(metrics.episode_oscr(S, C), [metrics.oscr(s, c) for s, c in zip(S, C)])
    np.testing.assert_allclose(metrics.episode_fpr95(S), [metrics.fpr95(s) for s in S])
