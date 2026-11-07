import numpy as np
from scipy.special import digamma

from src.estimator import KozachenkoLeonenkoEstimator
from src.theory import theoretical_marginal_entropy


def test_hand_example_k1():
    y = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    rho = np.array([1, 1, 2, 3, 4])  # nearest-neighbour distances, by hand
    expected = digamma(5) - digamma(1) + np.log(2) + np.mean(np.log(rho))
    assert np.isclose(KozachenkoLeonenkoEstimator(k_nn=1).estimate(y), expected)


def test_kdtree_matches_bruteforce():
    y = np.random.default_rng(1).normal(size=200)
    for k in (1, 3, 4):
        d = np.sort(np.abs(y[:, None] - y[None, :]), axis=1)[:, k]
        exp = digamma(200) - digamma(k) + np.log(2) + np.mean(np.log(d))
        assert np.isclose(KozachenkoLeonenkoEstimator(k_nn=k).estimate(y), exp)


def test_iid_gaussian():
    for s in (0.5, 1.0, 2.0):
        y = np.random.default_rng(0).normal(0, s, 200_000)
        assert (
            abs(
                KozachenkoLeonenkoEstimator(4).estimate(y)
                - theoretical_marginal_entropy(s)
            )
            < 0.02
        )
