"""Entropy Estimator Module with Abstract Interface for Future-Proofing."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from src.theory import theoretical_marginal_entropy


class BaseEntropyEstimator(ABC):
    name = "base"

    @abstractmethod
    def estimate(self, y: np.ndarray) -> float:
        """Estimates differential entropy given 1D scalar samples y."""

    def target(self, sigma: float, phi: float) -> float:
        """True value this estimator aims at. Default: marginal entropy h(X)."""
        return theoretical_marginal_entropy(sigma)


class KozachenkoLeonenkoEstimator(BaseEntropyEstimator):
    """Kozachenko-Leonenko (k-NN) estimator [Notes Part G, Eq 34]."""

    name = "kl"

    def __init__(self, k_nn: int = 3):
        self.k_nn = k_nn

    def estimate(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n <= self.k_nn:
            raise ValueError(
                f"Sample size N={n} must be strictly greater than k_nn={self.k_nn}."
            )
        pts = y[:, None]
        rho = cKDTree(pts).query(pts, k=self.k_nn + 1)[0][:, -1]
        rho = np.maximum(rho, 1e-12)
        return float(
            digamma(n) - digamma(self.k_nn) + np.log(2.0) + np.mean(np.log(rho))
        )


# Register new estimators here
ESTIMATORS = {"kl": KozachenkoLeonenkoEstimator}
