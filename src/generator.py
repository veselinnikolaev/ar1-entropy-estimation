"""AR(1) Process Simulation and Subsampling/Thinning Module

Based on Part A, B, and D of research notes.
"""

import numpy as np


def generate_ar1_process(
    n_samples: int, phi1: float, sigma: float, c: float = 0.0, seed: int = None
) -> np.ndarray:
    """Generates a stationary AR(1) process: X_t = C + phi_1 * X_{t-1} + epsilon_t

    Initialized from the exact stationary marginal distribution N(0, sigma^2).
    """
    if seed is not None:
        np.random.seed(seed)

    # Innovation variance s^2 = sigma^2 * (1 - phi_1^2) [Notes Part A, Eq 5]
    s2 = (sigma**2) * (1.0 - phi1**2)
    s = np.sqrt(max(0.0, s2))

    x = np.empty(n_samples)
    # Base case: Initialize X_0 from N(0, sigma^2) to avoid burn-in transients [Notes Part B]
    x[0] = np.random.normal(0.0, sigma)

    # Innovations for t = 1 to n_samples - 1
    eps = np.random.normal(0.0, s, size=n_samples)

    for t in range(1, n_samples):
        x[t] = c + phi1 * x[t - 1] + eps[t]

    return x


def thin_series(x: np.ndarray, k: int) -> np.ndarray:
    """Thins the time series by retaining every k-th observation: Y_j = X_{jk} [Notes Part D, Eq 14]."""
    return x[::k]