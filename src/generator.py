"""AR(1) process simulation and thinning [Notes Parts A, B, D]."""

import numpy as np
from scipy.signal import lfilter


def generate_ar1_process(n, phi1, sigma, rng):
    """Stationary AR(1), X_t = phi1 X_{t-1} + eps_t, X_0 ~ N(0, sigma^2)."""
    s = sigma * np.sqrt(1 - phi1**2)  # Eq 5
    x0 = rng.normal(0, sigma)
    x = np.empty(n)
    x[0] = x0
    if n > 1:
        eps = rng.normal(0, s, n - 1)
        x[1:] = lfilter([1.0], [1.0, -phi1], eps, zi=[phi1 * x0])[0]
    return x


def thin_series(x, k):
    """Y_j = X_{jk} [Eq 14]."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return x[::k]
