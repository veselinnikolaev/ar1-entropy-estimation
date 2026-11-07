"""Theoretical Marginal Entropy and Autocorrelation Formulas

Based on Part C and Part E of research notes.
"""

import numpy as np


def theoretical_marginal_entropy(sigma: float) -> float:
    """Computes theoretical marginal differential entropy h(X) = 0.5 * ln(2 * pi * e * sigma^2) [Notes Part C, Eq 13]."""
    return 0.5 * np.log(2.0 * np.pi * np.e * (sigma**2))


def integrated_autocorrelation_time(phi1: int, k: int = 1) -> float:
    """Computes tau_int for thinned series with step k [Notes Part E, Eq 27]."""
    phi_k = phi1**k
    return (1.0 + phi_k) / (1.0 - phi_k)

def entropy_rate(sigma, phi1):
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2 * (1 - phi1**2))