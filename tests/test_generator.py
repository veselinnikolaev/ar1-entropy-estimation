import numpy as np
from src.generator import generate_ar1_process, thin_series

def test_ar1_properties():
    rng = np.random.default_rng(0)
    phi, sigma, k = 0.9, 2.0, 3
    X = np.stack([generate_ar1_process(60, phi, sigma, rng) for _ in range(20000)])
    assert np.isclose(X[:, 0].var(), sigma**2, rtol=0.05)
    assert np.isclose(X[:, 40].var(), sigma**2, rtol=0.05)          # stationarity
    assert np.isclose(np.corrcoef(X[:, 30], X[:, 31])[0, 1], phi, atol=0.02)
    assert np.isclose(np.corrcoef(X[:, 30], X[:, 35])[0, 1], phi**5, atol=0.02)
    Y = X[:, ::k]                                                   # thinned lag-1
    assert np.isclose(np.corrcoef(Y[:, 5], Y[:, 6])[0, 1], phi**k, atol=0.02)

def test_thin_length():
    assert len(thin_series(np.arange(10), 3)) == 4