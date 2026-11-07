import numpy as np
import pandas as pd

from src.estimator import KozachenkoLeonenkoEstimator
from src.experiment import run_grid


def test_shift_invariance():
    y = np.random.default_rng(42).normal(size=500)
    est = KozachenkoLeonenkoEstimator(4)
    assert np.isclose(est.estimate(y + 7.3), est.estimate(y), atol=1e-9)


def test_scale_rule():
    y = np.random.default_rng(43).normal(size=500)
    est = KozachenkoLeonenkoEstimator(4)
    base = est.estimate(y)
    for a in (0.5, 3.0):
        assert np.isclose(est.estimate(a * y), base + np.log(a), atol=1e-9)


def test_run_grid_shape_and_reproducibility():
    kwargs = {
        "designs": ["fixed_n", "fixed_L"],
        "phis": [0.0, 0.9],
        "ks": [1, 2],
        "sizes": {"fixed_n": 50, "fixed_L": 100},
        "R": 5,
        "seed": 3,
    }
    df = run_grid(KozachenkoLeonenkoEstimator(4), **kwargs)

    assert len(df) == 16
    expected = {
        "estimator",
        "design",
        "phi",
        "k",
        "L",
        "n_ret",
        "kind",
        "R",
        "bias",
        "rmse",
        "std",
    }
    assert expected <= set(df.columns)

    df_again = run_grid(KozachenkoLeonenkoEstimator(4), **kwargs)
    pd.testing.assert_frame_equal(df, df_again)


def test_phi_zero_control():
    df = run_grid(
        KozachenkoLeonenkoEstimator(4),
        ["fixed_n"],
        [0.0],
        [5],
        {"fixed_n": 300},
        R=200,
        seed=0,
    )
    ar = df[df["kind"] == "ar1"].iloc[0]
    iid = df[df["kind"] == "iid"].iloc[0]
    se = np.sqrt(ar["std"] ** 2 / ar["R"] + iid["std"] ** 2 / iid["R"])
    assert abs(ar["bias"] - iid["bias"]) < 4 * se
