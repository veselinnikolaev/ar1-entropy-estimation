import numpy as np
import pandas as pd

from src.generator import generate_ar1_process, thin_series


def one_cell(design, phi, k, size, R, sigma, est, rng):
    """fixed_n: size = retained N (simulate size*k). fixed_L: size = total length."""
    L = size * k if design == "fixed_n" else size
    h = est.target(sigma, phi)
    ar, iid = np.empty(R), np.empty(R)
    for r in range(R):
        y = thin_series(generate_ar1_process(L, phi, sigma, rng), k)
        ar[r] = est.estimate(y)
        iid[r] = est.estimate(rng.normal(0, sigma, len(y)))  # matched size
    rows = []
    for kind, v in (("ar1", ar), ("iid", iid)):
        e = v - h
        rows.append(
            {
                "estimator": est.name,
                "design": design,
                "phi": phi,
                "k": k,
                "L": L,
                "n_ret": len(y),
                "kind": kind,
                "R": R,
                "bias": e.mean(),
                "rmse": np.sqrt((e**2).mean()),
                "std": e.std(ddof=1),
            }
        )
    return rows


def run_grid(est, designs, phis, ks, sizes, R=500, sigma=1.0, seed=0):
    cells = [(d, p, k) for d in designs for p in phis for k in ks]
    seeds = np.random.SeedSequence(seed).spawn(len(cells))
    rows = []
    for (d, p, k), s in zip(cells, seeds):
        rows += one_cell(d, p, k, sizes[d], R, sigma, est, np.random.default_rng(s))
        print(f"done {d} phi={p} k={k}", flush=True)
    return pd.DataFrame(rows)
