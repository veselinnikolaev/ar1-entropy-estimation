import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    ("bias", "Bias (est - true)"),
    ("rmse", "RMSE"),
    ("std", "Std across trajectories"),
]


def plot_design(df, design, estimator, outdir):
    d = df[(df.design == design) & (df.estimator == estimator)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, (col, label) in zip(axes, METRICS):
        for phi, g in d[d.kind == "ar1"].groupby("phi"):
            g = g.sort_values("k")
            if col == "bias":
                se = 1.96 * g["std"] / np.sqrt(g["R"])  # 95% CI of the mean error
                ax.errorbar(
                    g.k, g[col], yerr=se, marker="o", capsize=2, label=f"phi={phi}"
                )
            else:
                ax.plot(g.k, g[col], marker="o", label=f"phi={phi}")
        # iid control does not depend on phi, so average over phi
        iid = d[d.kind == "iid"].groupby("k")[col].mean().sort_index()
        ax.plot(iid.index, iid.values, "k--", label="iid control")
        ax.set_xscale("log")
        ax.set_xlabel("thinning step k")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col == "bias":
            ax.axhline(0, color="gray", lw=0.8)
        else:
            ax.set_yscale("log")
    axes[0].legend()
    fig.suptitle(f"{design} | estimator={estimator}")
    fig.tight_layout()
    out = Path(outdir) / f"{design}_{estimator}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/full_*.csv")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    files = sorted(Path(".").glob(a.pattern))
    if not files:
        raise SystemExit(f"No files match {a.pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    Path(a.outdir).mkdir(exist_ok=True)
    for est in df.estimator.unique():
        for design in df.design.unique():
            plot_design(df, design, est, a.outdir)


if __name__ == "__main__":
    main()
