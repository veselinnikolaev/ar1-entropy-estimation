import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy
import pandas
import scipy

from src.estimator import ESTIMATORS
from src.experiment import run_grid


def git_commit() -> str | None:
    """Return the current HEAD commit, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_metadata(path: Path, a, est, phis, ks, sizes, R) -> Path:
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "estimator": est.name,
        "k_nn": a.k_nn,
        "seed": a.seed,
        "R": R,
        "phis": phis,
        "ks": ks,
        "sizes": sizes,
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
        },
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimator", default="kl", choices=list(ESTIMATORS))
    ap.add_argument("--k_nn", type=int, default=4)
    a = ap.parse_args()

    if a.quick:
        phis, ks, sizes, R, name = (
            [0.0, 0.9],
            [1, 5],
            {"fixed_n": 200, "fixed_L": 1000},
            20,
            "quick",
        )
    else:
        phis, ks = [0.0, 0.5, 0.9, 0.99], [1, 2, 5, 10, 20, 50]
        sizes, R, name = {"fixed_n": 1000, "fixed_L": 20000}, 500, "full"

    est = ESTIMATORS[a.estimator](k_nn=a.k_nn)
    df = run_grid(est, ["fixed_n", "fixed_L"], phis, ks, sizes, R=R, seed=a.seed)
    Path("results").mkdir(exist_ok=True)
    out = Path(f"results/{name}_{est.name}_k{a.k_nn}_seed{a.seed}.csv")
    df.to_csv(out, index=False)
    meta = write_metadata(out.with_suffix(".json"), a, est, phis, ks, sizes, R)
    print(df.round(4).to_string())
    print("saved", out)
    print("saved", meta)


if __name__ == "__main__":
    main()
