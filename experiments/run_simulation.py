import argparse
from pathlib import Path
from src.estimator import ESTIMATORS
from src.experiment import run_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimator", default="kl", choices=list(ESTIMATORS))
    ap.add_argument("--k_nn", type=int, default=4)
    a = ap.parse_args()

    if a.quick:
        phis, ks, sizes, R, name = [0.0, 0.9], [1, 5], {"fixed_n": 200, "fixed_L": 1000}, 20, "quick"
    else:
        phis, ks = [0.0, 0.5, 0.9, 0.99], [1, 2, 5, 10, 20, 50]
        sizes, R, name = {"fixed_n": 1000, "fixed_L": 20000}, 500, "full"

    est = ESTIMATORS[a.estimator](k_nn=a.k_nn)
    df = run_grid(est, ["fixed_n", "fixed_L"], phis, ks, sizes, R=R, seed=a.seed)
    Path("results").mkdir(exist_ok=True)
    out = f"results/{name}_{est.name}_k{a.k_nn}_seed{a.seed}.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string())
    print("saved", out)


if __name__ == "__main__":
    main()