# Observation Frequency and Entropy Estimation for a Gaussian AR(1) Process

[![tests](https://github.com/veselinnikolaev/ar1-entropy-estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/veselinnikolaev/ar1-entropy-estimation/actions/workflows/tests.yml)

BSP3, University of Luxembourg. Studies how thinning (recording every k-th value)
affects k-NN entropy estimation for a stationary Gaussian AR(1) process with known
marginal entropy h(X) = 0.5 ln(2*pi*e*sigma^2).

## Layout
- `docs/`: project abstract and research notes (proofs, Parts A-G)
- `src/generator.py`: stationary AR(1) simulation and thinning (Parts A, B, D)
- `src/theory.py`: true entropy, tau_int, entropy rate (Parts C, E, F)
- `src/estimator.py`: Kozachenko-Leonenko estimator (Part G, Eq 34) behind `BaseEntropyEstimator`; register new estimators in `ESTIMATORS`
- `src/experiment.py`: grid runner (bias, RMSE, std over R trajectories, with iid controls)
- `src/plots.py`: figures from result CSVs
- `experiments/run_simulation.py`: command-line entry point
- `tests/`: estimator and generator validation
- `results/`: CSV outputs; each results CSV has a matching metadata JSON with the same stem; `figures/`: generated plots

## Setup
```bash
uv sync
```

## Commands (run from repo root)
```bash
uv run pytest
uv run python -m experiments.run_simulation --quick
uv run python -m experiments.run_simulation --estimator kl --k_nn 4 --seed 0
uv run python -m src.plots --pattern "results/full_kl_k4_seed0.csv"
```

## Experimental design
- phi in {0, 0.5, 0.9, 0.99}, k in {1, 2, 5, 10, 20, 50}, sigma = 1, R = 500 trajectories
- `fixed_n`: N = 1000 retained points (simulate N*k)
- `fixed_L`: total length L = 20000 (retain L/k)
- Each trajectory is paired with an iid N(0, sigma^2) control of the same retained size
- Reproducibility: per-cell RNG streams spawned from `SeedSequence(seed)`

## Change log
- Initial pipeline: generator, theory, KL estimator, tests (5 passing)
- Experiment runner, estimator registry, `--estimator/--k_nn/--seed` options
- Plots with 95% CI on bias
- Run metadata JSON saved next to each results CSV
- Property, reproducibility and phi=0 control tests
- MIT license and `CITATION.cff`
- ruff lint and format checks in CI