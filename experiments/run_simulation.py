"""Run script for AR(1) entropy estimation simulation.

Executes the workflow using modules from src/
"""

from src.estimator import KozachenkoLeonenkoEstimator
from src.generator import generate_ar1_process, thin_series
from src.theory import theoretical_marginal_entropy


def main():
    print("Starting AR(1) Entropy Estimation Simulation...")
    N = 5000
    phi1 = 0.95
    sigma = 1.0

    # 1. Generate AR(1) process
    series = generate_ar1_process(n_samples=N, phi1=phi1, sigma=sigma, seed=42)
    print(f"Generated AR(1) series of length {len(series)} with phi1={phi1}")

    # 2. Thin series (e.g., k=5)
    k = 5
    thinned = thin_series(series, k=k)
    print(
        f"Thinned series with step k={k}: length {len(thinned)} (Expected ~{N // k})"
    )

    # 3. Initialize Estimator
    estimator = KozachenkoLeonenkoEstimator(k_nn=3)

    # 4. Compute Results
    true_h = theoretical_marginal_entropy(sigma)
    est_h_full = estimator.estimate(series)
    est_h_thinned = estimator.estimate(thinned)

    print("\n--- Results Summary ---")
    print(f"True Marginal Entropy h(X):          {true_h:.4f}")
    print(
        f"Estimated k-NN Entropy (Full N={N}):   {est_h_full:.4f} (Error: {abs(true_h - est_h_full):.4f})"
    )
    print(
        f"Estimated k-NN Entropy (Thinned N={len(thinned)}): {est_h_thinned:.4f} (Error: {abs(true_h - est_h_thinned):.4f})"
    )


if __name__ == "__main__":
    main()