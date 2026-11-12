'''
This module performs Monte Carlo simulations to compare the finite-sample behavior of different inferential procedures for two-proportion comparisons.
It evaluates:
- Type I error rates for the two-proportion z-test and likelihood ratio test
- Coverage of Wald and score-style confidence intervals
- Approximate empirical behavior across multiple sample sizes and baseline probabilities

'''
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compute_z_test(
    successes_a: np.ndarray,
    n_a: int,
    successes_b: np.ndarray,
    n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates two-proportion Z-test statistics and asymptotic p-values."""
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    pooled_p = (successes_a + successes_b) / (n_a + n_b)

    standard_error = np.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / n_a + 1.0 / n_b))

    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = np.where(standard_error > 0.0, (p_a - p_b) / standard_error, 0.0)

    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stats)))
    return z_stats, p_values


def _calculate_safe_log_ratio(
    counts: np.ndarray,
    group_probs: np.ndarray,
    pooled_probs: np.ndarray,
) -> np.ndarray:
    """Helper: Computes counts * log(group_probs / pooled_probs) safely."""
    results = np.zeros_like(counts, dtype=float)
    valid_mask = (counts > 0) & (pooled_probs > 0)
    results[valid_mask] = counts[valid_mask] * np.log(
        group_probs[valid_mask] / pooled_probs[valid_mask]
    )
    return results


def compute_lrt(
    successes_a: np.ndarray,
    n_a: int,
    successes_b: np.ndarray,
    n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates pairwise likelihood ratio test (G^2) statistics and p-values."""
    counts_a = np.asarray(successes_a)
    counts_b = np.asarray(successes_b)

    p_a = counts_a / n_a
    p_b = counts_b / n_b
    pooled_p = (counts_a + counts_b) / (n_a + n_b)

    term_a = _calculate_safe_log_ratio(
        counts_a, p_a, pooled_p
    ) + _calculate_safe_log_ratio(n_a - counts_a, 1.0 - p_a, 1.0 - pooled_p)

    term_b = _calculate_safe_log_ratio(
        counts_b, p_b, pooled_p
    ) + _calculate_safe_log_ratio(n_b - counts_b, 1.0 - p_b, 1.0 - pooled_p)

    g2_stats = 2.0 * (term_a + term_b)
    p_values = 1.0 - stats.chi2.cdf(g2_stats, df=1)

    return g2_stats, p_values

def compute_fisher_exact(
    successes_a: np.ndarray,
    n_a: int,
    successes_b: np.ndarray,
    n_b: int,
) -> np.ndarray:
    """Calculates two-sided p-values using Fisher's Exact Test across simulated arrays."""
    p_values = np.zeros_like(successes_a, dtype=float)
    
    for i in range(len(successes_a)):
        contingency_table = [
            [successes_a[i], n_a - successes_a[i]],
            [successes_b[i], n_b - successes_b[i]]
        ]
        # Calculate the exact p-value
        _, p_values[i] = stats.fisher_exact(contingency_table, alternative='two-sided')
        
    return p_values

def run_single_simulation(
    sample_size: int,
    baseline_p: float,
    num_replications: int = 10000,
    alpha: float = 0.05,
    run_exact: bool = True,
) -> tuple[float, float, float]:
    """Simulates empirical Type I error rates, incorporating the exact test."""
    successes_a = np.random.binomial(n=sample_size, p=baseline_p, size=num_replications)
    successes_b = np.random.binomial(n=sample_size, p=baseline_p, size=num_replications)

    _, p_values_z = compute_z_test(successes_a, sample_size, successes_b, sample_size)
    _, p_values_lrt = compute_lrt(successes_a, sample_size, successes_b, sample_size)
    
    type1_error_z = float(np.mean(p_values_z <= alpha))
    type1_error_lrt = float(np.mean(p_values_lrt <= alpha))
    type1_error_fisher = np.nan
    
    if run_exact:
        p_values_fisher = compute_fisher_exact(successes_a, sample_size, successes_b, sample_size)
        type1_error_fisher = float(np.mean(p_values_fisher <= alpha))

    return type1_error_z, type1_error_lrt, type1_error_fisher

def generate_type1_error_data(
    sample_sizes: list[int],
    p_baselines: list[float],
    num_replications: int = 10000,
) -> pd.DataFrame:
    """Executes the full grid simulation, saving Z, LRT, and Fisher errors."""
    records: list[dict[str, float | int]] = []

    for p in p_baselines:
        for n in sample_sizes:
            # Fisher's exact can be computationally heavy in Python loops.
            # Consider limiting it to smaller sample sizes where asymptotic tests fail.
            run_exact = n <= 1000  
            
            err_z, err_lrt, err_fisher = run_single_simulation(
                sample_size=n,
                baseline_p=p,
                num_replications=num_replications,
                run_exact=run_exact
            )
            records.append(
                {
                    "sample_size": n,
                    "p_baseline": p,
                    "type1_z": err_z,
                    "type1_lrt": err_lrt,
                    "type1_fisher": err_fisher,
                }
            )

    return pd.DataFrame(records)

def compute_wald_ci_diff(
    successes_a: np.ndarray,
    n_a: int,
    successes_b: np.ndarray,
    n_b: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate Wald CI for the difference p_a - p_b."""
    p_a = np.asarray(successes_a, dtype=float) / n_a
    p_b = np.asarray(successes_b, dtype=float) / n_b
    diff = p_a - p_b
    se = np.sqrt(p_a * (1.0 - p_a) / n_a + p_b * (1.0 - p_b) / n_b)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    lower = diff - z_crit * se
    upper = diff + z_crit * se
    return lower, upper


def compute_score_ci_diff(
    successes_a: np.ndarray,
    n_a: int,
    successes_b: np.ndarray,
    n_b: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Score-style CI for the difference p_a - p_b under the null."""
    x_a = np.asarray(successes_a, dtype=float)
    x_b = np.asarray(successes_b, dtype=float)
    p_a = x_a / n_a
    p_b = x_b / n_b
    diff = p_a - p_b
    pooled_p = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / n_a + 1.0 / n_b))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    lower = diff - z_crit * se
    upper = diff + z_crit * se
    return lower, upper


def simulate_ci_coverage(
    sample_size: int,
    baseline_p: float,
    num_replications: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Empirical coverage of Wald and score-style 95% CIs under the null."""
    successes_a = np.random.binomial(n=sample_size, p=baseline_p, size=num_replications)
    successes_b = np.random.binomial(n=sample_size, p=baseline_p, size=num_replications)

    lower_wald, upper_wald = compute_wald_ci_diff(
        successes_a, sample_size, successes_b, sample_size, alpha=alpha
    )
    lower_score, upper_score = compute_score_ci_diff(
        successes_a, sample_size, successes_b, sample_size, alpha=alpha
    )

    coverage_wald = float(np.mean((lower_wald <= 0.0) & (upper_wald >= 0.0)))
    coverage_score = float(np.mean((lower_score <= 0.0) & (upper_score >= 0.0)))
    return coverage_wald, coverage_score


def generate_coverage_data(
    sample_sizes: list[int],
    p_baselines: list[float],
    num_replications: int = 10000,
) -> pd.DataFrame:
    """Runs the full grid for empirical CI coverage probabilities."""
    records: list[dict[str, float | int]] = []
    for p in p_baselines:
        for n in sample_sizes:
            cov_wald, cov_score = simulate_ci_coverage(
                sample_size=n,
                baseline_p=p,
                num_replications=num_replications,
            )
            records.append(
                {
                    "sample_size": n,
                    "p_baseline": p,
                    "coverage_wald": cov_wald,
                    "coverage_score": cov_score,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    np.random.seed(42)
    sample_sizes = [30, 50, 100, 250, 500, 1000, 5000, 20000, 50000, 100000]
    baseline_probabilities = [0.005, 0.01, 0.05, 0.1]

    output_dir = Path(__file__).resolve().parents[1] / "montecarlo_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation_results = generate_type1_error_data(
        sample_sizes=sample_sizes,
        p_baselines=baseline_probabilities,
        num_replications=10000,
    )

    coverage_results = generate_coverage_data(
        sample_sizes=sample_sizes,
        p_baselines=baseline_probabilities,
        num_replications=5000,
    )

    simulation_results.to_csv(output_dir / "simulation_results.csv", index=False)
    coverage_results.to_csv(output_dir / "coverage_results.csv", index=False)

    print(f"Saved simulation results to: {output_dir}")


if __name__ == "__main__":
    main()
