
"""
Monte Carlo Simulation of Family-Wise Error Rate (FWER)

Project: Three-Arm Email Marketing Experiment

Compares:
1. Three unadjusted pairwise Binomial LRTs.
2. Three pairwise LRTs with Holm's step-down correction.

Null hypothesis:
    p_A = p_B = p_C = p0

Author: Farhan
"""

import numpy as np
from scipy.stats import chi2
from scipy.special import xlogy


# --------------------------------------------------
# 1. SIMULATION PARAMETERS
# --------------------------------------------------

R = 100_000
N = 21_333
P0 = 0.00903
ALPHA = 0.05
SEED = 20261108

rng = np.random.default_rng(SEED)


# --------------------------------------------------
# 2. BINOMIAL LOG-LIKELIHOOD DEVIANCE
# --------------------------------------------------

def binomial_deviance(x, n, phat, pooled):
    """
    Calculate twice the log-likelihood difference
    between the unrestricted and restricted models.
    """

    success_ratio = np.divide(
        phat,
        pooled,
        out=np.ones_like(phat, dtype=float),
        where=pooled > 0
    )

    failure_ratio = np.divide(
        1 - phat,
        1 - pooled,
        out=np.ones_like(phat, dtype=float),
        where=pooled < 1
    )

    return 2 * (
        xlogy(x, success_ratio)
        + xlogy(n - x, failure_ratio)
    )


# --------------------------------------------------
# 3. GENERATE BINOMIAL DATA UNDER GLOBAL NULL
# --------------------------------------------------

def generate_data():

    """
    Generate independent Binomial conversion counts
    for groups A, B and C under the complete null.
    """

    X = rng.binomial(
        N,
        P0,
        size=(R, 3)
    )

    return X


# --------------------------------------------------
# 4. CALCULATE PAIRWISE LRT P-VALUES
# --------------------------------------------------

def calculate_pairwise_pvalues(X):

    """
    Calculate asymptotic chi-square p-values
    for A-B, A-C and B-C comparisons.
    """

    P = np.empty((R, 3))

    pairs = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]

    for k, (i, j) in enumerate(pairs):

        x1 = X[:, i]
        x2 = X[:, j]

        p1 = x1 / N
        p2 = x2 / N

        pooled = (x1 + x2) / (2 * N)

        G2 = (
            binomial_deviance(x1, N, p1, pooled)
            +
            binomial_deviance(x2, N, p2, pooled)
        )

        # Pairwise LRT has 1 degree of freedom
        P[:, k] = chi2.sf(G2, df=1)

    return P


# --------------------------------------------------
# 5. UNADJUSTED FAMILY-WISE ERROR RATE
# --------------------------------------------------

def calculate_unadjusted_fwer(P):

    """
    Reject if at least one of the three
    unadjusted p-values is below alpha.
    """

    any_rejection = np.any(
        P < ALPHA,
        axis=1
    )

    return np.mean(any_rejection)


# --------------------------------------------------
# 6. HOLM-ADJUSTED FAMILY-WISE ERROR RATE
# --------------------------------------------------

def calculate_holm_fwer(P):

    """
    Apply Holm's step-down procedure.

    Ordered thresholds:
        alpha/3, alpha/2, alpha
    """

    sorted_p = np.sort(P, axis=1)

    reject_1 = (
        sorted_p[:, 0] < ALPHA / 3
    )

    reject_2 = (
        reject_1
        & (sorted_p[:, 1] < ALPHA / 2)
    )

    reject_3 = (
        reject_2
        & (sorted_p[:, 2] < ALPHA)
    )

    any_rejection = (
        reject_1 | reject_2 | reject_3
    )

    return np.mean(any_rejection)


# --------------------------------------------------
# 7. MAIN SIMULATION
# --------------------------------------------------

def main():

    print("=" * 55)
    print("MONTE CARLO FWER SIMULATION")
    print("=" * 55)

    print("Replications:", R)
    print("Sample size per group:", N)
    print("Common conversion probability:", P0)
    print("Significance level:", ALPHA)
    print("Random seed:", SEED)

    # Generate data
    X = generate_data()

    # Calculate pairwise p-values
    P = calculate_pairwise_pvalues(X)

    # Calculate FWER
    raw_fwer = calculate_unadjusted_fwer(P)

    holm_fwer = calculate_holm_fwer(P)

    print("\nSIMULATION RESULTS")
    print("-" * 55)

    print(
        f"Unadjusted FWER: {raw_fwer:.5f}"
    )

    print(
        f"Holm-adjusted FWER: {holm_fwer:.5f}"
    )

    print(
        f"Nominal FWER: {ALPHA:.5f}"
    )

    print("\nINTERPRETATION")
    print("-" * 55)

    print(
        "Unadjusted testing produces a higher "
        "family-wise error rate."
    )

    print(
        "Holm's correction controls the family-wise "
        "error rate at or below the nominal level."
    )

    print("=" * 55)


if __name__ == "__main__":
    main()
