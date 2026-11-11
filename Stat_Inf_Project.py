
# ============================================================
# STATISTICAL INFERENCE PROJECT
# Pairwise LRT and Two-Proportion Z-Tests
# Hillstrom Email Marketing Experiment
# ============================================================

import pandas as pd
import numpy as np

from scipy.stats import chi2, norm
from scipy.special import xlogy, xlog1py


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_URL = 'https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/main/data/raw/hillstrom.csv'

df = pd.read_csv(DATA_URL)

# Map experimental groups
group_mapping = {
    "Mens E-Mail": "A",
    "Womens E-Mail": "B",
    "No E-Mail": "C"
}

df["group"] = df["segment"].map(group_mapping)

# Retain the three experimental groups
df = df.dropna(subset=["group", "conversion"])

# Check binary outcome
assert set(df["conversion"].unique()).issubset({0, 1})


# ============================================================
# 2. DATA SUMMARY
# ============================================================

summary = (
    df.groupby("group")["conversion"]
      .agg(n="count", X="sum")
)

summary["p_hat"] = summary["X"] / summary["n"]

print("\n========== DATA SUMMARY ==========")
print(summary)


# ============================================================
# 3. BINOMIAL LOG-LIKELIHOOD
# ============================================================

def log_likelihood(x, n, p):

    return (
        xlogy(x, p)
        + xlog1py(n - x, -p)
    )


# ============================================================
# 4. LIKELIHOOD RATIO TEST
# ============================================================

def likelihood_ratio_test(x1, n1, x2, n2):

    # Unrestricted MLEs
    p1 = x1 / n1
    p2 = x2 / n2

    # Restricted MLE under H0: p1 = p2
    p_pooled = (x1 + x2) / (n1 + n2)

    # Unrestricted log-likelihood
    ll_unrestricted = (
        log_likelihood(x1, n1, p1)
        + log_likelihood(x2, n2, p2)
    )

    # Restricted log-likelihood
    ll_restricted = (
        log_likelihood(x1, n1, p_pooled)
        + log_likelihood(x2, n2, p_pooled)
    )

    # LRT statistic
    G2 = max(0.0, 2 * (ll_unrestricted - ll_restricted))

    # One degree of freedom
    p_value = chi2.sf(G2, df=1)

    return G2, p_value


# ============================================================
# 5. TWO-PROPORTION Z-TEST
# ============================================================

def z_test(x1, n1, x2, n2):

    p1 = x1 / n1
    p2 = x2 / n2

    # Pooled proportion under H0
    p_pooled = (x1 + x2) / (n1 + n2)

    # Standard error under H0
    SE = np.sqrt(
        p_pooled * (1 - p_pooled)
        * (1/n1 + 1/n2)
    )

    Z = (p1 - p2) / SE

    # Two-sided p-value
    p_value = 2 * norm.sf(abs(Z))

    return Z, p_value


# ============================================================
# 6. HOLM CORRECTION
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(p_values)
    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    multipliers = np.arange(m, 0, -1)

    adjusted_sorted = multipliers * sorted_p

    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.minimum(adjusted_sorted, 1)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    return adjusted


# ============================================================
# 7. PAIRWISE COMPARISONS
# ============================================================

pairs = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C")
]

results = []

for g1, g2 in pairs:

    n1 = int(summary.loc[g1, "n"])
    x1 = int(summary.loc[g1, "X"])

    n2 = int(summary.loc[g2, "n"])
    x2 = int(summary.loc[g2, "X"])

    p1 = x1 / n1
    p2 = x2 / n2

    difference = p1 - p2

    relative_lift = (p1 / p2 - 1) * 100 if p2 > 0 else np.nan

    # LRT
    G2, p_lrt = likelihood_ratio_test(
        x1, n1, x2, n2
    )

    # Z-test
    Z, p_z = z_test(
        x1, n1, x2, n2
    )

    results.append({

        "Comparison": f"{g1} vs {g2}",

        "Rate 1": p1,
        "Rate 2": p2,

        "Difference": difference,
        "Relative lift (%)": relative_lift,

        "LRT statistic": G2,
        "LRT p-value": p_lrt,

        "Z statistic": Z,
        "Z p-value": p_z,

        "Z squared": Z**2,
        "G2 - Z2": G2 - Z**2

    })


results = pd.DataFrame(results)


# ============================================================
# 8. HOLM-ADJUSTED P-VALUES
# ============================================================

results["LRT Holm p-value"] = holm_correction(
    results["LRT p-value"]
)

results["Z Holm p-value"] = holm_correction(
    results["Z p-value"]
)


# ============================================================
# 9. STATISTICAL DECISIONS
# ============================================================

alpha = 0.01

results["LRT Decision"] = np.where(
    results["LRT Holm p-value"] < alpha,
    "Reject H0",
    "Fail to reject H0"
)

results["Z Decision"] = np.where(
    results["Z Holm p-value"] < alpha,
    "Reject H0",
    "Fail to reject H0"
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n========== PAIRWISE TEST RESULTS ==========")

print(results.round(8).to_string(index=False))


# ============================================================
# 11. COMPARE LRT AND Z-TEST
# ============================================================

print("\n========== LRT vs Z-TEST COMPARISON ==========")

for _, row in results.iterrows():

    print(f"\n{row['Comparison']}")

    print(f"LRT statistic = {row['LRT statistic']:.6f}")
    print(f"Z statistic   = {row['Z statistic']:.6f}")
    print(f"Z squared     = {row['Z squared']:.6f}")

    print(f"LRT - Z^2     = {row['G2 - Z2']:.6f}")

    print(f"LRT p-value   = {row['LRT p-value']:.10g}")
    print(f"Z p-value     = {row['Z p-value']:.10g}")


# ============================================================
# 12. FINAL CONCLUSIONS
# ============================================================

print("\n========== FINAL CONCLUSIONS ==========")

for _, row in results.iterrows():

    print(f"\nComparison: {row['Comparison']}")

    print(f"LRT adjusted p-value: {row['LRT Holm p-value']:.8g}")
    print(f"Z-test adjusted p-value: {row['Z Holm p-value']:.8g}")

    print(f"LRT: {row['LRT Decision']}")
    print(f"Z-test: {row['Z Decision']}")
