#!/usr/bin/env python3
"""Clean and validate the Hillstrom data for the dataset/CI part of the project.

Usage:
    pip install pandas numpy scipy matplotlib statsmodels
    curl -o hillstrom.csv "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    python validate_hillstrom.py --input hillstrom.csv --outdir results

The script standardizes column names and categorical whitespace, keeps rows with
valid treatment and conversion values for the primary analysis, and writes a
validation report. Exact duplicate rows are reported but not automatically
removed because the public file does not provide a unique customer identifier.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
)

ARMS = {
    "Mens E-Mail": "A",
    "Womens E-Mail": "B",
    "No E-Mail": "C",
}
LABELS = {"A": "Men's email", "B": "Women's email", "C": "No email (control)"}
ORDER = ["A", "B", "C"]
PRE_NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
PRE_CATEGORICAL = ["history_segment", "zip_code", "channel"]


def wilson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    return tuple(proportion_confint(x, n, alpha=alpha, method="wilson"))


def newcombe_ci(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05):
    try:
        return confint_proportions_2indep(
            x1, n1, x2, n2, method="newcomb", compare="diff", alpha=alpha
        )
    except (ValueError, FloatingPointError):
        return (float("nan"), float("nan"))


def mn_ci(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05):
    # Score interval with continuity correction: Miettinen--Nurminen method.
    # Some software versions may fail to invert the score test at boundary
    # cases (for example, zero conversions in a tiny sample); report NA rather
    # than terminating the rest of the audit.
    try:
        return confint_proportions_2indep(
            x1, n1, x2, n2, method="score", compare="diff",
            alpha=alpha, correction=True
        )
    except (ValueError, FloatingPointError):
        return (float("nan"), float("nan"))


def smd(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0 if x.mean() == y.mean() else float("inf")
    return float((x.mean() - y.mean()) / pooled)


def load_and_clean(path: Path):
    raw = pd.read_csv(path, skipinitialspace=True)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw = raw.copy()
    for col in ["segment", "history_segment", "zip_code", "channel"]:
        if col in raw.columns:
            raw[col] = raw[col].astype("string").str.strip()

    report = [f"Rows in source file: {len(raw):,}",
              f"Columns in source file: {len(raw.columns)}",
              "Missing values by column:"]
    report.extend([f"  {c}: {int(n)}" for c, n in raw.isna().sum().items()])
    report.append(f"Exact duplicate rows (flagged, not removed): {int(raw.duplicated().sum()):,}")

    if not {"segment", "conversion"}.issubset(raw.columns):
        raise ValueError("CSV must contain columns named 'segment' and 'conversion'.")
    raw["conversion"] = pd.to_numeric(raw["conversion"], errors="coerce")
    if "visit" in raw.columns:
        raw["visit"] = pd.to_numeric(raw["visit"], errors="coerce")
    if "spend" in raw.columns:
        raw["spend"] = pd.to_numeric(raw["spend"], errors="coerce")

    invalid_segment = ~raw["segment"].isin(ARMS)
    invalid_conversion = raw["conversion"].notna() & ~raw["conversion"].isin([0, 1])
    report.append(f"Rows with missing/unknown treatment label: {int(invalid_segment.sum()):,}")
    report.append(f"Rows with invalid non-binary conversion: {int(invalid_conversion.sum()):,}")

    valid = raw["segment"].isin(ARMS) & raw["conversion"].isin([0, 1])
    clean = raw.loc[valid].copy()
    clean["arm"] = clean["segment"].map(ARMS)
    report.append(f"Rows retained for conversion analysis: {len(clean):,}")
    report.append(f"Rows excluded for missing/invalid treatment or conversion: {len(raw)-len(clean):,}")

    if "recency" in clean.columns:
        clean["recency"] = pd.to_numeric(clean["recency"], errors="coerce")
        report.append(f"Recency outside 1--12: {int((~clean['recency'].between(1,12)).sum()):,}")
    if "history" in clean.columns:
        clean["history"] = pd.to_numeric(clean["history"], errors="coerce")
        report.append(f"Negative historical spend: {int((clean['history'] < 0).sum()):,}")
    if "spend" in clean.columns:
        report.append(f"Spend outside 0--499: {int((~clean['spend'].between(0,499)).sum()):,}")
        if "conversion" in clean.columns:
            inconsistency = ((clean["spend"] > 0) != (clean["conversion"] == 1))
            report.append(f"Conversion/spend inconsistencies: {int(inconsistency.sum()):,}")
    if "visit" in clean.columns:
        impossible = (clean["conversion"] == 1) & (clean["visit"] != 1)
        report.append(f"Conversions without visit flag: {int(impossible.sum()):,}")

    return raw, clean, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Original Hillstrom CSV file")
    parser.add_argument("--outdir", default="hillstrom_results", help="Output directory")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Could not find input CSV: {source}")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    raw, df, report = load_and_clean(source)
    df.to_csv(out / "hillstrom_cleaned.csv", index=False)

    group_rows = []
    for arm in ORDER:
        sub = df[df["arm"] == arm]
        n, x = len(sub), int(sub["conversion"].sum())
        if n == 0:
            raise ValueError(f"No valid records for arm {arm}.")
        p = x / n
        lo, hi = wilson(x, n)
        group_rows.append({
            "arm": arm, "group": LABELS[arm], "n": n, "conversions": x,
            "conversion_rate": p, "rate_percent": 100*p,
            "wilson_lower": lo, "wilson_upper": hi,
            "wilson_lower_percent": 100*lo, "wilson_upper_percent": 100*hi,
        })
    groups = pd.DataFrame(group_rows)
    groups.to_csv(out / "group_summary.csv", index=False)

    observed = groups.set_index("arm")["n"].reindex(ORDER).to_numpy()
    total = int(observed.sum())
    expected = total / 3
    srm = float(np.sum((observed - expected)**2 / expected))
    srm_p = float(chi2.sf(srm, 2))
    report += ["", "GROUP COUNTS"]
    report += [f"{r.arm}: n={r.n:,}, conversions={r.conversions:,}, rate={r.rate_percent:.6f}%"
               for r in groups.itertuples()]
    report += ["", f"Sample-ratio mismatch: chi-square(2)={srm:.6f}, p={srm_p:.6f}"]

    pairs = []
    stats = {r.arm: (int(r.conversions), int(r.n)) for r in groups.itertuples()}
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        x1, n1 = stats[a]
        x2, n2 = stats[b]
        p1, p2 = x1/n1, x2/n2
        d = p1 - p2
        se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        z = norm.ppf(0.975)
        wald = (d-z*se, d+z*se)
        newc = newcombe_ci(x1, n1, x2, n2)
        mn = mn_ci(x1, n1, x2, n2)
        pairs.append({
            "comparison": f"{a}-{b}", "p1": p1, "p2": p2,
            "difference": d, "difference_pp": 100*d,
            "relative_lift_percent": (100*d/p2) if p2 > 0 else float("nan"),
            "wald_lower_pp": 100*wald[0], "wald_upper_pp": 100*wald[1],
            "newcombe_lower_pp": 100*float(newc[0]), "newcombe_upper_pp": 100*float(newc[1]),
            "mn_lower_pp": 100*float(mn[0]), "mn_upper_pp": 100*float(mn[1]),
        })
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(out / "pairwise_effects.csv", index=False)

    # Pre-treatment covariate balance: numeric/binary fields and one-hot categories.
    numeric = [c for c in PRE_NUMERIC if c in df.columns]
    categorical = [c for c in PRE_CATEGORICAL if c in df.columns]
    balance_parts = []
    if numeric or categorical:
        base = df[numeric].apply(pd.to_numeric, errors="coerce") if numeric else pd.DataFrame(index=df.index)
        if categorical:
            dummies = pd.get_dummies(df[categorical], prefix=categorical, dtype=float)
            base = pd.concat([base, dummies], axis=1)
        rows = []
        for a, b in [("A", "C"), ("B", "C"), ("A", "B")]:
            for feature in base.columns:
                value = smd(base.loc[df.arm == a, feature], base.loc[df.arm == b, feature])
                rows.append({"comparison": f"{a}-{b}", "feature": feature, "SMD": value, "absolute_SMD": abs(value)})
        balance = pd.DataFrame(rows)
        balance.to_csv(out / "balance_smd.csv", index=False)
        report += ["", "PRE-TREATMENT COVARIATE BALANCE", f"Maximum absolute SMD: {balance.absolute_SMD.max():.6f}"]

    # Figure 1: individual conversion rates and Wilson intervals.
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=180)
    xloc = np.arange(len(groups))
    rates = groups.rate_percent.to_numpy()
    lo = groups.wilson_lower_percent.to_numpy()
    hi = groups.wilson_upper_percent.to_numpy()
    ax.errorbar(xloc, rates, yerr=np.vstack([rates-lo, hi-rates]), fmt="o", capsize=5, markersize=7)
    ax.set_xticks(xloc, ["A: Men's email", "B: Women's email", "C: No email"])
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Conversion rates by randomized group")
    ax.grid(axis="y", alpha=0.25)
    for k, row in groups.iterrows():
        ax.annotate(f"{row.rate_percent:.3f}% ({row.conversions}/{row.n})", (k, row.rate_percent),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure1_conversion_rates_wilson.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: three intervals for each pairwise difference.
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ypos = np.array([2, 1, 0], dtype=float)
    methods = [
        ("Wald", "wald_lower_pp", "wald_upper_pp", 0.18),
        ("Newcombe", "newcombe_lower_pp", "newcombe_upper_pp", 0.0),
        ("Miettinen--Nurminen", "mn_lower_pp", "mn_upper_pp", -0.18),
    ]
    for name, lower, upper, offset in methods:
        center = pair_df.difference_pp.to_numpy()
        l, u = pair_df[lower].to_numpy(), pair_df[upper].to_numpy()
        ax.errorbar(center, ypos+offset, xerr=np.vstack([center-l, u-center]), fmt="o", capsize=4, label=name)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(ypos, ["A - B", "A - C", "B - C"])
    ax.set_xlabel("Difference in conversion rate (percentage points)")
    ax.set_title("Pairwise differences with 95% confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure2_pairwise_difference_intervals.png", bbox_inches="tight")
    plt.close(fig)

    if pair_df[["mn_lower_pp", "mn_upper_pp"]].isna().any().any():
        report.append("WARNING: at least one Miettinen--Nurminen interval could not be numerically inverted; check boundary/small-sample groups and package version.")
    if pair_df["relative_lift_percent"].isna().any():
        report.append("NOTE: relative lift is undefined when the comparison denominator group's conversion rate is zero.")
    report += ["", "NOTES",
               "Exact duplicate rows are flagged but not removed automatically.",
               "Rows missing an allowed treatment label or a binary conversion outcome are excluded from the analysis CSV.",
               "No missing outcomes are imputed."]
    (out / "validation_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("Analysis complete. Files written to:", out.resolve())


if __name__ == "__main__":
    main()
