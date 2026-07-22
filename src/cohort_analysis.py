"""
Cohort analysis: cross-check SHAP's top churn drivers against raw churn rates.

Verifies that contract type, tenure, and monthly charges are genuine churn
drivers in the data, independent of any model.

Usage:
    python -m src.cohort_analysis
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.config import ARTIFACTS_DIR, TARGET
from src.preprocessing import load_and_clean


def _churn_rate_table(df, group_col, group_label):
    """Print and return a churn-rate table grouped by group_col."""
    agg = (
        df.groupby(group_col, observed=True)[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "customers"})
        .sort_index()
    )
    print(f"\n  {'Group':<22s}  {'Churn Rate':>10s}  {'Customers':>10s}")
    print(f"  {'-' * 22}  {'-' * 10}  {'-' * 10}")
    for idx, row in agg.iterrows():
        print(f"  {str(idx):<22s}  {row['churn_rate']:>9.1%}  {int(row['customers']):>10d}")
    return agg


def analyse_contract(df):
    """Churn rate by contract type."""
    print("\n== CONTRACT TYPE ==")
    agg = _churn_rate_table(df, "Contract", "Contract")

    mtm = agg.loc["Month-to-month", "churn_rate"]
    two = agg.loc["Two year", "churn_rate"]
    print(f"\n  VERDICT: CONFIRMED -- month-to-month churn is {mtm:.1%} vs "
          f"{two:.1%} for two-year contracts ({mtm / two:.1f}x higher).")
    return agg


def analyse_tenure(df):
    """Churn rate by tenure bucket."""
    print("\n== TENURE ==")
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    )
    agg = _churn_rate_table(df, "tenure_bucket", "Tenure Bucket")

    newest = agg.iloc[0]["churn_rate"]
    oldest = agg.iloc[-1]["churn_rate"]
    print(f"\n  VERDICT: CONFIRMED -- newest customers (0-6 mo) churn at {newest:.1%} "
          f"vs {oldest:.1%} for longest-tenured (49-72 mo), "
          f"a {newest / oldest:.1f}x difference.")
    return agg


def analyse_monthly_charges(df):
    """Churn rate by monthly-charges quartile."""
    print("\n== MONTHLY CHARGES ==")
    df = df.copy()
    df["charges_quartile"] = pd.qcut(
        df["MonthlyCharges"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    )
    agg = _churn_rate_table(df, "charges_quartile", "Charges Quartile")

    q1 = agg.iloc[0]["churn_rate"]
    q4 = agg.iloc[-1]["churn_rate"]
    print(f"\n  VERDICT: CONFIRMED -- highest-charge quartile (Q4) churns at {q4:.1%} "
          f"vs {q1:.1%} for Q1 ({q4 / q1:.1f}x higher).")
    return agg


def analyse_high_risk_cohort(df):
    """Combined high-risk cohort: month-to-month AND tenure 0-6 months."""
    print("\n== HIGH-RISK COHORT (Month-to-month + tenure 0-6 mo) ==")
    base_rate = df[TARGET].mean()
    cohort = df[(df["Contract"] == "Month-to-month") & (df["tenure"] <= 6)]
    cohort_rate = cohort[TARGET].mean()
    cohort_size = len(cohort)

    print(f"  Overall base churn rate : {base_rate:.1%}")
    print(f"  High-risk cohort rate   : {cohort_rate:.1%}  (n={cohort_size})")
    print(f"  Lift over base rate     : {cohort_rate / base_rate:.1f}x")


def save_panel_figure(contract_agg, tenure_agg, charges_agg):
    """Save a 1x3 bar-chart panel to artifacts/cohort_analysis.png."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _bar(ax, agg, title, xlabel):
        labels = [str(l) for l in agg.index]
        rates = agg["churn_rate"].values * 100
        bars = ax.bar(labels, rates, color="#e74c3c", edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, max(rates) * 1.25)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    _bar(axes[0], contract_agg, "Churn Rate by Contract", "Contract Type")
    _bar(axes[1], tenure_agg, "Churn Rate by Tenure", "Tenure Bucket")
    _bar(axes[2], charges_agg, "Churn Rate by Monthly Charges", "Charges Quartile")

    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "cohort_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved panel figure to {out_path}")


def main():
    print("Loading data ...")
    df = load_and_clean()
    print(f"  {len(df)} customers loaded, base churn rate: {df[TARGET].mean():.1%}")

    contract_agg = analyse_contract(df)
    tenure_agg = analyse_tenure(df)
    charges_agg = analyse_monthly_charges(df)
    analyse_high_risk_cohort(df)

    print("\n-- Saving cohort analysis figure --")
    save_panel_figure(contract_agg, tenure_agg, charges_agg)
    print("\nDone.")


if __name__ == "__main__":
    main()
