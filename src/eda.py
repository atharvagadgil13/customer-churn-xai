"""
Exploratory Data Analysis: summary statistics and churn-driver plots.

Usage:
    python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import ARTIFACTS_DIR, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import load_and_clean


def summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n── Shape ──")
    print(f"  {df.shape[0]} rows × {df.shape[1]} columns")

    print(f"\n── Churn distribution ──")
    counts = df[TARGET].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count}  ({pct:.1f}%)")

    print(f"\n── Numeric summary ──")
    print(df[NUMERIC_FEATURES].describe().round(2).to_string())

    print(f"\n── Missing values ──")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None (after cleaning)")
    else:
        print(missing.to_string())


def churn_driver_plots(df: pd.DataFrame):
    """Generate key churn-driver visualisations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Churn rate by Contract type
    ct = df.groupby("Contract")[TARGET].mean().sort_values(ascending=False)
    ct.plot.bar(ax=axes[0, 0], color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[0, 0].set_title("Churn Rate by Contract")
    axes[0, 0].set_ylabel("Churn Rate")
    axes[0, 0].tick_params(axis="x", rotation=0)

    # 2. Churn rate by Internet Service
    it = df.groupby("InternetService")[TARGET].mean().sort_values(ascending=False)
    it.plot.bar(ax=axes[0, 1], color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[0, 1].set_title("Churn Rate by Internet Service")
    axes[0, 1].set_ylabel("Churn Rate")
    axes[0, 1].tick_params(axis="x", rotation=0)

    # 3. Monthly charges distribution by churn
    for label, grp in df.groupby(TARGET):
        tag = "Churn" if label == 1 else "No Churn"
        axes[1, 0].hist(grp["MonthlyCharges"], bins=30, alpha=0.5, label=tag)
    axes[1, 0].set_title("Monthly Charges by Churn")
    axes[1, 0].set_xlabel("Monthly Charges")
    axes[1, 0].legend()

    # 4. Tenure distribution by churn
    for label, grp in df.groupby(TARGET):
        tag = "Churn" if label == 1 else "No Churn"
        axes[1, 1].hist(grp["tenure"], bins=30, alpha=0.5, label=tag)
    axes[1, 1].set_title("Tenure by Churn")
    axes[1, 1].set_xlabel("Tenure (months)")
    axes[1, 1].legend()

    fig.suptitle("Churn Driver Analysis", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "eda_churn_drivers.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved eda_churn_drivers.png to {ARTIFACTS_DIR}")


def main():
    df = load_and_clean()
    summary_stats(df)
    churn_driver_plots(df)
    print("\n✓ EDA complete.")


if __name__ == "__main__":
    main()
