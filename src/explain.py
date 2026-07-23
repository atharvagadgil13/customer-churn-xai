"""
Explainability: SHAP (global + per-customer) and LIME cross-check.

Usage:
    python -m src.explain
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.config import (
    MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    ARTIFACTS_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
)
from src.preprocessing import load_and_clean


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, preprocessor, feature_names


def shap_global(model, X_transformed, feature_names):
    """Global SHAP bar + beeswarm plots, saved to artifacts/."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names

    # Bar plot
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax_bar)
    fig_bar.tight_layout()
    fig_bar.savefig(ARTIFACTS_DIR / "shap_global_bar.png", dpi=150)
    plt.close(fig_bar)

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False, plot_size=None)
    fig_bee = plt.gcf()
    fig_bee.tight_layout()
    fig_bee.savefig(ARTIFACTS_DIR / "shap_beeswarm.png", dpi=150)
    plt.close(fig_bee)

    print("  Saved shap_global_bar.png and shap_beeswarm.png")
    return explainer, shap_values


def shap_waterfall(shap_values, idx=0):
    """Per-customer waterfall plot for a single observation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "shap_waterfall_sample.png", dpi=150)
    plt.close(fig)
    print(f"  Saved shap_waterfall_sample.png (customer index={idx})")


def lime_explanation(model, X_transformed, feature_names, idx=0):
    """LIME explanation for one customer as a cross-check."""
    explainer = LimeTabularExplainer(
        X_transformed,
        feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        mode="classification",
    )
    exp = explainer.explain_instance(
        X_transformed[idx],
        model.predict_proba,
        num_features=12,
    )
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "lime_sample.png", dpi=150)
    plt.close(fig)
    print(f"  Saved lime_sample.png (customer index={idx})")
    return exp


def main():
    print("Loading artifacts …")
    model, preprocessor, feature_names = load_artifacts()

    df = load_and_clean()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_transformed = preprocessor.transform(X)

    if not isinstance(X_transformed, np.ndarray):
        X_transformed = X_transformed.toarray()

    print("\n── SHAP global importance ──")
    explainer, shap_values = shap_global(model, X_transformed, feature_names)

    print("\n── SHAP per-customer waterfall ──")
    shap_waterfall(shap_values, idx=0)

    print("\n── LIME cross-check ──")
    lime_explanation(model, X_transformed, feature_names, idx=0)

    print(f"\n✓ Explanation plots saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
