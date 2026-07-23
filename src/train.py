"""
Model comparison, hyper-parameter tuning, evaluation, and artifact saving.

Usage:
    python -m src.train
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV,
    cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    confusion_matrix, classification_report, make_scorer,
    average_precision_score,
)

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, ID_COL,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, SCORING_METRIC,
    MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    METADATA_PATH, SCORED_CSV_PATH, ARTIFACTS_DIR,
)
from src.preprocessing import load_and_clean, build_preprocessor, get_feature_names

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Try TensorFlow/Keras MLP; fall back to sklearn MLPClassifier ──
try:
    import tensorflow as tf
    from sklearn.base import BaseEstimator, ClassifierMixin

    class KerasMLP(BaseEstimator, ClassifierMixin):
        """Thin sklearn-compatible wrapper around a Keras Sequential MLP."""

        def __init__(self, epochs=30, batch_size=32, verbose=0):
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            n_features = X.shape[1]
            self.model_ = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation="relu",
                                      input_shape=(n_features,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            self.model_.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["AUC"],
            )
            self.model_.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_split=0.1,
            )
            return self

        def predict(self, X):
            prob = self.model_.predict(X, verbose=0).ravel()
            return (prob >= 0.5).astype(int)

        def predict_proba(self, X):
            prob = self.model_.predict(X, verbose=0).ravel()
            return np.column_stack([1 - prob, prob])

    MLP_CLS = KerasMLP
    MLP_LABEL = "Keras MLP"
    print("[INFO] Using TensorFlow Keras MLP.")

except Exception:
    # TensorFlow unavailable on this Python version – fall back to sklearn
    from sklearn.neural_network import MLPClassifier

    class SklearnMLP(MLPClassifier):
        """MLPClassifier with sensible defaults (TensorFlow fallback)."""
        def __init__(self, **kwargs):
            defaults = dict(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
            defaults.update(kwargs)
            super().__init__(**defaults)

    MLP_CLS = SklearnMLP
    MLP_LABEL = "sklearn MLPClassifier"
    print("[INFO] TensorFlow not available; falling back to sklearn MLPClassifier.")


def _build_imb_pipeline(preprocessor, clf):
    """imblearn Pipeline: preprocess → SMOTE → classifier (per CV fold)."""
    return ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])


def compare_models(X, y, preprocessor):
    """Stratified 5-fold CV comparison; returns leaderboard DataFrame."""
    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE),
        MLP_LABEL: MLP_CLS(),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    scoring = {"auc": SCORING_METRIC, "avg_precision": "average_precision",
               "recall": "recall"}
    rows = []

    for name, clf in classifiers.items():
        pipe = _build_imb_pipeline(preprocessor, clf)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                                n_jobs=-1, error_score="raise")
        rows.append({
            "model": name,
            "mean_auc": scores["test_auc"].mean(),
            "std_auc": scores["test_auc"].std(),
            "mean_avg_precision": scores["test_avg_precision"].mean(),
            "std_avg_precision": scores["test_avg_precision"].std(),
            "mean_recall": scores["test_recall"].mean(),
            "std_recall": scores["test_recall"].std(),
        })
        r = rows[-1]
        print(f"  {name:25s}  AUC-ROC={r['mean_auc']:.4f}+-{r['std_auc']:.4f}  "
              f"AUC-PR={r['mean_avg_precision']:.4f}+-{r['std_avg_precision']:.4f}  "
              f"Recall={r['mean_recall']:.4f}+-{r['std_recall']:.4f}")

    return pd.DataFrame(rows)


def tune_xgboost(X, y, preprocessor):
    """GridSearchCV on XGBoost inside an imblearn pipeline."""
    pipe = _build_imb_pipeline(
        preprocessor,
        XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                      random_state=RANDOM_STATE),
    )

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe, param_grid, scoring=SCORING_METRIC,
        cv=cv, n_jobs=-1, verbose=0, refit=True,
    )
    search.fit(X, y)
    print(f"\n  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")
    return search


def tune_threshold(pipeline, X_train, y_train, min_recall=0.75):
    """Choose decision threshold from out-of-fold predictions on the TRAINING set.

    Sweeps 0.05..0.95 and picks the highest threshold with recall >= min_recall.
    Falls back to the threshold that maximises F1 if no threshold reaches min_recall.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                  method="predict_proba")[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": precision_score(y_train, preds, zero_division=0),
            "recall": recall_score(y_train, preds, zero_division=0),
            "f1": f1_score(y_train, preds, zero_division=0),
        })
    sweep_df = pd.DataFrame(results)

    recall_ok = sweep_df[sweep_df["recall"] >= min_recall]
    if not recall_ok.empty:
        best_row = recall_ok.loc[recall_ok["threshold"].idxmax()]
        reason = f"highest threshold with recall >= {min_recall:.2f}"
    else:
        best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
        reason = f"no threshold reached recall >= {min_recall:.2f}; picked max-F1"

    chosen = float(best_row["threshold"])
    print(f"\n  Chosen threshold : {chosen:.2f}")
    print(f"  Reason           : {reason}")
    print(f"  OOF Precision    : {best_row['precision']:.4f}")
    print(f"  OOF Recall       : {best_row['recall']:.4f}")
    print(f"  OOF F1           : {best_row['f1']:.4f}")
    return chosen, reason


def _evaluate_at_threshold(y_true, y_prob, threshold, label):
    """Compute and print metrics at a single threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"\n  [{label}]")
    for k, v in metrics.items():
        print(f"    {k:16s}: {v:.4f}")
    print(f"    Confusion matrix:\n      {cm[0]}\n      {cm[1]}")
    return metrics, cm.tolist(), report


def evaluate_on_test(pipeline, X_test, y_test, tuned_threshold):
    """Evaluate on the held-out test set at both 0.5 and the tuned threshold."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n── Test-set evaluation ──")
    m_default, cm_default, rep_default = _evaluate_at_threshold(
        y_test, y_prob, 0.5, "threshold=0.50 (default)")
    m_tuned, cm_tuned, rep_tuned = _evaluate_at_threshold(
        y_test, y_prob, tuned_threshold, f"threshold={tuned_threshold:.2f} (tuned)")

    return {
        "default_0.5": {"metrics": m_default, "confusion_matrix": cm_default,
                        "classification_report": rep_default},
        "tuned": {"metrics": m_tuned, "confusion_matrix": cm_tuned,
                  "classification_report": rep_tuned},
    }


def main():
    # ── Load & split ──
    print("Loading data …")
    df = load_and_clean()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"  Train {X_train.shape[0]}  |  Test {X_test.shape[0]}")

    # ── Compare models ──
    print("\n── Stratified 5-fold CV comparison ──")
    preprocessor = build_preprocessor()
    leaderboard = compare_models(X_train, y_train, preprocessor)

    # ── Tune XGBoost ──
    print("\n── Tuning XGBoost (GridSearchCV) ──")
    preprocessor_tune = build_preprocessor()
    search = tune_xgboost(X_train, y_train, preprocessor_tune)

    # ── AUC lift over logistic baseline ──
    lr_auc = leaderboard.loc[
        leaderboard["model"] == "LogisticRegression", "mean_auc"
    ].values[0]
    tuned_auc = search.best_score_
    print(f"\n  AUC lift (tuned XGBoost vs Logistic): "
          f"{tuned_auc:.4f} - {lr_auc:.4f} = {tuned_auc - lr_auc:+.4f}")

    best_pipeline = search.best_estimator_

    # ── Threshold tuning on TRAINING set (no leakage) ──
    print("\n-- Decision-threshold tuning (out-of-fold on training set) --")
    chosen_threshold, threshold_reason = tune_threshold(
        best_pipeline, X_train, y_train, min_recall=0.75)

    # ── Test-set evaluation at both thresholds ──
    test_results = evaluate_on_test(
        best_pipeline, X_test, y_test, chosen_threshold)

    # ── Save artifacts ──
    print("\nSaving artifacts ...")
    fitted_preprocessor = best_pipeline.named_steps["pre"]
    fitted_model = best_pipeline.named_steps["clf"]
    feature_names = get_feature_names(fitted_preprocessor)

    joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)
    joblib.dump(fitted_model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    metadata = {
        "leaderboard": leaderboard.to_dict(orient="records"),
        "best_params": search.best_params_,
        "best_cv_auc": search.best_score_,
        "threshold": chosen_threshold,
        "threshold_reason": threshold_reason,
        "test_metrics": test_results["tuned"]["metrics"],
        "test_metrics_default_0.5": test_results["default_0.5"]["metrics"],
        "test_metrics_tuned": test_results["tuned"]["metrics"],
        "confusion_matrix": test_results["tuned"]["confusion_matrix"],
        "confusion_matrix_default_0.5": test_results["default_0.5"]["confusion_matrix"],
        "confusion_matrix_tuned": test_results["tuned"]["confusion_matrix"],
        "classification_report": test_results["tuned"]["classification_report"],
        "classification_report_default_0.5": test_results["default_0.5"]["classification_report"],
        "classification_report_tuned": test_results["tuned"]["classification_report"],
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Score every customer ──
    print("Scoring all customers ...")
    df_score = load_and_clean()
    X_all = df_score[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_all_transformed = fitted_preprocessor.transform(X_all)
    df_score["churn_probability"] = fitted_model.predict_proba(
        X_all_transformed
    )[:, 1]
    df_score.to_csv(SCORED_CSV_PATH, index=False)

    print(f"\n* All artifacts saved to {ARTIFACTS_DIR}")
    print("  - preprocessor.joblib")
    print("  - xgb_model.joblib")
    print("  - feature_names.joblib")
    print("  - metadata.json")
    print("  - scored_customers.csv")


if __name__ == "__main__":
    main()
