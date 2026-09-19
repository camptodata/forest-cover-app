"""
CLI entry point: stratified 5-fold CV evaluation, written to models/metrics.json.

Compares two models:

* **baseline** -- default-hyperparameter XGBoost on the raw one-hot
  features, without sample weights (this reproduces the ~0.857 plain
  accuracy reported in the original notebook).
* **final** -- the population-weighted XGBoost model on the engineered
  feature set (see :mod:`forestcover.features`), matching what
  :mod:`forestcover.train` saves for the app.

Both plain accuracy and population-weighted accuracy (using
``sample_weight`` in :func:`sklearn.metrics.accuracy_score`, the same proxy
for Kaggle leaderboard performance the original team used) are reported,
along with per-class recall and confusion matrices. The output JSON has no
timestamps so re-running on unchanged data and code is diff-stable.

Usage::

    uv run python -m forestcover.evaluate
"""

from __future__ import annotations

import json
from importlib.metadata import version

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold

from forestcover.config import (
    CV_FOLDS,
    METRICS_PATH,
    RANDOM_SEED,
    TRAIN_CSV,
    XGB_PARAMS,
)
from forestcover.data import REQUIRED_COLUMNS, load_dataset, load_raw, validate_schema
from forestcover.features import ENGINEERED_FEATURE_COLUMNS, add_features
from forestcover.model import class_weights, sample_weights_for, train_model


def _cv_evaluate(X: pd.DataFrame, y: pd.Series, weighted: bool, feature_cols: list[str]) -> dict:
    """
    Run stratified k-fold CV and aggregate plain/weighted metrics.

    :param X: feature matrix (raw or engineered).
    :param y: Cover_Type target, values in 1..7.
    :param weighted: whether to fit with population-derived sample weights.
    :param feature_cols: ordered list of feature column names to use.
    :return: dict with plain_accuracy, weighted_accuracy, per_class_recall,
        confusion_matrix, and confusion_matrix_normalized.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    weights = sample_weights_for(y)

    plain_scores = []
    weighted_scores = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_val = weights.iloc[val_idx]

        model = _fit(X_tr, y_tr, weighted, feature_cols)

        preds = model.predict(X_val[feature_cols]) + 1

        plain_scores.append(accuracy_score(y_val, preds))
        weighted_scores.append(accuracy_score(y_val, preds, sample_weight=w_val))
        all_true.extend(y_val.tolist())
        all_pred.extend(preds.tolist())

    labels = list(range(1, 8))
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    per_class_recall = recall_score(all_true, all_pred, labels=labels, average=None)

    return {
        "plain_accuracy_mean": float(np.mean(plain_scores)),
        "plain_accuracy_std": float(np.std(plain_scores)),
        "weighted_accuracy_mean": float(np.mean(weighted_scores)),
        "weighted_accuracy_std": float(np.std(weighted_scores)),
        "per_class_recall": {
            str(c): float(r) for c, r in zip(labels, per_class_recall, strict=True)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "confusion_matrix_labels": labels,
    }


def _fit(
    X_tr: pd.DataFrame, y_tr: pd.Series, weighted: bool, feature_cols: list[str]
) -> xgb.XGBClassifier:
    """
    Fit one fold's model with either baseline or project hyperparameters.

    :param X_tr: training feature slice.
    :param y_tr: training target slice, values in 1..7.
    :param weighted: whether to use population-derived sample weights.
    :param feature_cols: feature columns to select from ``X_tr``.
    :return: the fitted classifier.
    """
    y_zero = y_tr - 1
    if weighted:
        params = XGB_PARAMS
        weights = sample_weights_for(y_tr)
    else:
        params = {"random_state": RANDOM_SEED, "tree_method": "hist", "n_jobs": -1}
        weights = None
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr[feature_cols], y_zero, sample_weight=weights)
    return model


def main() -> None:
    """Run baseline vs. final CV evaluation and write models/metrics.json."""
    print(f"Loading dataset from {TRAIN_CSV} ...")
    raw = load_raw(TRAIN_CSV)
    validate_schema(raw)
    y = raw["Cover_Type"]

    raw_feature_cols = [c for c in REQUIRED_COLUMNS if c not in ("Id", "Cover_Type")]
    X_raw = raw[raw_feature_cols]

    print("Evaluating baseline (raw features, unweighted, default hyperparameters) ...")
    baseline = _cv_evaluate(X_raw, y, weighted=False, feature_cols=raw_feature_cols)

    df = load_dataset(TRAIN_CSV)
    X_eng = add_features(df)
    print("Evaluating final model (engineered features, population-weighted) ...")
    final = _cv_evaluate(X_eng, y, weighted=True, feature_cols=ENGINEERED_FEATURE_COLUMNS)

    print("Fitting final model on full data for feature importances ...")
    final_model = train_model(X_eng, y, weighted=True)
    importances = dict(
        zip(ENGINEERED_FEATURE_COLUMNS, final_model.feature_importances_.tolist(), strict=True)
    )
    top_importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:20])

    metrics = {
        "cv_folds": CV_FOLDS,
        "random_seed": RANDOM_SEED,
        "class_weights": {str(k): v for k, v in class_weights().items()},
        "baseline": baseline,
        "final": final,
        "feature_importance_gain_top20": top_importances,
        "library_versions": {
            "python_pandas": version("pandas"),
            "numpy": version("numpy"),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote metrics to {METRICS_PATH}")
    print(
        f"Baseline plain accuracy: {baseline['plain_accuracy_mean']:.4f} | "
        f"weighted: {baseline['weighted_accuracy_mean']:.4f}"
    )
    print(
        f"Final plain accuracy: {final['plain_accuracy_mean']:.4f} | "
        f"weighted: {final['weighted_accuracy_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
