"""Train, save, load, and predict with the population-weighted XGBoost model.

Categorical Wilderness_Area/Soil_Type are kept as plain integer codes
(rather than pandas ``category`` dtype with ``enable_categorical=True``) so
that a saved model behaves identically when reloaded and predicting on a
single observation built from Streamlit widgets -- a category dtype would
otherwise need to carry the exact training-time category set around, which
is fragile across a save/load boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from forestcover.config import POPULATION_PROPORTIONS, TRAIN_PROPORTION, XGB_PARAMS
from forestcover.features import ENGINEERED_FEATURE_COLUMNS


def class_weights() -> dict[int, float]:
    """Compute per-class sample weights as population share over train share.

    :return: mapping from Cover_Type (1..7) to its sample weight.
    """
    return {k: v / TRAIN_PROPORTION for k, v in POPULATION_PROPORTIONS.items()}


def sample_weights_for(y: pd.Series) -> pd.Series:
    """Map a Cover_Type series (values 1..7) to per-row sample weights.

    :param y: series of Cover_Type values in 1..7.
    :return: series of sample weights aligned with ``y``'s index.
    """
    return y.map(class_weights())


def build_model(**overrides: object) -> xgb.XGBClassifier:
    """Construct an XGBoost classifier with the project's default hyperparameters.

    :param overrides: hyperparameters overriding the defaults in
        :data:`forestcover.config.XGB_PARAMS`.
    :return: an unfitted :class:`xgboost.XGBClassifier`.
    """
    params = {**XGB_PARAMS, **overrides}
    return xgb.XGBClassifier(**params)


def train_model(
    X: pd.DataFrame, y: pd.Series, weighted: bool = True, **overrides: object
) -> xgb.XGBClassifier:
    """Fit an XGBoost classifier on engineered features.

    :param X: engineered feature matrix (see :func:`forestcover.features.add_features`).
    :param y: Cover_Type target, values in 1..7 (internally zero-indexed for XGBoost).
    :param weighted: if True, fit with population-derived sample weights.
    :param overrides: extra hyperparameters passed to :func:`build_model`.
    :return: the fitted classifier.
    """
    model = build_model(**overrides)
    y_zero = y - 1
    weights = sample_weights_for(y) if weighted else None
    model.fit(X[ENGINEERED_FEATURE_COLUMNS], y_zero, sample_weight=weights)
    return model


def save_model(model: xgb.XGBClassifier, path: str | Path) -> None:
    """Save a fitted model to XGBoost's native JSON format.

    :param model: a fitted :class:`xgboost.XGBClassifier`.
    :param path: destination file path (``.json``).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: str | Path) -> xgb.XGBClassifier:
    """Load a model previously saved by :func:`save_model`.

    :param path: path to the saved JSON model file.
    :return: the reconstructed classifier, ready for prediction.
    :raises FileNotFoundError: if no file exists at ``path``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = (
            f"Model file not found at '{file_path}'. "
            "Run `uv run python -m forestcover.train` first."
        )
        raise FileNotFoundError(msg)
    model = xgb.XGBClassifier()
    model.load_model(str(file_path))
    return model


def predict_proba(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict per-class probabilities for one or more engineered observations.

    :param model: a fitted classifier (from :func:`train_model` or :func:`load_model`).
    :param X: engineered feature dataframe (one or more rows).
    :return: array of shape ``(n_rows, 7)`` with rows summing to 1.
    """
    return model.predict_proba(X[ENGINEERED_FEATURE_COLUMNS])


def predict_cover_type(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict the most likely Cover_Type (1..7) for one or more observations.

    :param model: a fitted classifier.
    :param X: engineered feature dataframe.
    :return: array of predicted Cover_Type values in 1..7.
    """
    probs = predict_proba(model, X)
    return probs.argmax(axis=1) + 1
