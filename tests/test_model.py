"""Tests for forestcover.model: weights, train/save/load, prediction."""

from __future__ import annotations

import numpy as np
import pytest

from forestcover.config import POPULATION_PROPORTIONS, TRAIN_PROPORTION
from forestcover.data import decode_categories, load_dataset
from forestcover.features import add_features
from forestcover.model import class_weights, load_model, predict_proba, save_model, train_model


def test_class_weights_equal_population_over_train_share():
    weights = class_weights()
    for k, v in POPULATION_PROPORTIONS.items():
        assert weights[k] == pytest.approx(v / TRAIN_PROPORTION)


@pytest.fixture(scope="module")
def small_engineered_data(real_train_csv_path):
    df = load_dataset(real_train_csv_path).sample(n=200, random_state=42).reset_index(drop=True)
    X = add_features(df)
    y = df["Cover_Type"]
    return X, y


def test_train_model_small_sample(small_engineered_data):
    X, y = small_engineered_data
    model = train_model(X, y, weighted=True, n_estimators=5, max_depth=3)
    preds = model.predict(X[X.columns])
    assert len(preds) == len(X)


def test_save_load_identical_predictions(tmp_path, small_engineered_data):
    X, y = small_engineered_data
    model = train_model(X, y, weighted=True, n_estimators=5, max_depth=3)
    path = tmp_path / "model.json"
    save_model(model, path)
    loaded = load_model(path)

    original_proba = predict_proba(model, X)
    loaded_proba = predict_proba(loaded, X)
    np.testing.assert_allclose(original_proba, loaded_proba)


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model(tmp_path / "nope.json")


def test_predict_proba_rows_sum_to_one(small_engineered_data):
    X, y = small_engineered_data
    model = train_model(X, y, weighted=True, n_estimators=5, max_depth=3)
    proba = predict_proba(model, X)
    assert proba.shape == (len(X), 7)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)


def test_predict_proba_single_row(small_engineered_data):
    X, y = small_engineered_data
    model = train_model(X, y, weighted=True, n_estimators=5, max_depth=3)
    single = X.iloc[[0]]
    proba = predict_proba(model, single)
    assert proba.shape == (1, 7)
    assert proba.sum() == pytest.approx(1.0, abs=1e-5)


def test_decode_categories_used_before_features(raw_df_factory):
    df = raw_df_factory(n=5, cover_type=2, wilderness_area=2, soil_type=5)
    decoded = decode_categories(df)
    X = add_features(decoded)
    assert not X.isna().any().any()
