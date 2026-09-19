"""Tests for forestcover.features.add_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forestcover.data import decode_categories
from forestcover.features import ENGINEERED_FEATURE_COLUMNS, add_features


@pytest.fixture
def decoded_df(balanced_raw_df):
    return decode_categories(balanced_raw_df)


def test_add_features_adds_expected_columns(decoded_df):
    out = add_features(decoded_df)
    assert list(out.columns) == ENGINEERED_FEATURE_COLUMNS


def test_add_features_is_deterministic(decoded_df):
    out1 = add_features(decoded_df)
    out2 = add_features(decoded_df)
    pd.testing.assert_frame_equal(out1, out2)


def test_add_features_no_nan_or_inf(decoded_df):
    out = add_features(decoded_df)
    numeric = out.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any()
    assert not np.isinf(numeric.to_numpy()).any()


def test_add_features_does_not_mutate_input(decoded_df):
    before = decoded_df.copy()
    add_features(decoded_df)
    pd.testing.assert_frame_equal(decoded_df, before)


def test_add_features_single_row(decoded_df):
    row = decoded_df.iloc[[0]]
    out = add_features(row)
    assert len(out) == 1
    assert list(out.columns) == ENGINEERED_FEATURE_COLUMNS


def test_add_features_known_values(raw_df_factory):
    df = raw_df_factory(n=1, cover_type=1, wilderness_area=1, soil_type=1)
    decoded = decode_categories(df)
    decoded.loc[0, "Aspect"] = 0
    decoded.loc[0, "Elevation"] = 3000
    decoded.loc[0, "Vertical_Distance_To_Hydrology"] = 100
    decoded.loc[0, "Horizontal_Distance_To_Hydrology"] = 0
    out = add_features(decoded)
    assert out.loc[0, "Aspect_sin"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc[0, "Aspect_cos"] == pytest.approx(1.0, abs=1e-9)
    assert out.loc[0, "Water_Elevation"] == 2900
    assert out.loc[0, "Water_above"] == 0
