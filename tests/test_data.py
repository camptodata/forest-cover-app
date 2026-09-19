"""Tests for forestcover.data: loading, validating, decoding, filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from forestcover.data import (
    REQUIRED_COLUMNS,
    class_distribution,
    decode_categories,
    filter_observations,
    load_dataset,
    load_raw,
    validate_schema,
)


# --------------------------------------------------------------------------
# load_raw
# --------------------------------------------------------------------------
def test_load_raw_real_file_shape_and_columns(real_train_csv_path):
    df = load_raw(real_train_csv_path)
    assert df.shape[0] == 15120
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_load_raw_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_raw(missing)


def test_load_raw_roundtrip(tmp_path, balanced_raw_df):
    p = tmp_path / "synthetic.csv"
    balanced_raw_df.to_csv(p, index=False)
    loaded = load_raw(p)
    pd.testing.assert_frame_equal(loaded, balanced_raw_df)


# --------------------------------------------------------------------------
# validate_schema
# --------------------------------------------------------------------------
def test_validate_schema_accepts_valid_frame(balanced_raw_df):
    validate_schema(balanced_raw_df)  # should not raise


def test_validate_schema_real_file(real_train_csv_path):
    df = load_raw(real_train_csv_path)
    validate_schema(df)  # should not raise


def test_validate_schema_missing_column(balanced_raw_df):
    bad = balanced_raw_df.drop(columns=["Elevation"])
    with pytest.raises(ValueError, match="Missing required column"):
        validate_schema(bad)


def test_validate_schema_nan_value(balanced_raw_df):
    bad = balanced_raw_df.copy()
    bad.loc[0, "Elevation"] = None
    with pytest.raises(ValueError, match="NaN"):
        validate_schema(bad)


def test_validate_schema_bad_cover_type(balanced_raw_df):
    bad = balanced_raw_df.copy()
    bad.loc[0, "Cover_Type"] = 9
    with pytest.raises(ValueError, match="Cover_Type"):
        validate_schema(bad)


def test_validate_schema_two_wilderness_flags(balanced_raw_df):
    bad = balanced_raw_df.copy()
    bad.loc[0, "Wilderness_Area2"] = 1
    bad.loc[0, "Wilderness_Area1"] = 1
    with pytest.raises(ValueError, match="Wilderness_Area"):
        validate_schema(bad)


def test_validate_schema_zero_soil_flags(balanced_raw_df):
    bad = balanced_raw_df.copy()
    soil_cols = [c for c in bad.columns if c.startswith("Soil_Type")]
    bad.loc[0, soil_cols] = 0
    with pytest.raises(ValueError, match="Soil_Type"):
        validate_schema(bad)


# --------------------------------------------------------------------------
# decode_categories
# --------------------------------------------------------------------------
def test_decode_categories_correctness(raw_df_factory):
    df = raw_df_factory(n=3, cover_type=5, wilderness_area=3, soil_type=17)
    decoded = decode_categories(df)
    assert (decoded["Wilderness_Area"] == 3).all()
    assert (decoded["Soil_Type"] == 17).all()
    assert (decoded["Wilderness_Name"] == "Comanche Peak").all()
    assert (decoded["Cover_Name"] == "Aspen").all()
    for c in decoded.columns:
        assert not c.startswith("Wilderness_Area1") or c == "Wilderness_Area"
    assert "Soil_Type1" not in decoded.columns


def test_decode_categories_does_not_mutate_input(balanced_raw_df):
    before = balanced_raw_df.copy()
    decode_categories(balanced_raw_df)
    pd.testing.assert_frame_equal(balanced_raw_df, before)


# --------------------------------------------------------------------------
# load_dataset
# --------------------------------------------------------------------------
def test_load_dataset_real_file(real_train_csv_path):
    df = load_dataset(real_train_csv_path)
    assert "Wilderness_Area" in df.columns
    assert "Soil_Type" in df.columns
    assert "Cover_Name" in df.columns
    assert df.shape[0] == 15120


# --------------------------------------------------------------------------
# class_distribution
# --------------------------------------------------------------------------
def test_class_distribution_real_file_is_balanced(real_train_csv_path):
    df = load_dataset(real_train_csv_path)
    dist = class_distribution(df)
    assert list(dist.index) == list(range(1, 8))
    assert (dist == 2160).all()


# --------------------------------------------------------------------------
# filter_observations
# --------------------------------------------------------------------------
@pytest.fixture
def decoded_df(balanced_raw_df):
    return decode_categories(balanced_raw_df)


def test_filter_none_is_passthrough(decoded_df):
    out = filter_observations(decoded_df)
    pd.testing.assert_frame_equal(out, decoded_df)


def test_filter_by_cover_types(decoded_df):
    out = filter_observations(decoded_df, cover_types=[1, 2])
    assert set(out["Cover_Type"].unique()) <= {1, 2}
    assert len(out) == 20


def test_filter_empty_list_selects_nothing(decoded_df):
    out = filter_observations(decoded_df, cover_types=[])
    assert len(out) == 0


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("wilderness_areas", [1]),
        ("soil_types", [3]),
    ],
)
def test_filter_single_criteria(decoded_df, kwarg, value):
    out = filter_observations(decoded_df, **{kwarg: value})
    col = "Wilderness_Area" if kwarg == "wilderness_areas" else "Soil_Type"
    assert set(out[col].unique()) <= set(value)


def test_filter_elevation_range_inclusive(decoded_df):
    lo, hi = int(decoded_df["Elevation"].min()), int(decoded_df["Elevation"].median())
    out = filter_observations(decoded_df, elevation_range=(lo, hi))
    assert (out["Elevation"] >= lo).all()
    assert (out["Elevation"] <= hi).all()
    assert (out["Elevation"] == lo).any() or lo not in decoded_df["Elevation"].values


def test_filter_slope_range_inclusive(decoded_df):
    lo, hi = int(decoded_df["Slope"].min()), int(decoded_df["Slope"].min())
    out = filter_observations(decoded_df, slope_range=(lo, hi))
    assert (out["Slope"] == lo).all()


def test_filter_invalid_range_raises(decoded_df):
    with pytest.raises(ValueError, match="Invalid elevation_range"):
        filter_observations(decoded_df, elevation_range=(100, 0))
    with pytest.raises(ValueError, match="Invalid slope_range"):
        filter_observations(decoded_df, slope_range=(50, 10))


def test_filter_combined_criteria(decoded_df):
    out = filter_observations(
        decoded_df,
        cover_types=[1, 2, 3],
        wilderness_areas=[1, 2, 3],
        elevation_range=(0, 100000),
    )
    assert set(out["Cover_Type"].unique()) <= {1, 2, 3}
    assert set(out["Wilderness_Area"].unique()) <= {1, 2, 3}


def test_filter_does_not_mutate_input(decoded_df):
    before = decoded_df.copy()
    filter_observations(decoded_df, cover_types=[1])
    pd.testing.assert_frame_equal(decoded_df, before)


def test_filter_result_is_a_copy_not_a_view(decoded_df):
    out = filter_observations(decoded_df, cover_types=[1])
    out.loc[out.index[0], "Elevation"] = -999
    assert decoded_df.loc[out.index[0], "Elevation"] != -999
