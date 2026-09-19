"""Load, validate, decode, and filter the Forest Cover Type dataset.

These are the "data importing / filtering" functions the course brief asks
to be tested: they are pure, side-effect free (besides reading a file) and
never mutate their input.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forestcover.config import (
    COVER_TYPE_NAMES,
    N_SOIL_TYPES,
    N_WILDERNESS_AREAS,
    WILDERNESS_AREA_NAMES,
)

REQUIRED_BASE_COLUMNS = [
    "Id",
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Cover_Type",
]

WILDERNESS_COLUMNS = [f"Wilderness_Area{i}" for i in range(1, N_WILDERNESS_AREAS + 1)]
SOIL_COLUMNS = [f"Soil_Type{i}" for i in range(1, N_SOIL_TYPES + 1)]

REQUIRED_COLUMNS = REQUIRED_BASE_COLUMNS + WILDERNESS_COLUMNS + SOIL_COLUMNS


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw Forest Cover Type CSV file.

    :param path: path to the CSV file.
    :return: the raw dataframe, unmodified.
    :raises FileNotFoundError: if no file exists at ``path``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Forest cover CSV not found at '{file_path}'. Did you download data/train.csv?"
        raise FileNotFoundError(msg)
    return pd.read_csv(file_path)


def validate_schema(df: pd.DataFrame) -> None:
    """Check that a dataframe matches the expected Forest Cover Type schema.

    :param df: dataframe to validate.
    :return: None if the dataframe is valid.
    :raises ValueError: with a specific message on the first violation found.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"Missing required column(s): {missing}"
        raise ValueError(msg)

    if df[REQUIRED_COLUMNS].isna().any().any():
        bad_cols = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isna().any()].tolist()
        msg = f"Found NaN values in column(s): {bad_cols}"
        raise ValueError(msg)

    non_integer = [c for c in REQUIRED_COLUMNS if not pd.api.types.is_integer_dtype(df[c])]
    if non_integer:
        msg = f"Column(s) expected to be integer-typed are not: {non_integer}"
        raise ValueError(msg)

    if not df["Cover_Type"].isin(range(1, 8)).all():
        msg = "Column 'Cover_Type' must only contain integers in 1..7."
        raise ValueError(msg)

    wilderness_sum = df[WILDERNESS_COLUMNS].sum(axis=1)
    if not (wilderness_sum == 1).all():
        bad_rows = wilderness_sum[wilderness_sum != 1].index.tolist()
        msg = f"Row(s) do not have exactly one Wilderness_Area flag set: {bad_rows[:10]}"
        raise ValueError(msg)

    soil_sum = df[SOIL_COLUMNS].sum(axis=1)
    if not (soil_sum == 1).all():
        bad_rows = soil_sum[soil_sum != 1].index.tolist()
        msg = f"Row(s) do not have exactly one Soil_Type flag set: {bad_rows[:10]}"
        raise ValueError(msg)


def decode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the one-hot Wilderness/Soil columns into readable categories.

    Adds integer ``Wilderness_Area`` (1..4) and ``Soil_Type`` (1..40) columns
    plus human-readable ``Wilderness_Name`` and ``Cover_Name`` columns. The
    original one-hot columns are dropped. The input is never mutated.

    :param df: dataframe with one-hot Wilderness_Area*/Soil_Type* columns.
    :return: a new dataframe with decoded categorical columns.
    """
    out = df.copy()

    out["Wilderness_Area"] = out[WILDERNESS_COLUMNS].to_numpy() @ range(1, N_WILDERNESS_AREAS + 1)
    out["Soil_Type"] = out[SOIL_COLUMNS].to_numpy() @ range(1, N_SOIL_TYPES + 1)
    out = out.drop(columns=WILDERNESS_COLUMNS + SOIL_COLUMNS)

    out["Wilderness_Name"] = out["Wilderness_Area"].map(WILDERNESS_AREA_NAMES)
    if "Cover_Type" in out.columns:
        out["Cover_Name"] = out["Cover_Type"].map(COVER_TYPE_NAMES)

    return out


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load, validate, and decode the Forest Cover Type dataset in one call.

    This is the function the Streamlit app caches with ``st.cache_data``.

    :param path: path to the raw CSV file.
    :return: the validated, decoded dataframe.
    :raises FileNotFoundError: if the file does not exist.
    :raises ValueError: if the schema is invalid.
    """
    df = load_raw(path)
    validate_schema(df)
    return decode_categories(df)


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Compute the row count per Cover_Type value.

    :param df: dataframe containing a ``Cover_Type`` column.
    :return: a series indexed by Cover_Type (1..7) with row counts, sorted by index.
    """
    return df["Cover_Type"].value_counts().sort_index()


def filter_observations(
    df: pd.DataFrame,
    cover_types: list[int] | None = None,
    wilderness_areas: list[int] | None = None,
    elevation_range: tuple[int, int] | None = None,
    slope_range: tuple[int, int] | None = None,
    soil_types: list[int] | None = None,
) -> pd.DataFrame:
    """Filter observations by cover type, wilderness area, elevation, slope, and soil type.

    Every criterion is independent and combined with logical AND. ``None``
    means "no filter" on that criterion; an empty list means "select no
    rows". Ranges are inclusive on both ends. The input dataframe is never
    mutated and a new copy is always returned.

    :param df: decoded dataframe (see :func:`decode_categories`).
    :param cover_types: list of Cover_Type values to keep, or None.
    :param wilderness_areas: list of Wilderness_Area values to keep, or None.
    :param elevation_range: inclusive ``(min, max)`` elevation bounds, or None.
    :param slope_range: inclusive ``(min, max)`` slope bounds, or None.
    :param soil_types: list of Soil_Type values to keep, or None.
    :return: a filtered copy of ``df``.
    :raises ValueError: if a range's minimum exceeds its maximum.
    """
    mask = pd.Series(True, index=df.index)

    if cover_types is not None:
        mask &= df["Cover_Type"].isin(cover_types)

    if wilderness_areas is not None:
        mask &= df["Wilderness_Area"].isin(wilderness_areas)

    if soil_types is not None:
        mask &= df["Soil_Type"].isin(soil_types)

    if elevation_range is not None:
        lo, hi = elevation_range
        if lo > hi:
            msg = f"Invalid elevation_range: min ({lo}) > max ({hi})."
            raise ValueError(msg)
        mask &= df["Elevation"].between(lo, hi)

    if slope_range is not None:
        lo, hi = slope_range
        if lo > hi:
            msg = f"Invalid slope_range: min ({lo}) > max ({hi})."
            raise ValueError(msg)
        mask &= df["Slope"].between(lo, hi)

    return df.loc[mask].copy()
