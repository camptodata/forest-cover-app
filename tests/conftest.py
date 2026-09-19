"""Shared pytest fixtures: synthetic dataframe factories for forestcover tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forestcover.config import N_SOIL_TYPES, N_WILDERNESS_AREAS


def _make_raw_rows(
    n: int,
    cover_type: int = 1,
    wilderness_area: int = 1,
    soil_type: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wilderness = np.zeros((n, N_WILDERNESS_AREAS), dtype=int)
    wilderness[:, wilderness_area - 1] = 1
    soil = np.zeros((n, N_SOIL_TYPES), dtype=int)
    soil[:, soil_type - 1] = 1

    data = {
        "Id": np.arange(1, n + 1),
        "Elevation": rng.integers(1900, 3800, n),
        "Aspect": rng.integers(0, 360, n),
        "Slope": rng.integers(0, 60, n),
        "Horizontal_Distance_To_Hydrology": rng.integers(0, 1400, n),
        "Vertical_Distance_To_Hydrology": rng.integers(-200, 700, n),
        "Horizontal_Distance_To_Roadways": rng.integers(0, 7000, n),
        "Hillshade_9am": rng.integers(0, 255, n),
        "Hillshade_Noon": rng.integers(0, 255, n),
        "Hillshade_3pm": rng.integers(0, 255, n),
        "Horizontal_Distance_To_Fire_Points": rng.integers(0, 7000, n),
    }
    df = pd.DataFrame(data)
    for i in range(N_WILDERNESS_AREAS):
        df[f"Wilderness_Area{i + 1}"] = wilderness[:, i]
    for i in range(N_SOIL_TYPES):
        df[f"Soil_Type{i + 1}"] = soil[:, i]
    df["Cover_Type"] = cover_type
    return df


@pytest.fixture
def raw_df_factory():
    """Return a factory building synthetic raw (one-hot) Forest Cover dataframes."""
    return _make_raw_rows


@pytest.fixture
def balanced_raw_df(raw_df_factory) -> pd.DataFrame:
    """Build a small synthetic dataframe with all 7 cover types represented."""
    frames = [
        raw_df_factory(
            n=10, cover_type=ct, wilderness_area=((ct - 1) % 4) + 1, soil_type=ct, seed=ct
        )
        for ct in range(1, 8)
    ]
    df = pd.concat(frames, ignore_index=True)
    df["Id"] = np.arange(1, len(df) + 1)
    return df


@pytest.fixture(scope="module")
def real_train_csv_path():
    """Path to the real committed training CSV, for a few end-to-end tests."""
    from forestcover.config import TRAIN_CSV

    return TRAIN_CSV
