"""Feature engineering, ported from the original notebook's ``add_features``.

The port keeps every engineered feature from CELL 22 of the original
Colab notebook (see ``reference/notebook_dump.md``) but cleans up naming and
typing:

* ``Soil_Type`` / ``Wilderness_Area`` are integer codes (not pandas
  ``category`` dtype) so that saving/loading the model and predicting on a
  single row stay simple and dtype-stable.
* ``Water Elevation`` (with a space, in the original) is renamed
  ``Water_Elevation``.
* The function accepts either the raw one-hot dataframe or the already
  decoded one (with ``Wilderness_Area``/``Soil_Type`` integer columns), so
  it can be reused both at training time (from :func:`forestcover.data.load_dataset`)
  and at prediction time (single-row input from the app).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ELU (Ecological Land Unit) code per soil type, from the original dataset
# documentation (Blackard & Dean, 1999).
ELU_CODE: dict[int, int] = {
    1: 2702, 2: 2703, 3: 2704, 4: 2705, 5: 2706, 6: 2717, 7: 3501, 8: 3502,
    9: 4201, 10: 4703, 11: 4704, 12: 4744, 13: 4758, 14: 5101, 15: 5151,
    16: 6101, 17: 6102, 18: 6731, 19: 7101, 20: 7102, 21: 7103, 22: 7201,
    23: 7202, 24: 7700, 25: 7701, 26: 7702, 27: 7709, 28: 7710, 29: 7745,
    30: 7746, 31: 7755, 32: 7756, 33: 7757, 34: 7790, 35: 8703, 36: 8707,
    37: 8708, 38: 8771, 39: 8772, 40: 8776,
}  # fmt: skip

_NO_DESC = [7, 8, 14, 15, 16, 17, 19, 20, 21, 23, 35]
_STONY = [6, 12]
_VERY_STONY = [2, 9, 18, 26]
_EXTREMELY_STONY = [1, 22, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
_RUBBLY = [3, 4, 5, 10, 11, 13]

_SURFACE_COVER: dict[int, int] = {i: 0 for i in _NO_DESC}
_SURFACE_COVER.update({i: 1 for i in _STONY})
_SURFACE_COVER.update({i: 2 for i in _VERY_STONY})
_SURFACE_COVER.update({i: 3 for i in _EXTREMELY_STONY})
_SURFACE_COVER.update({i: 4 for i in _RUBBLY})
_SURFACE_COVER[0] = 0

_STONES = [1, 2, 6, 9, 12, 18, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
_BOULDERS = [22]
_RUBBLE = [3, 4, 5, 10, 11, 13]

_ROCK_SIZE: dict[int, int] = {i: 0 for i in _NO_DESC}
_ROCK_SIZE.update({i: 1 for i in _STONES})
_ROCK_SIZE.update({i: 2 for i in _BOULDERS})
_ROCK_SIZE.update({i: 3 for i in _RUBBLE})
_ROCK_SIZE[0] = 0

#: Feature columns produced by :func:`add_features`, in a fixed order. This
#: is also the column order the model is trained and predicted on.
ENGINEERED_FEATURE_COLUMNS: list[str] = [
    "Elevation",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Soil_Type",
    "Wilderness_Area",
    "Aspect_sin",
    "Aspect_cos",
    "Water_distance",
    "Average_shadow",
    "Water_above",
    "Climatic_zone",
    "Geologic_zone",
    "Horizontal_Distance_To_Roadways_Log",
    "Water_Elevation",
    "Hydro_Fire_1",
    "Hydro_Fire_2",
    "Hydro_Road_1",
    "Hydro_Road_2",
    "Fire_Road_1",
    "Fire_Road_2",
    "EHiElv",
    "EHDtH",
    "Elev_3Horiz",
    "Elev_Road_1",
    "Elev_Road_2",
    "Elev_Fire_1",
    "Elev_Fire_2",
    "Surface_Cover",
    "Rock_Size",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer geospatial, solar, and soil-derived features.

    Expects a dataframe already decoded by :func:`forestcover.data.decode_categories`
    (i.e. with integer ``Wilderness_Area`` and ``Soil_Type`` columns rather
    than one-hot columns). Deterministic and side-effect free: the input is
    never mutated.

    :param df: decoded dataframe with at least the raw numeric columns plus
        integer ``Soil_Type`` and ``Wilderness_Area``.
    :return: a new dataframe with the engineered feature columns, in the
        order given by :data:`ENGINEERED_FEATURE_COLUMNS`.
    :raises KeyError: if a required input column is missing.
    """
    out = df.copy()

    out["Aspect_sin"] = np.sin(out["Aspect"] * (np.pi / 180))
    out["Aspect_cos"] = np.cos(out["Aspect"] * (np.pi / 180))

    out["Water_distance"] = np.sqrt(
        out["Horizontal_Distance_To_Hydrology"] ** 2 + out["Vertical_Distance_To_Hydrology"] ** 2
    )
    out["Average_shadow"] = out[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].mean(axis=1)
    out["Water_above"] = (out["Vertical_Distance_To_Hydrology"] < 0).astype(int)

    elu_code = out["Soil_Type"].map(ELU_CODE).fillna(0).astype(int)
    out["Climatic_zone"] = elu_code.astype(str).str.zfill(1).str[0].astype(int)
    out["Geologic_zone"] = elu_code.astype(str).str.zfill(2).str[1].astype(int)

    out["Horizontal_Distance_To_Roadways_Log"] = np.log1p(out["Horizontal_Distance_To_Roadways"])
    out["Water_Elevation"] = out["Elevation"] - out["Vertical_Distance_To_Hydrology"]
    out["Hydro_Fire_1"] = (
        out["Horizontal_Distance_To_Hydrology"] + out["Horizontal_Distance_To_Fire_Points"]
    )
    out["Hydro_Fire_2"] = (
        out["Horizontal_Distance_To_Hydrology"] - out["Horizontal_Distance_To_Fire_Points"]
    ).abs()
    out["Hydro_Road_1"] = (
        out["Horizontal_Distance_To_Hydrology"] + out["Horizontal_Distance_To_Roadways"]
    ).abs()
    out["Hydro_Road_2"] = (
        out["Horizontal_Distance_To_Hydrology"] - out["Horizontal_Distance_To_Roadways"]
    ).abs()
    out["Fire_Road_1"] = (
        out["Horizontal_Distance_To_Fire_Points"] + out["Horizontal_Distance_To_Roadways"]
    ).abs()
    out["Fire_Road_2"] = (
        out["Horizontal_Distance_To_Fire_Points"] - out["Horizontal_Distance_To_Roadways"]
    ).abs()
    out["EHiElv"] = out["Horizontal_Distance_To_Roadways"] * out["Elevation"]
    out["EHDtH"] = out["Elevation"] - out["Horizontal_Distance_To_Hydrology"] * 0.2
    out["Elev_3Horiz"] = (
        out["Elevation"]
        + out["Horizontal_Distance_To_Roadways"]
        + out["Horizontal_Distance_To_Fire_Points"]
        + out["Horizontal_Distance_To_Hydrology"]
    )
    out["Elev_Road_1"] = out["Elevation"] + out["Horizontal_Distance_To_Roadways"]
    out["Elev_Road_2"] = out["Elevation"] - out["Horizontal_Distance_To_Roadways"]
    out["Elev_Fire_1"] = out["Elevation"] + out["Horizontal_Distance_To_Fire_Points"]
    out["Elev_Fire_2"] = out["Elevation"] - out["Horizontal_Distance_To_Fire_Points"]

    out["Surface_Cover"] = out["Soil_Type"].map(_SURFACE_COVER).fillna(0).astype(int)
    out["Rock_Size"] = out["Soil_Type"].map(_ROCK_SIZE).fillna(0).astype(int)

    return out[ENGINEERED_FEATURE_COLUMNS]
