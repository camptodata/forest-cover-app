# Data

## Provenance

`train.csv` is the balanced 15,120-row training sample used in Kaggle's
["Forest Cover Type Prediction"](https://www.kaggle.com/c/forest-cover-type-prediction)
class competition. It is a sampled/relabelled subset of the original UCI
**Covertype** dataset:

> Blackard, J. A. and Dean, D. J. (1999). *Comparative accuracies of
> artificial neural networks and discriminant analysis in predicting forest
> cover types from cartographic variables.* Computers and Electronics in
> Agriculture 24(3):131-151.
>
> UCI Machine Learning Repository: [Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype),
> licensed CC BY 4.0.

Data originates from four wilderness areas in the Roosevelt National Forest,
northern Colorado: Rawah, Neota, Comanche Peak, and Cache la Poudre.

## Licence / attribution

CC BY 4.0. Attribution: Blackard & Dean (1999), UCI Machine Learning
Repository.

## Balance

The 15,120 rows are perfectly balanced: exactly 2,160 rows per `Cover_Type`
(1 through 7). The true population is heavily imbalanced -- see
`src/forestcover/config.py::POPULATION_PROPORTIONS`, estimated by the
original team via Kaggle leaderboard "probe" submissions. This mismatch is
the central twist the app explains and corrects for with sample weights.

## Column dictionary

| Column | Type | Description |
| --- | --- | --- |
| `Id` | int | Row identifier |
| `Elevation` | int | Elevation in meters |
| `Aspect` | int | Aspect in degrees azimuth |
| `Slope` | int | Slope in degrees |
| `Horizontal_Distance_To_Hydrology` | int | Horizontal distance to nearest surface water (m) |
| `Vertical_Distance_To_Hydrology` | int | Vertical distance to nearest surface water (m) |
| `Horizontal_Distance_To_Roadways` | int | Horizontal distance to nearest roadway (m) |
| `Hillshade_9am` | int | Hillshade index at 9am, summer solstice (0-255) |
| `Hillshade_Noon` | int | Hillshade index at noon, summer solstice (0-255) |
| `Hillshade_3pm` | int | Hillshade index at 3pm, summer solstice (0-255) |
| `Horizontal_Distance_To_Fire_Points` | int | Horizontal distance to nearest wildfire ignition point (m) |
| `Wilderness_Area1..4` | int (0/1) | One-hot wilderness area: 1 Rawah, 2 Neota, 3 Comanche Peak, 4 Cache la Poudre |
| `Soil_Type1..40` | int (0/1) | One-hot soil type (ELU code, see USFS ELU classification) |
| `Cover_Type` | int (1-7) | Target: 1 Spruce/Fir, 2 Lodgepole Pine, 3 Ponderosa Pine, 4 Cottonwood/Willow, 5 Aspen, 6 Douglas-fir, 7 Krummholz |

`forestcover.data.decode_categories` collapses the one-hot `Wilderness_Area*`
and `Soil_Type*` columns into single integer `Wilderness_Area` (1-4) and
`Soil_Type` (1-40) columns, plus human-readable `Wilderness_Name` and
`Cover_Name` columns.
