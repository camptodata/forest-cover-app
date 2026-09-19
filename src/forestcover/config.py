"""Central configuration: paths, class names, and model hyperparameters.

Every path is resolved relative to the repository root so the same code
works both on a local checkout and inside the Docker image, where the
project is copied to ``/app``.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

TRAIN_CSV = DATA_DIR / "train.csv"
MODEL_PATH = MODELS_DIR / "xgb_model.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------
RANDOM_SEED = 42

COVER_TYPE_NAMES: dict[int, str] = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

WILDERNESS_AREA_NAMES: dict[int, str] = {
    1: "Rawah",
    2: "Neota",
    3: "Comanche Peak",
    4: "Cache la Poudre",
}

N_SOIL_TYPES = 40
N_WILDERNESS_AREAS = 4

# Population class proportions estimated by the original team via Kaggle
# leaderboard "probe" submissions (one class predicted for every row).
# The training set itself is perfectly balanced (1/7 per class).
POPULATION_PROPORTIONS: dict[int, float] = {
    1: 0.364,
    2: 0.486,
    3: 0.062,
    4: 0.005,
    5: 0.015,
    6: 0.030,
    7: 0.036,
}

TRAIN_PROPORTION = 1.0 / 7.0

# --------------------------------------------------------------------------
# Model hyperparameters (moderate, chosen for a small, fast-to-train,
# reproducible model rather than maximum leaderboard accuracy).
# --------------------------------------------------------------------------
XGB_PARAMS: dict[str, object] = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

CV_FOLDS = 5
