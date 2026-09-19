# Forest Cover Type Classifier

[![CI](https://github.com/camptodata/forest-cover-app/actions/workflows/ci.yml/badge.svg)](https://github.com/camptodata/forest-cover-app/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A Streamlit app that predicts forest cover type from cartographic variables,
built around a population-weighted XGBoost model, containerized and tested
end to end. This is Nick Hartmann's individual final project for HEC Paris'
*Tooling for the Data Scientist* course, packaging a Machine Learning II
group project.

## What this is

The data comes from a Kaggle class competition for the Machine Learning II
course, a variant of the public
["Forest Cover Type Prediction"](https://www.kaggle.com/c/forest-cover-type-prediction)
competition. It provides a 15,120-row training sample, **perfectly
balanced** across the 7 cover-type classes (2,160 rows each). The real
forest is nothing like that: our original team estimated the true
population distribution by submitting seven single-class "probe" files to
the Kaggle leaderboard (each probe predicts one class for every row, so its
reported accuracy equals that class's true population share). The result:

| Cover type | Population share |
| --- | --- |
| 1 Spruce/Fir | 36.4% |
| 2 Lodgepole Pine | 48.6% |
| 3 Ponderosa Pine | 6.2% |
| 4 Cottonwood/Willow | 0.5% |
| 5 Aspen | 1.5% |
| 6 Douglas-fir | 3.0% |
| 7 Krummholz | 3.6% |

Lodgepole Pine and Spruce/Fir alone are ~85% of the real forest. Training
naively on the balanced set would waste model capacity on rare classes that
barely matter, and plain validation accuracy would misrepresent real-world
(and Kaggle leaderboard) performance. So the final model is trained with
**sample weights = population share / training share** and evaluated with
**population-weighted accuracy** as the honest metric. The app walks through
this story end to end: explore the data, see the balanced-vs-real
distributions, inspect model performance under both metrics, and get live
predictions.

## Quickstart with Docker

```bash
docker run -p 8501:8501 ghcr.io/camptodata/forest-cover-app:latest
```

Then open <http://localhost:8501>.

Build locally instead of pulling:

```bash
docker build -t forest-cover-app .
docker run -p 8501:8501 forest-cover-app
```

## Requirements / local setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch the
interpreter for you if needed).

```bash
git clone https://github.com/camptodata/forest-cover-app.git
cd forest-cover-app
uv sync
uv run streamlit run app/streamlit_app.py
```

The app needs a trained model at `models/xgb_model.json` and metrics at
`models/metrics.json` -- both are committed to this repository, so this
just works out of the box. To reproduce them yourself, see below.

## Training

```bash
uv run python -m forestcover.train
```

Loads `data/train.csv`, validates and decodes it, engineers features (see
`src/forestcover/features.py`), fits an XGBoost classifier with
population-derived sample weights and a fixed seed (42), and writes
`models/xgb_model.json` (native XGBoost JSON format, not pickle).

## Evaluation

```bash
uv run python -m forestcover.evaluate
```

Runs stratified 5-fold cross-validation for two models -- a default-
hyperparameter baseline on raw one-hot features without weights, and the
final weighted model on engineered features -- computing plain accuracy,
population-weighted accuracy, per-class recall, and confusion matrices for
each. Writes everything, plus gain-based feature importances and library
versions, to `models/metrics.json`. The output has no timestamps, so
re-running on unchanged code/data is diff-stable.

## Pre-trained model

`models/xgb_model.json` is committed to the repo (native XGBoost JSON
format, well under 10 MB) so the app and tests work without training first.
It was produced by the `train` command above with seed 42 and the
hyperparameters in `src/forestcover/config.py::XGB_PARAMS`
(`n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.7,
colsample_bytree=0.6`).

## Results

Real numbers from `models/metrics.json` (produced by `uv run python -m
forestcover.evaluate`, stratified 5-fold CV, seed 42, xgboost 3.4.1 /
scikit-learn 1.9.1):

| Model | Plain accuracy | Weighted accuracy |
| --- | --- | --- |
| Baseline (raw one-hot features, default hyperparameters, unweighted) | 85.75% | 72.47% |
| Final (engineered features, population-weighted, this repo's shipped model) | 84.45% | 80.01% |

The final model gives up about one point of plain accuracy for a large gain
in population-weighted accuracy (72.5% -> 80.0%). That is the point of
sample weighting: mistakes on the two dominant classes cost the most, so the
model is pushed to get those right, even if rare classes are classified a
little less cleanly. Weighted accuracy is the metric that reflects the real,
imbalanced population. The app's
"Model performance" section renders this same table plus per-class recall
and confusion matrices live from `models/metrics.json`.

For reference, the **original notebook** (with SHAP feature selection,
Optuna tuning, and calibrated fold ensembling -- none of which this
simplified, reproducible version includes) reported:

- Default XGBoost, plain 5-fold CV accuracy on raw features: **~0.857**
- Population-weighted CV accuracy: **~0.80** with default hyperparameters,
  **~0.824** after Optuna tuning.

Reproduce: `uv run python -m forestcover.train && uv run python -m forestcover.evaluate`.

## Project structure

```
forest-cover-app/
├── app/streamlit_app.py       # Streamlit entry point (5 sections)
├── src/forestcover/
│   ├── config.py               # paths, class/wilderness names, weights, hyperparameters
│   ├── data.py                 # load / validate / decode / filter (tested)
│   ├── features.py             # add_features, ported from the original notebook
│   ├── model.py                # weights, train, save/load, predict
│   ├── train.py                # CLI: train and save the model
│   └── evaluate.py             # CLI: CV evaluation -> models/metrics.json
├── data/train.csv, data/README.md
├── models/xgb_model.json, models/metrics.json
├── tests/                      # pytest suite
├── .github/workflows/ci.yml
├── Dockerfile, .dockerignore
└── pyproject.toml, uv.lock
```

## Tests & CI

`tests/test_data.py` and `tests/test_features.py` cover the importing,
validation, decoding, and filtering functions the course brief asks to be
tested, using synthetic fixtures built by a factory in `tests/conftest.py`
plus checks against the real CSV. `tests/test_model.py` covers weights,
train/save/load round-trips, and prediction shape/sum invariants.
`tests/test_app.py` uses Streamlit's `AppTest` harness to smoke-test every
app section end to end.

Run locally:

```bash
uv run pytest --cov=forestcover --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) on every push/PR to `main` and on `v*` tags:

1. **lint-test**: `uv sync`, `ruff check`, `ruff format --check`, `pytest`
   with coverage (uploaded as an artifact).
2. **docker**: builds the image, runs it, and polls
   `http://localhost:8501/_stcore/health` until healthy (or fails with the
   container logs) as a real smoke test; on push to `main` or a version tag,
   pushes the image to `ghcr.io/camptodata/forest-cover-app`.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format .
uv run pre-commit install   # run hooks automatically before each commit
```

## Data & licence/attribution

`data/train.csv` derives from the UCI **Covertype** dataset (Blackard, J.
A. and Dean, D. J., 1999, *Comparative accuracies of artificial neural
networks and discriminant analysis in predicting forest cover types from
cartographic variables*, Computers and Electronics in Agriculture
24(3):131-151), licensed CC BY 4.0. See `data/README.md` for the full
column dictionary and provenance.

## Credits

Original Machine Learning II group project by **Hae In Keum, Nikolai
Levin, Julius Enderwitz, and Nick Hartmann**. This repository is Nick
Hartmann's individual *Tooling for the Data Scientist* final project,
packaging that group work as a tested, containerized Streamlit app.

## Simplifications vs. the original notebook

To keep the shipped model small, fast to train, and fully reproducible from
a clean checkout, this project deliberately drops three things from the
original notebook:

- **SHAP-based feature selection** -- the shipped model uses the full
  engineered feature set instead of the SHAP-ranked top-16 subset.
- **Optuna hyperparameter tuning** -- hyperparameters are fixed, moderate
  values chosen for a sub-2-minute training run and a sub-10 MB model,
  rather than the result of a tuning study.
- **Calibrated fold ensembling** -- the shipped model is a single fit on
  the full training set rather than an average of 5 fold-specific models.

These simplifications trade a few points of accuracy for a build that
anyone can reproduce with two commands and that fits comfortably in a
container image.
