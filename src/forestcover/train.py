"""
CLI entry point: train the final model and write it to models/.

Usage::

    uv run python -m forestcover.train
"""

from __future__ import annotations

from forestcover.config import MODEL_PATH, TRAIN_CSV
from forestcover.data import load_dataset
from forestcover.features import add_features
from forestcover.model import save_model, train_model


def main() -> None:
    """Load the dataset, engineer features, fit the weighted model, and save it."""
    print(f"Loading dataset from {TRAIN_CSV} ...")
    df = load_dataset(TRAIN_CSV)

    y = df["Cover_Type"]
    X = add_features(df)

    print(f"Training weighted XGBoost model on {len(X)} rows, {X.shape[1]} features ...")
    model = train_model(X, y, weighted=True)

    save_model(model, MODEL_PATH)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Saved model to {MODEL_PATH} ({size_kb:.1f} KB).")


if __name__ == "__main__":
    main()
