"""Streamlit entry point for the Forest Cover Type classifier app.

Thin UI layer: all data/feature/model logic lives in ``src/forestcover``.
Run with ``uv run streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forestcover.config import (  # noqa: E402
    COVER_TYPE_NAMES,
    METRICS_PATH,
    MODEL_PATH,
    N_SOIL_TYPES,
    POPULATION_PROPORTIONS,
    TRAIN_CSV,
    TRAIN_PROPORTION,
    WILDERNESS_AREA_NAMES,
)
from forestcover.data import class_distribution, filter_observations, load_dataset  # noqa: E402
from forestcover.features import add_features  # noqa: E402
from forestcover.model import class_weights, load_model, predict_proba  # noqa: E402

st.set_page_config(layout="wide", page_title="Forest Cover Type Classifier", page_icon="🌲")

RAW_NUMERIC_FEATURES = [
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
]


@st.cache_data
def get_data() -> pd.DataFrame:
    """Load and decode the training dataset, cached across reruns."""
    return load_dataset(TRAIN_CSV)


@st.cache_resource
def get_model():
    """Load the pre-trained model, cached across reruns."""
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict:
    """Load models/metrics.json produced by ``forestcover.evaluate``."""
    import json

    with METRICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def section_overview(df: pd.DataFrame, metrics: dict) -> None:
    """Render the Overview section."""
    st.title("🌲 Forest Cover Type Classifier")
    st.markdown(
        "This app packages a Machine Learning II group project (Kaggle "
        "'Forest Cover Type Prediction') into a Streamlit application, as the "
        "final project for HEC Paris' *Tooling for the Data Scientist* course."
    )

    st.subheader("The twist: a balanced training set, an imbalanced world")
    st.markdown(
        "The 15,120-row training set is **perfectly balanced**: exactly "
        "2,160 rows per cover type. But the real forest is not balanced at "
        "all. The original team estimated the true population distribution "
        "by submitting seven single-class 'probe' files to the Kaggle "
        "leaderboard and reading off each one's accuracy. The result: "
        "Lodgepole Pine and Spruce/Fir alone make up about 85% of the real "
        "forest, while Cottonwood/Willow is under 1%. Training on the "
        "balanced set naively would over-predict the rare classes, so the "
        "final model is trained with **sample weights = population share / "
        "training share** and evaluated with **population-weighted "
        "accuracy** -- the honest proxy for real-world (and Kaggle "
        "leaderboard) performance."
    )

    final = metrics["final"]
    baseline = metrics["baseline"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline plain accuracy", f"{baseline['plain_accuracy_mean']:.1%}")
    col2.metric("Final plain accuracy", f"{final['plain_accuracy_mean']:.1%}")
    col3.metric("Baseline weighted accuracy", f"{baseline['weighted_accuracy_mean']:.1%}")
    col4.metric("Final weighted accuracy", f"{final['weighted_accuracy_mean']:.1%}")

    st.subheader("Dataset")
    st.markdown(
        f"- {len(df):,} rows, {df.shape[1]} columns after decoding\n"
        "- Source: UCI Covertype dataset (Blackard & Dean, 1999, CC BY 4.0), "
        "sampled by Kaggle for the 'Forest Cover Type Prediction' class competition\n"
        "- 4 wilderness areas, 40 soil types, 7 cover type classes"
    )

    st.subheader("Credits")
    st.markdown(
        "Original Machine Learning II group project by **Hae In Keum, "
        "Nikolai Levin, Julius Enderwitz, and Nick Hartmann**. This "
        "repository packages that work as an individual *Tooling for the "
        "Data Scientist* final project by **Nick Hartmann**."
    )


def section_explore(df: pd.DataFrame) -> None:
    """Render the Explore the data section."""
    st.title("Explore the data")

    with st.sidebar:
        st.header("Filters")
        cover_types = st.multiselect(
            "Cover type",
            options=list(COVER_TYPE_NAMES.keys()),
            default=list(COVER_TYPE_NAMES.keys()),
            format_func=lambda c: f"{c} - {COVER_TYPE_NAMES[c]}",
            key="filter_cover_types",
        )
        wilderness_areas = st.multiselect(
            "Wilderness area",
            options=list(WILDERNESS_AREA_NAMES.keys()),
            default=list(WILDERNESS_AREA_NAMES.keys()),
            format_func=lambda w: f"{w} - {WILDERNESS_AREA_NAMES[w]}",
            key="filter_wilderness_areas",
        )
        elevation_min, elevation_max = int(df["Elevation"].min()), int(df["Elevation"].max())
        elevation_range = st.slider(
            "Elevation (m)",
            elevation_min,
            elevation_max,
            (elevation_min, elevation_max),
            key="filter_elevation",
        )
        slope_min, slope_max = int(df["Slope"].min()), int(df["Slope"].max())
        slope_range = st.slider(
            "Slope (deg)", slope_min, slope_max, (slope_min, slope_max), key="filter_slope"
        )
        soil_types = st.multiselect(
            "Soil type",
            options=list(range(1, N_SOIL_TYPES + 1)),
            default=list(range(1, N_SOIL_TYPES + 1)),
            key="filter_soil_types",
        )

    filtered = filter_observations(
        df,
        cover_types=cover_types,
        wilderness_areas=wilderness_areas,
        elevation_range=elevation_range,
        slope_range=slope_range,
        soil_types=soil_types,
    )

    st.metric("Filtered rows", f"{len(filtered):,}", delta=f"of {len(df):,} total")

    if filtered.empty:
        st.warning("No observations match the current filters. Widen a filter to see data.")
        return

    dist = class_distribution(filtered)
    dist_df = pd.DataFrame(
        {
            "Cover_Type": dist.index,
            "Name": [COVER_TYPE_NAMES[c] for c in dist.index],
            "Count": dist.values,
        }
    )
    st.subheader("Class distribution")
    st.plotly_chart(px.bar(dist_df, x="Name", y="Count"), width="stretch")

    st.subheader("Elevation by cover type")
    filtered_named = filtered.assign(Cover_Name=filtered["Cover_Type"].map(COVER_TYPE_NAMES))
    st.plotly_chart(
        px.box(filtered_named, x="Cover_Name", y="Elevation", color="Cover_Name"), width="stretch"
    )

    st.subheader("Scatter: two numeric features")
    c1, c2 = st.columns(2)
    x_feat = c1.selectbox("X axis", RAW_NUMERIC_FEATURES, index=0, key="scatter_x")
    y_feat = c2.selectbox("Y axis", RAW_NUMERIC_FEATURES, index=4, key="scatter_y")
    sample = filtered_named.sample(n=min(3000, len(filtered_named)), random_state=42)
    st.plotly_chart(
        px.scatter(sample, x=x_feat, y=y_feat, color="Cover_Name", opacity=0.6), width="stretch"
    )

    st.subheader("Preview")
    st.dataframe(filtered.head(200), width="stretch")

    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="forest_cover_filtered.csv",
        mime="text/csv",
    )


def section_train_vs_real() -> None:
    """Render the Train vs. real world section."""
    st.title("Train vs. real world")
    st.markdown(
        "The training data is artificially balanced at 1/7 per class. The "
        "team probed the real Kaggle leaderboard test distribution by "
        "submitting seven single-class predictions -- each submission's "
        "reported accuracy reveals that class's true population share."
    )

    labels = [COVER_TYPE_NAMES[c] for c in sorted(POPULATION_PROPORTIONS)]
    train_share = [TRAIN_PROPORTION] * 7
    pop_share = [POPULATION_PROPORTIONS[c] for c in sorted(POPULATION_PROPORTIONS)]

    fig = go.Figure()
    fig.add_bar(name="Training set (uniform)", x=labels, y=train_share)
    fig.add_bar(name="Probed population", x=labels, y=pop_share)
    fig.update_layout(barmode="group", yaxis_title="Share")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Resulting sample weights")
    weights = class_weights()
    weights_df = pd.DataFrame(
        {
            "Cover_Type": list(weights.keys()),
            "Name": [COVER_TYPE_NAMES[c] for c in weights],
            "Population share": [POPULATION_PROPORTIONS[c] for c in weights],
            "Sample weight": list(weights.values()),
        }
    )
    st.dataframe(weights_df, width="stretch", hide_index=True)

    st.markdown(
        "**Why weighted accuracy is the honest metric:** plain accuracy on "
        "the balanced training set rewards a model equally for every class, "
        "even though rare classes barely matter in the real forest and "
        "common classes matter a lot. Population-weighted accuracy "
        "re-weights each validation example by its real-world class "
        "frequency, so it approximates what the Kaggle leaderboard (built "
        "on the true population) actually measures."
    )


def section_performance(metrics: dict) -> None:
    """Render the Model performance section."""
    st.title("Model performance")

    baseline = metrics["baseline"]
    final = metrics["final"]
    results_df = pd.DataFrame(
        {
            "Model": ["Baseline (raw features, unweighted)", "Final (engineered, weighted)"],
            "Plain accuracy": [baseline["plain_accuracy_mean"], final["plain_accuracy_mean"]],
            "Weighted accuracy": [
                baseline["weighted_accuracy_mean"],
                final["weighted_accuracy_mean"],
            ],
        }
    )
    st.subheader("Results")
    st.dataframe(
        results_df.style.format({"Plain accuracy": "{:.2%}", "Weighted accuracy": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Per-class recall (final model)")
    recall = final["per_class_recall"]
    recall_df = pd.DataFrame(
        {
            "Cover type": [COVER_TYPE_NAMES[int(c)] for c in recall],
            "Recall": list(recall.values()),
        }
    )
    st.plotly_chart(px.bar(recall_df, x="Cover type", y="Recall"), width="stretch")

    st.subheader("Confusion matrix (final model)")
    normalized = st.toggle("Show row-normalized values", value=True)
    cm = np.array(final["confusion_matrix_normalized" if normalized else "confusion_matrix"])
    labels = [COVER_TYPE_NAMES[c] for c in final["confusion_matrix_labels"]]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Greens",
        text_auto=".2f" if normalized else True,
        labels={"x": "Predicted", "y": "True"},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 20 feature importance (gain, final model)")
    importances = metrics["feature_importance_gain_top20"]
    imp_df = pd.DataFrame(
        {"Feature": list(importances.keys()), "Gain": list(importances.values())}
    ).sort_values("Gain", ascending=True)
    st.plotly_chart(px.bar(imp_df, x="Gain", y="Feature", orientation="h"), width="stretch")


def section_predict(df: pd.DataFrame, model) -> None:
    """Render the Predict section."""
    st.title("Predict")
    st.markdown(
        "Set the raw observation features and get a prediction, computed "
        "through the exact same `add_features` + model code path used at "
        "training time."
    )

    if st.button("Use a random observation from the dataset"):
        row = df.sample(n=1, random_state=None).iloc[0]
        for feat in RAW_NUMERIC_FEATURES:
            st.session_state[f"predict_{feat}"] = int(row[feat])
        st.session_state["predict_wilderness"] = int(row["Wilderness_Area"])
        st.session_state["predict_soil"] = int(row["Soil_Type"])
        st.session_state["predict_true_label"] = int(row["Cover_Type"])

    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(RAW_NUMERIC_FEATURES):
        lo, hi = int(df[feat].min()), int(df[feat].max())
        default = st.session_state.get(f"predict_{feat}", int(df[feat].median()))
        inputs[feat] = cols[i % 2].slider(feat, lo, hi, default, key=f"predict_{feat}")

    wilderness = st.selectbox(
        "Wilderness area",
        options=list(WILDERNESS_AREA_NAMES.keys()),
        format_func=lambda w: f"{w} - {WILDERNESS_AREA_NAMES[w]}",
        index=list(WILDERNESS_AREA_NAMES.keys()).index(
            st.session_state.get("predict_wilderness", 1)
        ),
        key="predict_wilderness",
    )
    soil = st.selectbox(
        "Soil type",
        options=list(range(1, N_SOIL_TYPES + 1)),
        index=st.session_state.get("predict_soil", 1) - 1,
        key="predict_soil",
    )

    if "predict_true_label" in st.session_state:
        st.info(
            f"True label of the sampled observation: "
            f"{COVER_TYPE_NAMES[st.session_state['predict_true_label']]}"
        )

    row_df = pd.DataFrame(
        [
            {
                **inputs,
                "Wilderness_Area": wilderness,
                "Soil_Type": soil,
            }
        ]
    )
    engineered = add_features(row_df)
    proba = predict_proba(model, engineered)[0]
    predicted_class = int(np.argmax(proba)) + 1

    st.subheader("Prediction")
    st.success(f"Predicted cover type: **{COVER_TYPE_NAMES[predicted_class]}**")

    proba_df = pd.DataFrame(
        {"Cover type": [COVER_TYPE_NAMES[c] for c in range(1, 8)], "Probability": proba}
    )
    st.plotly_chart(px.bar(proba_df, x="Cover type", y="Probability"), width="stretch")


def main() -> None:
    """Wire up sidebar navigation and dispatch to each section."""
    df = get_data()
    metrics = get_metrics()

    page = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Explore the data",
            "Train vs. real world",
            "Model performance",
            "Predict",
        ],
        key="nav_page",
    )

    if page == "Overview":
        section_overview(df, metrics)
    elif page == "Explore the data":
        section_explore(df)
    elif page == "Train vs. real world":
        section_train_vs_real()
    elif page == "Model performance":
        section_performance(metrics)
    elif page == "Predict":
        model = get_model()
        section_predict(df, model)


main()
