"""Smoke tests for the Streamlit app, using Streamlit's AppTest harness.

Requires a committed model artefact (models/xgb_model.json) and metrics
(models/metrics.json), produced by ``forestcover.train`` / ``forestcover.evaluate``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from forestcover.config import METRICS_PATH, MODEL_PATH

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.is_file() and METRICS_PATH.is_file()),
    reason="requires models/xgb_model.json and models/metrics.json (run train.py/evaluate.py)",
)

SECTIONS = [
    "Overview",
    "Explore the data",
    "Train vs. real world",
    "Model performance",
    "Predict",
]


def _run(page: str) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", SECTIONS)
def test_each_section_renders_without_exception(page):
    at = _run(page)
    assert not at.exception


def test_explore_filter_changes_row_count():
    at = _run("Explore the data")
    before = at.metric[0].value

    multiselects = at.sidebar.multiselect
    cover_type_select = multiselects[0]
    cover_type_select.set_value([1]).run()

    after = at.metric[0].value
    assert not at.exception
    assert before != after
