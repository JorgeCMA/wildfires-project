"""Headless smoke test of the Streamlit demo (skipped if the demo extra is absent)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("folium")
pytest.importorskip("streamlit_folium")

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "demo" / "app.py"


def test_app_runs_without_exception():
    at = AppTest.from_file(str(APP_PATH), default_timeout=180).run()
    assert not at.exception
    assert at.title, "the app should render its title"
