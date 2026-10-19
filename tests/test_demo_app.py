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


def test_plan_c_map_click_updates_coordinates(monkeypatch):
    """A Plan-C map click must push the clicked point into the lat/lon inputs."""
    clicked = {"last_clicked": {"lat": 41.1234, "lng": 2.5678}}
    monkeypatch.setattr("streamlit_folium.st_folium", lambda *args, **kwargs: clicked)

    at = AppTest.from_file(str(APP_PATH), default_timeout=180).run()

    assert not at.exception
    values = {widget.label: widget.value for widget in at.number_input}
    assert values["Latitud"] == pytest.approx(41.1234)
    assert values["Longitud"] == pytest.approx(2.5678)
