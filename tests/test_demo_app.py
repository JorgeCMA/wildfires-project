"""Headless smoke test of the Streamlit demo (skipped if the demo extra is absent)."""

from __future__ import annotations

import ast
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


def test_fire_marker_callback_is_a_function_expression():
    """Folium wraps the callback as `var callback = {callback};`.

    Declaring `var callback = ...` inside the string renders invalid JS
    (`var callback = var callback = ...`) and the whole map script fails.
    """
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    callback = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "FIRE_MARKER_CALLBACK" for t in node.targets)
    )
    assert callback.strip().startswith("function (row)")
    assert "var callback" not in callback
