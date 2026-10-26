"""Static checks for the demo's corporate theme and assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from wildfire.analysis.demo_data import PALETTE, RISK_BAND_COLORS
from wildfire.analysis.deployment import RISK_BAND_LABELS

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"
LOGO_PATH = ROOT / "demo" / "assets" / "wildfires_logo_1.png"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    assert value.startswith("#") and len(value) == 7, value
    return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))


def _relative_luminance(value: str) -> float:
    channels = []
    for channel in _hex_to_rgb(value):
        c = channel / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(foreground: str, background: str) -> float:
    high, low = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (high + 0.05) / (low + 0.05)


def test_palette_tokens_are_valid_and_complete():
    expected = {
        "bg",
        "secondary_bg",
        "text",
        "muted",
        "primary",
        "accent",
        "gold",
        "deep",
        "link",
        "grid",
    }
    assert set(PALETTE) == expected
    for name, value in PALETTE.items():
        assert len(_hex_to_rgb(value)) == 3, name


def test_risk_band_ramp_is_ordered_low_to_high():
    assert list(RISK_BAND_COLORS) == RISK_BAND_LABELS[5]
    assert len(set(RISK_BAND_COLORS.values())) == 5
    luminances = [_relative_luminance(value) for value in RISK_BAND_COLORS.values()]
    assert luminances == sorted(luminances, reverse=True)


def test_corporate_hexes_are_preserved():
    assert PALETTE["bg"].upper() == "#FFF9EA"
    assert PALETTE["accent"].upper() == "#FFC074"
    assert PALETTE["primary"].upper() == "#F65C2C"
    assert PALETTE["gold"].upper() == "#FDAA15"
    assert PALETTE["text"].upper() == "#000000"


def test_streamlit_theme_mirrors_the_palette():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    theme = config["theme"]
    assert theme["base"] == "light"
    assert theme["primaryColor"].upper() == PALETTE["primary"].upper()
    assert theme["backgroundColor"].upper() == PALETTE["bg"].upper()
    assert theme["secondaryBackgroundColor"].upper() == PALETTE["secondary_bg"].upper()
    assert theme["textColor"].upper() == PALETTE["text"].upper()
    assert theme["baseFontSize"] == 18


@pytest.mark.parametrize(
    ("foreground", "background", "minimum"),
    [
        (PALETTE["text"], PALETTE["bg"], 4.5),
        (PALETTE["text"], PALETTE["secondary_bg"], 4.5),
        (PALETTE["muted"], PALETTE["bg"], 4.5),
        (PALETTE["link"], PALETTE["bg"], 4.5),
        (PALETTE["primary"], PALETTE["bg"], 3.0),  # UI component, not body text
        (PALETTE["text"], PALETTE["primary"], 4.5),  # primary-button label
        ("#FFFFFF", PALETTE["deep"], 4.5),  # "Muy alto" label
        (PALETTE["text"], PALETTE["accent"], 4.5),  # "Bajo" label
        (PALETTE["text"], PALETTE["gold"], 4.5),  # "Medio" label
    ],
)
def test_palette_contrast(foreground: str, background: str, minimum: float):
    assert _contrast(foreground, background) >= minimum


def test_logo_asset_is_present():
    assert LOGO_PATH.is_file()
    assert LOGO_PATH.stat().st_size > 0
