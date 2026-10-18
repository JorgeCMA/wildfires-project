"""Empirical CCAA-day fire rates — deployment unit (PLAN §Plan D)."""

from __future__ import annotations

import pandas as pd
import pytest

from wildfire.analysis.deployment import ALL_LABEL, ccaa_day_fire_rates


def _detections():
    # Span 2023-01-01 .. 2023-01-03 = 3 days, two CCAAs.
    return pd.DataFrame(
        {
            "acq_date": [
                "2023-01-01",
                "2023-01-01",  # duplicate cell -> counted once
                "2023-01-03",
                "2023-01-02",
            ],
            "ccaa": ["A", "A", "A", "B"],
        }
    )


class TestCcaaDayFireRates:
    def test_counts_and_rates(self):
        table = ccaa_day_fire_rates(_detections())

        per_ccaa = table[table["ccaa"] != ALL_LABEL].set_index("ccaa")
        assert per_ccaa.loc["A", "n_ccaa_days"] == 3
        assert per_ccaa.loc["A", "n_fire_days"] == 2
        assert per_ccaa.loc["A", "fire_rate"] == pytest.approx(2 / 3)
        assert per_ccaa.loc["B", "n_fire_days"] == 1
        assert per_ccaa.loc["B", "fire_rate"] == pytest.approx(1 / 3)

    def test_all_row_is_pooled_cell_rate_and_last(self):
        table = ccaa_day_fire_rates(_detections())

        assert table.iloc[-1]["ccaa"] == ALL_LABEL
        all_row = table.iloc[-1]
        assert all_row["n_ccaa_days"] == 3 * 2
        assert all_row["n_fire_days"] == 3
        assert all_row["fire_rate"] == pytest.approx(0.5)

    def test_sorted_descending(self):
        rates = [
            r
            for r, c in zip(
                ccaa_day_fire_rates(_detections())["fire_rate"],
                ccaa_day_fire_rates(_detections())["ccaa"],
            )
            if c != ALL_LABEL
        ]
        assert rates == sorted(rates, reverse=True)

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="ccaa"):
            ccaa_day_fire_rates(_detections().drop(columns=["ccaa"]))

    def test_unparseable_date_raises(self):
        bad = _detections()
        bad.loc[0, "acq_date"] = "not-a-date"
        with pytest.raises(ValueError, match="unparseable"):
            ccaa_day_fire_rates(bad)
