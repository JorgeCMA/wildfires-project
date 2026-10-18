"""Deployment unit (PLAN §Plan D): CCAA-day rates + relative risk bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.analysis.deployment import (
    ALL_LABEL,
    RISK_BAND_LABELS,
    add_risk_bands,
    ccaa_day_fire_rates,
    join_ccaa_fire_rates,
    quantile_bands,
)


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


class TestQuantileBands:
    def test_five_bands_counts_and_labels(self):
        out = quantile_bands(pd.Series(np.arange(100)))

        assert list(out["risk_band"].cat.categories) == RISK_BAND_LABELS[5]
        assert out["risk_band"].nunique() == 5
        assert out["risk_band"].value_counts().eq(20).all()

    def test_bands_are_monotone_in_score(self):
        scores = pd.Series(np.linspace(0, 1, 50))
        out = quantile_bands(scores)
        codes = out["risk_band"].cat.codes.to_numpy()
        assert (np.diff(codes) >= 0).all()

    def test_ties_share_a_band(self):
        out = quantile_bands(pd.Series([0.1, 0.5, 0.5, 0.5, 0.9]), n_bands=3)
        middle = out.loc[[1, 2, 3], "risk_band"]
        assert middle.nunique() == 1

    def test_percentile_span(self):
        out = quantile_bands(pd.Series(np.arange(10)))
        assert out["risk_percentile"].max() == pytest.approx(100.0)

    def test_three_bands_default_labels(self):
        out = quantile_bands(pd.Series(np.arange(9)), n_bands=3)
        assert list(out["risk_band"].cat.categories) == RISK_BAND_LABELS[3]

    def test_nan_scores_raise(self):
        with pytest.raises(ValueError, match="finite"):
            quantile_bands(pd.Series([0.1, np.nan]))

    def test_unknown_n_bands_without_labels_raises(self):
        with pytest.raises(ValueError, match="no default labels"):
            quantile_bands(pd.Series(np.arange(8)), n_bands=4)

    def test_explicit_labels_length_checked(self):
        with pytest.raises(ValueError, match="expected 5"):
            quantile_bands(pd.Series(np.arange(10)), labels=["a", "b"])


class TestAddRiskBands:
    def _frame(self):
        return pd.DataFrame(
            {
                "protocol": ["a"] * 10 + ["b"] * 10,
                "score": list(np.arange(10)) + list(np.arange(10) * 2),
            }
        )

    def test_adds_columns_and_bands_per_group(self):
        out = add_risk_bands(self._frame(), n_bands=5)

        assert {"risk_band", "risk_percentile"} <= set(out.columns)
        assert out["risk_band"].dtype.name == "category"
        # same relative position in each group -> same bands in both
        a = out.loc[out["protocol"] == "a", "risk_band"].astype(str).to_numpy()
        b = out.loc[out["protocol"] == "b", "risk_band"].astype(str).to_numpy()
        assert list(a) == list(b)

    def test_group_col_none_bands_whole_frame(self):
        out = add_risk_bands(self._frame(), group_col=None, n_bands=5)
        assert out["risk_band"].nunique() == 5

    def test_missing_score_column_raises(self):
        with pytest.raises(KeyError, match="score"):
            add_risk_bands(self._frame().drop(columns=["score"]))

    def test_missing_group_column_raises(self):
        with pytest.raises(KeyError, match="protocol"):
            add_risk_bands(self._frame().drop(columns=["protocol"]))


class TestJoinCcaaFireRates:
    def _rates(self):
        return pd.DataFrame(
            {
                "ccaa": ["A", "B", ALL_LABEL],
                "fire_rate": [0.5, 0.25, 0.4],
            }
        )

    def test_attaches_region_rate(self):
        records = pd.DataFrame({"record_id": ["r1", "r2", "r3"]})
        ccaa_of_record = pd.DataFrame(
            {"record_id": ["r1", "r2", "r3"], "ccaa": ["A", "B", "A"]}
        )
        out = join_ccaa_fire_rates(records, ccaa_of_record, self._rates())

        assert out.set_index("record_id").loc["r1", "ccaa_fire_rate"] == 0.5
        assert out.set_index("record_id").loc["r2", "ccaa_fire_rate"] == 0.25
        # same CCAA repeated -> same rate; ALL row never used
        assert out.set_index("record_id").loc["r3", "ccaa_fire_rate"] == 0.5
        assert not out["ccaa_fire_rate"].eq(0.4).any()

    def test_unknown_region_is_nan(self):
        records = pd.DataFrame({"record_id": ["r1"]})
        ccaa_of_record = pd.DataFrame({"record_id": ["r1"], "ccaa": ["Z"]})
        out = join_ccaa_fire_rates(records, ccaa_of_record, self._rates())
        assert pd.isna(out.loc[0, "ccaa_fire_rate"])

    def test_missing_rate_column_raises(self):
        records = pd.DataFrame({"record_id": ["r1"]})
        ccaa_of_record = pd.DataFrame({"record_id": ["r1"], "ccaa": ["A"]})
        with pytest.raises(KeyError, match="fire_rate"):
            join_ccaa_fire_rates(
                records, ccaa_of_record, self._rates().drop(columns=["fire_rate"])
            )
