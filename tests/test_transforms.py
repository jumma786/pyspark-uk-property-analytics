"""Tests for the cleaning and enrichment rules.

Each test names the decision it is protecting. A test that only asserts "the
function ran" would pass on a function that silently dropped every row -- which
is the failure this suite exists to prevent.
"""
from __future__ import annotations

from pyspark.sql import functions as F

from src.transforms import (
    MAX_PLAUSIBLE_PRICE, MIN_PLAUSIBLE_PRICE, clean, enrich, quality_report,
)


class TestClean:
    def test_keeps_only_plausible_prices(self, sample_df):
        prices = [r["price"] for r in clean(sample_df).select("price").collect()]
        assert prices, "cleaning removed every row"
        assert all(MIN_PLAUSIBLE_PRICE <= p <= MAX_PLAUSIBLE_PRICE for p in prices)

    def test_drops_one_pound_transfers(self, sample_df):
        ids = {r["transaction_id"] for r in clean(sample_df).collect()}
        assert "t7" not in ids, "a GBP 1 related-party transfer survived cleaning"

    def test_drops_portfolio_scale_outlier(self, sample_df):
        ids = {r["transaction_id"] for r in clean(sample_df).collect()}
        assert "t8" not in ids, "a 90M bulk sale survived and will distort medians"

    def test_drops_null_price_and_null_date(self, sample_df):
        ids = {r["transaction_id"] for r in clean(sample_df).collect()}
        assert "t9" not in ids and "t10" not in ids

    def test_drops_deleted_records(self, sample_df):
        """record_status D means the transfer was withdrawn, not that it happened."""
        ids = {r["transaction_id"] for r in clean(sample_df).collect()}
        assert "t11" not in ids

    def test_keeps_row_at_the_price_floor(self, sample_df):
        """The boundary is inclusive; an off-by-one here silently loses rows."""
        ids = {r["transaction_id"] for r in clean(sample_df).collect()}
        assert "t6" in ids

    def test_keeps_row_with_missing_postcode(self, sample_df):
        """A missing postcode is not a reason to discard a real transaction."""
        ids = {r["transaction_id"] for r in clean(sample_df).collect()}
        assert "t12" in ids


class TestEnrich:
    def test_extracts_postcode_area(self, sample_df):
        rows = {r["transaction_id"]: r["postcode_area"]
                for r in enrich(clean(sample_df)).collect()}
        assert rows["t1"] == "SW"      # two-letter area
        assert rows["t2"] == "M"       # single-letter area
        assert rows["t3"] == "LS"

    def test_missing_postcode_yields_null_area_not_empty_string(self, sample_df):
        """An empty string would form its own group in every aggregation."""
        rows = {r["transaction_id"]: r["postcode_area"]
                for r in enrich(clean(sample_df)).collect()}
        assert rows["t12"] is None

    def test_derives_calendar_parts(self, sample_df):
        rows = {r["transaction_id"]: r for r in enrich(clean(sample_df)).collect()}
        assert rows["t1"]["year"] == 2020
        assert rows["t1"]["month"] == 1
        assert rows["t1"]["quarter"] == 1
        assert str(rows["t1"]["transfer_date"]) == "2020-01-15"

    def test_maps_property_type_codes(self, sample_df):
        rows = {r["transaction_id"]: r["property_type_name"]
                for r in enrich(clean(sample_df)).collect()}
        assert rows["t1"] == "Detached"
        assert rows["t2"] == "Flat/Maisonette"
        assert rows["t3"] == "Terraced"

    def test_new_build_flag_is_boolean_not_string(self, sample_df):
        rows = {r["transaction_id"]: r["is_new_build"]
                for r in enrich(clean(sample_df)).collect()}
        assert rows["t4"] is True
        assert rows["t5"] is False


class TestQualityReport:
    def test_counts_what_cleaning_removed(self, sample_df):
        cleaned = clean(sample_df)
        report = quality_report(sample_df, cleaned)
        assert report["rows_raw"] == 12
        assert report["rows_dropped"] == report["rows_raw"] - report["rows_clean"]
        assert report["rows_dropped"] == 5    # t7 t8 t9 t10 t11
        assert 0 < report["pct_dropped"] < 100

    def test_report_is_not_silently_zero(self, sample_df):
        """A quality report that always reports zero loss is worse than none."""
        report = quality_report(sample_df, clean(sample_df))
        assert report["rows_dropped"] > 0
