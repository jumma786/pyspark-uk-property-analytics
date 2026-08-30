"""Tests for the analytical layer and join behaviour."""
from __future__ import annotations

import datetime as dt

import pytest
from pyspark.sql import functions as F

from src.analytics import (
    fastest_growing_districts, median_price_by_district_year, new_build_premium,
    price_growth_by_district, rolling_median_by_area,
)
from src.joins import join_regions, region_lookup, skew_profile
from src.schema import PRICE_PAID_SCHEMA
from src.transforms import clean, enrich


def _priced_years(pairs, district="GROWTH DISTRICT", postcode="SW1A 1AA",
                  per_year=40):
    """Rows for (year, price) pairs; per_year clears the >=30 transaction floor."""
    rows = []
    for year, price in pairs:
        for i in range(per_year):
            rows.append((
                f"x{year}{i}", price, dt.datetime(year, 6, 1), postcode,
                "D", "N", "F", str(i), None, "St", None, "Town",
                district, "County", "A", "A",
            ))
    return rows


@pytest.fixture(scope="module")
def trend_df(spark):
    """A district with a known price path, so growth maths can be checked."""
    rows = _priced_years([(2018, 100_000), (2019, 110_000), (2020, 121_000)])
    df = spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)
    return enrich(clean(df))


class TestMedian:
    def test_returns_one_row_per_district_year(self, trend_df):
        out = median_price_by_district_year(trend_df).collect()
        assert len(out) == 3
        assert {r["year"] for r in out} == {2018, 2019, 2020}

    def test_median_matches_known_value(self, trend_df):
        out = {r["year"]: r["median_price"]
               for r in median_price_by_district_year(trend_df).collect()}
        assert out[2018] == 100_000
        assert out[2020] == 121_000

    def test_suppresses_thin_cells(self, spark):
        """Fewer than 30 transactions makes a median that means nothing."""
        rows = [(f"y{i}", 200_000, dt.datetime(2020, 1, 1), "M1 1AA", "D", "N",
                 "F", str(i), None, "St", None, "Town", "THIN DISTRICT",
                 "County", "A", "A") for i in range(5)]
        df = enrich(clean(spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)))
        assert median_price_by_district_year(df).count() == 0


class TestGrowth:
    def test_year_on_year_percentage(self, trend_df):
        out = {r["year"]: r["yoy_pct"]
               for r in price_growth_by_district(trend_df).collect()}
        assert out[2018] is None          # no prior year to compare against
        assert out[2019] == pytest.approx(10.0, abs=0.01)
        assert out[2020] == pytest.approx(10.0, abs=0.01)

    def test_cumulative_growth_from_first_year(self, trend_df):
        out = {r["year"]: r["cumulative_growth_pct"]
               for r in price_growth_by_district(trend_df).collect()}
        assert out[2018] == pytest.approx(0.0, abs=0.01)
        assert out[2020] == pytest.approx(21.0, abs=0.01)

    def test_yoy_is_null_across_a_missing_year(self, spark):
        """A district-year suppressed by the transaction floor must break the
        chain, not silently become the "previous year"."""
        rows = (_priced_years([(2018, 100_000), (2020, 200_000)])
                # 2019 has only 5 sales, so it is suppressed as a thin cell.
                + _priced_years([(2019, 150_000)], per_year=5))
        df = enrich(clean(spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)))
        out = {r["year"]: r["yoy_pct"]
               for r in price_growth_by_district(df).collect()}
        assert set(out) == {2018, 2020}
        assert out[2020] is None, "2020 compared against 2018 and called it YoY"


class TestFastestGrowing:
    def test_since_year_sets_the_baseline(self, spark):
        """since_year must rebase the growth, not just filter the output rows."""
        rows = _priced_years([(1995, 50_000), (2000, 100_000), (2020, 200_000)])
        df = enrich(clean(spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)))
        out = fastest_growing_districts(df, since_year=2000, top_n=5).collect()
        assert len(out) == 1
        # 100k -> 200k since 2000 is 100%. Measured from 1995 it would be 300%.
        assert out[0]["cumulative_growth_pct"] == pytest.approx(100.0, abs=0.01)


class TestRollingMedian:
    def test_window_widens_then_holds(self, trend_df):
        out = {r["year"]: r["rolling_avg_median"]
               for r in rolling_median_by_area(trend_df, window_years=3).collect()}
        # Year 1 averages one value, year 2 averages two, year 3 averages three.
        assert out[2018] == pytest.approx(100_000, abs=1)
        assert out[2019] == pytest.approx(105_000, abs=1)
        assert out[2020] == pytest.approx(110_333, abs=2)

    def test_window_spans_years_not_rows(self, spark):
        """A 3-year window must stay 3 years wide when a year has no sales.

        With a row-based window, an area with 2010 and then nothing until 2020
        averages those two rows as though they were adjacent years.
        """
        rows = _priced_years([(2010, 100_000), (2020, 200_000)],
                             postcode="ZE1 0AA", district="GAP AREA")
        df = enrich(clean(spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)))
        out = {r["year"]: r["rolling_avg_median"]
               for r in rolling_median_by_area(df, window_years=3).collect()}
        assert out[2010] == pytest.approx(100_000, abs=1)
        assert out[2020] == pytest.approx(200_000, abs=1), (
            "2020 averaged with a 2010 median ten years earlier"
        )


class TestJoins:
    def test_broadcast_and_shuffle_produce_identical_results(self, trend_df, spark):
        """A join hint must change the plan, never the answer."""
        lookup = region_lookup(spark)
        a = join_regions(trend_df, lookup, broadcast=True)
        b = join_regions(trend_df, lookup, broadcast=False)
        assert a.count() == b.count()
        assert {r["region"] for r in a.collect()} == {r["region"] for r in b.collect()}

    def test_maps_area_to_region(self, trend_df, spark):
        joined = join_regions(trend_df, region_lookup(spark))
        assert {r["region"] for r in joined.collect()} == {"London"}

    def test_left_join_keeps_rows_with_no_region_match(self, spark):
        """An unmatched postcode area must not silently delete the transaction."""
        rows = [("z1", 200_000, dt.datetime(2020, 1, 1), "ZZ9 9ZZ", "D", "N",
                 "F", "1", None, "St", None, "Town", "D", "C", "A", "A")]
        df = enrich(clean(spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)))
        joined = join_regions(df, region_lookup(spark))
        assert joined.count() == 1
        assert joined.first()["region"] is None

    def test_skew_profile_percentages_are_bounded(self, trend_df):
        rows = skew_profile(trend_df).collect()
        assert rows
        assert all(0 < r["pct_of_total"] <= 100 for r in rows)


class TestNewBuildPremium:
    def test_premium_is_computed_against_existing_stock(self, spark):
        rows = []
        for i in range(40):
            rows.append((f"n{i}", 300_000, dt.datetime(2021, 1, 1), "SW1A 1AA",
                         "D", "Y", "F", str(i), None, "St", None, "T",
                         "D", "C", "A", "A"))
            rows.append((f"o{i}", 250_000, dt.datetime(2021, 1, 1), "SW1A 1AA",
                         "D", "N", "F", str(i), None, "St", None, "T",
                         "D", "C", "A", "A"))
        df = enrich(clean(spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)))
        out = new_build_premium(df).collect()
        assert len(out) == 1
        assert out[0]["premium_pct"] == pytest.approx(20.0, abs=0.01)
