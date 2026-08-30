"""Analytical layer: window functions and aggregations at scale.

These are the queries the engineering exists to serve. Each runs over the full
~30M-row history rather than a sample.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F


def median_price_by_district_year(df: DataFrame) -> DataFrame:
    """Median price per district per year.

    ``percentile_approx`` rather than an exact median: an exact percentile
    requires a full sort within each group, which across ~400 districts and 30
    years is a large shuffle for a number nobody reads to the penny. The
    accuracy parameter (10,000) bounds the error well below rounding noise.
    """
    return (
        df.groupBy("district", "year")
          .agg(
              F.percentile_approx("price", 0.5, 10_000).alias("median_price"),
              F.count("*").alias("transactions"),
              F.avg("price").cast("long").alias("mean_price"),
          )
          .filter(F.col("transactions") >= 30)   # thin cells make noisy medians
    )


def price_growth_by_district(df: DataFrame) -> DataFrame:
    """Year-on-year and cumulative growth per district, via window functions.

    ``prev_median`` is taken with a range window over ``year``, not ``lag``.
    The input has already dropped district-years with fewer than 30
    transactions, so the rows are not contiguous: ``lag`` would hand a district
    whose 2006 was suppressed the 2005 median and label the result year-on-year
    growth. A range window asks for the preceding *year* rather than the
    preceding *row*, so a gap yields NULL and the cell is honestly empty.
    """
    prev_year = (Window.partitionBy("district").orderBy(F.col("year").cast("long"))
                 .rangeBetween(-1, -1))
    whole = (Window.partitionBy("district").orderBy("year")
             .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))
    return (
        median_price_by_district_year(df)
        .withColumn("prev_median", F.max("median_price").over(prev_year))
        .withColumn(
            "yoy_pct",
            F.round(100.0 * (F.col("median_price") - F.col("prev_median"))
                    / F.col("prev_median"), 2),
        )
        .withColumn("first_median", F.first("median_price").over(whole))
        .withColumn(
            "cumulative_growth_pct",
            F.round(100.0 * (F.col("median_price") - F.col("first_median"))
                    / F.col("first_median"), 2),
        )
    )


def rolling_median_by_area(df: DataFrame, window_years: int = 3) -> DataFrame:
    """Rolling multi-year smoother of the yearly median, by postcode area.

    The output column is ``rolling_avg_median`` and it is exactly that: the
    mean of the last ``window_years`` yearly medians. It is not a median of the
    underlying prices over that span -- computing one would mean re-percentiling
    the raw rows in every window, and the point here is to damp thin-year noise
    cheaply. Naming it ``rolling_median`` would misdescribe the number.

    The window is ``rangeBetween`` over ``year``, so a postcode area with no
    transactions in some year gets a window that is still ``window_years`` wide
    in time. ``rowsBetween`` would quietly stretch it across a longer span.
    """
    base = (
        df.groupBy("postcode_area", "year")
          .agg(F.percentile_approx("price", 0.5, 10_000).alias("median_price"),
               F.count("*").alias("transactions"))
    )
    window = (Window.partitionBy("postcode_area").orderBy(F.col("year").cast("long"))
              .rangeBetween(-(window_years - 1), 0))
    return base.withColumn("rolling_avg_median",
                           F.round(F.avg("median_price").over(window), 0))


def new_build_premium(df: DataFrame) -> DataFrame:
    """Premium paid for new builds over existing stock, by year and type."""
    grouped = (
        df.groupBy("year", "property_type_name", "is_new_build")
          .agg(F.percentile_approx("price", 0.5, 10_000).alias("median_price"),
               F.count("*").alias("transactions"))
    )
    # Both filters drop NULL is_new_build (see transforms.enrich): a record
    # with no build flag is not evidence either way, so it is excluded from
    # both medians rather than silently counted as existing stock.
    new = grouped.filter(F.col("is_new_build")).drop("is_new_build")
    existing = grouped.filter(~F.col("is_new_build")).drop("is_new_build")
    return (
        new.alias("n")
        .join(existing.alias("o"), on=["year", "property_type_name"], how="inner")
        .select(
            "year", "property_type_name",
            F.col("n.median_price").alias("new_build_median"),
            F.col("o.median_price").alias("existing_median"),
            F.round(100.0 * (F.col("n.median_price") - F.col("o.median_price"))
                    / F.col("o.median_price"), 2).alias("premium_pct"),
            F.col("n.transactions").alias("new_build_transactions"),
        )
    )


def transaction_volume_by_quarter(df: DataFrame) -> DataFrame:
    """Quarterly transaction counts -- the shape of 2008 and 2020 in one table."""
    return (
        df.groupBy("year", "quarter")
          .agg(F.count("*").alias("transactions"),
               F.percentile_approx("price", 0.5, 10_000).alias("median_price"))
          .orderBy("year", "quarter")
    )


def fastest_growing_districts(df: DataFrame, since_year: int = 2000,
                              top_n: int = 20) -> DataFrame:
    """Districts with the largest cumulative growth since a base year.

    ``since_year`` filters the transactions *before* growth is computed, so the
    baseline is the district's median in that year. Filtering afterwards --
    which this did previously -- leaves ``cumulative_growth_pct`` measured from
    the district's first year anywhere in the file (1995) and only changes
    which rows survive, so the argument silently did nothing.
    """
    growth = price_growth_by_district(df.filter(F.col("year") >= since_year))
    latest = (growth.groupBy("district")
              .agg(F.max("year").alias("year")))
    return (
        growth.join(latest, on=["district", "year"], how="inner")
        .select("district", "year", "median_price", "cumulative_growth_pct",
                "transactions")
        .orderBy(F.desc("cumulative_growth_pct"))
        .limit(top_n)
    )
