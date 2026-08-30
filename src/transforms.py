"""Cleaning and derived columns.

Every rule here is a decision about what the data actually means, not a tidy-up.
Where a rule discards rows, it is because including them would corrupt an
aggregate -- and the pipeline counts what it drops so the loss is visible.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from .schema import PROPERTY_TYPES, DURATIONS

# Land Registry includes a small number of transfers at implausible prices --
# £1 transfers between related parties, and bulk portfolio sales recorded as a
# single line. Both distort a median badly at district level.
MIN_PLAUSIBLE_PRICE = 1_000
MAX_PLAUSIBLE_PRICE = 50_000_000


def clean(df: DataFrame) -> DataFrame:
    """Apply cleaning rules, returning only rows safe to aggregate."""
    return (
        df
        # record_status: A = addition, C = change, D = delete. Only additions
        # and changes represent live records; deletions are withdrawals.
        .filter(F.col("record_status").isin("A", "C"))
        .filter(F.col("price").isNotNull())
        .filter(F.col("price").between(MIN_PLAUSIBLE_PRICE, MAX_PLAUSIBLE_PRICE))
        .filter(F.col("date_of_transfer").isNotNull())
    )


def enrich(df: DataFrame) -> DataFrame:
    """Add derived columns used by the analytics layer."""
    return (
        df
        # Reduce the parsed timestamp to a date: the time component is
        # always 00:00 in this feed and carrying it invites timezone bugs.
        .withColumn("transfer_date", F.to_date("date_of_transfer"))
        .withColumn("year", F.year("date_of_transfer"))
        .withColumn("month", F.month("date_of_transfer"))
        .withColumn("quarter", F.quarter("date_of_transfer"))
        # Postcode area is the alphabetic prefix: "SW1A 1AA" -> "SW". It is the
        # coarsest useful geography and the join key for the reference table.
        .withColumn(
            "postcode_area",
            F.upper(F.regexp_extract(F.coalesce(F.col("postcode"), F.lit("")),
                                     r"^([A-Za-z]{1,2})", 1)),
        )
        .withColumn(
            "postcode_area",
            F.when(F.col("postcode_area") == "", None).otherwise(F.col("postcode_area")),
        )
        # Three-valued on purpose: NULL old_new stays NULL rather than
        # becoming False, so a record with no build flag is not counted
        # as existing stock. Such rows fall out of both sides of
        # new_build_premium rather than skewing the "existing" median.
        .withColumn("is_new_build", F.col("old_new") == F.lit("Y"))
        .withColumn(
            "property_type_name",
            F.create_map([F.lit(x) for kv in PROPERTY_TYPES.items()
                          for x in kv])[F.col("property_type")],
        )
        .withColumn(
            "duration_name",
            F.create_map([F.lit(x) for kv in DURATIONS.items()
                          for x in kv])[F.col("duration")],
        )
    )


def quality_report(raw: DataFrame, cleaned: DataFrame) -> dict:
    """Count what cleaning removed, so the loss is stated rather than hidden."""
    raw_n = raw.count()
    clean_n = cleaned.count()
    return {
        "rows_raw": raw_n,
        "rows_clean": clean_n,
        "rows_dropped": raw_n - clean_n,
        "pct_dropped": round(100.0 * (raw_n - clean_n) / raw_n, 4) if raw_n else 0.0,
    }
