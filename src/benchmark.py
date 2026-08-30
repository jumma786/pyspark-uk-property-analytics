"""Measured comparisons.

Every claim this project makes about performance is produced here and written
to ``docs/results.json`` by ``run_pipeline.py``. Nothing is asserted from
memory or from what the documentation says ought to be true.

Method notes, because they change the numbers:
  * Each timing is a full action (``count`` or a write), not a lazy plan.
  * Spark is stopped and restarted between configurations that need different
    session-level settings, since several cannot be changed at runtime.
  * The first run of anything on a cold OS page cache is slower. Where a
    comparison is sensitive to that, both arms are run in the same state and
    the caveat is recorded next to the result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from pyspark.sql import SparkSession, functions as F

from .config import PARQUET, RAW_CSV, SHUFFLE_PARTITIONS, build_spark
from .ingest import directory_size_bytes, read_parquet, read_raw_csv
from .joins import join_regions, region_lookup


@dataclass
class Result:
    name: str
    seconds: float
    rows: int | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        row_part = f", {self.rows:,} rows" if self.rows is not None else ""
        return f"{self.name}: {self.seconds:.1f}s{row_part}"


def timed(name: str, fn: Callable[[], object], *, notes: str = "") -> Result:
    """Run ``fn`` and record wall-clock seconds and, if returned, a row count."""
    started = time.perf_counter()
    out = fn()
    seconds = time.perf_counter() - started
    rows = out if isinstance(out, int) else None
    return Result(name, seconds, rows, notes)


# --------------------------------------------------------------------------
# 1. Schema inference cost
# --------------------------------------------------------------------------

SAMPLED_INFERENCE_RATIO = 0.1


def bench_schema_inference(spark: SparkSession, *,
                           sample_only: bool = True) -> list[Result]:
    """Explicit schema vs ``inferSchema`` on the raw CSV.

    Inference requires a full extra pass over the file before the real job can
    plan. On a multi-gigabyte CSV that pass is the dominant cost.

    ``sample_only`` limits the inference arm to a sampled inference pass, where
    a full one would take longer than the rest of the suite combined. It makes
    the measured gap *smaller* than the real one, so the comparison is
    understated rather than overstated, and the ratio used is recorded in the
    result notes.
    """
    ratio = SAMPLED_INFERENCE_RATIO if sample_only else None
    infer_notes = (
        f"inferSchema=true, samplingRatio={ratio}; a sampled inference pass, so "
        "this understates the cost of full inference"
        if ratio is not None else
        "inferSchema=true; requires an extra full pass before planning"
    )
    return [
        timed("read_csv_explicit_schema",
              lambda: read_raw_csv(spark, infer=False).count(),
              notes="schema declared in code; single pass"),
        timed("read_csv_infer_schema",
              lambda: read_raw_csv(spark, infer=True,
                                   sampling_ratio=ratio).count(),
              notes=infer_notes),
    ]


# --------------------------------------------------------------------------
# 2. Storage format: CSV vs partitioned Parquet
# --------------------------------------------------------------------------

def bench_format(spark: SparkSession) -> list[Result]:
    """Same filtered aggregate over raw CSV and over partitioned Parquet."""
    results: list[Result] = []

    def csv_query() -> int:
        df = read_raw_csv(spark)
        return (df.filter(F.year("date_of_transfer") == 2019)
                  .groupBy("property_type").count().count())

    def parquet_query() -> int:
        df = read_parquet(spark)
        return (df.filter(F.col("year") == 2019)
                  .groupBy("property_type").count().count())

    # The two arms are not row-for-row identical: the CSV arm reads the raw
    # feed, the Parquet arm reads the cleaned layer. That difference is stated
    # rather than corrected, because it is the honest comparison -- the Parquet
    # side has already paid for cleaning at ingest, and re-cleaning the CSV
    # here would charge the CSV arm for work the Parquet arm is not doing now.
    results.append(timed("filtered_aggregate_csv", csv_query,
                         notes="raw rows; no pushdown, whole file parsed"))
    results.append(timed("filtered_aggregate_parquet_partitioned", parquet_query,
                         notes=("cleaned rows; partition pruning + projection "
                                "pushdown. Row sets differ: cleaning was paid "
                                "at ingest, not here")))

    csv_bytes = directory_size_bytes(RAW_CSV)
    pq_bytes = directory_size_bytes(PARQUET)
    results.append(Result(
        "storage_footprint", 0.0, None,
        notes="on-disk size after snappy-compressed columnar rewrite",
        extra={
            "csv_bytes": csv_bytes,
            "parquet_bytes": pq_bytes,
            "compression_ratio": round(csv_bytes / pq_bytes, 2) if pq_bytes else None,
        },
    ))
    return results


# --------------------------------------------------------------------------
# 3. Join strategy: broadcast vs shuffle
# --------------------------------------------------------------------------

def bench_join(spark: SparkSession) -> list[Result]:
    """Broadcast hash join vs a forced sort-merge join on a skewed key."""
    df = read_parquet(spark)
    lookup = region_lookup(spark)

    def broadcast_join() -> int:
        return (join_regions(df, lookup, broadcast=True)
                .groupBy("region").count().count())

    def shuffle_join() -> int:
        # Disable auto-broadcast so Spark cannot rescue the comparison.
        prior = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
        try:
            return (join_regions(df, lookup, broadcast=False)
                    .groupBy("region").count().count())
        finally:
            spark.conf.set("spark.sql.autoBroadcastJoinThreshold", prior)

    return [
        timed("join_broadcast", broadcast_join,
              notes="small side broadcast to every executor; no shuffle of the 30M-row side"),
        timed("join_sort_merge_forced", shuffle_join,
              notes="autoBroadcastJoinThreshold=-1; both sides shuffled on a skewed key"),
    ]


# --------------------------------------------------------------------------
# 4. Shuffle partition count
# --------------------------------------------------------------------------

def bench_shuffle_partitions(
        counts: tuple[int, ...] = (200, SHUFFLE_PARTITIONS)) -> list[Result]:
    """Spark's default of 200 shuffle partitions vs a value tuned to the box.

    Needs a fresh session per value: ``spark.sql.shuffle.partitions`` is read
    when a plan is created, and mixing values within one session muddies the
    comparison.
    """
    results: list[Result] = []
    for n in counts:
        spark = build_spark(f"shuffle-{n}", aqe=False, shuffle_partitions=n)
        try:
            df = read_parquet(spark)
            results.append(timed(
                f"groupby_shuffle_partitions_{n}",
                lambda: df.groupBy("district", "year").count().count(),
                notes=f"spark.sql.shuffle.partitions={n}, AQE off",
            ))
        finally:
            spark.stop()
    return results


# --------------------------------------------------------------------------
# 5. Adaptive Query Execution
# --------------------------------------------------------------------------

def bench_aqe() -> list[Result]:
    """AQE on vs off for a skewed shuffle join.

    ``broadcast=False`` only removes the hint. The lookup is a few kilobytes,
    far under the 10 MB auto-broadcast threshold, so without disabling that
    threshold Spark broadcasts it anyway and both arms measure a join with no
    shuffle at all -- which is not what a result named ``skewed_join`` claims
    to be. ``bench_join`` disables it for the same reason; this did not.
    """
    results: list[Result] = []
    for enabled in (False, True):
        spark = build_spark(f"aqe-{enabled}", aqe=enabled, shuffle_partitions=200)
        try:
            spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
            df = read_parquet(spark)
            lookup = region_lookup(spark)
            results.append(timed(
                f"skewed_join_aqe_{'on' if enabled else 'off'}",
                lambda: (join_regions(df, lookup, broadcast=False)
                         .groupBy("postcode_area").count().count()),
                notes=("autoBroadcastJoinThreshold=-1; " + (
                    "AQE coalesces partitions and splits skewed ones"
                    if enabled else "fixed 200 partitions, no skew handling")),
            ))
        finally:
            spark.stop()
    return results
