"""Orchestration: ingest, analyse, benchmark, write results.

Usage
-----
    python -m src.run_pipeline --stage ingest      # CSV -> partitioned Parquet
    python -m src.run_pipeline --stage analyse     # write analytical outputs
    python -m src.run_pipeline --stage benchmark   # measured comparisons
    python -m src.run_pipeline --stage all
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import functions as F

from .analytics import (
    fastest_growing_districts, median_price_by_district_year,
    new_build_premium, price_growth_by_district, rolling_median_by_area,
    transaction_volume_by_quarter,
)
from .config import DRIVER_MEMORY, OUTPUT, PARQUET, RAW_CSV, build_spark
from .ingest import (
    build_parquet_layer, directory_size_bytes, read_parquet, read_raw_csv,
)
from .joins import join_regions, region_lookup, region_summary, skew_profile
from .transforms import quality_report

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


# Substrings that identify the Windows Hadoop-shim failure, and only it. A
# bare ``except Exception`` here would also swallow a genuine logic error in an
# aggregate and quietly write the wrong numbers via the fallback path.
_HADOOP_SHIM_MARKERS = ("winutils", "hadoop_home", "getsetpermissioncommand",
                        "nativeio", "unsatisfiedlinkerror")


def _is_hadoop_shim_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _HADOOP_SHIM_MARKERS)


def _write_csv(df, name: str, *, limit: int | None = None) -> int:
    """Write a single-file CSV summary, returning its row count.

    Spark on a local Windows filesystem requires ``winutils.exe`` and
    ``hadoop.dll`` on HADOOP_HOME. Without them the write fails in
    ``Shell.<clinit>`` ("HADOOP_HOME and hadoop.home.dir are unset"). These
    outputs are small aggregates, so where the Spark writer is unavailable we
    fall back to collecting the summary and writing it with pandas -- which
    touches no Hadoop code at all.

    Note what the fallback does *not* rescue, because the previous wording here
    claimed otherwise: it is not the case that "every read still works". Only
    single-file reads do. Reading a partitioned Parquet directory needs
    ``NativeIO$Windows.access0`` to list leaf files and fails with
    ``UnsatisfiedLinkError`` without the shim, and so does reading a lone
    ``.parquet`` file. So on a box with no shim, ``stage_analyse`` cannot get
    as far as calling this function, and on a box with one the Spark writer
    works and the fallback never fires. Reaching it means the DataFrame came
    from somewhere other than the Parquet layer -- CSV, as in
    ``scripts/analyse_from_csv.py``.

    The two paths do not produce the same layout, and callers should expect
    either: the Spark writer produces a directory ``output/<name>/part-*.csv``,
    the fallback a single file ``output/<name>.csv``. Which one ran is printed.

    The fallback is deliberately limited to summaries. The Parquet layer in
    ``ingest.py`` is 31.5M rows and is not collected to the driver under any
    circumstances; that stage genuinely needs a working Hadoop shim.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = df.limit(limit) if limit else df
    # Without caching, the count below recomputes the entire aggregate: a
    # second full job over the Parquet layer for every output written.
    out.cache()
    try:
        try:
            (out.coalesce(1).write.mode("overwrite")
                .option("header", "true").csv(str(OUTPUT / name)))
        except Exception as exc:
            if not _is_hadoop_shim_failure(exc):
                raise
            print(f"    {name}: no Hadoop shim; writing {name}.csv via pandas",
                  flush=True)
            # Spark creates the target directory before it fails, leaving an
            # empty one next to the file the fallback is about to write.
            shutil.rmtree(OUTPUT / name, ignore_errors=True)
            out.toPandas().to_csv(OUTPUT / f"{name}.csv", index=False)
        return out.count()
    finally:
        out.unpersist()


def stage_ingest() -> dict:
    spark = build_spark("ingest")
    try:
        # One pass over the CSV: read, clean, enrich, write Parquet.
        timings = build_parquet_layer(spark)
        # Counting the written layer is Parquet footer arithmetic rather than a
        # scan, so the only further pass over the CSV is the raw count. This
        # was previously three full passes over 5.5 GB -- raw.count(),
        # cleaned.count(), and then build_parquet_layer reading and cleaning
        # the whole file a second time from scratch.
        quality = quality_report(read_raw_csv(spark), read_parquet(spark))
        sizes = {
            "csv_bytes": directory_size_bytes(RAW_CSV),
            "parquet_bytes": directory_size_bytes(PARQUET),
        }
        sizes["compression_ratio"] = (
            round(sizes["csv_bytes"] / sizes["parquet_bytes"], 2)
            if sizes["parquet_bytes"] else None
        )
        return {"quality": quality, "timings": timings, "sizes": sizes}
    finally:
        spark.stop()


def stage_analyse() -> dict:
    spark = build_spark("analyse")
    try:
        df = read_parquet(spark)
        lookup = region_lookup(spark)
        joined = join_regions(df, lookup)

        written = {
            "median_price_by_district_year":
                _write_csv(median_price_by_district_year(df),
                           "median_price_by_district_year"),
            "price_growth_by_district":
                _write_csv(price_growth_by_district(df),
                           "price_growth_by_district"),
            "rolling_median_by_area":
                _write_csv(rolling_median_by_area(df), "rolling_median_by_area"),
            "new_build_premium":
                _write_csv(new_build_premium(df), "new_build_premium"),
            "transaction_volume_by_quarter":
                _write_csv(transaction_volume_by_quarter(df),
                           "transaction_volume_by_quarter"),
            "region_summary":
                _write_csv(region_summary(joined), "region_summary"),
            "fastest_growing_districts":
                _write_csv(fastest_growing_districts(df),
                           "fastest_growing_districts"),
        }

        # One aggregate, one job. As five separate actions this was five full
        # scans of the Parquet layer for five scalars.
        headline = df.agg(
            F.count(F.lit(1)).alias("total_rows"),
            F.min("year").alias("year_min"),
            F.max("year").alias("year_max"),
            F.countDistinct("district").alias("districts"),
            F.countDistinct("postcode_area").alias("postcode_areas"),
        ).first().asDict()
        skew = [r.asDict() for r in skew_profile(df).collect()]
        return {"outputs": written, "headline": headline, "skew_top10": skew}
    finally:
        spark.stop()


def stage_benchmark() -> dict:
    from .benchmark import (
        bench_aqe, bench_format, bench_join, bench_schema_inference,
        bench_shuffle_partitions,
    )
    results: list = []

    spark = build_spark("benchmark-main")
    try:
        results += bench_schema_inference(spark)
        results += bench_format(spark)
        results += bench_join(spark)
    finally:
        spark.stop()

    results += bench_shuffle_partitions()
    results += bench_aqe()

    return {"results": [
        {"name": r.name, "seconds": round(r.seconds, 2), "rows": r.rows,
         "notes": r.notes, **({"extra": r.extra} if r.extra else {})}
        for r in results
    ]}


def _environment() -> dict:
    import os
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "spark_driver_memory": DRIVER_MEMORY,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UK property price PySpark pipeline")
    parser.add_argument("--stage", default="all",
                        choices=["ingest", "analyse", "benchmark", "all"])
    args = parser.parse_args()

    report: dict = {"environment": _environment(), "stages": {}}
    started = time.perf_counter()

    if args.stage in ("ingest", "all"):
        print("[1/3] ingest: CSV -> partitioned Parquet ...", flush=True)
        report["stages"]["ingest"] = stage_ingest()
    if args.stage in ("analyse", "all"):
        print("[2/3] analyse: window functions and aggregations ...", flush=True)
        report["stages"]["analyse"] = stage_analyse()
    if args.stage in ("benchmark", "all"):
        print("[3/3] benchmark: measured comparisons ...", flush=True)
        report["stages"]["benchmark"] = stage_benchmark()

    report["total_seconds"] = round(time.perf_counter() - started, 1)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {DOCS / 'results.json'}")


if __name__ == "__main__":
    main()
