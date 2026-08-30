"""CSV -> partitioned Parquet conversion.

This is the step that makes everything downstream fast. The source is a single
5.5 GB uncompressed CSV with no header: no statistics, no column pruning, and
every query re-parses every byte. Converting once to Parquet partitioned by
year buys predicate pushdown, projection pushdown and columnar compression for
every query that follows.
"""
from __future__ import annotations

import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from .config import PARQUET, RAW_CSV
from .schema import PRICE_PAID_SCHEMA
from .transforms import clean, enrich

QUOTE_CHAR = chr(34)  # double quote; kept as a constant to avoid quoting noise


def read_raw_csv(spark: SparkSession, path: Path = RAW_CSV,
                 *, infer: bool = False,
                 sampling_ratio: float | None = None) -> DataFrame:
    """Read the Price Paid CSV.

    ``infer=True`` exists only so the benchmark can measure what schema
    inference costs on a file this size. Production reads pass the schema.

    ``sampling_ratio`` applies only when inferring: it is Spark's
    ``samplingRatio``, the fraction of rows the inference pass reads before it
    decides the types. It makes the inference arm of the benchmark cheaper and
    therefore *understates* what full inference costs -- which is the safe
    direction for a claim, and is recorded in the result notes.
    """
    reader = (spark.read
              .option("header", "false")
              .option("quote", QUOTE_CHAR)
              .option("timestampFormat", "yyyy-MM-dd HH:mm"))
    if infer:
        reader = reader.option("inferSchema", "true")
        if sampling_ratio is not None:
            reader = reader.option("samplingRatio", str(sampling_ratio))
        return reader.csv(str(path))
    return reader.schema(PRICE_PAID_SCHEMA).csv(str(path))


def write_parquet(df: DataFrame, path: Path = PARQUET,
                  *, partition_by: str = "year") -> float:
    """Write partitioned Parquet, returning elapsed seconds.

    Partitioning by year gives ~30 partitions across a 1995-2026 span: coarse
    enough to avoid the small-file problem, selective enough that any query
    carrying a date filter touches a fraction of the data.
    """
    started = time.perf_counter()
    (df.write
       .mode("overwrite")
       .partitionBy(partition_by)
       .option("compression", "snappy")
       .parquet(str(path)))
    return time.perf_counter() - started


def build_parquet_layer(spark: SparkSession) -> dict:
    """Full ingest: read CSV, clean, enrich, write partitioned Parquet."""
    started = time.perf_counter()
    raw = read_raw_csv(spark)
    cleaned = enrich(clean(raw))
    write_seconds = write_parquet(cleaned)
    return {
        "total_seconds": round(time.perf_counter() - started, 1),
        "write_seconds": round(write_seconds, 1),
    }


def read_parquet(spark: SparkSession, path: Path = PARQUET) -> DataFrame:
    return spark.read.parquet(str(path))


def directory_size_bytes(path: Path) -> int:
    """Total bytes on disk for a file or directory tree."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
