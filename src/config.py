"""Central configuration: paths and Spark session construction.

The Spark settings here are tuned for a single machine with 12 cores and 8 GB
of RAM. That constraint is the point of this project: the source file is 5.5 GB,
so nothing can be collected to the driver and every stage has to stream.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

# Spark launches Python workers by invoking "python3" by default. That name does
# not exist on a standard Windows install, so any operation needing a worker --
# createDataFrame from Python objects, Python UDFs -- dies with
# "Cannot run program python3". Pointing both variables at the running
# interpreter makes the project portable across Windows, macOS and Linux.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "pp-complete.csv"
PARQUET = DATA / "parquet" / "price_paid"
OUTPUT = DATA / "output"

# Driver memory deliberately below total RAM: Spark needs headroom for the JVM
# overhead and the OS. Setting this to 6g on an 8 GB box causes GC thrashing.
DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "4g")

# 200 is the Spark default and is wrong for a single-node run: it produces 200
# tiny shuffle files per stage. 2x the core count keeps every core busy without
# drowning the scheduler in task overhead. Derived rather than hardcoded so the
# ratio stays true on a box that is not the 12-core machine this was tuned on.
CORES = os.cpu_count() or 4
SHUFFLE_PARTITIONS = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", str(2 * CORES)))


def build_spark(app_name: str = "uk-property-analytics", *,
                aqe: bool = True,
                shuffle_partitions: int | None = None) -> SparkSession:
    """Create a local SparkSession.

    ``aqe`` is exposed so the benchmark can measure Adaptive Query Execution
    on and off rather than asserting that it helps.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{CORES}]")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions",
                shuffle_partitions or SHUFFLE_PARTITIONS)
        .config("spark.sql.adaptive.enabled", str(aqe).lower())
        .config("spark.sql.adaptive.skewJoin.enabled", str(aqe).lower())
        .config("spark.sql.adaptive.coalescePartitions.enabled", str(aqe).lower())
        # Parquet is written with the same timestamp semantics Spark reads back.
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        # Keep the local UI off during benchmarks so it does not skew timings.
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
