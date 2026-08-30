"""Shared pytest fixtures.

Tests run against a small in-memory DataFrame, not the 5.5 GB file. The point
of a test here is that the transformation logic is correct and the edge cases
are handled -- correctness on 12 hand-built rows where the expected answer is
known by inspection, not on 30M where it is not.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import pytest
from pyspark.sql import SparkSession

from src.schema import PRICE_PAID_SCHEMA

# Same interpreter pin as src/config.py -- tests build DataFrames from Python
# objects, which requires a working Python worker.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder
        .appName("tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _row(txn, price, date, postcode, ptype="D", old_new="N", duration="F",
         district="TEST DISTRICT", status="A"):
    return (txn, price, date, postcode, ptype, old_new, duration,
            "1", None, "Test Street", None, "Test Town", district,
            "Test County", "A", status)


@pytest.fixture(scope="session")
def sample_df(spark):
    """Twelve rows covering every branch the cleaning rules care about."""
    # TimestampType, so datetime not date -- see src/schema.py
    d = dt.datetime
    rows = [
        # --- rows that must survive cleaning ---
        _row("t1", 250_000, d(2020, 1, 15), "SW1A 1AA"),
        _row("t2", 180_000, d(2020, 6, 30), "M1 2AB", ptype="F"),
        _row("t3", 95_000, d(2015, 3, 1), "LS1 4XY", ptype="T"),
        _row("t4", 420_000, d(2021, 9, 9), "SW2 3CD", old_new="Y"),
        _row("t5", 310_000, d(2021, 9, 9), "SW2 3CD", old_new="N"),
        _row("t6", 1_500, d(1999, 1, 1), "B1 1AA"),        # just above floor
        # --- rows that must be dropped ---
        _row("t7", 1, d(2020, 1, 1), "SW1A 1AA"),           # below price floor
        _row("t8", 90_000_000, d(2020, 1, 1), "SW1A 1AA"),  # above price ceiling
        _row("t9", None, d(2020, 1, 1), "SW1A 1AA"),        # null price
        _row("t10", 200_000, None, "SW1A 1AA"),             # null date
        _row("t11", 200_000, d(2020, 1, 1), "SW1A 1AA", status="D"),  # deleted
        # --- edge case: missing postcode, must survive but yield null area ---
        _row("t12", 175_000, d(2018, 4, 4), None),
    ]
    return spark.createDataFrame(rows, schema=PRICE_PAID_SCHEMA)
