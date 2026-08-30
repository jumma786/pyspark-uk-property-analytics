"""Explicit schema for the Land Registry Price Paid file.

Note on ``date_of_transfer``: the published file writes this as
``"1995-01-31 00:00"`` -- a timestamp, not a bare date. Declaring it as
``DateType`` parses to NULL for every row without raising, which is the kind of
silent fault that reaches a dashboard looking healthy. It is read as a
timestamp with an explicit format and reduced to a date downstream.

``inferSchema=True`` on a 5.5 GB CSV costs a full extra pass over the data
before the real job starts. Declaring the schema removes that pass entirely --
the benchmark measures the difference rather than assuming it.

Column order follows the Land Registry's published Price Paid Data
specification. The file has no header row.
"""
from __future__ import annotations

from pyspark.sql.types import (
    IntegerType, StringType, StructField, StructType, TimestampType,
)

PRICE_PAID_SCHEMA = StructType([
    StructField("transaction_id", StringType(), nullable=False),
    StructField("price", IntegerType(), nullable=True),
    StructField("date_of_transfer", TimestampType(), nullable=True),
    StructField("postcode", StringType(), nullable=True),
    StructField("property_type", StringType(), nullable=True),   # D S T F O
    StructField("old_new", StringType(), nullable=True),         # Y N
    StructField("duration", StringType(), nullable=True),        # F L U
    StructField("paon", StringType(), nullable=True),
    StructField("saon", StringType(), nullable=True),
    StructField("street", StringType(), nullable=True),
    StructField("locality", StringType(), nullable=True),
    StructField("town_city", StringType(), nullable=True),
    StructField("district", StringType(), nullable=True),
    StructField("county", StringType(), nullable=True),
    StructField("ppd_category_type", StringType(), nullable=True),  # A B
    StructField("record_status", StringType(), nullable=True),      # A C D
])

# Human-readable lookups. Small enough to broadcast rather than shuffle.
PROPERTY_TYPES = {
    "D": "Detached", "S": "Semi-Detached", "T": "Terraced",
    "F": "Flat/Maisonette", "O": "Other",
}
DURATIONS = {"F": "Freehold", "L": "Leasehold", "U": "Unknown"}
PPD_CATEGORIES = {"A": "Standard Price Paid", "B": "Additional Price Paid"}
