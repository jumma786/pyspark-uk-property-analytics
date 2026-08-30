"""Tests for the declared schema.

The schema is the contract with a 5.5 GB file that has no header row. If a
column moves or a type is wrong, every downstream number is wrong quietly
rather than loudly -- so these tests check the contract itself.
"""
from __future__ import annotations

from pyspark.sql.types import IntegerType, StringType, TimestampType

from src.schema import (
    DURATIONS, PPD_CATEGORIES, PRICE_PAID_SCHEMA, PROPERTY_TYPES,
)


class TestSchema:
    def test_has_sixteen_columns_in_published_order(self):
        """Land Registry publishes 16 columns; the file has no header to check."""
        names = [f.name for f in PRICE_PAID_SCHEMA.fields]
        assert len(names) == 16
        assert names[0] == "transaction_id"
        assert names[1] == "price"
        assert names[2] == "date_of_transfer"
        assert names[3] == "postcode"
        assert names[-1] == "record_status"

    def test_price_is_integer_not_string(self):
        """A string price sorts lexically: 9 > 100000. Silent and wrong."""
        field = PRICE_PAID_SCHEMA["price"]
        assert isinstance(field.dataType, IntegerType)

    def test_date_of_transfer_is_timestamp_not_date(self):
        """The feed writes 1995-01-31 00:00. DateType parses that to NULL for
        every row without raising -- the fault this project exists to avoid."""
        field = PRICE_PAID_SCHEMA["date_of_transfer"]
        assert isinstance(field.dataType, TimestampType)

    def test_transaction_id_is_non_nullable(self):
        assert PRICE_PAID_SCHEMA["transaction_id"].nullable is False

    def test_postcode_is_nullable(self):
        """Some genuine transactions have no postcode; they are still real."""
        assert PRICE_PAID_SCHEMA["postcode"].nullable is True
        assert isinstance(PRICE_PAID_SCHEMA["postcode"].dataType, StringType)


class TestLookups:
    def test_property_type_codes_are_complete(self):
        assert set(PROPERTY_TYPES) == {"D", "S", "T", "F", "O"}

    def test_duration_codes_are_complete(self):
        assert set(DURATIONS) == {"F", "L", "U"}

    def test_ppd_category_codes_are_complete(self):
        assert set(PPD_CATEGORIES) == {"A", "B"}

    def test_lookups_have_no_blank_labels(self):
        for mapping in (PROPERTY_TYPES, DURATIONS, PPD_CATEGORIES):
            assert all(v and v.strip() for v in mapping.values())
