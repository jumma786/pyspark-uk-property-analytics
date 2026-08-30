"""Tests for the region reference table.

The lookup is small and hand-maintained, which is exactly the kind of asset
that rots quietly. These tests fail loudly if it does.
"""
from __future__ import annotations

from src.joins import POSTCODE_AREA_REGIONS


class TestRegionLookup:
    def test_postcode_areas_are_unique(self):
        """A duplicate key silently multiplies rows in a left join."""
        areas = [a for a, _ in POSTCODE_AREA_REGIONS]
        assert len(areas) == len(set(areas)), "duplicate postcode area in lookup"

    def test_areas_are_uppercase_alphabetic(self):
        for area, _ in POSTCODE_AREA_REGIONS:
            assert area.isalpha() and area.isupper()
            assert 1 <= len(area) <= 2

    def test_regions_are_from_a_closed_set(self):
        expected = {
            "London", "South East", "South West", "East of England",
            "East Midlands", "West Midlands", "North West", "North East",
            "Yorkshire", "Wales", "Scotland", "Northern Ireland",
        }
        assert {r for _, r in POSTCODE_AREA_REGIONS} <= expected

    def test_covers_major_areas(self):
        areas = {a for a, _ in POSTCODE_AREA_REGIONS}
        for a in ("SW", "M", "B", "LS", "EH", "CF", "BT"):
            assert a in areas, f"missing major postcode area {a}"

    def test_border_area_td_is_not_mapped_to_scotland(self):
        """Price Paid is England and Wales only, so every TD row in this feed
        is English (Berwick-upon-Tweed and the Northumberland border). Mapping
        the area to Scotland produced a Scotland region made entirely of
        English transactions."""
        lookup = dict(POSTCODE_AREA_REGIONS)
        assert lookup["TD"] == "North East"

    def test_lookup_is_small_enough_to_broadcast(self):
        """The whole point of the broadcast join is that this stays tiny."""
        assert len(POSTCODE_AREA_REGIONS) < 1000
