"""Join strategy: broadcast vs shuffle, and key skew.

The reference table here is tiny -- around 120 rows. Joining it to ~30M rows is
the textbook case for a broadcast hash join, and the benchmark measures the
difference against a forced shuffle rather than asserting it.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, functions as F

# Postcode area -> broad UK region. Deliberately small: this is the object the
# project broadcasts. Areas are the alphabetic prefix of a UK postcode.
#
# Scope note, because it changes how the output reads: Land Registry Price Paid
# covers England and Wales only. Fifteen entries below (AB DD DG EH FK G HS IV
# KA KW KY ML PA PH ZE, plus BT for Northern Ireland) therefore never match a
# row in this feed -- verified against a 3M-row sample of the source file. They
# are kept so the table is a correct UK-wide reference, but `region_summary`
# will not contain Scotland or Northern Ireland, and that is the data, not a
# join fault.
#
# TD is the exception that matters. The area straddles the border, and in a
# feed restricted to England and Wales every TD row is English (Berwick-upon-
# Tweed and the Northumberland border). Mapping it to Scotland -- as this table
# previously did -- produced a "Scotland" region built entirely from English
# transactions. It is mapped to North East accordingly.
POSTCODE_AREA_REGIONS = [
    ("AB", "Scotland"), ("AL", "East of England"), ("B", "West Midlands"),
    ("BA", "South West"), ("BB", "North West"), ("BD", "Yorkshire"),
    ("BH", "South West"), ("BL", "North West"), ("BN", "South East"),
    ("BR", "London"), ("BS", "South West"), ("BT", "Northern Ireland"),
    ("CA", "North West"), ("CB", "East of England"), ("CF", "Wales"),
    ("CH", "North West"), ("CM", "East of England"), ("CO", "East of England"),
    ("CR", "London"), ("CT", "South East"), ("CV", "West Midlands"),
    ("CW", "North West"), ("DA", "London"), ("DD", "Scotland"),
    ("DE", "East Midlands"), ("DG", "Scotland"), ("DH", "North East"),
    ("DL", "North East"), ("DN", "Yorkshire"), ("DT", "South West"),
    ("DY", "West Midlands"), ("E", "London"), ("EC", "London"),
    ("EH", "Scotland"), ("EN", "London"), ("EX", "South West"),
    ("FK", "Scotland"), ("FY", "North West"), ("G", "Scotland"),
    ("GL", "South West"), ("GU", "South East"), ("HA", "London"),
    ("HD", "Yorkshire"), ("HG", "Yorkshire"), ("HP", "South East"),
    ("HR", "West Midlands"), ("HS", "Scotland"), ("HU", "Yorkshire"),
    ("HX", "Yorkshire"),
    ("IG", "London"), ("IP", "East of England"), ("IV", "Scotland"),
    ("KA", "Scotland"), ("KT", "South East"), ("KW", "Scotland"),
    ("KY", "Scotland"), ("L", "North West"), ("LA", "North West"),
    ("LD", "Wales"), ("LE", "East Midlands"), ("LL", "Wales"),
    ("LN", "East Midlands"), ("LS", "Yorkshire"), ("LU", "East of England"),
    ("M", "North West"), ("ME", "South East"), ("MK", "South East"),
    ("ML", "Scotland"), ("N", "London"), ("NE", "North East"),
    ("NG", "East Midlands"), ("NN", "East Midlands"), ("NP", "Wales"),
    ("NR", "East of England"), ("NW", "London"), ("OL", "North West"),
    ("OX", "South East"), ("PA", "Scotland"), ("PE", "East of England"),
    ("PH", "Scotland"), ("PL", "South West"), ("PO", "South East"),
    ("PR", "North West"), ("RG", "South East"), ("RH", "South East"),
    ("RM", "London"), ("S", "Yorkshire"), ("SA", "Wales"),
    ("SE", "London"), ("SG", "East of England"), ("SK", "North West"),
    ("SL", "South East"), ("SM", "London"), ("SN", "South West"),
    ("SO", "South East"), ("SP", "South West"), ("SR", "North East"),
    ("SS", "East of England"), ("ST", "West Midlands"), ("SW", "London"),
    ("SY", "Wales"), ("TA", "South West"), ("TD", "North East"),
    ("TF", "West Midlands"), ("TN", "South East"), ("TQ", "South West"),
    ("TR", "South West"), ("TS", "North East"), ("TW", "London"),
    ("UB", "London"), ("W", "London"), ("WA", "North West"),
    ("WC", "London"), ("WD", "East of England"), ("WF", "Yorkshire"),
    ("WN", "North West"), ("WR", "West Midlands"), ("WS", "West Midlands"),
    ("WV", "West Midlands"), ("YO", "Yorkshire"), ("ZE", "Scotland"),
]


def region_lookup(spark: SparkSession) -> DataFrame:
    """The small side of the join."""
    return spark.createDataFrame(POSTCODE_AREA_REGIONS,
                                 ["postcode_area", "region"])


def join_regions(df: DataFrame, lookup: DataFrame, *,
                 broadcast: bool = True) -> DataFrame:
    """Attach region.

    ``broadcast=False`` forces a sort-merge join so the benchmark can compare.
    In a real pipeline this would always be broadcast: the right side is a few
    kilobytes, and shuffling 30M rows to match it is pure waste.
    """
    right = F.broadcast(lookup) if broadcast else lookup
    return df.join(right, on="postcode_area", how="left")


def skew_profile(df: DataFrame, key: str = "postcode_area",
                 top_n: int = 10) -> DataFrame:
    """Show how uneven the join key is.

    UK property transactions concentrate heavily: a handful of postcode areas
    carry a disproportionate share of rows. That imbalance is what makes a naive
    shuffle join on this key slow -- one task receives far more rows than the
    rest and the stage waits on it.
    """
    total = df.count()
    return (
        df.groupBy(key)
          .agg(F.count("*").alias("rows"))
          .withColumn("pct_of_total", F.round(100.0 * F.col("rows") / total, 3))
          .orderBy(F.desc("rows"))
          .limit(top_n)
    )


def region_summary(df: DataFrame) -> DataFrame:
    """Median price and volume by region and year -- the joined output."""
    return (
        df.filter(F.col("region").isNotNull())
          .groupBy("region", "year")
          .agg(F.percentile_approx("price", 0.5, 10_000).alias("median_price"),
               F.count("*").alias("transactions"))
          .orderBy("region", "year")
    )
