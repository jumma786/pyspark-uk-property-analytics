"""Full analysis straight from the 5.5 GB CSV, with timings.

Why this exists
---------------
The normal path is ``run_pipeline --stage ingest`` to build a partitioned
Parquet layer, then query that. Parquet on Windows needs ``winutils.exe`` and
``hadoop.dll`` on ``HADOOP_HOME``. Without them it is not only writes that
fail: reading a partitioned Parquet directory needs the same native shim to
list leaf files and dies with ``UnsatisfiedLinkError``. Reading a single CSV
file is the one path that still works, which is what this script is built on.

This script takes the slower road deliberately -- reading the raw CSV for each
action -- so the analysis can run and be measured on a machine without the
Hadoop shim. The cost of that choice is visible in the timings it prints, which
is itself the argument for the Parquet layer.

    python scripts/analyse_from_csv.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyspark.sql import functions as F                      # noqa: E402

from src.analytics import (                                  # noqa: E402
    fastest_growing_districts, new_build_premium,
    transaction_volume_by_quarter,
)
from src.config import build_spark                           # noqa: E402
from src.ingest import read_raw_csv                          # noqa: E402
from src.joins import join_regions, region_lookup, region_summary  # noqa: E402
from src.transforms import clean, enrich                     # noqa: E402

OUT = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT = ROOT / "data" / "output"
CSV_OUT.mkdir(parents=True, exist_ok=True)

report: dict = {"timings": {}, "results": {}}


def step(name: str):
    """Context-managed timing that records into the report."""
    class _T:
        def __enter__(self):
            self.t = time.perf_counter()
            print(f"  -> {name} ...", flush=True)
            return self

        def __exit__(self, *exc):
            secs = round(time.perf_counter() - self.t, 1)
            report["timings"][name] = secs
            print(f"     {name}: {secs}s", flush=True)
    return _T()


def save(df, name: str, limit: int | None = None):
    """Collect a summary to pandas and write it -- no Hadoop writer involved."""
    pdf = (df.limit(limit) if limit else df).toPandas()
    pdf.to_csv(CSV_OUT / f"{name}.csv", index=False)
    return len(pdf)


def main() -> None:
    spark = build_spark("analyse-from-csv")
    try:
        raw = read_raw_csv(spark)
        df = enrich(clean(raw))

        with step("count_raw_rows"):
            raw_n = raw.count()
        with step("count_clean_rows"):
            clean_n = df.count()

        report["results"]["quality"] = {
            "rows_raw": raw_n,
            "rows_clean": clean_n,
            "rows_dropped": raw_n - clean_n,
            "pct_dropped": round(100.0 * (raw_n - clean_n) / raw_n, 4),
        }

        with step("headline_stats"):
            row = df.agg(
                F.min("year").alias("year_min"),
                F.max("year").alias("year_max"),
                F.countDistinct("district").alias("districts"),
                F.countDistinct("postcode_area").alias("postcode_areas"),
                F.percentile_approx("price", 0.5, 10_000).alias("median_price_all_time"),
            ).first()
            report["results"]["headline"] = row.asDict()

        with step("national_median_by_year"):
            nat = (df.groupBy("year")
                     .agg(F.percentile_approx("price", 0.5, 10_000).alias("median_price"),
                          F.count("*").alias("transactions"))
                     .orderBy("year"))
            save(nat, "national_median_by_year")
            report["results"]["national_median_by_year"] = [
                r.asDict() for r in nat.collect()
            ]

        with step("skew_profile_postcode_area"):
            total = clean_n
            skew = (df.groupBy("postcode_area")
                      .agg(F.count("*").alias("rows"))
                      .withColumn("pct_of_total",
                                  F.round(100.0 * F.col("rows") / total, 3))
                      .orderBy(F.desc("rows")).limit(10))
            report["results"]["skew_top10"] = [r.asDict() for r in skew.collect()]

        with step("new_build_premium"):
            nb = new_build_premium(df)
            save(nb, "new_build_premium")
            report["results"]["new_build_premium_recent"] = [
                r.asDict() for r in nb.filter(F.col("year") >= 2020)
                                      .orderBy("year", "property_type_name").collect()
            ]

        with step("transaction_volume_by_quarter"):
            vol = transaction_volume_by_quarter(df)
            save(vol, "transaction_volume_by_quarter")
            crisis = [r.asDict() for r in vol.filter(
                F.col("year").isin(2007, 2008, 2009, 2019, 2020, 2021)).collect()]
            report["results"]["volume_crisis_years"] = crisis

        with step("fastest_growing_districts"):
            fast = fastest_growing_districts(df, since_year=2000, top_n=15)
            save(fast, "fastest_growing_districts")
            report["results"]["fastest_growing_districts"] = [
                r.asDict() for r in fast.collect()
            ]

        lookup = region_lookup(spark)

        with step("join_broadcast"):
            n_broadcast = (join_regions(df, lookup, broadcast=True)
                           .groupBy("region").count().count())
        with step("join_sort_merge_forced"):
            prior = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
            spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
            try:
                n_shuffle = (join_regions(df, lookup, broadcast=False)
                             .groupBy("region").count().count())
            finally:
                spark.conf.set("spark.sql.autoBroadcastJoinThreshold", prior)
        report["results"]["join_regions_equal"] = (n_broadcast == n_shuffle)

        with step("region_summary"):
            reg = region_summary(join_regions(df, lookup))
            save(reg, "region_summary")
            report["results"]["region_latest"] = [
                r.asDict() for r in reg.filter(F.col("year") == 2024).collect()
            ]

    finally:
        spark.stop()

    (OUT / "results_from_csv.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n" + json.dumps(report["results"]["quality"], indent=2))
    print("\nTimings:", json.dumps(report["timings"], indent=2))
    print(f"\nWrote {OUT / 'results_from_csv.json'}")


if __name__ == "__main__":
    main()
