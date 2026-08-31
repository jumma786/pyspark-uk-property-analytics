# PySpark UK Property Analytics

[![Tests](https://github.com/jumma786/pyspark-uk-property-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/jumma786/pyspark-uk-property-analytics/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PySpark](https://img.shields.io/badge/pyspark-3.5.3-orange)
![Data](https://img.shields.io/badge/data-HM%20Land%20Registry-0b5fff)

A single-machine PySpark analytics project for the UK Land Registry Price Paid
dataset. It converts the 5.5 GB raw CSV into a partitioned Parquet layer, runs
district, postcode-area, and regional housing-market analytics, and benchmarks
the engineering choices that make the pipeline practical on local hardware.

The project is built to demonstrate production-style data engineering habits:
explicit schemas, reproducible stages, measured performance claims, tested
business logic, and clear separation between raw data, transformed storage, and
analytical outputs.

## What This Project Does

- Reads the official HM Land Registry Price Paid CSV with an explicit Spark
  schema.
- Cleans invalid, deleted, missing-date, and implausible-price records.
- Enriches transactions with year, month, quarter, postcode area, property type,
  tenure, and new-build flags.
- Writes a Snappy-compressed Parquet layer partitioned by year.
- Produces housing-market summaries using aggregations, joins, and window
  functions.
- Benchmarks CSV vs Parquet, schema inference vs explicit schema, broadcast vs
  shuffle joins, shuffle partition counts, and Adaptive Query Execution.
- Runs a focused PySpark test suite in GitHub Actions.

## Dataset

The pipeline expects the HM Land Registry Price Paid single-file CSV at:

```text
data/raw/pp-complete.csv
```

Download it from the official source:

- Dataset page: [Price Paid Data downloads](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
- Single-file CSV: [pp-complete.csv](https://price-paid-data.publicdata.landregistry.gov.uk/pp-complete.csv)
- Data terms: [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

The raw CSV, generated Parquet files, and analytical outputs are intentionally
ignored by Git because they are large and reproducible.

## Architecture

```text
data/raw/pp-complete.csv
        |
        v
src/ingest.py
  explicit schema -> clean -> enrich -> partitioned Parquet
        |
        v
data/parquet/price_paid/year=YYYY/*.parquet
        |
        v
src/analytics.py + src/joins.py
  aggregations, windows, region joins, skew profiling
        |
        v
data/output/
docs/results.json
```

## Repository Layout

```text
.
+-- .github/workflows/tests.yml   # CI for the PySpark test suite
+-- scripts/analyse_from_csv.py   # Windows-friendly CSV analysis fallback
+-- src/
|   +-- analytics.py              # Analytical outputs and window functions
|   +-- benchmark.py              # Measured performance comparisons
|   +-- config.py                 # Spark/session/path configuration
|   +-- ingest.py                 # CSV read and Parquet write layer
|   +-- joins.py                  # Region reference join and skew profiling
|   +-- run_pipeline.py           # Main orchestration entry point
|   +-- schema.py                 # Land Registry schema and code mappings
|   +-- transforms.py             # Cleaning and enrichment rules
+-- tests/                        # Unit tests over transforms, joins, schema, analytics
+-- pytest.ini
+-- requirements.txt
```

## Analytical Outputs

Running the analysis stage writes the following outputs under `data/output/`:

| Output | Purpose |
| --- | --- |
| `median_price_by_district_year` | Median, mean, and transaction count by district and year |
| `price_growth_by_district` | Year-on-year and cumulative district-level growth |
| `rolling_median_by_area` | Three-year rolling average of postcode-area yearly medians |
| `new_build_premium` | New-build premium versus existing stock by year and property type |
| `transaction_volume_by_quarter` | Quarterly transaction volume and median price |
| `region_summary` | Regional median price and volume after postcode-area lookup |
| `fastest_growing_districts` | Highest cumulative district growth since the configured base year |

The full pipeline also writes a structured execution report to
`docs/results.json`.

## Engineering Highlights

### Explicit Schema

The Price Paid file has no header row. The schema in `src/schema.py` avoids an
extra inference pass over a multi-gigabyte CSV and prevents silent date parsing
failures.

### Partitioned Parquet

The raw CSV is converted once into Snappy-compressed Parquet partitioned by
`year`, enabling partition pruning and columnar reads for downstream analysis.

### Honest Aggregates

The cleaning layer keeps only live records, removes implausible prices, and
preserves unknown new-build flags as nulls instead of treating them as existing
stock.

### Broadcast Join Strategy

The postcode-area-to-region lookup is small enough to broadcast. The benchmark
compares that plan against a forced sort-merge join so the performance claim is
measured, not assumed.

### Window Functions

Growth and rolling-median calculations use Spark windows carefully. Year-on-year
growth uses a range window so missing district-years do not get mislabeled as
adjacent-year growth.

## Quick Start

### 1. Create an environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Java

PySpark 3.5.3 supports Java 8, 11, and 17. This project is developed and tested
with Java 11.

### 3. Add the source CSV

Place the downloaded file here:

```text
data/raw/pp-complete.csv
```

### 4. Run the pipeline

```bash
python -m src.run_pipeline --stage all
```

You can also run individual stages:

```bash
python -m src.run_pipeline --stage ingest
python -m src.run_pipeline --stage analyse
python -m src.run_pipeline --stage benchmark
```

## Windows Note

Spark Parquet reads and writes on a local Windows filesystem may require the
Hadoop native shim (`winutils.exe` and `hadoop.dll`) configured through
`HADOOP_HOME`. If the Parquet route is not available on your machine, this
project includes a CSV-based fallback:

```bash
python scripts/analyse_from_csv.py
```

The fallback is slower because it reads the raw CSV for each action, but it
avoids the local Hadoop filesystem dependency and writes small summary CSVs via
pandas.

## Testing

Run the full test suite:

```bash
pytest
```

The tests use small hand-built Spark DataFrames, so they do not require the
5.5 GB source file. The current suite covers schema behavior, cleaning and
enrichment, region joins, skew profiling, growth calculations, rolling windows,
and new-build premium logic.

## CI

GitHub Actions runs the test suite on every push and pull request to `main`.
The workflow installs Java 11, Python 3.11, project dependencies, and then runs:

```bash
pytest
```

## Example Commands

Build the Parquet layer only:

```bash
python -m src.run_pipeline --stage ingest
```

Run analytics after Parquet exists:

```bash
python -m src.run_pipeline --stage analyse
```

Run measured performance comparisons:

```bash
python -m src.run_pipeline --stage benchmark
```

Run the Windows-friendly CSV fallback:

```bash
python scripts/analyse_from_csv.py
```

## Notes On Reproducibility

- `data/raw/`, `data/parquet/`, and `data/output/` are excluded from version
  control because they are large generated artifacts.
- `docs/results.json` is generated by the full pipeline and captures runtime
  environment details, stage timings, output row counts, headline metrics, and
  benchmark results.
- Performance numbers depend on CPU count, memory, storage speed, OS cache
  state, Java version, and Spark configuration.

## License

This repository does not currently include a code license. The source data is
published by HM Land Registry and is available under the Open Government
Licence, subject to the terms linked above.
