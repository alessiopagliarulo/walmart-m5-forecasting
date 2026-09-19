# walmart-m5-forecasting

Forecasting 28 days of daily unit sales for every product in one Walmart store, using
the real M5 competition data, with an evaluation harness built to be honest: verified
data, leakage-free features and walk-forward validation. The full plan is in
[docs/PLAN.md](docs/PLAN.md).

## Status

Only the foundation is built (issue #1):

- **Data download and verification** - fetches the four M5 files from a pinned public
  mirror, checks each file's sha256 against [provenance/m5_data.json](provenance/m5_data.json),
  and checks the data against documented M5 facts (30,490 series, 3,049 products,
  10 stores, 3 states, 1,941 days) before anything uses it.
- **One-store subset** - default store `CA_1`, selectable with `--store`.
- **Feature pipeline** - 52 model inputs: sales lags, rolling mean/std, zero-sales share,
  price, calendar, events and SNAP. A test proves no feature reads the future.

No model is trained yet and there are no forecast results. Baselines, scikit-learn,
XGBoost/LightGBM, SHAP and the write-up are issues #2-#6.

## Run it

Needs [uv](https://docs.astral.sh/uv/). About 330 MB download; the feature build takes
about 12 seconds and peaks at about 3.5 GB of memory on an Apple-silicon laptop.

```sh
uv sync
uv run m5-download          # fetch into data/raw (gitignored), verify hashes and facts
uv run m5-features          # build data/processed/features_CA_1.parquet
uv run m5-features --store TX_2
```

`m5-features` writes a small manifest, `results/features_<store>_manifest.json`, that
records the store, row counts, date range, the feature list and the input file hashes.

## Checks

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

The tests run on a tiny **synthetic** fixture in `tests/fixtures/synthetic_m5/` that
only copies the M5 file layout. It is never used to produce a published number.

## Data

Walmart M5 data from the Kaggle
[M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
competition, fetched from the HuggingFace mirror
[denephew/M5_Forecasting](https://huggingface.co/datasets/denephew/M5_Forecasting)
(license field: cc) at a pinned revision. Raw data is never committed.

## Layout

- `src/m5/` - `download.py`, `verify.py`, `data.py` (one-store loaders),
  `features.py`, `leakage.py` (the look-ahead check), `pipeline.py`, `config.py`
- `tests/` - pytest suite and the synthetic fixture
- `provenance/` - source URLs and sha256 of the raw files
- `results/` - small committed artifacts produced by the scripts
