# walmart-m5-forecasting

Forecasting 28 days of daily unit sales for every product in one Walmart store, using
the real M5 competition data, with an evaluation harness built to be honest: verified
data, leakage-free features and walk-forward validation. The full plan is in
[docs/PLAN.md](docs/PLAN.md).

## Results

The full write-up, with the method, what drives the forecasts, where the models are
weak and what comes next, is [docs/RESULTS.md](docs/RESULTS.md). The numbers below
are regenerated from the committed artifacts by `uv run m5-writeup`, and a test fails
if any of them drifts.

<!-- GENERATED:headline:BEGIN -->

- **xgboost** has the lowest mean WRMSSE, **0.524**, over 3 walk-forward folds on the 3,049 item series of store CA_1: **32.7% lower than the seasonal-naive baseline** (0.778).
- **lightgbm** is 0.0018 behind (0.525), too close to call either model the winner. Ridge and the hurdle model score 0.592 and 0.578; the plain linear regression (0.814) is worse than the baseline.

<!-- GENERATED:headline:END -->

<!-- GENERATED:model_table:BEGIN -->

| Model | Family | Fold 1 | Fold 2 | Fold 3 | Mean WRMSSE | vs seasonal-naive | Mean MAE (units) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive | naive baseline | 0.7661 | 0.9075 | 0.6598 | 0.7778 | - | 1.3228 |
| linear_regression | linear | 0.8570 | 0.7976 | 0.7885 | 0.8144 | 4.7% worse | 1.1726 |
| ridge | linear | 0.6205 | 0.5351 | 0.6193 | 0.5916 | 23.9% lower | 1.1315 |
| hurdle_logistic_ridge | logistic hurdle (classifier x regressor) | 0.6045 | 0.5269 | 0.6033 | 0.5783 | 25.7% lower | 1.1429 |
| lightgbm | gradient boosting | 0.5728 | 0.4740 | 0.5292 | 0.5253 | 32.5% lower | 1.1175 |
| xgboost | gradient boosting | 0.5778 | 0.4738 | 0.5189 | **0.5235** | 32.7% lower | 1.1146 |

<!-- GENERATED:model_table:END -->

![Whole-store daily units, actual vs forecast, over the three test windows](results/forecast_vs_actual_CA_1.png)

**What drives the forecasts.** SHAP on both boosted models shows they forecast an item
mostly from its sales over the last two to four weeks, adjusted by which item it is and
the day of the week; price, holidays and the SNAP flag matter at the margin.

![Mean |SHAP| of the top 15 features, LightGBM and XGBoost](results/shap_importance_CA_1.png)

**SNAP days, measured from the data (no model).** Food sells more on the days SNAP
(food stamp) benefits are paid, compared within the same month and weekday. Part of that
is a start-of-month effect that lifts non-food too, so the last columns net it out:

<!-- GENERATED:snap_table:BEGIN -->

| Scope | Food lift | 95% CI | Non-food lift | Food beyond non-food | 95% CI |
| --- | --- | --- | --- | --- | --- |
| store CA_1 | +11.8% | +10.3% to +13.4% | +3.8% | +7.8% | +5.8% to +9.7% |
| state CA (all stores) | +9.8% | +8.4% to +11.2% | +2.2% | +7.5% | +5.8% to +9.1% |

SNAP days in California are the 1st to the 10th of each month. Compared within 448 same-month, same-weekday cells from 2011-01-29 to 2016-05-22; the intervals resample whole months (2,000 draws).

<!-- GENERATED:snap_table:END -->

**What this does not show.**

- **One store, not all ten.** Everything is measured on store CA_1 only.
- **A backtest, not a live deployment.** No forecast was used to order stock or checked
  against sales that had not yet happened.
- **Not comparable to the Kaggle leaderboard**, which scores all ten stores on a hidden
  period; on one store the metric's 12 levels reduce to 4. The scores only rank the
  models against each other.
- **Three folds** rank models that differ clearly but cannot separate the two boosted
  models or put an error bar on a score.

## Status

Built: the foundation (issue #1), the evaluation harness, baselines (#2),
scikit-learn models (#3), gradient boosting (#4), SHAP explanations (#5) and the
results write-up (#6).

- **Data download and verification** - fetches the four M5 files from a pinned public
  mirror, checks each file's sha256 against [provenance/m5_data.json](provenance/m5_data.json),
  and checks the data against documented M5 facts (30,490 series, 3,049 products,
  10 stores, 3 states, 1,941 days) before anything uses it.
- **One-store subset** - default store `CA_1`, selectable with `--store`.
- **Feature pipeline** - 52 model inputs: sales lags, rolling mean/std, zero-sales share,
  price, calendar, events and SNAP. A test proves no feature reads the future.

- **Evaluation harness** - 3 back-to-back walk-forward folds of 28 days ending on the
  last day of sales, scored with the competition's WRMSSE over its 12 aggregation levels
  (plus RMSSE and MAE per series). A test proves no model's forecast changes when every
  sale after the cutoff is scrambled.
- **Models** - seasonal-naive (repeat last week), a simple linear regression on five
  sales-history features, Ridge on the feature pipeline, and a hurdle model where a
  LogisticRegression predicts whether an item sells at all on a day (about half of all
  item-days sell nothing) and a Ridge predicts how much when it does.
- **Gradient boosting** - LightGBM and XGBoost with Tweedie loss on the same features
  and folds. Each fold runs a small grid search (Tweedie variance power x tree size)
  inside its own training days: it validates on the last 28 training days, picks the
  number of boosting rounds by early stopping there, then refits the best candidate on
  all training days. The fold's test window never takes part.
- **Explanations** - exact SHAP values for both boosted models on every forecast row
  of every fold, from models rebuilt to reproduce the backtest exactly: global and
  per-group importance, three series read in detail, and error by segment (new items,
  intermittent sellers, price changes). Plus the SNAP-day sales lift measured from the
  raw data alone, per category, matched by month and weekday.

Measured results, regenerated by `m5-backtest`: [results/metrics.md](results/metrics.md)
(full detail, including every search candidate, in
[results/metrics.json](results/metrics.json)). `m5-plots` draws
[results/forecast_vs_actual_CA_1.png](results/forecast_vs_actual_CA_1.png) from that
file. `m5-explain` writes [results/explanations.md](results/explanations.md) (with
[results/shap_summary.json](results/shap_summary.json) and
[results/snap_lift.json](results/snap_lift.json)) and the SHAP plots, for example
[results/shap_summary_xgboost_CA_1.png](results/shap_summary_xgboost_CA_1.png).
`m5-writeup` rewrites every number in the Results section above,
[docs/RESULTS.md](docs/RESULTS.md) and [docs/RESUME_CLAIMS.md](docs/RESUME_CLAIMS.md)
(what each resume claim rests on) from those files.

## Run it

Needs [uv](https://docs.astral.sh/uv/). About 330 MB download; the feature build takes
about 12 seconds and peaks at about 3.5 GB of memory on an Apple-silicon laptop.

```sh
uv sync
uv run m5-download          # fetch into data/raw (gitignored), verify hashes and facts
uv run m5-features          # build data/processed/features_CA_1.parquet
uv run m5-features --store TX_2
uv run m5-backtest          # every model on the 3 folds -> results/metrics.json + .md
uv run m5-plots             # forecast-vs-actual plot from results/metrics.json
uv run m5-explain           # SHAP + SNAP lift -> results/shap_summary.json, snap_lift.json,
                            # explanations.md and the SHAP plots
uv run m5-writeup           # rewrite the write-up's numbers from results/ (--check: only compare)
```

`m5-backtest` takes about 25 minutes and peaks at about 18 GB of memory on the same
laptop. Most of the time goes to the boosted models' per-fold grid searches. It refuses a feature table that
was not built from the verified raw files. It logs every model and fold to MLflow in
the gitignored `mlruns/` folder (browse with
`uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`) and records package
versions, the git commit, the seed, fold boundaries and runtimes in `metrics.json`.

`m5-explain` takes about 4 minutes. It needs `m5-backtest`'s `metrics.json` and the raw
files, and stops unless every rebuilt model reproduces its recorded WRMSSE.

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
  `features.py`, `leakage.py` (the look-ahead check), `pipeline.py`, `config.py`,
  `evaluation.py` (folds and WRMSSE), `models.py`, `backtest.py`, `plots.py`,
  `explain.py` (SHAP), `snap.py` (SNAP lift from the raw data), `writeup.py` (the
  write-up's generated regions)
- `tests/` - pytest suite and the synthetic fixture
- `docs/` - the results write-up, the plan and the resume-claim evidence
- `provenance/` - source URLs and sha256 of the raw files
- `results/` - small committed artifacts produced by the scripts
