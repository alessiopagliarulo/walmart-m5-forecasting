# Resume claims and their evidence

Each resume bullet is checked against the committed artifacts. Status is one of
**verified** (true as worded), **differs** (the measured fact differs from the wording;
use the suggested wording instead) or **not yet built**.

Every number below is copied from a script-generated file. To regenerate them:
`uv run m5-features`, then `uv run m5-backtest`, then `uv run m5-plots`.

## Scope: what the numbers cover

- **One store only: `CA_1`** (California), 3,049 item series. Nothing here is measured
  on the other 9 stores or the full 30,490-series competition data, so no claim may
  imply more than one store.
- **3 walk-forward folds**, each training on every day up to its cutoff and forecasting
  the next 28 days (`folds` in `results/metrics.json`):

  | Fold | Train through | Forecast |
  | --- | --- | --- |
  | 1 | 2016-02-28 (d_1857) | 2016-02-29 to 2016-03-27 |
  | 2 | 2016-03-27 (d_1885) | 2016-03-28 to 2016-04-24 |
  | 3 | 2016-04-24 (d_1913) | 2016-04-25 to 2016-05-22 |

- **Metric**: the competition's WRMSSE over its 12 levels, lower is better. On one
  store the 12 levels reduce to 4 distinct ones, so the score is not comparable with
  the Kaggle leaderboard, which covers all 10 stores. "Mean WRMSSE" is the average
  over the 3 folds.
- **Evidence files**: `results/metrics.json` (per-model, per-fold WRMSSE, search
  candidates, seed, versions, git commit), `results/metrics.md` (the same as tables),
  `results/features_CA_1_manifest.json` (store, series count, feature list),
  `results/forecast_vs_actual_CA_1.png` (plot).

## Measured results (mean WRMSSE over 3 folds, from `results/metrics.json`)

| Model | Family | Mean WRMSSE | vs seasonal-naive |
| --- | --- | --- | --- |
| seasonal_naive | naive baseline | 0.7778 | - |
| linear_regression | linear | 0.8144 | 4.7% worse |
| ridge | linear | 0.5916 | 23.9% lower |
| hurdle_logistic_ridge | logistic hurdle (classifier x regressor) | 0.5783 | 25.7% lower |
| lightgbm | gradient boosting | 0.5253 | 32.5% lower |
| xgboost | gradient boosting | 0.5235 | 32.7% lower |

The simple linear regression loses to seasonal-naive. The boosted models beat Ridge
by about 11% (0.5253 and 0.5235 vs 0.5916) and the hurdle model by about 9%.
XGBoost and LightGBM are within 0.002 of each other: XGBoost is ahead on folds 2 and 3,
LightGBM on fold 1. That gap is too small to call either model the winner.

## Claim 1 - verified, with a scope note

> "Forecast 28-day daily sales for 3,049 Walmart products with a LightGBM and XGBoost
> pipeline on a Tweedie loss"

- 28-day horizon: `horizon_days: 28` in `results/metrics.json` and the manifest.
- 3,049 products: `data.n_series: 3049` in `results/metrics.json` and `n_series: 3049`
  in `results/features_CA_1_manifest.json`. These are the 3,049 items sold in store CA_1.
- LightGBM and XGBoost with Tweedie loss: `models.lightgbm.fixed_params.objective:
  "tweedie"` and `models.xgboost.fixed_params.objective: "reg:tweedie"` in
  `results/metrics.json`. The code is in `src/m5/models.py`.

Accurate as worded. To stop it reading as all of Walmart, you can add the store:
"Forecast 28-day daily sales for 3,049 products in one Walmart store with a LightGBM
and XGBoost pipeline on a Tweedie loss".

## Claim 2 - differs

> "Cut WRMSSE error 27% vs. a seasonal-naive baseline using 40+ lag, rolling-mean and
> SNAP/holiday features"

- Measured improvement: the best model, XGBoost, scores a mean WRMSSE of 0.5235 against
  0.7778 for seasonal-naive, which is **32.7% lower**. LightGBM scores 0.5253, which is
  **32.5% lower**. The 27% figure was a target set before this repo existed. Nothing was
  tuned toward it: the search picks candidates by WRMSSE on each fold's own training
  days, never on the test windows.
- Features: `n_features: 52` in `results/features_CA_1_manifest.json`. Only 33 of them
  are the lag, rolling-window and SNAP/holiday kind named in the claim: 12 sales lags,
  6 rolling means, 6 rolling standard deviations, 2 zero-sales shares, 1 same-weekday
  mean, 5 event fields and 1 SNAP flag. The other 19 are 7 price, 7 calendar, 3 product
  identity (item, department, category) and 2 other fields (year, days since release).
  So "40+ lag, rolling-mean and SNAP/holiday features" overstates that group. "52
  features" is true of the whole set.

Suggested wording: **"Cut WRMSSE 33% vs. a seasonal-naive baseline (0.52 vs. 0.78,
3-fold walk-forward backtest) using 52 lag, rolling, price, calendar and SNAP/holiday
features"**.

## Claim 3 - partly not yet built

> "Benchmarked 4 model families under leakage-free walk-forward CV, ranking drivers
> with SHAP across 3,049 series"

- 4 model families: verified if the grouping in the table above is accepted. It has 6
  models in 4 families: naive baseline, linear (OLS and Ridge), logistic hurdle and
  gradient boosting. All 6 models are in `results/metrics.json` with identical folds.
- Leakage-free walk-forward CV: verified by tests, which run on synthetic data in CI.
  `tests/test_leakage.py` scrambles every input after a cutoff and requires every
  feature up to the cutoff to stay the same. `test_forecasts_ignore_sales_after_the_cutoff`
  in `tests/test_backtest.py` requires every model's forecast, the boosted models' grid
  search included, to stay the same when every sale after the cutoff is scrambled.
  `test_boosted_search_stays_inside_the_training_window` checks that the search only
  validates on the last 28 training days.
- Across 3,049 series: verified (`data.n_series` in `results/metrics.json`).
- **Ranking drivers with SHAP: not yet built.** This is issue #5, and no SHAP artifact
  exists in the repo yet.

Wording until #5 ships: **"Benchmarked 6 models in 4 families (naive, linear, logistic
hurdle, gradient boosting) under leakage-free walk-forward CV across 3,049 series"**.
Add "ranking drivers with SHAP" back only when #5 commits its SHAP artifact.

## Plots

- **Forecast vs actual**: `results/forecast_vs_actual_CA_1.png`. `m5-plots` draws it
  from the `store_daily_units` in `results/metrics.json`. It shows whole-store daily
  units over the 84 test days, with the actual sales and the forecasts of
  seasonal-naive, Ridge, LightGBM and XGBoost.
  - All models follow the weekly cycle: weekend peaks of about 6,000-6,800 units and
    weekday troughs of about 3,500-4,000 units.
  - LightGBM and XGBoost lie almost on top of each other. They track weekdays closely
    but fall short of the tallest weekend peaks: on Sunday 6 March XGBoost forecasts
    5,713 units against 6,829 sold, on 3 April 5,970 against 6,496, and on 14-15 May
    about 5,700-5,800 against 6,245 and 6,707. This comes from `store_daily_units`
    in `results/metrics.json`.
  - Seasonal-naive repeats its training week's quirks. Fold 2 copies the week ending on
    Easter Sunday (27 March), when sales dipped, so it forecasts 4,669 units for every
    Sunday in fold 2 against actual Sundays of about 6,000-6,500 (6,496 on 3 April).
  - Ridge is close to the boosted models on most days but sits lower on several
    troughs and peaks, most visibly in fold 3.
- **SHAP summary plot: pending issue #5.**
