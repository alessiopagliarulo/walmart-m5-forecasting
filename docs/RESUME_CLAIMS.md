# Resume claims and their evidence

Each resume bullet is checked against the committed artifacts. Status is one of
**verified** (true as worded), **differs** (the measured fact differs from the wording;
use the suggested wording instead) or **not yet built**.

Every number below is copied from a script-generated file. To regenerate them:
`uv run m5-features`, then `uv run m5-backtest`, then `uv run m5-plots` and
`uv run m5-explain`.

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
  `results/forecast_vs_actual_CA_1.png` (plot), `results/shap_summary.json` and
  `results/explanations.md` (SHAP), `results/snap_lift.json` (SNAP lift),
  `results/shap_summary_{lightgbm,xgboost}_CA_1.png` and
  `results/shap_importance_CA_1.png` (SHAP plots).

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

## Claim 3 - verified

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
- Ranking drivers with SHAP: verified (issue #5). `m5-explain` rebuilds the LightGBM and
  XGBoost model of every fold from the choice recorded in `results/metrics.json` and
  refuses to go on unless the rebuilt model reproduces the recorded WRMSSE (it does,
  exactly: `wrmsse_rebuilt` equals `wrmsse_recorded` in `results/shap_summary.json`).
  It then computes exact TreeSHAP values for every row of every fold's 28-day test
  window, 85,372 rows per fold (3,049 series x 28 days), so no sampling is involved.
  The ranking is in `mean_importance` in `results/shap_summary.json`.

Suggested wording: **"Benchmarked 6 models in 4 model families under leakage-free
walk-forward CV across 3,049 series, ranking forecast drivers with SHAP"**. The
original wording is also accurate.

## What SHAP says drives the forecasts

From `results/shap_summary.json` (mean |SHAP| over all 3 folds, every test row; SHAP
values are in log units because the Tweedie models forecast log(expected units)).
The plan's target was "the 28-day rolling mean, 28-day lag and sell price drive most
predictions". **That differs from what was measured:**

| Rank | LightGBM | share | XGBoost | share |
| --- | --- | --- | --- | --- |
| 1 | roll_mean_14 | 15.5% | roll_mean_14 | 19.2% |
| 2 | roll_mean_28 | 12.6% | item_id | 15.4% |
| 3 | item_id | 12.0% | roll_mean_28 | 11.1% |
| 4 | roll_std_182 | 8.0% | roll_mean_7 | 8.2% |
| 5 | day_of_week | 5.0% | roll_std_182 | 5.2% |

- The 28-day rolling mean is a top-3 driver in both models (12.6% and 11.1% of total
  mean |SHAP|), but the 14-day rolling mean ranks first in both.
- The 28-day lag (`lag_28`) ranks 20th in LightGBM and 14th in XGBoost (1.3% and
  1.6%). Sell price ranks 18th and 16th (1.4% and 1.5%). Together with `roll_mean_28`
  the three features account for about 15% and 14% of mean |SHAP|, not "most".
- By feature group (`mean_group_importance`): rolling means 41.4% (LightGBM) and
  45.9% (XGBoost), product identity (item, department, category) 16.7% and 19.7%, all
  7 price features together 4.2% and 3.3%.
- The ranking is stable: `roll_mean_14`, `roll_mean_28` and `item_id` are the top 3 in
  every fold of both models; only their order changes (per-fold table in
  `results/explanations.md`).

Where the models look fragile (fold 3, from `latest_fold.segments` and
`latest_fold.series`):

- Mostly-zero items (at least 75% zero days in the 112 days before the forecast
  origin; 31% of rows) are forecast about 24% too low (LightGBM -24.7%, XGBoost
  -24.1%), and their error is larger than their mean sales (MAE / mean 1.10 and 1.09).
  Regular sellers are almost unbiased (-0.8% and -1.0%).
- Sudden demand jumps are missed. The largest miss, FOODS_3_566, sold 160 units in the
  last 28 training days and 750 in the test window; the models forecast 167 and 156,
  because every sales feature is at least 28 days old.
- The models lean on item identity (`item_id` is 2nd or 3rd), which carries no
  information for a new item. Only 4 items (112 rows) were on sale for fewer than 182
  days in fold 3, too few to measure that well; XGBoost under-forecasts them by 19.9%,
  LightGBM by 4.2%.

Suggested wording if you want the finding on the resume: **"SHAP showed recent 14- and
28-day rolling means and item identity drive the forecasts; lags and price matter
little"**.

## SNAP-day lift, measured from the data (no model)

From `results/snap_lift.json`, computed by `m5-explain` from the raw sales, not from
any model. Method (in full in the file): in California SNAP benefits fall on the 1st to
the 10th of every month. Daily units per category are compared between SNAP and
non-SNAP days of the same calendar month and weekday (448 month-weekday cells,
2011-01-29 to 2016-05-22, Christmas Day dropped because the stores close). 95%
intervals come from a bootstrap over months (2,000 draws, seed 0).

| Scope | FOODS lift | 95% CI | Non-food lift | FOODS vs non-food |
| --- | --- | --- | --- | --- |
| Store CA_1 | +11.8% | +10.3% to +13.4% | +3.8% | +7.8% (+5.8% to +9.7%) |
| All 4 CA stores | +9.8% | +8.4% to +11.2% | +2.2% | +7.5% (+5.8% to +9.1%) |

The plan's target was "SNAP days lift food sales about 10%". **Verified, with a
caveat:** food sells 11.8% more on SNAP days at CA_1 (9.8% across California). Because
SNAP days are always the first 10 days of the month, part of that is a start-of-month
effect that lifts everything: non-food also sells 3.8% more on those days. Food's lift
beyond non-food is 7.8%. The models barely use the `snap` flag itself (0.4% and 0.1%
of mean |SHAP|).

Suggested wording: **"Measured a 12% lift in food sales on SNAP benefit days (8% above
non-food on the same days)"**.

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
- **SHAP summary plots**: `results/shap_summary_xgboost_CA_1.png` and
  `results/shap_summary_lightgbm_CA_1.png`, drawn by `m5-explain`. Each is a beeswarm
  of the fold 3 test window: 5,000 of the 85,372 rows, drawn at random with seed 0 (the
  numbers in the JSON use all rows). One dot per row; its position is the feature's
  SHAP value and its colour the feature's value (grey for categorical features).
  - In both models `roll_mean_14`, `item_id` and `roll_mean_28` have the widest
    spreads, roughly -1 to +1.3 in log units; `item_id` reaches about -2.4 (XGBoost)
    and -1.9 (LightGBM) for a few items.
  - For the rolling means, red (high recent sales) sits right of zero and blue left of
    it: the forecast follows recent sales, as expected.
  - `zero_frac_112` is mostly red to the right: given the other features, a larger
    share of recent zero days raises the forecast a little (up to about +0.4).
  - `day_of_week` pushes weekend days (red) up by about +0.1; `wday` is the same
    information in M5's own numbering (1 = Saturday), so its colours are reversed.
  - `price_rel_item_mean` is blue on the right: an item priced below its own average
    (a discount) gets a higher forecast, up to about +0.5.
- **SHAP importance, both models**: `results/shap_importance_CA_1.png`, bars of mean
  |SHAP| over every test row of all 3 folds for the top 15 features. The two models
  agree on the top 3; XGBoost puts more weight on `roll_mean_14`, `item_id` and
  `roll_mean_7`, LightGBM more on `roll_std_182`.

