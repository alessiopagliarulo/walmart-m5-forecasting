# Resume claims and their evidence

Each resume bullet is checked against the committed artifacts. Status is one of
**verified** (true as worded), **differs** (the measured fact differs from the wording;
use the suggested wording instead) or **not yet built**.

Every number below comes from a script-generated file. To regenerate them:
`uv run m5-features`, then `uv run m5-backtest`, then `uv run m5-plots`,
`uv run m5-explain` and `uv run m5-writeup`. The tables and the findings marked
`GENERATED` are rewritten by `m5-writeup` and checked by `tests/test_writeup.py`; the
numbers quoted in the claim sections are copied from them and from the artifacts named
next to each one. The full write-up is [docs/RESULTS.md](RESULTS.md).

## Scope: what the numbers cover

<!-- GENERATED:scope:BEGIN -->

- **Store CA_1** (state CA), 3,049 item series, daily sales from 2011-01-29 to 2016-05-22.
- **28-day horizon**, forecast directly from 52 features whose sales inputs are at least 28 days old.
- **3 walk-forward folds**, each training on every day up to its cutoff and forecasting the next 28 days; the last fold ends on the last day of sales:

| Fold | Train through | Forecast | Series x days scored |
| --- | --- | --- | --- |
| 1 | 2016-02-28 | 2016-02-29 to 2016-03-27 | 85,372 |
| 2 | 2016-03-27 | 2016-03-28 to 2016-04-24 | 85,372 |
| 3 | 2016-04-24 | 2016-04-25 to 2016-05-22 | 85,372 |

<!-- GENERATED:scope:END -->

- **One store only.** Nothing here is measured on the other 9 stores or the full
  30,490-series competition data, so no claim may imply more than one store.
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

<!-- GENERATED:model_table:BEGIN -->

| Model | Family | Fold 1 | Fold 2 | Fold 3 | Mean WRMSSE | vs seasonal-naive | Mean MAE (units) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive | naive baseline | 0.7661 | 0.9075 | 0.6598 | 0.7778 | - | 1.3228 |
| linear_regression | linear | 0.8570 | 0.7976 | 0.7885 | 0.8144 | 4.7% worse | 1.1726 |
| ridge | linear | 0.6205 | 0.5351 | 0.6193 | 0.5916 | 23.9% lower | 1.1315 |
| hurdle_logistic_ridge | logistic hurdle | 0.6045 | 0.5269 | 0.6033 | 0.5783 | 25.7% lower | 1.1429 |
| lightgbm | gradient boosting | 0.5728 | 0.4740 | 0.5292 | 0.5253 | 32.5% lower | 1.1175 |
| xgboost | gradient boosting | 0.5778 | 0.4738 | 0.5189 | **0.5235** | 32.7% lower | 1.1146 |

<!-- GENERATED:model_table:END -->

<!-- GENERATED:model_findings:BEGIN -->

- The simple linear regression on a few sales-history features is 4.7% worse than seasonal-naive (0.8144 vs 0.7778) and beats it in 1 of 3 folds. Ridge on the full feature set is 23.9% lower, so the gain comes from the full feature set, not from regression as such.
- Adding a LogisticRegression "does it sell today" stage (the hurdle model) takes Ridge from 0.5916 to 0.5783, better than Ridge in every fold.
- Both boosted models beat Ridge and the hurdle model in every fold. Even the weaker one, lightgbm, is 11.2% lower than Ridge and 9.2% lower than the hurdle model.
- Between the two boosted models the fold winners are lightgbm in fold 1, xgboost in fold 2, xgboost in fold 3: each model wins at least one fold, so 3 folds cannot separate them.

<!-- GENERATED:model_findings:END -->

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

## Bullet 3 replacement and bullet 2 upgrade

The resume's third bullet now reads "Ranked rolling demand and item price as top SHAP
predictors, quantifying a 10% food-sales lift on SNAP days". It is to be replaced,
because SHAP already appears elsewhere on the resume. (Claim 3 above checks a different
wording.) Below are four verified candidates and an upgrade for the second bullet.

Rules for every wording: true as worded against the committed artifacts, one store only,
past-tense verb first, no trailing period, at most 118 characters. Each count is Python
`len()` of the exact string. Percentages printed by `m5-writeup` in the generated tables
above are quoted as printed (one decimal, standard rounding; the unrounded value is given
next to each). A percentage computed here for the first time is rounded down, never up.

### Summary

| # | Wording | Chars |
| --- | --- | --- |
| 1 | Benchmarked 6 models across 4 families under leakage-free walk-forward CV on 3,049 series from one Walmart store | 112 |
| 2 | Built a logistic-plus-Ridge hurdle model for 53%-zero daily demand, cutting WRMSSE 2.2% vs. Ridge and 25.7% vs. naive | 117 |
| 3 | Wrote CI tests that scramble post-cutoff sales and require all 6 models' forecasts unchanged; sha256-pinned raw data | 116 |
| 4 | Measured an 11.8% SNAP-day food lift at one store, 7.8% above non-food (bootstrap 95% CI 5.8-9.7%), 448 matched cells | 117 |
| Bullet 2 | Cut WRMSSE 32.7% vs. a seasonal-naive baseline using 52 features incl. lags, rolling means, price and SNAP/holidays | 115 |

**Pick for machine-learning and data-science roles: candidate 3.** Bullets 1 and 2
already cover the models and the gain over the baseline, and the SNAP lift was already
in the old bullet 3; candidate 3 adds what the resume does not show at all, leakage
testing and reproducible data, the most common failure in time-series ML.

### Candidate 1 - model bake-off and validation rigor - verified

> "Benchmarked 6 models across 4 families under leakage-free walk-forward CV on 3,049
> series from one Walmart store" (112 characters)

- 6 models: the 6 keys of `models` in `results/metrics.json` (seasonal_naive,
  linear_regression, ridge, hurdle_logistic_ridge, lightgbm, xgboost), the same 6 as
  `MODELS` in `src/m5/models.py`, all scored on the same 3 folds.
- 4 families: `FAMILIES` in `src/m5/writeup.py`, shown in the Family column of the
  table above: naive baseline, linear (OLS and Ridge), logistic hurdle, gradient
  boosting (LightGBM and XGBoost).
- Leakage-free walk-forward CV: the 3 folds in `folds` of `results/metrics.json` each
  train on every day up to the cutoff and score the next 28. "Leakage-free" rests on
  the tests listed under Claim 3 and candidate 3, which run in CI on synthetic data.
- 3,049 series from one store: `data.n_series: 3049` in `results/metrics.json`,
  `store_id: "CA_1"`.
- Overlap: close to the older Claim 3 wording, and bullet 1 already names LightGBM and
  XGBoost, so this adds the least new ground.

### Candidate 2 - intermittent demand, the hurdle model - verified, with a caveat

> "Built a logistic-plus-Ridge hurdle model for 53%-zero daily demand, cutting WRMSSE
> 2.2% vs. Ridge and 25.7% vs. naive" (117 characters)

- The model: `hurdle_logistic_ridge` in `results/metrics.json`, "LogisticRegression
  (C=1.0) for P(sales > 0) times Ridge (alpha=1.0) fitted on selling days"
  (`HurdleLogisticRidge` in `src/m5/models.py`).
- 53% zero: the share of scored item-days (3 test windows, 85,372 each) with zero
  sales is 52.9%, from `classifier.share_sold` of the hurdle model's folds in
  `results/metrics.json` (0.4552, 0.4666, 0.4911; 1 minus their mean is 0.5290).
  Confirmed from the sha256-verified raw file (read-only, run from the repo root):

  ```python
  import numpy as np
  import pandas as pd

  s = pd.read_csv("data/raw/sales_train_evaluation.csv")
  d = s[s.store_id == "CA_1"][[f"d_{i}" for i in range(1, 1942)]].to_numpy()
  first = np.argmax(d > 0, axis=1)
  after = np.arange(d.shape[1])[None, :] >= first[:, None]
  print((d == 0).mean(), (d[after] == 0).mean(), (d[:, 1857:] == 0).mean())
  # 0.6376 0.5514 0.5290
  ```

  So 63.8% of all 5,918,109 CA_1 item-days are zero, 55.1% counting each item only
  from its first sale (before that it was not on sale yet), and 52.9% in the scored
  test windows. The wording uses 53%, the smallest of the three and the one that
  traces to a committed artifact. Reading it as "53% of item-days sell nothing" is
  accurate; rounding 52.9% to 53% overstates by a tenth of a point, well inside the
  spread between the three measures.
- 2.2% vs. Ridge: mean WRMSSE 0.5783 against 0.5916 (table above), 2.26% lower,
  rounded down. Better than Ridge in every fold: 2.5%, 1.5% and 2.5% lower (rounded down)
  (`folds[].wrmsse` of both models).
- 25.7% vs. naive: printed in the table above (0.5783 vs. 0.7778; unrounded 25.65%).
  "naive" means the seasonal-naive baseline.
- What it does not buy: mean MAE is 1.0% worse than Ridge (1.1429 vs. 1.1315 units),
  so the gain is on the scaled, sales-weighted WRMSSE only; and both boosted models
  beat it in every fold (LightGBM is 9.2% lower). The wording claims neither.
- The classifier is useful on its own: AUC 0.7844, 0.7850 and 0.7588 and Brier skill
  0.2454, 0.2460 and 0.2045 over the base rate, in `results/metrics.md`.

### Candidate 3 - leakage and reproducibility engineering - verified

> "Wrote CI tests that scramble post-cutoff sales and require all 6 models' forecasts
> unchanged; sha256-pinned raw data" (116 characters)

- The test: `test_forecasts_ignore_sales_after_the_cutoff` in `tests/test_backtest.py`,
  parametrized over every entry of `MODELS` (6), replaces every sale after the fold's
  cutoff with random integers and asserts each model's forecast is exactly equal
  (`np.testing.assert_array_equal`); it first asserts the scramble really changed the
  later targets, so it cannot pass vacuously. The boosted models' grid search runs
  inside that fit.
- Related gates: `tests/test_leakage.py` (via `src/m5/leakage.py`) scrambles every sale
  after a cutoff and requires features up to the cutoff plus the horizon to be
  unchanged, then scrambles every input after the cutoff and requires features up to
  the cutoff to be unchanged; a deliberately leaky builder must be caught; `test_boosted_search_stays_inside_the_training_window`
  checks the search validates only on the last 28 training days.
- CI: `.github/workflows/ci.yml` runs `uv run pytest` on every push and pull request.
  The tests run on the synthetic fixture (`tests/fixtures/make_synthetic_m5.py`), not
  the real data; "CI tests" does not claim otherwise.
- sha256-pinned raw data: `provenance/m5_data.json` pins the sha256 of all 4 raw files
  (`test_committed_provenance_pins_all_four_files` in `tests/test_download.py`).
  `m5-download` refuses a mismatching file (`src/m5/download.py`), `m5-features`
  refuses one (`src/m5/pipeline.py`), and `m5-backtest` stops on features built from
  unverified data (`test_backtest_stops_on_unverified_features`). The hashes are
  recorded again in `data.raw_sha256` of `results/metrics.json`.
- Beyond the line: `m5-explain` refuses to run unless the rebuilt boosted models
  reproduce the recorded WRMSSE exactly (`wrmsse_rebuilt` equals `wrmsse_recorded` in
  `results/shap_summary.json`).

### Candidate 4 - SNAP lift as a statistical measurement - verified

> "Measured an 11.8% SNAP-day food lift at one store, 7.8% above non-food (bootstrap
> 95% CI 5.8-9.7%), 448 matched cells" (117 characters)

- From `scopes."store CA_1"` in `results/snap_lift.json` (the SNAP table above):
  food `lift` 0.1185 (11.8%, CI 10.3% to 13.4%); `foods_vs_non_food.lift` 0.0776
  (7.8%, printed as +7.8%), `lift_ci95` 0.0575 to 0.0969 (printed as +5.8% to +9.7%);
  `n_cells` 448 same-month, same-weekday cells; bootstrap of 2,000 draws over whole
  months, seed 0.
- "7.8% above non-food" is the lift of food beyond the non-food lift in the same
  cells (non-food itself rises 3.8% on SNAP days, a start-of-month effect); the
  interval in brackets belongs to that 7.8%.
- No model is involved: `m5-explain` computes it from the raw sales. The older
  suggestion above rounds these to 12% and 8%; this wording keeps the printed
  decimals so nothing is rounded up.
- Scope: store CA_1. Across all of California the figures are 9.8% and 7.5%. The
  resume's current "10%" was the plan's target; do not mix the two scopes.

### Bullet 2 upgrade - verified

> Current: "Cut WRMSSE error 27% vs. a seasonal-naive baseline using 40+ lag,
> rolling-mean and SNAP/holiday features"

> Upgrade: "Cut WRMSSE 32.7% vs. a seasonal-naive baseline using 52 features incl. lags,
> rolling means, price and SNAP/holidays" (115 characters)

- 32.7%: XGBoost against seasonal-naive in the table above (0.5235 vs. 0.7778,
  unrounded 32.69%).
- 52 features: `n_features: 52` in `results/features_CA_1_manifest.json`. "incl." is
  needed: 5 of the 52 are product identity, year and days since release, which the
  list does not name. Lags (12), rolling means (6), price (7) and SNAP plus holiday
  events (1 + 5) are all among them, per the breakdown under Claim 2.
- It replaces the longer suggestion under Claim 2, which rounds to 33% and runs past
  118 characters.

## What SHAP says drives the forecasts

From `results/shap_summary.json` (mean |SHAP| over all 3 folds, every test row; SHAP
values are in log units because the Tweedie models forecast log(expected units)).
The plan's target was "the 28-day rolling mean, 28-day lag and sell price drive most
predictions". **That differs from what was measured:**

<!-- GENERATED:shap_top:BEGIN -->

| Rank | lightgbm | share | xgboost | share |
| --- | --- | --- | --- | --- |
| 1 | roll_mean_14 | 15.5% | roll_mean_14 | 19.2% |
| 2 | roll_mean_28 | 12.6% | item_id | 15.4% |
| 3 | item_id | 12.0% | roll_mean_28 | 11.1% |
| 4 | roll_std_182 | 8.0% | roll_mean_7 | 8.2% |
| 5 | day_of_week | 5.0% | roll_std_182 | 5.2% |

Share of total mean |SHAP| by feature group:

| Group | lightgbm | xgboost |
| --- | --- | --- |
| rolling means | 41.4% | 45.9% |
| product identity | 16.7% | 19.7% |
| price | 4.2% | 3.3% |
| sales lags | 3.1% | 3.8% |
| snap | 0.4% | 0.2% |

<!-- GENERATED:shap_top:END -->

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

<!-- GENERATED:snap_table:BEGIN -->

| Scope | Food lift | 95% CI | Non-food lift | Food beyond non-food | 95% CI |
| --- | --- | --- | --- | --- | --- |
| store CA_1 | +11.8% | +10.3% to +13.4% | +3.8% | +7.8% | +5.8% to +9.7% |
| state CA (all stores) | +9.8% | +8.4% to +11.2% | +2.2% | +7.5% | +5.8% to +9.1% |

SNAP days in California are the 1st to the 10th of each month. Compared within 448 same-month, same-weekday cells from 2011-01-29 to 2016-05-22; the intervals resample whole months (2,000 draws).

<!-- GENERATED:snap_table:END -->

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
    but fall short of the tallest weekend peaks; the largest shortfalls, from
    `store_daily_units` in `results/metrics.json`, follow this list.
  - Seasonal-naive repeats its training week's quirks. Fold 2 copies the week ending on
    Easter Sunday (27 March), when sales dipped, so it forecasts 4,669 units for every
    Sunday in fold 2 against actual Sundays of about 6,000-6,500 (6,496 on 3 April).
  - Ridge is close to the boosted models on most days but sits lower on several
    troughs and peaks, most visibly in fold 3.

<!-- GENERATED:peak_misses:BEGIN -->

- **Peaks are under-forecast.** Summed over the whole store, xgboost forecasts below actual sales on 56 of 84 test days. Its largest shortfalls: Sun 6 Mar 2016: 5,658 forecast vs 6,829 sold; Sat 26 Mar 2016: 5,370 forecast vs 6,139 sold; Sun 15 May 2016: 5,777 forecast vs 6,707 sold.

<!-- GENERATED:peak_misses:END -->

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

