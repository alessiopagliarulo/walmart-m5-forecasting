"""How much more sells on SNAP days, measured from the raw sales alone (no model).

SNAP (the US food-stamp programme) pays benefits on fixed days of the month, set per
state; `calendar.csv` flags them in `snap_CA`, `snap_TX` and `snap_WI`. SNAP buys food
only, so a real SNAP effect should lift the FOODS category and not HOBBIES or
HOUSEHOLD.

Method:

1. Daily units summed over each category's items, for one store or a whole state.
   NON_FOOD is HOBBIES plus HOUSEHOLD.
2. Christmas Day is dropped: the stores close (CA_1 sells nothing, the other stores a
   handful of units). So is any other day on which nothing sold at all.
3. Days are grouped into cells of the same calendar month and weekday (for example the
   Tuesdays of March 2014). Only cells holding both SNAP and non-SNAP days are kept, so
   season, trend and the weekly cycle are the same on both sides of each comparison.
4. In every cell: log(mean units on SNAP days) - log(mean units on other days). A
   category's lift is exp(mean of those log ratios) - 1.
5. FOODS vs NON_FOOD: the same with each cell's FOODS log ratio minus its NON_FOOD log
   ratio, i.e. how much more the SNAP days lift food than everything else.
6. 95% intervals come from a bootstrap that resamples whole calendar months.

Caveat: in each state SNAP falls on the same days every month (California: the 1st to
the 10th), so step 4 compares the start of the month with the rest of it. Anything else
tied to the start of the month, such as paydays, is mixed into a category's lift. The
non-food categories show how big that mix is, and step 5 nets it out, assuming it
lifts food and non-food alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 0


def snap_method() -> str:
    """The method section of this module's docstring, for the published artifact."""
    return (__doc__ or "").split("Method:")[1].strip()


def daily_category_units(sales_wide: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """One row per day: calendar fields plus total units per category and in `total`."""
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    by_cat = sales_wide.groupby(sales_wide["cat_id"].astype(str))[day_cols].sum().T
    by_cat.index = by_cat.index.str.removeprefix("d_").astype("int16")
    by_cat["total"] = by_cat.sum(axis=1)
    cal = calendar.set_index("d")
    return cal.loc[by_cat.index].join(by_cat.astype("float64")).reset_index()


def _cell_log_ratios(days: pd.DataFrame, column: str, snap_col: str) -> pd.Series:
    """Per (year, month, wday) cell with both kinds of day: log(SNAP mean / other mean)."""
    means = days.pivot_table(
        index=["year", "month", "wday"], columns=snap_col, values=column, aggfunc="mean"
    ).dropna()
    means = means[(means[0] > 0) & (means[1] > 0)]
    return pd.Series(np.log(means[1]) - np.log(means[0]), index=means.index)


def _summary(log_ratios: pd.Series, draws: list[np.ndarray]) -> dict[str, Any]:
    """Lift from the cells' log ratios, with a month-block bootstrap interval."""
    month_key = log_ratios.index.get_level_values("year").to_numpy("int64") * 100
    month_key += log_ratios.index.get_level_values("month").to_numpy("int64")
    values = log_ratios.to_numpy()
    by_month = {k: values[month_key == k] for k in np.unique(month_key)}
    boot = [
        np.expm1(np.concatenate([by_month[k] for k in draw if k in by_month]).mean())
        for draw in draws
    ]
    return {
        "lift": float(np.expm1(values.mean())),
        "lift_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_cells": len(values),
        "share_of_cells_higher_on_snap_days": float((values > 0).mean()),
    }


def snap_lift(
    daily: pd.DataFrame,
    state: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Matched SNAP-day lift per category and for FOODS vs NON_FOOD (module docstring)."""
    snap_col = f"snap_{state}"
    daily = daily.assign(NON_FOOD=daily["total"] - daily["FOODS"])
    closed = (daily["event_name_1"].astype("object") == "Christmas") | (daily["total"] <= 0)
    open_days = daily[~closed]
    month_keys = np.unique(open_days["year"].to_numpy("int64") * 100 + open_days["month"])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(month_keys, size=len(month_keys)) for _ in range(reps)]

    names = [c for c in ("FOODS", "HOBBIES", "HOUSEHOLD", "NON_FOOD") if c in open_days]
    ratios = {name: _cell_log_ratios(open_days, name, snap_col) for name in names}
    out: dict[str, Any] = {}
    snap_day = open_days[snap_col] == 1
    for name in names:
        out[name] = {
            **_summary(ratios[name], draws),
            "unmatched_lift": float(
                open_days.loc[snap_day, name].mean() / open_days.loc[~snap_day, name].mean() - 1
            ),
            "mean_daily_units_snap_days": float(open_days.loc[snap_day, name].mean()),
            "mean_daily_units_other_days": float(open_days.loc[~snap_day, name].mean()),
        }
    food_vs_non_food = (ratios["FOODS"] - ratios["NON_FOOD"]).dropna()
    return {
        "snap_column": snap_col,
        "n_days": len(open_days),
        "n_snap_days": int(snap_day.sum()),
        "first_date": str(open_days["date"].min().date()),
        "last_date": str(open_days["date"].max().date()),
        "dropped_days": [str(d.date()) for d in daily.loc[closed, "date"]],
        "snap_days_of_month": sorted(
            int(d) for d in open_days.loc[snap_day, "date"].dt.day.unique()
        ),
        "categories": out,
        "foods_vs_non_food": _summary(food_vs_non_food, draws),
    }
