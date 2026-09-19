"""Feature engineering for one store: lags, rolling stats, price, calendar and SNAP.

Information rule (checked by tests/test_leakage.py): a row for day t may use
  * sales only from days <= t - horizon (the forecast origin for a direct 28-day forecast),
  * prices only from weeks up to and including the week of t (prices are set in advance),
  * calendar/event/SNAP data only for days <= t (the calendar is published in advance).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from m5.config import HORIZON
from m5.data import EVENT_COLS

ROLL_WINDOWS = (7, 14, 28, 56, 112, 182)
ZERO_FRAC_WINDOWS = (28, 112)
# Extra lags beyond the horizon: the week after the origin, then weekly and yearly.
LAG_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 336)

KEY_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "date"]
TARGET = "sales"
CATEGORICAL_FEATURES = ["item_id", "dept_id", "cat_id", *EVENT_COLS]
CALENDAR_FEATURES = [
    "wday",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    *EVENT_COLS,
    "has_event",
    "snap",
]
PRICE_FEATURES = [
    "sell_price",
    "price_change_1w",
    "price_rel_item_max",
    "price_rel_item_mean",
    "price_item_std",
    "price_n_changes",
    "price_rel_dept",
]


def lags_for(horizon: int) -> list[int]:
    return [horizon + k for k in LAG_OFFSETS]


def sales_features(horizon: int = HORIZON) -> list[str]:
    names = [f"lag_{k}" for k in lags_for(horizon)]
    names += [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    names += [f"roll_std_{w}" for w in ROLL_WINDOWS]
    names += [f"zero_frac_{w}" for w in ZERO_FRAC_WINDOWS]
    names += ["same_dow_mean_4", "days_since_release"]
    return names


def feature_columns(horizon: int = HORIZON) -> list[str]:
    """Every model input column, in output order (identity categoricals first)."""
    return [
        "item_id",
        "dept_id",
        "cat_id",
        *CALENDAR_FEATURES,
        *PRICE_FEATURES,
        *sales_features(horizon),
    ]


def _sales_matrices(y: pd.DataFrame, horizon: int) -> dict[str, pd.DataFrame]:
    """y: days x series, NaN where unknown or before release. Returns days x series frames."""
    out: dict[str, pd.DataFrame] = {}
    for k in lags_for(horizon):
        out[f"lag_{k}"] = y.shift(k)
    origin = y.shift(horizon)
    for w in ROLL_WINDOWS:
        roll = origin.rolling(w, min_periods=w)
        out[f"roll_mean_{w}"] = roll.mean()
        out[f"roll_std_{w}"] = roll.std()
    is_zero = origin.eq(0).astype("float32").where(origin.notna())
    for w in ZERO_FRAC_WINDOWS:
        out[f"zero_frac_{w}"] = is_zero.rolling(w, min_periods=w).mean()
    first_same_dow = math.ceil(horizon / 7) * 7
    same_dow = [y.shift(first_same_dow + 7 * i) for i in range(4)]
    out["same_dow_mean_4"] = sum(same_dow[1:], same_dow[0]) / 4
    return {name: frame.astype("float32") for name, frame in out.items()}


def _price_matrices(p: pd.DataFrame, dept_of: pd.Series) -> dict[str, pd.DataFrame]:
    """p: days x series daily sell price (NaN when the item is not on sale)."""
    prev = p.shift(1)
    changed = (p.ne(prev) & p.notna() & prev.notna()).astype("float32")
    dept_mean = p.T.groupby(dept_of.to_numpy(), observed=True).transform("mean").T
    features = {
        "sell_price": p,
        "price_change_1w": p / p.shift(7) - 1,
        "price_rel_item_max": p / p.cummax(),
        "price_rel_item_mean": p / p.expanding(min_periods=1).mean(),
        "price_item_std": p.expanding(min_periods=2).std(),
        "price_n_changes": changed.cumsum(),
        "price_rel_dept": p / dept_mean,
    }
    return {name: frame.astype("float32") for name, frame in features.items()}


def build_features(
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    store_id: str,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Long feature table: one row per (series, day) from each item's release onward.

    Covers the sales history d_1..d_N plus the next `horizon` days (sales = NaN) when the
    calendar reaches that far, so the same table serves backtests and the real forecast.
    The store used is recorded in the `store_id` column and in `DataFrame.attrs`.
    """
    stores = sales_wide["store_id"].astype(str).unique()
    if list(stores) != [store_id]:
        raise ValueError(f"expected sales for store {store_id!r} only, got {list(stores)}")
    state = str(sales_wide["state_id"].astype(str).iloc[0])
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    n_hist = len(day_cols)
    cal = calendar.sort_values("d").reset_index(drop=True)
    n_days = min(len(cal), n_hist + horizon)
    cal = cal.iloc[:n_days]

    ids = sales_wide["id"].astype(str).to_numpy()
    y = pd.DataFrame(np.full((n_days, len(ids)), np.nan, dtype="float32"), columns=ids)
    y.iloc[:n_hist] = sales_wide[day_cols].to_numpy(dtype="float32").T

    price_week = prices.pivot_table(
        index="wm_yr_wk", columns="item_id", values="sell_price", observed=True
    )
    item_of = sales_wide["item_id"].astype(str).to_numpy()
    p = price_week.reindex(index=cal["wm_yr_wk"].to_numpy(), columns=item_of)
    p = p.reset_index(drop=True).astype("float32")
    p.columns = ids

    # An item exists from its first priced week; the zeros before that are not sales.
    released = p.notna().cummax()
    y = y.where(released)
    first_day = released.to_numpy().argmax(axis=0)

    matrices = _sales_matrices(y, horizon)
    matrices.update(_price_matrices(p, sales_wide["dept_id"].astype(str)))

    item_idx, day_idx = np.nonzero(released.to_numpy().T)  # ordered by series, then day
    out = pd.DataFrame(
        {
            "id": pd.Categorical(ids[item_idx]),
            **{
                c: pd.Categorical(sales_wide[c].to_numpy()[item_idx])
                for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
            },
            "d": cal["d"].to_numpy()[day_idx].astype("int16"),
            "date": cal["date"].to_numpy()[day_idx],
            TARGET: y.to_numpy()[day_idx, item_idx],
        }
    )

    date = pd.DatetimeIndex(cal["date"])
    cal_feats: dict[str, np.ndarray] = {
        "wday": cal["wday"].to_numpy().astype("int8"),
        "day_of_week": date.dayofweek.to_numpy().astype("int8"),
        "day_of_month": date.day.to_numpy().astype("int8"),
        "week_of_year": date.isocalendar().week.to_numpy().astype("int8"),
        "month": date.month.to_numpy().astype("int8"),
        "quarter": date.quarter.to_numpy().astype("int8"),
        "year": date.year.to_numpy().astype("int16"),
        "is_weekend": np.asarray(date.dayofweek >= 5).astype("int8"),
        "has_event": cal["event_name_1"].notna().to_numpy().astype("int8"),
        "snap": cal[f"snap_{state}"].to_numpy().astype("int8"),
    }
    for col in CALENDAR_FEATURES:
        if col in EVENT_COLS:
            out[col] = pd.Categorical(
                cal[col].to_numpy()[day_idx], categories=cal[col].astype("category").cat.categories
            )
        else:
            out[col] = cal_feats[col][day_idx]
    for name in PRICE_FEATURES:
        out[name] = matrices.pop(name).to_numpy(dtype="float32")[day_idx, item_idx]
    for name in sales_features(horizon):
        if name == "days_since_release":
            out[name] = (day_idx - first_day[item_idx]).astype("int16")
        else:
            out[name] = matrices.pop(name).to_numpy(dtype="float32")[day_idx, item_idx]

    out.attrs.update({"store_id": store_id, "state_id": state, "horizon": horizon})
    return out
