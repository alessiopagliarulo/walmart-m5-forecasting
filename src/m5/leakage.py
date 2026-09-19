"""Perturbation check that a feature builder never reads the future.

Two experiments on the same inputs, both around a cutoff day T that ends a price week:

1. Sales only: scramble every sale after T. Features on days <= T + horizon must not
   change, because sales-derived features may only see sales up to t - horizon.
2. Everything: scramble sales, prices (weeks after T) and calendar events/SNAP after T.
   Features on days <= T must not change.

Any feature that changes is leaking and is returned by name.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Builder = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]
KEYS = ["id", "d"]


def _perturb_sales(sales_wide: pd.DataFrame, cutoff: int, rng: np.random.Generator) -> pd.DataFrame:
    out = sales_wide.copy()
    later = [c for c in out.columns if c.startswith("d_") and int(c[2:]) > cutoff]
    out.loc[:, later] = rng.integers(0, 50, size=(len(out), len(later))).astype("int16")
    return out


def _perturb_calendar(
    calendar: pd.DataFrame, cutoff: int, rng: np.random.Generator
) -> pd.DataFrame:
    out = calendar.copy()
    later = out["d"] > cutoff
    for col in [c for c in out.columns if c.startswith("event_")]:
        values = out[col].astype("object")
        values[later] = rng.choice(["PerturbedA", "PerturbedB", None], size=int(later.sum()))
        out[col] = values.astype("category")
    for col in [c for c in out.columns if c.startswith("snap_")]:
        out.loc[later, col] = (1 - out.loc[later, col]).astype(out[col].dtype)
    return out


def _perturb_prices(
    prices: pd.DataFrame, after_week: int, rng: np.random.Generator
) -> pd.DataFrame:
    out = prices.copy()
    later = out["wm_yr_wk"] > after_week
    factor = rng.uniform(0.5, 1.5, size=int(later.sum())).astype("float32")
    out.loc[later, "sell_price"] = out.loc[later, "sell_price"] * factor
    return out


def _changed_columns(base: pd.DataFrame, other: pd.DataFrame, max_day: int) -> list[str]:
    a = base[base["d"] <= max_day].set_index(KEYS).sort_index()
    b = other[other["d"] <= max_day].set_index(KEYS).sort_index()
    if not a.index.equals(b.index):
        return ["<row set>"]
    changed = []
    for col in a.columns:
        x = a[col].astype("object") if isinstance(a[col].dtype, pd.CategoricalDtype) else a[col]
        y = b[col].astype("object") if isinstance(b[col].dtype, pd.CategoricalDtype) else b[col]
        same = (x == y) | (x.isna() & y.isna())
        if not bool(same.all()):
            changed.append(str(col))
    return changed


def find_leaks(
    build: Builder,
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    cutoff: int,
    horizon: int,
    target: str = "sales",
    seed: int = 0,
) -> dict[str, list[str]]:
    """Return {experiment: [leaking feature names]}; empty lists mean no leak detected."""
    week = calendar.set_index("d")["wm_yr_wk"]
    if cutoff + 1 in week.index and week[cutoff + 1] == week[cutoff]:
        raise ValueError(f"cutoff d_{cutoff} must be the last day of a price week")
    rng = np.random.default_rng(seed)
    base = build(sales_wide, calendar, prices).drop(columns=[target])

    sales_only = build(_perturb_sales(sales_wide, cutoff, rng), calendar, prices)
    everything = build(
        _perturb_sales(sales_wide, cutoff, rng),
        _perturb_calendar(calendar, cutoff, rng),
        _perturb_prices(prices, int(week[cutoff]), rng),
    )
    return {
        "sales_after_cutoff": _changed_columns(
            base, sales_only.drop(columns=[target]), cutoff + horizon
        ),
        "all_inputs_after_cutoff": _changed_columns(
            base, everything.drop(columns=[target]), cutoff
        ),
    }
