"""The leakage gate: features for day t must not depend on anything after t (or, for
sales, after t - horizon). A deliberately leaky builder proves the check can fail."""

from __future__ import annotations

from functools import partial

import pandas as pd
import pytest

from m5.features import build_features
from m5.leakage import find_leaks
from tests.conftest import FIXTURE_HORIZON, FIXTURE_STORE

build = partial(build_features, store_id=FIXTURE_STORE, horizon=FIXTURE_HORIZON)


@pytest.mark.parametrize("cutoff", [196, 385])
def test_features_do_not_leak(
    cutoff: int, store_sales: pd.DataFrame, calendar: pd.DataFrame, store_prices: pd.DataFrame
) -> None:
    leaks = find_leaks(build, store_sales, calendar, store_prices, cutoff, FIXTURE_HORIZON)
    assert leaks == {"sales_after_cutoff": [], "all_inputs_after_cutoff": []}


def test_perturbation_really_reaches_later_features(
    store_sales: pd.DataFrame, calendar: pd.DataFrame, store_prices: pd.DataFrame
) -> None:
    # Guard against a vacuous pass: claiming a window one week longer than the real
    # horizon must expose the lags whose inputs were scrambled.
    leaks = find_leaks(build, store_sales, calendar, store_prices, 385, FIXTURE_HORIZON + 7)
    assert {"lag_28", "lag_34", "roll_mean_7"} <= set(leaks["sales_after_cutoff"])
    assert "lag_35" not in leaks["sales_after_cutoff"]


def _leaky(sales_wide: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = build(sales_wide, cal, prices)
    out["lag_27"] = out.groupby("id", observed=True)["sales"].shift(FIXTURE_HORIZON - 1)
    out["next_week_price"] = out.groupby("id", observed=True)["sell_price"].shift(-7)
    return out


def test_leaky_features_are_caught(
    store_sales: pd.DataFrame, calendar: pd.DataFrame, store_prices: pd.DataFrame
) -> None:
    leaks = find_leaks(_leaky, store_sales, calendar, store_prices, 385, FIXTURE_HORIZON)
    assert "lag_27" in leaks["sales_after_cutoff"]
    assert "next_week_price" in leaks["all_inputs_after_cutoff"]


def test_cutoff_must_end_a_price_week(
    store_sales: pd.DataFrame, calendar: pd.DataFrame, store_prices: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="last day of a price week"):
        find_leaks(build, store_sales, calendar, store_prices, 200, FIXTURE_HORIZON)
