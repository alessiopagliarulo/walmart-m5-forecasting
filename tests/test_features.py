from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5.features import (
    TARGET,
    build_features,
    feature_columns,
    lags_for,
    sales_features,
)
from tests.conftest import FIXTURE_HORIZON, FIXTURE_STORE


@pytest.fixture(scope="module")
def feats(
    store_sales: pd.DataFrame, calendar: pd.DataFrame, store_prices: pd.DataFrame
) -> pd.DataFrame:
    return build_features(store_sales, calendar, store_prices, FIXTURE_STORE, FIXTURE_HORIZON)


def _series(feats: pd.DataFrame, item: str) -> pd.DataFrame:
    return feats[feats["item_id"] == item].set_index("d")


def test_at_least_40_features_and_all_present(feats: pd.DataFrame) -> None:
    cols = feature_columns(FIXTURE_HORIZON)
    assert len(cols) >= 40
    assert len(set(cols)) == len(cols)
    assert set(cols) <= set(feats.columns)


def test_store_is_recorded(feats: pd.DataFrame) -> None:
    assert feats.attrs["store_id"] == FIXTURE_STORE
    assert feats["store_id"].astype(str).unique().tolist() == [FIXTURE_STORE]


def test_rows_start_at_release_and_cover_horizon(feats: pd.DataFrame) -> None:
    per_item = feats.groupby("item_id", observed=True)["d"].agg(["min", "max"])
    assert per_item.loc["HOBBIES_1_002", "min"] == 20 * 7 + 1  # first priced week
    assert per_item.loc["FOODS_1_001", "min"] == 1
    assert (per_item["max"] == 420 + FIXTURE_HORIZON).all()
    future = feats[feats["d"] > 420]
    assert future[TARGET].isna().all()
    assert feats.loc[feats["d"] <= 420, TARGET].notna().all()


def test_lags_and_rolling_match_hand_computation(
    feats: pd.DataFrame, store_sales: pd.DataFrame
) -> None:
    row = store_sales[store_sales["item_id"] == "FOODS_1_001"].iloc[0]
    y = np.array([row[f"d_{i}"] for i in range(1, 421)], dtype=float)
    s = _series(feats, "FOODS_1_001")
    t = 400  # late enough that every lag, including lag_364, is defined
    for k in lags_for(FIXTURE_HORIZON):
        assert s.loc[t, f"lag_{k}"] == y[t - k - 1]
    window = y[t - 28 - 7 : t - 28]
    assert s.loc[t, "roll_mean_7"] == pytest.approx(window.mean(), rel=1e-6)
    assert s.loc[t, "roll_std_7"] == pytest.approx(window.std(ddof=1), rel=1e-5)
    assert s.loc[t, "zero_frac_28"] == pytest.approx((y[t - 56 : t - 28] == 0).mean())
    same_dow = [y[t - k - 1] for k in (28, 35, 42, 49)]
    assert s.loc[t, "same_dow_mean_4"] == pytest.approx(np.mean(same_dow))
    # Future rows still get lag features from known history.
    assert s.loc[448, "lag_28"] == y[420 - 1]


def test_pre_release_zeros_are_not_treated_as_sales(feats: pd.DataFrame) -> None:
    s = _series(feats, "HOBBIES_1_002")
    first = 141
    assert s.loc[first, "days_since_release"] == 0
    assert pd.isna(s.loc[first + FIXTURE_HORIZON - 1, "lag_28"])
    assert pd.notna(s.loc[first + FIXTURE_HORIZON, "lag_28"])


def test_price_features(feats: pd.DataFrame, store_prices: pd.DataFrame) -> None:
    s = _series(feats, "FOODS_1_001")
    p = store_prices[store_prices["item_id"] == "FOODS_1_001"].set_index("wm_yr_wk")["sell_price"]
    t = 200
    week = 11101 + (t - 1) // 7
    assert s.loc[t, "sell_price"] == pytest.approx(p[week])
    assert s.loc[t, "price_change_1w"] == pytest.approx(p[week] / p[week - 1] - 1)
    history = p[p.index <= week]
    assert s.loc[t, "price_rel_item_max"] == pytest.approx(p[week] / history.max())
    assert s["price_rel_item_max"].max() <= 1.0 + 1e-6


def test_calendar_and_snap_use_store_state(feats: pd.DataFrame, calendar: pd.DataFrame) -> None:
    s = _series(feats, "FOODS_1_001")
    cal = calendar.set_index("d")
    assert (s["snap"].to_numpy() == cal.loc[s.index, "snap_CA"].to_numpy()).all()
    assert (s["snap"].to_numpy() != cal.loc[s.index, "snap_TX"].to_numpy()).any()
    assert s.loc[1, "wday"] == 1 and s.loc[1, "day_of_week"] == 5  # a Saturday
    assert (s["has_event"] == cal.loc[s.index, "event_name_1"].notna().astype(int)).all()


def test_feature_dtypes_are_downcast(feats: pd.DataFrame) -> None:
    for col in sales_features(FIXTURE_HORIZON):
        assert feats[col].dtype in (np.float32, np.int16), col
    assert feats["id"].dtype == "category"


def test_wrong_store_is_rejected(
    store_sales: pd.DataFrame, calendar: pd.DataFrame, store_prices: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="expected sales for store"):
        build_features(store_sales, calendar, store_prices, "TX_1")
