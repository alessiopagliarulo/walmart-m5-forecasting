"""The walk-forward folds and the WRMSSE metric, checked against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from m5.backtest import evaluator_for, hierarchy_of, split, to_matrix
from m5.evaluation import M5_LEVELS, WRMSSEEvaluator, make_folds, naive_scale
from m5.features import TARGET
from m5.models import SeasonalNaive


def test_folds_are_back_to_back_and_end_on_last_day() -> None:
    folds = make_folds(1941)
    assert [(f.train_end, f.test_start, f.test_end) for f in folds] == [
        (1857, 1858, 1885),
        (1885, 1886, 1913),
        (1913, 1914, 1941),
    ]
    with pytest.raises(ValueError, match="need more than"):
        make_folds(80)


def test_naive_scale_starts_at_first_nonzero_sale() -> None:
    train = np.array(
        [
            [0, 2, 4, 2],  # from index 1: diffs 2, -2 -> (4 + 4) / 2 = 4
            [1, 1, 3, 1],  # diffs 0, 2, -2 -> (0 + 4 + 4) / 3 = 8/3
            [0, 0, 0, 5],  # nothing after the first sale: undefined
            [0, 0, 0, 0],  # never sold: undefined
        ],
        dtype=float,
    )
    scale = naive_scale(train)
    assert scale[:2] == pytest.approx([4.0, 8 / 3])
    assert np.isnan(scale[2:]).all()


# Two items of one department in one store. Hand computation (weights from the last
# 2 training days):
#   A: scale 4, errors (1, -1) -> MSE 1 -> RMSSE sqrt(1/4) = 0.5, dollars (4+2)*1 = 6
#   B: scale 8/3, errors (0, -2) -> MSE 2 -> RMSSE sqrt(2/(8/3)) = 0.866025, dollars (3+1)*2 = 8
#   item level: (6*0.5 + 8*0.866025) / 14 = 0.709157
#   total: train 1,3,7,3 -> diffs 2,4,-4 -> scale 12; errors (1, -3) -> MSE 5
#          -> RMSSE sqrt(5/12) = 0.645497
#   9 of the 12 levels equal the total (one store, dept and category), 3 equal the items:
#   WRMSSE = (9*0.645497 + 3*0.709157) / 12 = 0.661412
HAND_HIERARCHY = pd.DataFrame(
    {
        "id": ["A_CA_1", "B_CA_1"],
        "item_id": ["FOODS_1_001", "FOODS_1_002"],
        "dept_id": ["FOODS_1", "FOODS_1"],
        "cat_id": ["FOODS", "FOODS"],
        "store_id": ["CA_1", "CA_1"],
        "state_id": ["CA", "CA"],
    }
)
HAND_TRAIN = np.array([[0, 2, 4, 2], [1, 1, 3, 1]], dtype=float)
HAND_PRICES = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=float)
HAND_ACTUAL = np.array([[2, 2], [1, 3]], dtype=float)
HAND_FORECAST = np.array([[3, 1], [1, 1]], dtype=float)


def test_wrmsse_matches_hand_computation() -> None:
    evaluator = WRMSSEEvaluator(HAND_HIERARCHY, HAND_TRAIN, HAND_PRICES, weight_days=2)
    score = evaluator.score(HAND_ACTUAL, HAND_FORECAST)
    assert score.rmsse == pytest.approx([0.5, 0.866025], abs=1e-6)
    assert score.by_level["total"] == pytest.approx(0.645497, abs=1e-6)
    assert score.by_level["item_store"] == pytest.approx(0.709157, abs=1e-6)
    assert score.wrmsse == pytest.approx(0.661412, abs=1e-6)
    assert score.mae == pytest.approx((1 + 1 + 0 + 2) / 4)
    assert score.rmse == pytest.approx(math.sqrt((1 + 1 + 0 + 4) / 4))


def test_perfect_forecast_scores_zero_and_shape_is_checked() -> None:
    evaluator = WRMSSEEvaluator(HAND_HIERARCHY, HAND_TRAIN, HAND_PRICES, weight_days=2)
    assert evaluator.score(HAND_ACTUAL, HAND_ACTUAL).wrmsse == 0.0
    with pytest.raises(ValueError, match="shaped like actual"):
        evaluator.score(HAND_ACTUAL, HAND_FORECAST[:, :1])
    with pytest.raises(ValueError, match="finite"):
        evaluator.score(HAND_ACTUAL, np.full_like(HAND_ACTUAL, np.nan))


def test_series_without_usable_history_are_reported_not_hidden() -> None:
    # B's only sale is on the last training day, so its scale is undefined while it still
    # carries dollar weight. It is left out of its level, and its share is reported.
    train = np.array([[0, 2, 4, 2], [0, 0, 0, 3]], dtype=float)
    evaluator = WRMSSEEvaluator(HAND_HIERARCHY, train, HAND_PRICES, weight_days=2)
    assert evaluator.undefined_weight_share["item_store"] == pytest.approx(6 / 12)
    assert evaluator.undefined_weight_share["total"] == 0.0
    score = evaluator.score(HAND_ACTUAL, HAND_FORECAST)
    assert score.by_level["item_store"] == pytest.approx(0.5)


def test_sales_without_price_in_weight_window_are_rejected() -> None:
    prices = HAND_PRICES.copy()
    prices[0, -1] = np.nan
    with pytest.raises(ValueError, match="without a sell price"):
        WRMSSEEvaluator(HAND_HIERARCHY, HAND_TRAIN, prices, weight_days=2)


HIERARCHY = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _reference_wrmsse(panel: pd.DataFrame, forecast: pd.Series, train_end: int) -> float:
    """A plain, loop-based WRMSSE straight from the competition guidelines."""
    hist = panel[panel[TARGET].notna()][[*HIERARCHY, "d", TARGET, "sell_price"]].copy()
    hist[HIERARCHY] = hist[HIERARCHY].astype(str)
    level_scores = []
    for keys in M5_LEVELS.values():
        group_keys = keys or (lambda _: 0)
        per_group = []
        for _, rows in hist.groupby(group_keys, observed=True):
            daily = rows.groupby("d")[TARGET].sum()
            train = [float(daily.get(d, 0.0)) for d in range(1, train_end + 1)]
            test_days = range(train_end + 1, train_end + 29)
            actual = [float(daily.get(d, 0.0)) for d in test_days]
            fc_rows = forecast[forecast.index.get_level_values("id").isin(rows["id"].unique())]
            fc_daily = fc_rows.groupby(level="d").sum()
            predicted = [float(fc_daily.get(d, 0.0)) for d in test_days]
            recent = rows[(rows["d"] > train_end - 28) & (rows["d"] <= train_end)]
            dollars = float((recent[TARGET] * recent["sell_price"]).sum())
            first = next(i for i, v in enumerate(train) if v != 0)
            diffs = [(train[i] - train[i - 1]) ** 2 for i in range(first + 1, len(train))]
            scale = sum(diffs) / len(diffs)
            mse = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / 28
            per_group.append((dollars, math.sqrt(mse / scale)))
        total = sum(w for w, _ in per_group)
        level_scores.append(sum(w * e for w, e in per_group) / total)
    return sum(level_scores) / len(level_scores)


@pytest.mark.parametrize("fold_index", [0, 2])
def test_evaluator_matches_reference_on_synthetic_fixture(
    feature_panel: pd.DataFrame, fold_index: int
) -> None:
    fold = make_folds(int(feature_panel.loc[feature_panel[TARGET].notna(), "d"].max()))[fold_index]
    hierarchy = hierarchy_of(feature_panel)
    train, test = split(feature_panel, fold)
    prediction = SeasonalNaive().fit(train).predict(test)

    ids = pd.Index(hierarchy["id"])
    truth = feature_panel.loc[test.index]
    actual = to_matrix(truth, truth[TARGET].to_numpy(dtype="float64"), ids, fold)
    score = evaluator_for(feature_panel, hierarchy, fold).score(
        actual, to_matrix(test, prediction, ids, fold)
    )

    forecast = pd.Series(
        prediction,
        index=pd.MultiIndex.from_arrays(
            [test["id"].astype(str).to_numpy(), test["d"].to_numpy()], names=["id", "d"]
        ),
    )
    assert score.wrmsse == pytest.approx(_reference_wrmsse(feature_panel, forecast, fold.train_end))
    assert np.isfinite(score.rmsse).all()
