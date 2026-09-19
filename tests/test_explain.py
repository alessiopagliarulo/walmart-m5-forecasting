"""SHAP explanations and the SNAP lift, on the synthetic fixture and constructed data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from m5 import backtest, explain
from m5.data import load_state_sales
from m5.evaluation import make_folds
from m5.features import feature_columns
from m5.models import MODELS, BoostedModel
from m5.snap import daily_category_units, snap_lift
from tests.conftest import FIXTURE_HORIZON, FIXTURE_STORE

LAST_DAY = 420
BOOSTED = [n for n, m in MODELS.items() if issubclass(m, BoostedModel)]


@pytest.mark.parametrize("name", BOOSTED)
def test_refit_rebuilds_the_fitted_model(name: str, feature_panel: pd.DataFrame) -> None:
    """m5-explain rebuilds each fold's model from its recorded choice; that must give
    back the very model `fit` produced, not a similar one."""
    fold = make_folds(LAST_DAY)[1]
    train, test = backtest.split(feature_panel, fold)
    fitted = MODELS[name]().fit(train)
    assert isinstance(fitted, BoostedModel)
    rebuilt = MODELS[name]()
    assert isinstance(rebuilt, BoostedModel)
    rebuilt.refit(train, fitted.tuning_["chosen"], fitted.tuning_["chosen_rounds"])
    np.testing.assert_array_equal(rebuilt.predict(test), fitted.predict(test))


@pytest.mark.parametrize("name", BOOSTED)
def test_shap_values_add_up_to_the_log_forecast(name: str, feature_panel: pd.DataFrame) -> None:
    fold = make_folds(LAST_DAY)[2]
    train, test = backtest.split(feature_panel, fold)
    model = MODELS[name]()
    assert isinstance(model, BoostedModel)
    model.refit(train, model.candidates()[0], 50)
    values, base = model.shap_values(test)
    assert values.shape == (len(test), len(feature_columns(FIXTURE_HORIZON)))
    np.testing.assert_allclose(values.sum(axis=1) + base, np.log(model.predict(test)), atol=1e-4)


def constructed_days(food_lift: float, other_lift: float) -> pd.DataFrame:
    """Two years of noise-free daily sales with a weekly cycle, month-to-month growth, a
    closed Christmas, and SNAP on the 1st to the 10th multiplying sales by a known factor."""
    dates = pd.date_range("2013-01-01", "2014-12-31", freq="D")
    snap = (dates.day <= 10).astype("int8")
    months_in = (dates.year - 2013) * 12 + dates.month
    level = (1 + 0.2 * np.sin(2 * np.pi * dates.dayofweek / 7)) * (1 + 0.02 * months_in)
    christmas = (dates.month == 12) & (dates.day == 25)
    foods = np.where(christmas, 0.0, 100 * level * np.where(snap == 1, 1 + food_lift, 1.0))
    hobbies = np.where(christmas, 0.0, 30 * level * np.where(snap == 1, 1 + other_lift, 1.0))
    household = np.where(christmas, 0.0, 50 * level * np.where(snap == 1, 1 + other_lift, 1.0))
    return pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "wday": dates.dayofweek,
            "event_name_1": ["Christmas" if c else None for c in christmas],
            "snap_CA": snap,
            "FOODS": foods,
            "HOBBIES": hobbies,
            "HOUSEHOLD": household,
            "total": foods + hobbies + household,
        }
    )


def test_snap_lift_recovers_a_known_effect() -> None:
    result = snap_lift(constructed_days(0.20, 0.05), "CA", reps=200)
    cats = result["categories"]
    assert cats["FOODS"]["lift"] == pytest.approx(0.20)
    assert cats["HOBBIES"]["lift"] == pytest.approx(0.05)
    assert cats["NON_FOOD"]["lift"] == pytest.approx(0.05)
    assert result["foods_vs_non_food"]["lift"] == pytest.approx(1.20 / 1.05 - 1)
    lo, hi = cats["FOODS"]["lift_ci95"]
    assert lo == pytest.approx(0.20) and hi == pytest.approx(0.20)
    assert result["snap_days_of_month"] == list(range(1, 11))
    assert result["dropped_days"] == ["2013-12-25", "2014-12-25"]
    # 24 months x 7 weekdays, every one holding SNAP and other days.
    assert cats["FOODS"]["n_cells"] == 24 * 7
    assert cats["FOODS"]["share_of_cells_higher_on_snap_days"] == 1.0


def test_snap_lift_is_zero_without_an_effect() -> None:
    result = snap_lift(constructed_days(0.0, 0.0), "CA", reps=50)
    for r in result["categories"].values():
        assert r["lift"] == pytest.approx(0.0, abs=1e-12)


def test_daily_category_units_sums_each_category(fixture_dir: Path, calendar: pd.DataFrame) -> None:
    sales = load_state_sales(fixture_dir, "CA")
    daily = daily_category_units(sales, calendar)
    assert len(daily) == LAST_DAY
    foods = sales[sales["cat_id"] == "FOODS"][[f"d_{d}" for d in range(1, LAST_DAY + 1)]]
    np.testing.assert_array_equal(daily["FOODS"], foods.sum().to_numpy())
    np.testing.assert_array_equal(
        daily["total"], daily[["FOODS", "HOBBIES", "HOUSEHOLD"]].sum(axis=1)
    )


def run_backtest(features: Path, results: Path, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
    args = ["--features-dir", str(features), "--results-dir", str(results)]
    assert backtest.main([*args, "--models", *BOOSTED, "--tracking-uri", uri]) == 0


def test_explain_end_to_end_on_fixture(
    fixture_features: Path, fixture_dir: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    run_backtest(fixture_features, results, tmp_path)
    args = ["--features-dir", str(fixture_features), "--raw-dir", str(fixture_dir)]
    assert explain.main([*args, "--results-dir", str(results)]) == 0

    summary = json.loads((results / "shap_summary.json").read_text())
    metrics = json.loads((results / "metrics.json").read_text())
    assert summary["store_id"] == FIXTURE_STORE
    assert list(summary["models"]) == BOOSTED
    features = feature_columns(FIXTURE_HORIZON)
    for name, model in summary["models"].items():
        assert sorted(r["feature"] for r in model["mean_importance"]) == sorted(features)
        assert sum(r["share"] for r in model["mean_importance"]) == pytest.approx(1.0)
        shap_means = [r["mean_abs_shap"] for r in model["mean_importance"]]
        assert shap_means == sorted(shap_means, reverse=True)
        for entry, recorded in zip(model["folds"], metrics["models"][name]["folds"], strict=True):
            assert entry["wrmsse_rebuilt"] == pytest.approx(recorded["wrmsse"], abs=1e-12)
            assert entry["n_rows"] == 5 * FIXTURE_HORIZON  # every test row, no sampling
            assert entry["max_additivity_error"] < 1e-4
        latest = model["latest_fold"]
        assert latest["fold"] == 3 and len(latest["series"]) == 3
        assert latest["segments"][0]["segment"] == "all rows"
        assert latest["segments"][0]["n_rows"] == 5 * FIXTURE_HORIZON
        png = results / f"shap_summary_{name}_{FIXTURE_STORE}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    png = results / f"shap_importance_{FIXTURE_STORE}.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    snap = json.loads((results / "snap_lift.json").read_text())
    assert set(snap["scopes"]) == {f"store {FIXTURE_STORE}", "state CA (all stores)"}
    for scope in snap["scopes"].values():
        assert set(scope["categories"]) == {"FOODS", "HOBBIES", "HOUSEHOLD", "NON_FOOD"}
        assert np.isfinite(scope["foods_vs_non_food"]["lift"])
    markdown = (results / "explanations.md").read_text()
    assert "do not edit by hand" in markdown and "SNAP-day lift" in markdown


def test_explain_stops_when_the_rebuilt_model_differs(
    fixture_features: Path, fixture_dir: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    run_backtest(fixture_features, results, tmp_path)
    metrics_path = results / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["models"][BOOSTED[0]]["folds"][0]["wrmsse"] += 0.01
    metrics_path.write_text(json.dumps(metrics))
    args = ["--features-dir", str(fixture_features), "--raw-dir", str(fixture_dir)]
    assert explain.main([*args, "--results-dir", str(results)]) == 1
    assert not (results / "shap_summary.json").exists()
