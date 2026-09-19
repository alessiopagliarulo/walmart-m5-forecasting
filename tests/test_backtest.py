"""Models and the backtest runner, on the synthetic fixture only."""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from m5 import backtest, config, pipeline, plots
from m5.download import sha256_file
from m5.evaluation import make_folds
from m5.features import TARGET, build_features
from m5.models import MODELS, BoostedModel, LightGBMModel, LinearDesign, SeasonalNaive
from m5.verify import verify_raw
from tests.conftest import FIXTURE_FACTS, FIXTURE_HORIZON, FIXTURE_STORE

LAST_DAY = 420  # last day with sales in the fixture


def test_seasonal_naive_repeats_last_week(feature_panel: pd.DataFrame) -> None:
    fold = make_folds(LAST_DAY)[1]
    train, test = backtest.split(feature_panel, fold)
    prediction = SeasonalNaive().fit(train).predict(test)
    sales = feature_panel.set_index(["id", "d"])[TARGET]
    for series, day, predicted in zip(test["id"], test["d"], prediction, strict=True):
        source = fold.train_end - 7 + (day - fold.train_end - 1) % 7 + 1
        assert predicted == sales[(series, source)]


def test_split_never_hands_the_target_to_predict(feature_panel: pd.DataFrame) -> None:
    fold = make_folds(LAST_DAY)[0]
    train, test = backtest.split(feature_panel, fold)
    assert TARGET not in test.columns
    assert train["d"].max() == fold.train_end
    assert (test["d"].min(), test["d"].max()) == (fold.test_start, fold.test_end)


@pytest.mark.parametrize("name", list(MODELS))
def test_forecasts_ignore_sales_after_the_cutoff(
    name: str,
    store_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    store_prices: pd.DataFrame,
    feature_panel: pd.DataFrame,
) -> None:
    """End-to-end leakage check of the harness: scrambling every sale after the cutoff
    (which reaches both the features and the targets of later days) must not change a
    single forecast."""
    fold = make_folds(LAST_DAY)[1]
    scrambled = store_sales.copy()
    later = [c for c in scrambled.columns if c.startswith("d_") and int(c[2:]) > fold.train_end]
    rng = np.random.default_rng(0)
    scrambled[later] = rng.integers(0, 50, size=(len(scrambled), len(later))).astype("int16")
    other = build_features(scrambled, calendar, store_prices, FIXTURE_STORE, FIXTURE_HORIZON)
    assert not other[TARGET].equals(feature_panel[TARGET])

    forecasts = []
    for panel in (feature_panel, other):
        train, test = backtest.split(panel, fold)
        forecasts.append(MODELS[name]().fit(train).predict(test))
    np.testing.assert_array_equal(forecasts[0], forecasts[1])
    assert np.isfinite(forecasts[0]).all() and (forecasts[0] >= 0).all()


@pytest.fixture
def fixture_features(fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build the fixture's feature table with the real pipeline, as m5-features would."""
    hashes = {name: sha256_file(fixture_dir / name) for name in config.RAW_FILES}
    provenance = {"files": {name: {"sha256": h} for name, h in hashes.items()}}
    monkeypatch.setattr(pipeline, "load_provenance", lambda: provenance)
    monkeypatch.setattr(pipeline, "verify_raw", partial(verify_raw, expected=FIXTURE_FACTS))
    monkeypatch.setattr(backtest, "load_provenance", lambda: provenance)
    out = tmp_path / "processed"
    args = ["--raw-dir", str(fixture_dir), "--out-dir", str(out), "--manifest-dir", str(out)]
    assert pipeline.main(args) == 0
    return out


def test_backtest_end_to_end_on_fixture(fixture_features: Path, tmp_path: Path) -> None:
    results = tmp_path / "results"
    uri = f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
    args = ["--features-dir", str(fixture_features), "--results-dir", str(results)]
    assert backtest.main([*args, "--tracking-uri", uri]) == 0

    report = json.loads((results / "metrics.json").read_text())
    assert report["store_id"] == FIXTURE_STORE
    assert report["seed"] == 0
    assert set(report["environment"]["packages"]) >= {"numpy", "pandas", "scikit-learn"}
    assert [(f["train_end_d"], f["test_end_d"]) for f in report["folds"]] == [
        (336, 364),
        (364, 392),
        (392, 420),
    ]
    assert list(report["models"]) == list(MODELS)
    for model in report["models"].values():
        assert [r["fold"] for r in model["folds"]] == [1, 2, 3]
        for r in model["folds"]:
            assert np.isfinite(r["wrmsse"]) and r["runtime_seconds"] >= 0
            assert len(r["wrmsse_by_level"]) == 12
        assert model["mean_wrmsse"] == pytest.approx(np.mean([r["wrmsse"] for r in model["folds"]]))
    clf = report["models"]["hurdle_logistic_ridge"]["folds"][0]["classifier"]
    assert 0.0 <= clf["auc"] <= 1.0
    assert set(report["environment"]["packages"]) >= {"lightgbm", "xgboost"}
    boosted = [n for n, m in MODELS.items() if issubclass(m, BoostedModel)]
    assert boosted == ["lightgbm", "xgboost"]
    n_candidates = 0
    for name in boosted:
        model = report["models"][name]
        assert model["search_space"] and model["fixed_params"]["seed"] == 0
        for r in model["folds"]:
            tuning = r["tuning"]
            assert len(tuning["candidates"]) == math.prod(map(len, model["search_space"].values()))
            n_candidates += len(tuning["candidates"])

    for model in report["models"].values():
        daily = model["folds"][0]["store_daily_units"]
        assert len(daily["actual"]) == len(daily["forecast"]) == FIXTURE_HORIZON
    assert plots.main(["--results-dir", str(results)]) == 0
    png = results / f"forecast_vs_actual_{FIXTURE_STORE}.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    markdown = (results / "metrics.md").read_text()
    assert "do not edit by hand" in markdown
    assert all(name in markdown for name in MODELS)

    runs = mlflow.search_runs(experiment_names=[f"m5-backtest-{FIXTURE_STORE}"])
    # One parent run, one run per model and fold, one per search candidate.
    assert len(runs) == 1 + 3 * len(MODELS) + n_candidates


def test_backtest_stops_on_unverified_features(
    fixture_features: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {name: {"sha256": "0" * 64} for name in config.RAW_FILES}
    monkeypatch.setattr(backtest, "load_provenance", lambda: {"files": files})
    results = tmp_path / "results"
    args = ["--features-dir", str(fixture_features), "--results-dir", str(results)]
    assert backtest.main(args) == 1
    assert not results.exists()


def test_linear_design_uses_training_statistics_only() -> None:
    train = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, np.nan], "event": pd.Categorical(["A", None, "A", "B"])}
    )
    test = pd.DataFrame({"x": [np.nan, 5.0], "event": pd.Categorical(["C", "B"])})
    design = LinearDesign(["x"], ["event"]).fit(train)
    assert design.names_ == ["x", "x_missing", "event=A", "event=B", "event=none"]

    # x filled with 0 -> [1, 0, 3, 0]: mean 1, std 1.2247; missing flag [0, 1, 0, 1].
    out = design.transform(test)
    std_x = np.std([1.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(out[:, 0], [(0 - 1) / std_x, (5 - 1) / std_x], rtol=1e-6)
    np.testing.assert_allclose(out[:, 1], [(1 - 0.5) / 0.5, (0 - 0.5) / 0.5])
    # The unseen category "C" gets no indicator at all.
    raw_onehot = out[:, 2:] * np.array(design.std_[2:]) + np.array(design.mean_[2:])
    np.testing.assert_allclose(raw_onehot, [[0, 0, 0], [0, 1, 0]], atol=1e-6)


def test_boosted_search_stays_inside_the_training_window(feature_panel: pd.DataFrame) -> None:
    """The grid search validates on the last 28 training days, picks the candidate with
    the lowest inner WRMSSE and refits it; the fold's test days play no part."""
    fold = make_folds(LAST_DAY)[1]
    train, test = backtest.split(feature_panel, fold)
    model = LightGBMModel().fit(train)
    tuning = model.tuning_
    assert tuning["inner_train_end_d"] == fold.train_end - FIXTURE_HORIZON
    assert tuning["inner_valid_d"] == [fold.train_end - FIXTURE_HORIZON + 1, fold.train_end]
    assert [c["params"] for c in tuning["candidates"]] == model.candidates()
    assert len(tuning["candidates"]) == 6
    best = min(tuning["candidates"], key=lambda c: c["inner_wrmsse"])
    assert tuning["chosen"] == best["params"]
    assert model.params_ == {**model.fixed_params, **best["params"]}
    assert model.booster_.num_trees() == tuning["chosen_rounds"]
    assert 1 <= tuning["chosen_rounds"] <= 3000

    prediction = model.predict(test)
    assert prediction.shape == (len(test),) and (prediction >= 0).all()
    # Deterministic: the same training rows give the same search and the same forecast.
    again = LightGBMModel().fit(train)
    assert again.tuning_ == tuning
    np.testing.assert_array_equal(again.predict(test), prediction)
