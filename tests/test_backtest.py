"""Models and the backtest runner, on the synthetic fixture only."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from m5 import backtest, config, pipeline
from m5.download import sha256_file
from m5.evaluation import make_folds
from m5.features import TARGET, build_features
from m5.models import MODELS, SeasonalNaive
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

    markdown = (results / "metrics.md").read_text()
    assert "do not edit by hand" in markdown
    assert all(name in markdown for name in MODELS)

    runs = mlflow.search_runs(experiment_names=[f"m5-backtest-{FIXTURE_STORE}"])
    assert len(runs) == 1 + 3 * len(MODELS)


def test_backtest_stops_on_unverified_features(
    fixture_features: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {name: {"sha256": "0" * 64} for name in config.RAW_FILES}
    monkeypatch.setattr(backtest, "load_provenance", lambda: {"files": files})
    results = tmp_path / "results"
    args = ["--features-dir", str(fixture_features), "--results-dir", str(results)]
    assert backtest.main(args) == 1
    assert not results.exists()
