"""Shared fixtures. Tests only ever touch the SYNTHETIC fixture, never real M5 data."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from m5 import backtest, config, pipeline
from m5.data import load_calendar, load_store_prices, load_store_sales
from m5.download import sha256_file
from m5.features import build_features
from m5.verify import verify_raw

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic_m5"
FIXTURE_STORE = "CA_1"
FIXTURE_HORIZON = 28

# What the synthetic fixture contains, by construction (see make_synthetic_m5.py).
FIXTURE_FACTS: dict[str, Any] = {
    "n_series": 10,
    "n_items": 5,
    "n_stores": 2,
    "n_states": 2,
    "n_days_sales": 420,
    "n_days_calendar": 448,
    "states": ("CA", "TX"),
    "stores": ("CA_1", "TX_1"),
    "n_submission_rows": 20,
}


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def calendar() -> pd.DataFrame:
    return load_calendar(FIXTURE_DIR)


@pytest.fixture(scope="session")
def store_sales() -> pd.DataFrame:
    return load_store_sales(FIXTURE_DIR, FIXTURE_STORE, chunksize=3)


@pytest.fixture(scope="session")
def store_prices() -> pd.DataFrame:
    return load_store_prices(FIXTURE_DIR, FIXTURE_STORE, chunksize=100)


@pytest.fixture(scope="session")
def feature_panel(
    store_sales: pd.DataFrame, calendar: pd.DataFrame, store_prices: pd.DataFrame
) -> pd.DataFrame:
    return build_features(store_sales, calendar, store_prices, FIXTURE_STORE, FIXTURE_HORIZON)


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
