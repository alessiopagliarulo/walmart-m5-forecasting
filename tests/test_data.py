from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from m5.data import load_store_sales


def test_store_sales_are_one_store_and_downcast(store_sales: pd.DataFrame) -> None:
    assert store_sales["store_id"].astype(str).unique().tolist() == ["CA_1"]
    assert len(store_sales) == 5
    day_cols = [c for c in store_sales.columns if c.startswith("d_")]
    assert len(day_cols) == 420
    assert set(store_sales[day_cols].dtypes) == {pd.Series(dtype="int16").dtype}
    assert isinstance(store_sales["item_id"].dtype, pd.CategoricalDtype)


def test_chunk_size_does_not_change_result(fixture_dir: Path, store_sales: pd.DataFrame) -> None:
    whole = load_store_sales(fixture_dir, "CA_1", chunksize=10_000)
    pd.testing.assert_frame_equal(whole, store_sales)


def test_unknown_store_raises(fixture_dir: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_store_sales(fixture_dir, "WI_9")


def test_store_prices_are_one_store(store_prices: pd.DataFrame) -> None:
    assert store_prices["store_id"].astype(str).unique().tolist() == ["CA_1"]
    assert store_prices["sell_price"].dtype == "float32"


def test_calendar_types(calendar: pd.DataFrame) -> None:
    assert calendar["d"].tolist() == list(range(1, 449))
    assert calendar["snap_CA"].dtype == "int8"
