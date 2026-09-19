"""Memory-conscious loaders for the raw M5 files, subset to one store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from m5.verify import ID_COLS, day_columns

EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    cal = pd.read_csv(raw_dir / "calendar.csv", dtype={c: "category" for c in EVENT_COLS})
    cal["date"] = pd.to_datetime(cal["date"])
    cal["d"] = cal["d"].str.removeprefix("d_").astype("int16")
    for col in ["wm_yr_wk", "year"]:
        cal[col] = cal[col].astype("int16")
    for col in ["wday", "month", *[c for c in cal.columns if c.startswith("snap_")]]:
        cal[col] = cal[col].astype("int8")
    return cal.drop(columns=["weekday"])


def load_store_sales(raw_dir: Path, store_id: str, chunksize: int = 2_000) -> pd.DataFrame:
    """Wide sales for one store: one row per series, int16 day columns d_1..d_N.

    The full file is read in chunks and filtered, so peak memory stays far below the
    size of the whole table (30,490 x 1,941 values).
    """
    path = raw_dir / "sales_train_evaluation.csv"
    days = day_columns(path)
    dtypes: dict[str, str] = {c: "str" for c in ID_COLS} | {d: "int16" for d in days}
    parts = [
        chunk[chunk["store_id"] == store_id]
        for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize)
    ]
    sales = pd.concat(parts, ignore_index=True)
    if sales.empty:
        raise ValueError(f"store {store_id!r} not found in {path.name}")
    for col in ID_COLS:
        sales[col] = sales[col].astype("category")
    return sales


def load_store_prices(raw_dir: Path, store_id: str, chunksize: int = 1_000_000) -> pd.DataFrame:
    path = raw_dir / "sell_prices.csv"
    dtypes = {"store_id": "str", "item_id": "str", "wm_yr_wk": "int16", "sell_price": "float32"}
    parts = [
        chunk[chunk["store_id"] == store_id]
        for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize)
    ]
    prices = pd.concat(parts, ignore_index=True)
    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    return prices
