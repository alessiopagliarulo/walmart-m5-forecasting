"""Check raw M5 files against documented facts before anything uses them."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from m5 import config

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


class DataVerificationError(RuntimeError):
    """Raised when raw data does not match what the pipeline expects."""


def day_columns(sales_path: Path) -> list[str]:
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    if header[: len(ID_COLS)] != ID_COLS:
        raise DataVerificationError(f"unexpected id columns in {sales_path.name}: {header[:6]}")
    return header[len(ID_COLS) :]


def measure_raw(raw_dir: Path) -> dict[str, Any]:
    """Measure the facts that verification compares; reads only small slices of the big files."""
    sales_path = raw_dir / "sales_train_evaluation.csv"
    days = day_columns(sales_path)
    if days != [f"d_{i}" for i in range(1, len(days) + 1)]:
        raise DataVerificationError("sales day columns are not the contiguous sequence d_1..d_N")
    ids = pd.read_csv(sales_path, usecols=ID_COLS, dtype="category")

    calendar = pd.read_csv(raw_dir / "calendar.csv", usecols=["date", "d"])
    if calendar["d"].tolist() != [f"d_{i}" for i in range(1, len(calendar) + 1)]:
        raise DataVerificationError("calendar d column is not the contiguous sequence d_1..d_N")
    dates = pd.to_datetime(calendar["date"])
    if not (dates.diff().dropna() == pd.Timedelta(days=1)).all():
        raise DataVerificationError("calendar dates are not consecutive days")

    prices = pd.read_csv(
        raw_dir / "sell_prices.csv",
        usecols=["store_id", "item_id", "sell_price"],
        dtype={"store_id": "category", "item_id": "category", "sell_price": "float32"},
    )
    if prices["sell_price"].isna().any() or (prices["sell_price"] <= 0).any():
        raise DataVerificationError("sell_prices has missing or non-positive prices")

    submission = pd.read_csv(raw_dir / "sample_submission.csv", usecols=["id"])

    return {
        "n_series": int(len(ids)),
        "n_unique_ids": int(ids["id"].nunique()),
        "n_items": int(ids["item_id"].nunique()),
        "n_stores": int(ids["store_id"].nunique()),
        "n_states": int(ids["state_id"].nunique()),
        "n_days_sales": len(days),
        "n_days_calendar": len(calendar),
        "states": tuple(sorted(ids["state_id"].unique().astype(str))),
        "stores": tuple(sorted(ids["store_id"].unique().astype(str))),
        "price_stores": tuple(sorted(prices["store_id"].unique().astype(str))),
        "price_items_subset_of_sales": bool(
            set(prices["item_id"].unique()) <= set(ids["item_id"].unique())
        ),
        "n_submission_rows": int(len(submission)),
        "calendar_first_date": str(dates.iloc[0].date()),
        "calendar_last_date": str(dates.iloc[-1].date()),
    }


def verify_raw(raw_dir: Path, expected: Mapping[str, Any] = config.KNOWN_FACTS) -> dict[str, Any]:
    """Return measured facts, or raise DataVerificationError listing every mismatch."""
    for name in config.RAW_FILES:
        if not (raw_dir / name).exists():
            raise DataVerificationError(f"missing raw file {raw_dir / name}")
    measured = measure_raw(raw_dir)
    problems = [
        f"{key}: expected {want!r}, found {measured[key]!r}"
        for key, want in expected.items()
        if measured[key] != want
    ]
    if measured["n_unique_ids"] != measured["n_series"]:
        problems.append("series ids are not unique")
    if measured["price_stores"] != measured["stores"]:
        problems.append("sell_prices stores differ from sales stores")
    if not measured["price_items_subset_of_sales"]:
        problems.append("sell_prices has items that are not in the sales file")
    if measured["n_days_calendar"] < measured["n_days_sales"] + config.HORIZON:
        problems.append("calendar does not cover the sales history plus the forecast horizon")
    if problems:
        raise DataVerificationError("; ".join(problems))
    return {k: list(v) if isinstance(v, tuple) else v for k, v in measured.items()}
