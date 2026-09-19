"""Regenerate the SYNTHETIC test fixture in tests/fixtures/synthetic_m5/.

The fixture only mimics the M5 file layout. Every value is random and made up: it is for
unit tests only and must never be used to produce a published number.

    uv run python tests/fixtures/make_synthetic_m5.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "synthetic_m5"
N_HISTORY = 420
HORIZON = 28
STORES = {"CA_1": "CA", "TX_1": "TX"}
ITEMS = ["FOODS_1_001", "FOODS_1_002", "HOBBIES_1_001", "HOBBIES_1_002", "HOUSEHOLD_1_001"]
LATE_ITEM, LATE_WEEK = "HOBBIES_1_002", 20  # first priced week (0-based) for the late item


def main() -> None:
    rng = np.random.default_rng(20260919)
    n_days = N_HISTORY + HORIZON
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")  # starts on a Saturday
    week = np.arange(n_days) // 7
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + week,
            "weekday": dates.day_name(),
            "wday": (dates.dayofweek + 2) % 7 + 1,  # M5 convention: Saturday = 1
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, n_days + 1)],
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
        }
    )
    for day in rng.choice(n_days, size=30, replace=False):
        calendar.loc[day, ["event_name_1", "event_type_1"]] = rng.choice(
            [["FixtureFest", "Cultural"], ["FixtureDay", "National"]]
        )
    for day in rng.choice(n_days, size=5, replace=False):
        calendar.loc[day, ["event_name_2", "event_type_2"]] = ["FixtureExtra", "Religious"]
    # Different SNAP windows per state so tests can tell which state's flag was used.
    for state, (lo, hi) in {"CA": (1, 10), "TX": (1, 15), "WI": (5, 14)}.items():
        calendar[f"snap_{state}"] = ((dates.day >= lo) & (dates.day <= hi)).astype(int)

    sales_rows, price_rows = [], []
    for store, state in STORES.items():
        for item in ITEMS:
            dept, cat = item.rsplit("_", 1)[0], item.split("_")[0]
            first_week = LATE_WEEK if item == LATE_ITEM else 0
            level = rng.uniform(0.5, 6)
            sales = rng.poisson(level, size=N_HISTORY)
            sales[: first_week * 7] = 0
            sales_rows.append([f"{item}_{store}_evaluation", item, dept, cat, store, state, *sales])
            price = round(float(rng.uniform(1, 10)), 2)
            for w in range(first_week, week.max() + 1):
                if rng.random() < 0.05:
                    price = round(price * float(rng.uniform(0.8, 1.2)), 2)
                price_rows.append([store, item, 11101 + w, price])

    day_cols = [f"d_{i}" for i in range(1, N_HISTORY + 1)]
    sales_df = pd.DataFrame(
        sales_rows,
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *day_cols],
    )
    prices_df = pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    ids = [
        i.replace("_evaluation", s) for s in ("_validation", "_evaluation") for i in sales_df["id"]
    ]
    submission = pd.DataFrame({"id": ids, **{f"F{i}": 0 for i in range(1, HORIZON + 1)}})

    OUT.mkdir(exist_ok=True)
    calendar.to_csv(OUT / "calendar.csv", index=False)
    sales_df.to_csv(OUT / "sales_train_evaluation.csv", index=False)
    prices_df.to_csv(OUT / "sell_prices.csv", index=False)
    submission.to_csv(OUT / "sample_submission.csv", index=False)


if __name__ == "__main__":
    main()
