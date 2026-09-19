"""Project-wide constants: paths, data source, documented M5 facts and defaults."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
PROVENANCE_PATH = REPO_ROOT / "provenance" / "m5_data.json"

# Public HuggingFace mirror of the Kaggle "M5 Forecasting - Accuracy" competition data.
# The revision is pinned so the URLs always resolve to the same bytes.
HF_REPO = "denephew/M5_Forecasting"
HF_REVISION = "727e8b594ced7825091038cb47823bd3dccdd806"
ORIGINAL_SOURCE = "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"

RAW_FILES = (
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv",
)


def source_url(filename: str, revision: str = HF_REVISION) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/{filename}"


STORES: tuple[str, ...] = (
    "CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3",
)  # fmt: skip

# Facts about the canonical M5 data, from the competition guidelines
# (Makridakis, Spiliotis & Assimakopoulos, "The M5 competition"). The mirror is
# checked against these before any pipeline step uses it.
KNOWN_FACTS: dict[str, object] = {
    "n_series": 30_490,
    "n_items": 3_049,
    "n_stores": 10,
    "n_states": 3,
    "n_days_sales": 1_941,
    "n_days_calendar": 1_969,
    "states": ("CA", "TX", "WI"),
    "stores": STORES,
    "n_submission_rows": 60_980,
}

# One store keeps the problem laptop-sized (~3,049 series). CA_1 is the first store
# in the file and a common choice in public M5 work; any store id above is valid.
DEFAULT_STORE = "CA_1"

# The M5 forecast horizon. Sales-derived features only use sales at least this
# many days old, so one model can forecast all 28 days directly.
HORIZON = 28
