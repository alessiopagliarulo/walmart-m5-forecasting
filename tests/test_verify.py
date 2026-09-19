from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from m5 import config
from m5.verify import DataVerificationError, verify_raw
from tests.conftest import FIXTURE_FACTS


def test_fixture_passes_with_its_own_facts(fixture_dir: Path) -> None:
    facts = verify_raw(fixture_dir, FIXTURE_FACTS)
    assert facts["n_series"] == 10
    assert facts["calendar_first_date"] == "2011-01-29"


def test_fixture_fails_against_real_m5_facts(fixture_dir: Path) -> None:
    # The gate must refuse data that is not the real M5 data.
    with pytest.raises(DataVerificationError, match="n_series: expected 30490"):
        verify_raw(fixture_dir, config.KNOWN_FACTS)


def test_missing_file_is_rejected(fixture_dir: Path, tmp_path: Path) -> None:
    for name in config.RAW_FILES[:-1]:
        shutil.copy(fixture_dir / name, tmp_path / name)
    with pytest.raises(DataVerificationError, match="missing raw file"):
        verify_raw(tmp_path, FIXTURE_FACTS)


def test_dropped_series_is_rejected(fixture_dir: Path, tmp_path: Path) -> None:
    for name in config.RAW_FILES:
        shutil.copy(fixture_dir / name, tmp_path / name)
    sales = tmp_path / "sales_train_evaluation.csv"
    lines = sales.read_text().splitlines(keepends=True)
    sales.write_text("".join(lines[:-1]))
    with pytest.raises(DataVerificationError, match="n_series"):
        verify_raw(tmp_path, FIXTURE_FACTS)


def test_gap_in_day_columns_is_rejected(fixture_dir: Path, tmp_path: Path) -> None:
    for name in config.RAW_FILES:
        shutil.copy(fixture_dir / name, tmp_path / name)
    sales = tmp_path / "sales_train_evaluation.csv"
    sales.write_text(sales.read_text().replace(",d_7,", ",d_77,", 1))
    with pytest.raises(DataVerificationError, match="contiguous"):
        verify_raw(tmp_path, FIXTURE_FACTS)
