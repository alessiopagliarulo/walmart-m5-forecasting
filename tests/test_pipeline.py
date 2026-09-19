from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from m5 import config, pipeline
from m5.download import sha256_file
from m5.verify import verify_raw
from tests.conftest import FIXTURE_FACTS


@pytest.fixture
def fixture_provenance(fixture_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    hashes = {name: sha256_file(fixture_dir / name) for name in config.RAW_FILES}
    files = {name: {"sha256": h} for name, h in hashes.items()}
    monkeypatch.setattr(pipeline, "load_provenance", lambda: {"files": files})
    monkeypatch.setattr(pipeline, "verify_raw", partial(verify_raw, expected=FIXTURE_FACTS))
    return hashes


def test_end_to_end_on_fixture(
    fixture_dir: Path, tmp_path: Path, fixture_provenance: dict[str, str]
) -> None:
    args = ["--raw-dir", str(fixture_dir), "--out-dir", str(tmp_path), "--manifest-dir"]
    assert pipeline.main([*args, str(tmp_path), "--store", "TX_1"]) == 0

    manifest = json.loads((tmp_path / "features_TX_1_manifest.json").read_text())
    assert manifest["store_id"] == "TX_1"
    assert manifest["state_id"] == "TX"
    assert manifest["raw_sha256"] == fixture_provenance
    assert manifest["n_series"] == 5
    assert manifest["n_future_rows"] == 5 * 28

    table = pq.read_table(tmp_path / "features_TX_1.parquet")
    assert json.loads(table.schema.metadata[b"m5_manifest"])["store_id"] == "TX_1"
    frame = table.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == manifest["n_rows"]
    assert set(manifest["features"]) <= set(frame.columns)


def test_hash_mismatch_stops_pipeline(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {name: {"sha256": "0" * 64} for name in config.RAW_FILES}
    monkeypatch.setattr(pipeline, "load_provenance", lambda: {"files": files})
    code = pipeline.main(["--raw-dir", str(fixture_dir), "--out-dir", str(tmp_path)])
    assert code == 1
    assert not list(tmp_path.iterdir())
