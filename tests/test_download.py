from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from m5 import config, download
from m5.verify import DataVerificationError


def _fake_fetch(fixture_dir: Path):  # type: ignore[no-untyped-def]
    def fetch(url: str, dest: Path) -> None:
        shutil.copy(fixture_dir / url.rsplit("/", 1)[1], dest)

    return fetch


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"m5" * 1000)
    assert download.sha256_file(path, chunk_size=7) == hashlib.sha256(b"m5" * 1000).hexdigest()


def test_download_records_hash_and_url(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "fetch", _fake_fetch(fixture_dir))
    records = download.download_all(tmp_path, expected={})
    assert [r.filename for r in records] == list(config.RAW_FILES)
    for r in records:
        assert r.sha256 == download.sha256_file(fixture_dir / r.filename)
        assert r.url == config.source_url(r.filename)


def test_hash_mismatch_stops_download(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "fetch", _fake_fetch(fixture_dir))
    with pytest.raises(DataVerificationError, match="does not match provenance"):
        download.download_all(tmp_path, expected={"calendar.csv": "0" * 64})


def test_present_file_with_matching_hash_is_not_refetched(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in config.RAW_FILES:
        shutil.copy(fixture_dir / name, tmp_path / name)
    expected = {name: download.sha256_file(tmp_path / name) for name in config.RAW_FILES}

    def fail(url: str, dest: Path) -> None:
        raise AssertionError("should not fetch")

    monkeypatch.setattr(download, "fetch", fail)
    assert len(download.download_all(tmp_path, expected)) == 4


def test_committed_provenance_pins_all_four_files() -> None:
    provenance = json.loads(config.PROVENANCE_PATH.read_text())
    assert provenance["mirror"]["revision"] == config.HF_REVISION
    assert set(provenance["files"]) == set(config.RAW_FILES)
    for name, rec in provenance["files"].items():
        assert rec["url"] == config.source_url(name)
        assert len(rec["sha256"]) == 64
    facts = provenance["verified_facts"]
    for key, want in config.KNOWN_FACTS.items():
        assert facts[key] == (list(want) if isinstance(want, tuple) else want)
