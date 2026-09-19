"""Download the four M5 files, check them against the provenance record, verify them.

Usage:
    m5-download                    # fetch into data/raw, require hashes to match provenance
    m5-download --record           # (re)write provenance/m5_data.json from verified files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from m5 import config
from m5.verify import DataVerificationError, verify_raw


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileRecord:
    filename: str
    url: str
    sha256: str
    bytes: int


def load_provenance(path: Path = config.PROVENANCE_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def expected_hashes(provenance: dict[str, Any] | None) -> dict[str, str]:
    if provenance is None:
        return {}
    return {name: rec["sha256"] for name, rec in provenance["files"].items()}


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def download_all(raw_dir: Path, expected: dict[str, str]) -> list[FileRecord]:
    """Fetch each file unless an identical copy (same sha256) is already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in config.RAW_FILES:
        url = config.source_url(name)
        dest = raw_dir / name
        if dest.exists() and name in expected and sha256_file(dest) == expected[name]:
            print(f"{name}: already present, sha256 matches provenance")
        else:
            print(f"{name}: downloading {url}")
            fetch(url, dest)
        digest = sha256_file(dest)
        if name in expected and digest != expected[name]:
            raise DataVerificationError(
                f"{name}: sha256 {digest} does not match provenance {expected[name]}"
            )
        records.append(FileRecord(name, url, digest, dest.stat().st_size))
    return records


def write_provenance(
    records: list[FileRecord], facts: dict[str, Any], path: Path = config.PROVENANCE_PATH
) -> None:
    payload = {
        "dataset": "M5 Forecasting - Accuracy (Walmart)",
        "original_source": config.ORIGINAL_SOURCE,
        "mirror": {
            "huggingface_repo": config.HF_REPO,
            "revision": config.HF_REVISION,
            "license_field": "cc",
        },
        "files": {
            r.filename: {"url": r.url, "sha256": r.sha256, "bytes": r.bytes} for r in records
        },
        "verified_facts": facts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument(
        "--record",
        action="store_true",
        help="write provenance from the downloaded files (only after they pass verification)",
    )
    args = parser.parse_args(argv)

    provenance = load_provenance()
    if provenance is None and not args.record:
        print("No provenance record found; rerun with --record to create one.", file=sys.stderr)
        return 2
    expected = {} if args.record else expected_hashes(provenance)
    try:
        records = download_all(args.raw_dir, expected)
        facts = verify_raw(args.raw_dir)
    except DataVerificationError as exc:
        print(f"STOP: data verification failed: {exc}", file=sys.stderr)
        return 1
    if args.record:
        write_provenance(records, facts)
        print(f"Wrote {config.PROVENANCE_PATH}")
    print("All files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
