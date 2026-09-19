"""Build the one-store feature table from verified raw M5 data.

Usage:
    m5-features                     # default store (config.DEFAULT_STORE)
    m5-features --store TX_2

Writes data/processed/features_<store>.parquet (gitignored) and a small committed
manifest, results/features_<store>_manifest.json, describing what was built from what.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from m5 import config
from m5.data import load_calendar, load_store_prices, load_store_sales
from m5.download import expected_hashes, load_provenance, sha256_file
from m5.features import TARGET, build_features, feature_columns
from m5.verify import DataVerificationError, verify_raw


def check_hashes(raw_dir: Path) -> dict[str, str]:
    expected = expected_hashes(load_provenance())
    if not expected:
        raise DataVerificationError("no provenance record; run m5-download first")
    actual = {}
    for name in config.RAW_FILES:
        actual[name] = sha256_file(raw_dir / name)
        if actual[name] != expected[name]:
            raise DataVerificationError(f"{name}: sha256 differs from provenance")
    return actual


def build_manifest(
    features: pd.DataFrame, store_id: str, horizon: int, hashes: dict[str, str]
) -> dict[str, Any]:
    history = features[features[TARGET].notna()]
    return {
        "store_id": store_id,
        "state_id": features.attrs["state_id"],
        "horizon_days": horizon,
        "n_rows": len(features),
        "n_series": int(features["id"].nunique()),
        "n_history_rows": len(history),
        "n_future_rows": len(features) - len(history),
        "first_date": str(features["date"].min().date()),
        "last_history_date": str(history["date"].max().date()),
        "last_date": str(features["date"].max().date()),
        "n_features": len(feature_columns(horizon)),
        "features": feature_columns(horizon),
        "target": TARGET,
        "raw_sha256": hashes,
    }


def write_parquet(features: pd.DataFrame, path: Path, manifest: dict[str, Any]) -> None:
    table = pa.Table.from_pandas(features, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"m5_manifest"] = json.dumps(manifest).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=config.DEFAULT_STORE, choices=config.STORES)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)

    try:
        hashes = check_hashes(args.raw_dir)
        verify_raw(args.raw_dir)
    except DataVerificationError as exc:
        print(f"STOP: data verification failed: {exc}", file=sys.stderr)
        return 1

    calendar = load_calendar(args.raw_dir)
    sales = load_store_sales(args.raw_dir, args.store)
    prices = load_store_prices(args.raw_dir, args.store)
    features = build_features(sales, calendar, prices, args.store, args.horizon)

    manifest = build_manifest(features, args.store, args.horizon, hashes)
    out_path = args.out_dir / f"features_{args.store}.parquet"
    write_parquet(features, out_path, manifest)
    manifest_path = args.manifest_dir / f"features_{args.store}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_path} ({len(features):,} rows) and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
