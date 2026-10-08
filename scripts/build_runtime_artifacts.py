"""Build the compact runtime files in data/runtime/ from the weekly CSV exports.

The app, training and evaluation read these Parquet files instead of the raw CSVs:
historical_features_<year>.parquet, historical_crop_map_<year>.parquet, historical_reference.parquet,
monthly_baselines.parquet, stable_crop_map.parquet and manifest.json.

Usage:
    python scripts/build_runtime_artifacts.py
    python scripts/build_runtime_artifacts.py --exports data/weekly_exports --out data/runtime
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.io import load_exports
from core.runtime_artifacts import build_runtime_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, default=ROOT / "data" / "weekly_exports")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "runtime")
    args = parser.parse_args()

    files = sorted(args.exports.glob("*.csv"))
    signature = [{"name": f.name, "bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in files]
    manifest = build_runtime_artifacts(load_exports(args.exports), args.out, source_signature=signature)
    print(f"{len(files)} CSV files -> {args.out}")
    print(f"years {manifest['years']}, rows {manifest['historical_rows']:,}, "
          f"{manifest['total_artifact_bytes'] / 1e6:.1f} MB in {manifest['build_seconds']:.0f}s")


if __name__ == "__main__":
    main()
