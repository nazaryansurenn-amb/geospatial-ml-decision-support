"""End-to-end check on real data: runtime files, rules and one next-week forecast for the latest live week.

Usage:
    python scripts/smoke_check.py
    python scripts/smoke_check.py --root path/to/project   # a folder holding data/ and models/
Exits with code 1 when a required file is missing or the forecast cannot be made.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.features import build_temporal_features
from core.io import assemble_live_season, normalize_cell_id
from core.next_week import build_forecast_features, load_forecast_bundle, predict_next_week
from core.rules import CLASS_ORDER, assign_rule_classes
from core.runtime_artifacts import artifact_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    root = parser.parse_args().root
    runtime = root / "data" / "runtime"

    required = [artifact_paths(runtime)["monthly_baselines"], artifact_paths(runtime)["stable_crop_map"],
                root / "models" / "next_week_forecast.joblib"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[FAIL] missing: " + ", ".join(missing))
        return 1

    season, report = assemble_live_season(root / "data" / "live_exports")
    if season.empty or report.errors:
        print("[FAIL] no usable live exports: " + "; ".join(report.messages("error")))
        return 1
    baselines = pd.read_parquet(artifact_paths(runtime)["monthly_baselines"])
    baselines["cell_id"] = baselines["cell_id"].map(normalize_cell_id)
    crop_map = pd.read_parquet(artifact_paths(runtime)["stable_crop_map"])
    crop_map["cell_id"] = crop_map["cell_id"].map(normalize_cell_id)

    classified = assign_rule_classes(build_temporal_features(season, history_monthly_baselines=baselines))
    features = build_forecast_features(classified, crop_map)
    latest = features[features["week_start"] == features["week_start"].max()]
    forecast = predict_next_week(latest, load_forecast_bundle(root / "models"))

    week = pd.Timestamp(latest["week_start"].max()).date()
    now = forecast["rule_class"].value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict()
    nxt = forecast["next_class"].value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict()
    print(f"[OK] latest live week {week}: {len(forecast):,} vegetated cells")
    print(f"     this week (rules): {now}")
    print(f"     next week (model): {nxt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
