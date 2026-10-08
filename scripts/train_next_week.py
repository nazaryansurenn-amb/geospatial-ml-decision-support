"""Train the next-week priority forecast and record how it compares with persistence.

The target is next week's rule class (core/rules.py). Each test year is predicted by a
model trained only on earlier years; the shipped model is then trained on all
history years and checked on the live season, which it has never seen.

Each evaluation also scores two baselines that need no training: no-change (persistence)
and the rule-based trend forecast (core/baselines.py).

Usage:
    python scripts/train_next_week.py
    python scripts/train_next_week.py --years 2022 2023 2024 2025 --test-years 2024 2025
    python scripts/train_next_week.py --data-dir path/to/data --models-dir path/to/models
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from core.baselines import trend_rule_forecast
from core.features import build_temporal_features
from core.io import assemble_live_season, normalize_cell_id
from core.next_week import (
    FORECAST_FEATURES,
    FORECAST_METADATA_FILE,
    FORECAST_MODEL_FILE,
    MODEL_PARAMS,
    label_next_week_pairs,
)
from core.rules import CLASS_ORDER, DEFAULT_RULE_PARAMS, RULE_VERSION
from core.runtime_artifacts import artifact_paths, historical_features_path

STARTED = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - STARTED:5.0f}s] {message}", flush=True)


def load_crop_map(runtime_dir: Path) -> pd.DataFrame:
    crop_map = pd.read_parquet(artifact_paths(runtime_dir)["stable_crop_map"])
    crop_map["cell_id"] = crop_map["cell_id"].map(normalize_cell_id)
    return crop_map


def load_history_pairs(years: list[int], crop_map: pd.DataFrame, runtime_dir: Path) -> pd.DataFrame:
    parts = []
    for year in years:
        path = historical_features_path(runtime_dir, year)
        if not path.exists():
            raise FileNotFoundError(f"Missing runtime features for {year}: {path}")
        parts.append(label_next_week_pairs(pd.read_parquet(path), crop_map).assign(year=year))
        log(f"{year}: {len(parts[-1]):,} cell-weeks with a matched next week")
    return pd.concat(parts, ignore_index=True)


def load_live_pairs(crop_map: pd.DataFrame, runtime_dir: Path, live_dir: Path) -> pd.DataFrame:
    season, report = assemble_live_season(live_dir)
    if season.empty or report.errors:
        return pd.DataFrame()
    baselines = pd.read_parquet(artifact_paths(runtime_dir)["monthly_baselines"])
    baselines["cell_id"] = baselines["cell_id"].map(normalize_cell_id)
    return label_next_week_pairs(build_temporal_features(season, history_monthly_baselines=baselines), crop_map)


def score(truth: np.ndarray, predicted: np.ndarray, current: np.ndarray) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(truth, predicted, labels=["NOW"], zero_division=0)
    changed = truth != current
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, labels=list(CLASS_ORDER), average="macro")), 4),
        "now_precision": round(float(precision[0]), 4),
        "now_recall": round(float(recall[0]), 4),
        "changing_cells_correct": round(float((predicted[changed] == truth[changed]).mean()), 4) if changed.any() else None,
    }


def evaluate(model, pairs: pd.DataFrame) -> dict:
    truth = pairs["target_next_class"].to_numpy()
    current = pairs["rule_class"].to_numpy()
    return {
        "rows": int(len(pairs)),
        "share_changing_class": round(float((truth != current).mean()), 4),
        "model": score(truth, model.predict(pairs[FORECAST_FEATURES]), current),
        "persistence": score(truth, current, current),
        "trend_rule": score(truth, trend_rule_forecast(pairs), current),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025],
                        help="History years for training (2021 lacks prior-year baselines).")
    parser.add_argument("--test-years", type=int, nargs="+", default=[2024, 2025],
                        help="Years evaluated with a model trained on earlier years only.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    runtime_dir, models_dir = args.data_dir / "runtime", args.models_dir

    crop_map = load_crop_map(runtime_dir)
    history = load_history_pairs(args.years, crop_map, runtime_dir)

    validation = {}
    for test_year in args.test_years:
        train = history[history["year"] < test_year]
        test = history[history["year"] == test_year]
        if train.empty or test.empty:
            continue
        model = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(train[FORECAST_FEATURES], train["target_next_class"])
        validation[str(test_year)] = {"train_years": sorted(train["year"].unique().tolist()), **evaluate(model, test)}
        log(f"test {test_year}: model {validation[str(test_year)]['model']['accuracy']:.3f} "
            f"vs persistence {validation[str(test_year)]['persistence']['accuracy']:.3f}")

    model = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(history[FORECAST_FEATURES], history["target_next_class"])
    log(f"final model trained on {args.years} ({len(history):,} cell-weeks, {model.n_iter_} boosting rounds)")

    live = load_live_pairs(crop_map, runtime_dir, args.data_dir / "live_exports")
    if not live.empty:
        live_year = int(pd.to_datetime(live["week_start"]).dt.year.max())
        validation[f"{live_year}_live"] = {"train_years": args.years, **evaluate(model, live)}
        log(f"live {live_year}: model {validation[f'{live_year}_live']['model']['accuracy']:.3f} "
            f"vs persistence {validation[f'{live_year}_live']['persistence']['accuracy']:.3f}")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FORECAST_FEATURES}, models_dir / FORECAST_MODEL_FILE, compress=3)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": "next week's rule class (report starting 5-10 days after the latest one)",
        "rule_version": RULE_VERSION,
        "rule_params": DEFAULT_RULE_PARAMS,
        "model": "HistGradientBoostingClassifier",
        "model_params": MODEL_PARAMS,
        "sklearn_version": sklearn.__version__,
        "train_years": args.years,
        "train_rows": int(len(history)),
        "classes": list(CLASS_ORDER),
        "features": FORECAST_FEATURES,
        "uses_weather_forecast": False,
        "validation": validation,
        "baseline": "persistence: next week keeps this week's rule class",
        "other_baseline": "trend_rule: extend the 2-week NDMI trend one week and apply the class rules",
    }
    (models_dir / FORECAST_METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    size_mb = (models_dir / FORECAST_MODEL_FILE).stat().st_size / 1e6
    log(f"saved {FORECAST_MODEL_FILE} ({size_mb:.1f} MB) and {FORECAST_METADATA_FILE}")


if __name__ == "__main__":
    main()
