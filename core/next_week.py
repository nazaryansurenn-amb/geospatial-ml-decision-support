from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.io import normalize_cell_id
from core.rules import CLASS_ORDER, assign_rule_classes, matched_week_values, previous_week_values, rule_conditions

# Next-week forecast of the rule classes from core/rules.py.
#
# Every input is known at the end of the latest reporting week: the cell's
# temporal features, this week's and last week's rule class, the rule conditions,
# and the 3x3 neighbourhood in the same week. No weather forecast is used.
# Training (scripts/train_next_week.py) and the app both build features here.

FORECAST_MODEL_FILE = "next_week_forecast.joblib"
FORECAST_METADATA_FILE = "next_week_forecast_metadata.json"

# HistGradientBoostingClassifier settings, fixed in advance (never tuned on a test year).
MODEL_PARAMS = {
    "max_iter": 500,
    "learning_rate": 0.08,
    "max_leaf_nodes": 63,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 25,
    "random_state": 42,
}

BASE_FEATURES = [
    "NDVI_med", "NDMI_med", "dNDMI", "ndmi_drop_ytd", "ndmi_mean_2w", "ndmi_mean_3w",
    "ndmi_mean_4w", "dndmi_mean_2w", "dndmi_mean_4w", "Rain_mm_7d", "Rain2w", "Rain3w",
    "Rain4w", "Rain6w", "dry_weeks_3w", "dry_weeks_6w", "rain_cum_ytd", "Temp_C_7d",
    "temp_mean_2w", "temp_mean_4w", "hot2w", "hot4w", "hot_weeks_ytd", "drying3w",
    "drying6w", "drying_weeks_ytd", "ndmi_cell_month_hist_med", "ndmi_vs_cell_month_hist_med",
    "ndvi_cell_month_hist_med", "ndvi_vs_cell_month_hist_med", "rain_cell_month_hist_med",
    "rain_vs_cell_month_hist_med", "temp_cell_month_hist_med", "temp_vs_cell_month_hist_med",
    "month", "doy",
]
CONDITION_COLUMNS = ("very_dry", "dry", "drying", "below_normal", "rain_relief")
NEIGHBOUR_COLUMNS = ("is_now", "is_soon", "ndmi_mean_2w", "dndmi_mean_2w")
CLASS_CODE = {"WAIT": 0, "SOON": 1, "NOW": 2}
CROP_CODE = {"ANNUAL": 0, "PERENNIAL": 1, "NON_AGRI": 2}

FORECAST_FEATURES = (
    BASE_FEATURES
    + ["crop_code", "cur_code", "raw_code", "prev_code", "ndmi_prev"]
    + [f"c_{name}" for name in CONDITION_COLUMNS]
    + [f"nb_{name}" for name in NEIGHBOUR_COLUMNS]
    + ["nb_count"]
)


def _add_neighbour_means(d: pd.DataFrame) -> pd.DataFrame:
    xy = d["cell_id"].str.split(",", n=1, expand=True)
    d["gx"] = pd.to_numeric(xy[0], errors="coerce").fillna(-10**9).astype(np.int64)
    d["gy"] = pd.to_numeric(xy[1], errors="coerce").fillna(-10**9).astype(np.int64)
    d["is_now"] = d["cur_code"].eq(2).astype(float)
    d["is_soon"] = d["cur_code"].eq(1).astype(float)

    columns = list(NEIGHBOUR_COLUMNS)
    key = d[["week_start", "gx", "gy"]]
    sums = {column: np.zeros(len(d)) for column in columns}
    count = np.zeros(len(d))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = d[["week_start", "gx", "gy"] + columns].drop_duplicates(["week_start", "gx", "gy"]).copy()
            shifted["gx"] -= dx
            shifted["gy"] -= dy
            matched = key.merge(shifted, on=["week_start", "gx", "gy"], how="left")
            count += matched["is_now"].notna().to_numpy()
            for column in columns:
                values = matched[column].to_numpy(dtype=float)
                present = ~np.isnan(values)
                sums[column][present] += values[present]
    for column in columns:
        d[f"nb_{column}"] = np.where(count > 0, sums[column] / np.maximum(count, 1), np.nan)
    d["nb_count"] = count
    return d.drop(columns=["gx", "gy", "is_now", "is_soon"])


def build_forecast_features(frame: pd.DataFrame, crop_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Vegetated rows of a rule-classified frame (assign_rule_classes) with all forecast inputs."""
    d = frame[frame["rule_class"].notna()].copy().reset_index(drop=True)
    d["cell_id"] = d["cell_id"].astype(str)
    d["week_start"] = pd.to_datetime(d["week_start"], errors="coerce")

    if crop_map is not None and not crop_map.empty and "crop_type" in crop_map.columns:
        crops = crop_map[["cell_id", "crop_type"]].assign(cell_id=lambda c: c["cell_id"].astype(str))
        d = d.drop(columns=[c for c in ["crop_type"] if c in d.columns]).merge(crops, on="cell_id", how="left")
    if "crop_type" not in d.columns:
        d["crop_type"] = "ANNUAL"
    d["crop_type"] = d["crop_type"].fillna("ANNUAL")
    d["crop_code"] = d["crop_type"].map(CROP_CODE).fillna(0)

    conditions = rule_conditions(d)
    for name in CONDITION_COLUMNS:
        d[f"c_{name}"] = conditions[name].astype(int)
    d["cur_code"] = d["rule_class"].map(CLASS_CODE)
    d["raw_code"] = d["rule_class_raw"].map(CLASS_CODE)
    d["prev_code"] = pd.to_numeric(d["rule_class_prev"].map(CLASS_CODE), errors="coerce")
    d["ndmi_prev"] = pd.to_numeric(
        previous_week_values(d, pd.to_numeric(d["NDMI_med"], errors="coerce")), errors="coerce"
    )
    for column in BASE_FEATURES:
        d[column] = pd.to_numeric(d[column], errors="coerce") if column in d.columns else np.nan
    return _add_neighbour_means(d)


def following_week_values(frame: pd.DataFrame, values: pd.Series, min_days: int = 5, max_days: int = 10) -> pd.Series:
    """Value from the same cell's report that starts 5-10 days later (training target)."""
    return matched_week_values(frame, values, min_days, max_days, "forward")


def label_next_week_pairs(frame: pd.DataFrame, crop_map: pd.DataFrame | None) -> pd.DataFrame:
    """Forecast inputs for every vegetated cell-week whose next-week rule class is known (target_next_class)."""
    frame = frame.copy()
    frame["cell_id"] = frame["cell_id"].map(normalize_cell_id)
    features = build_forecast_features(assign_rule_classes(frame), crop_map)
    features["target_next_class"] = following_week_values(features, features["rule_class"])
    return features[features["target_next_class"].notna()].reset_index(drop=True)


def load_forecast_bundle(model_dir: Path) -> dict | None:
    model_path = Path(model_dir) / FORECAST_MODEL_FILE
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    metadata_path = Path(model_dir) / FORECAST_METADATA_FILE
    bundle["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return bundle


def predict_next_week(features: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Forecast class and class probabilities for next week, one row per cell-week in features."""
    model = bundle["model"]
    probabilities = model.predict_proba(features[bundle["features"]])
    keep = [column for column in ("cell_id", "week_start", "week_end", "crop_type", "rule_class") if column in features.columns]
    out = features[keep].copy()
    classes = [str(name) for name in model.classes_]
    for name in CLASS_ORDER:
        out[f"p_next_{name}"] = probabilities[:, classes.index(name)] if name in classes else 0.0
    out["next_class"] = np.array(classes)[probabilities.argmax(axis=1)]
    return out.reset_index(drop=True)
