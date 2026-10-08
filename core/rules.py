from __future__ import annotations

import numpy as np
import pandas as pd

# Explicit irrigation-priority rules (WAIT / SOON / NOW).
#
# Input: a multi-week frame from build_temporal_features() (the 2-week persistence
# check needs the previous week of each cell to be present in the same frame).
# Only vegetated cells (NDVI_med >= ndvi_min) receive a class; others get None.
#
# NOW  = very dry 2-week NDMI, drying or below the cell's own normal for the month,
#        no recent rain relief, and SOON/NOW already in the previous week.
# SOON = dry 2-week NDMI without rain relief, very dry despite rain, or drying while
#        below the cell's own normal.
# WAIT = everything else.
#
# Thresholds were calibrated on 2022-2025 so that a typical June-August week has
# about 15% NOW and 35-42% SOON, choosing the most week-to-week stable setting.

RULE_VERSION = "rules-v1-2026-10"

DEFAULT_RULE_PARAMS = {
    "ndvi_min": 0.30,
    "ndmi_very_dry": 0.11,
    "ndmi_dry": 0.15,
    "drying_max_trend": -0.015,
    "below_normal_max_anomaly": -0.05,
    "rain_relief_7d_mm": 15.0,
    "rain_relief_14d_mm": 25.0,
    # Monthly export batches restart on day 1, so after a 31-day month the previous
    # non-overlapping report starts 10 days earlier (e.g. 05-22 before 06-01).
    "previous_week_min_days": 5,
    "previous_week_max_days": 10,
}

CLASS_ORDER = ("WAIT", "SOON", "NOW")

# Priority score: each class keeps its own band and drier canopies rank higher
# inside the band (2-week NDMI from 0.30 down to -0.10).
SCORE_BANDS = {"WAIT": (0.0, 0.3), "SOON": (0.5, 0.25), "NOW": (0.8, 0.2)}
SCORE_NDMI_WET = 0.30
SCORE_NDMI_DRY = -0.10


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def rule_conditions(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULT_RULE_PARAMS, **(params or {})}
    level = _num(df, "ndmi_mean_2w").fillna(_num(df, "NDMI_med"))
    trend = _num(df, "dndmi_mean_2w").fillna(_num(df, "dNDMI"))
    anomaly = _num(df, "ndmi_vs_cell_month_hist_med")
    rain_7d = _num(df, "Rain_mm_7d")
    rain_14d = _num(df, "Rain2w")

    return pd.DataFrame(
        {
            "vegetated": _num(df, "NDVI_med").ge(p["ndvi_min"]),
            "very_dry": level.le(p["ndmi_very_dry"]),
            "dry": level.le(p["ndmi_dry"]),
            "drying": trend.le(p["drying_max_trend"]),
            "below_normal": anomaly.le(p["below_normal_max_anomaly"]),
            "rain_relief": rain_7d.ge(p["rain_relief_7d_mm"]) | rain_14d.ge(p["rain_relief_14d_mm"]),
        },
        index=df.index,
    )


def classify_week_rules(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Single-week class before the persistence check."""
    c = rule_conditions(df, params)
    now = c["very_dry"] & (c["drying"] | c["below_normal"]) & ~c["rain_relief"]
    soon = ~now & (
        (c["dry"] & ~c["rain_relief"])
        | (c["very_dry"] & c["rain_relief"])
        | (c["drying"] & c["below_normal"])
    )
    raw = pd.Series(np.select([now, soon], ["NOW", "SOON"], default="WAIT"), index=df.index, dtype=object)
    return raw.where(c["vegetated"], None)


def matched_week_values(
    df: pd.DataFrame, values: pd.Series, min_days: int, max_days: int, direction: str
) -> pd.Series:
    """Value from the same cell's report that starts min_days..max_days earlier ("backward") or later ("forward")."""
    frame = pd.DataFrame(
        {
            "row": np.arange(len(df)),
            "cell_id": df["cell_id"].astype(str).to_numpy(),
            "week_start": pd.to_datetime(df["week_start"], errors="coerce").to_numpy(),
            "value": values.to_numpy(),
        }
    ).dropna(subset=["week_start"])
    step = pd.Timedelta(days=min_days)
    left = frame[["row", "cell_id", "week_start"]].copy()
    left["lookup"] = left["week_start"] - step if direction == "backward" else left["week_start"] + step
    right = frame[["cell_id", "week_start", "value"]].rename(columns={"week_start": "other_start"})
    right = right.dropna(subset=["value"])

    matched = pd.merge_asof(
        left.sort_values("lookup"),
        right.sort_values("other_start"),
        left_on="lookup",
        right_on="other_start",
        by="cell_id",
        direction=direction,
        tolerance=pd.Timedelta(days=max_days - min_days),
    )
    result = pd.Series([None] * len(df), index=df.index, dtype=object)
    result.iloc[matched["row"].to_numpy()] = matched["value"].to_numpy()
    return result


def previous_week_values(df: pd.DataFrame, values: pd.Series, params: dict | None = None) -> pd.Series:
    """Value of the same cell's report that started 5-10 days earlier (skips overlapping batches)."""
    p = {**DEFAULT_RULE_PARAMS, **(params or {})}
    return matched_week_values(
        df, values, int(p["previous_week_min_days"]), int(p["previous_week_max_days"]), "backward"
    )


def assign_rule_classes(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Add rule_class_raw, rule_class_prev and rule_class (with 2-week persistence for NOW)."""
    out = df.copy()
    out["rule_class_raw"] = classify_week_rules(out, params)
    out["rule_class_prev"] = previous_week_values(out, out["rule_class_raw"], params)
    persistent = out["rule_class_prev"].isin(["SOON", "NOW"])
    out["rule_class"] = out["rule_class_raw"].where(~(out["rule_class_raw"].eq("NOW") & ~persistent), "SOON")
    return out


CALIBRATION_GRID = {
    "ndmi_very_dry": [round(0.06 + 0.01 * i, 2) for i in range(11)],   # 0.06 .. 0.16
    "dry_gap": [0.04, 0.06, 0.08],                                       # ndmi_dry = very_dry + gap
    "drying_max_trend": [-0.015, -0.01, -0.005, 0.0],
    "below_normal_max_anomaly": [-0.05, -0.03, -0.02],
}


def calibrate_rule_thresholds(
    history: pd.DataFrame,
    now_share: tuple[float, float] = (0.14, 0.16),
    soon_share: tuple[float, float] = (0.35, 0.42),
    months: tuple[int, ...] = (6, 7, 8),
) -> tuple[dict, pd.DataFrame]:
    """Grid-search the four NDMI thresholds on a history frame.

    Keeps the settings whose June-August class shares fall in the target bands (about 15% NOW,
    35-42% SOON) and returns the one with the highest week-to-week stability, together with the
    whole grid. Rain-relief and persistence settings stay as in DEFAULT_RULE_PARAMS.
    """
    p = DEFAULT_RULE_PARAMS
    d = history[_num(history, "NDVI_med").ge(p["ndvi_min"])].reset_index(drop=True)
    pos = pd.Series(np.arange(len(d)), index=d.index)
    lo, hi = int(p["previous_week_min_days"]), int(p["previous_week_max_days"])
    prev_idx = pd.to_numeric(matched_week_values(d, pos, lo, hi, "backward"), errors="coerce").fillna(-1).astype(int).to_numpy()
    next_idx = pd.to_numeric(matched_week_values(d, pos, lo, hi, "forward"), errors="coerce").fillna(-1).astype(int).to_numpy()

    level = _num(d, "ndmi_mean_2w").fillna(_num(d, "NDMI_med")).to_numpy()
    trend = _num(d, "dndmi_mean_2w").fillna(_num(d, "dNDMI")).to_numpy()
    anomaly = np.nan_to_num(_num(d, "ndmi_vs_cell_month_hist_med").to_numpy(dtype=float), nan=np.inf)
    relief = (_num(d, "Rain_mm_7d").ge(p["rain_relief_7d_mm"]) | _num(d, "Rain2w").ge(p["rain_relief_14d_mm"])).to_numpy()
    month = pd.to_datetime(d["week_start"], errors="coerce").dt.month.isin(months).to_numpy()
    has_next = next_idx >= 0

    rows = []
    g = CALIBRATION_GRID
    for very_dry in g["ndmi_very_dry"]:
        for gap in g["dry_gap"]:
            for drying_t in g["drying_max_trend"]:
                for below_t in g["below_normal_max_anomaly"]:
                    is_very_dry, is_dry = level <= very_dry, level <= very_dry + gap
                    drying, below = trend <= drying_t, anomaly <= below_t
                    now = is_very_dry & (drying | below) & ~relief
                    soon = ~now & ((is_dry & ~relief) | (is_very_dry & relief) | (drying & below))
                    raw = np.where(now, 2, np.where(soon, 1, 0))
                    prev = np.where(prev_idx >= 0, raw[np.maximum(prev_idx, 0)], -1)
                    cls = np.where((raw == 2) & (prev < 1), 1, raw)
                    rows.append({
                        "ndmi_very_dry": very_dry,
                        "ndmi_dry": round(very_dry + gap, 2),
                        "drying_max_trend": drying_t,
                        "below_normal_max_anomaly": below_t,
                        "now_share": float((cls[month] == 2).mean()),
                        "soon_share": float((cls[month] == 1).mean()),
                        "stability": float((cls[has_next] == cls[next_idx[has_next]]).mean()),
                    })
    grid = pd.DataFrame(rows)
    band = grid[grid["now_share"].between(*now_share) & grid["soon_share"].between(*soon_share)]
    if band.empty:   # fall back to the setting closest to the middle of the NOW band
        band = grid.assign(_gap=(grid["now_share"] - sum(now_share) / 2).abs()).nsmallest(1, "_gap")
    best = band.sort_values("stability", ascending=False).iloc[0]
    keys = ("ndmi_very_dry", "ndmi_dry", "drying_max_trend", "below_normal_max_anomaly")
    return {k: float(best[k]) for k in keys}, grid


def rule_priority_score(df: pd.DataFrame) -> pd.Series:
    """0-1 score from rule_class: WAIT 0-0.3, SOON 0.5-0.75, NOW 0.8-1.0, ranked by dryness."""
    level = _num(df, "ndmi_mean_2w").fillna(_num(df, "NDMI_med"))
    dryness = ((SCORE_NDMI_WET - level) / (SCORE_NDMI_WET - SCORE_NDMI_DRY)).clip(0, 1).fillna(0)
    classes = df["rule_class"]
    base = classes.map({name: band[0] for name, band in SCORE_BANDS.items()})
    width = classes.map({name: band[1] for name, band in SCORE_BANDS.items()})
    return (base + width * dryness).astype(float)
