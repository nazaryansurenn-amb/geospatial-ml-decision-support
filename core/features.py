from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_NUMERIC_COLS = ["NDVI_med", "NDMI_med", "Rain_mm_7d", "Temp_C_7d", "dNDMI", "dVV", "WetScore", "Wind_ms_7d"]
MONTHLY_BASELINE_SPECS = {
    "NDMI_med": "ndmi",
    "NDVI_med": "ndvi",
    "Rain_mm_7d": "rain",
    "Temp_C_7d": "temp",
}


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week_start" in out.columns:
        out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    if "week_end" in out.columns:
        out["week_end"] = pd.to_datetime(out["week_end"], errors="coerce")
    if "year" not in out.columns and "week_start" in out.columns:
        out["year"] = out["week_start"].dt.year
    if "month" not in out.columns and "week_start" in out.columns:
        out["month"] = out["week_start"].dt.month
    if "doy" not in out.columns and "week_start" in out.columns:
        out["doy"] = out["week_start"].dt.dayofyear
    if "cell_id" in out.columns:
        out["cell_id"] = out["cell_id"].astype(str)
    return out


def _add_within_year_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["year", "cell_id", "week_start"]).reset_index(drop=True)

    for col in ROLLING_NUMERIC_COLS:
        if col in out.columns:
            out[col] = _safe_num(out[col])

    grp = out.groupby(["year", "cell_id"], sort=False)

    if "NDMI_med" in out.columns:
        out["ndmi_peak_ytd"] = grp["NDMI_med"].cummax()
        out["ndmi_drop_ytd"] = (out["ndmi_peak_ytd"] - out["NDMI_med"]).clip(lower=0)
        out["ndmi_mean_2w"] = grp["NDMI_med"].transform(lambda s: s.rolling(2, min_periods=1).mean())
        out["ndmi_mean_3w"] = grp["NDMI_med"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out["ndmi_mean_4w"] = grp["NDMI_med"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    else:
        out["ndmi_peak_ytd"] = np.nan
        out["ndmi_drop_ytd"] = np.nan
        out["ndmi_mean_2w"] = np.nan
        out["ndmi_mean_3w"] = np.nan
        out["ndmi_mean_4w"] = np.nan

    if "Rain_mm_7d" in out.columns:
        out["Rain2w"] = grp["Rain_mm_7d"].transform(lambda s: s.rolling(2, min_periods=1).sum())
        out["Rain3w"] = grp["Rain_mm_7d"].transform(lambda s: s.rolling(3, min_periods=1).sum())
        out["Rain4w"] = grp["Rain_mm_7d"].transform(lambda s: s.rolling(4, min_periods=1).sum())
        out["Rain6w"] = grp["Rain_mm_7d"].transform(lambda s: s.rolling(6, min_periods=1).sum())
        out["dry_weeks_3w"] = grp["Rain_mm_7d"].transform(lambda s: s.lt(5).astype(int).rolling(3, min_periods=1).sum())
        out["dry_weeks_6w"] = grp["Rain_mm_7d"].transform(lambda s: s.lt(5).astype(int).rolling(6, min_periods=1).sum())
        out["rain_cum_ytd"] = grp["Rain_mm_7d"].cumsum()
    else:
        out["Rain2w"] = np.nan
        out["Rain3w"] = np.nan
        out["Rain4w"] = np.nan
        out["Rain6w"] = np.nan
        out["dry_weeks_3w"] = np.nan
        out["dry_weeks_6w"] = np.nan
        out["rain_cum_ytd"] = np.nan

    if "dNDMI" in out.columns:
        out["drying3w"] = grp["dNDMI"].transform(lambda s: s.lt(0).astype(int).rolling(3, min_periods=1).sum())
        out["drying6w"] = grp["dNDMI"].transform(lambda s: s.lt(0).astype(int).rolling(6, min_periods=1).sum())
        out["drying_weeks_ytd"] = grp["dNDMI"].transform(lambda s: s.lt(0).astype(int).cumsum())
        out["dndmi_mean_2w"] = grp["dNDMI"].transform(lambda s: s.rolling(2, min_periods=1).mean())
        out["dndmi_mean_4w"] = grp["dNDMI"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    else:
        out["drying3w"] = np.nan
        out["drying6w"] = np.nan
        out["drying_weeks_ytd"] = np.nan
        out["dndmi_mean_2w"] = np.nan
        out["dndmi_mean_4w"] = np.nan

    if "Temp_C_7d" in out.columns:
        out["hot2w"] = grp["Temp_C_7d"].transform(lambda s: s.ge(28).astype(int).rolling(2, min_periods=1).sum())
        out["hot4w"] = grp["Temp_C_7d"].transform(lambda s: s.ge(28).astype(int).rolling(4, min_periods=1).sum())
        out["hot_weeks_ytd"] = grp["Temp_C_7d"].transform(lambda s: s.ge(28).astype(int).cumsum())
        out["temp_mean_2w"] = grp["Temp_C_7d"].transform(lambda s: s.rolling(2, min_periods=1).mean())
        out["temp_mean_4w"] = grp["Temp_C_7d"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    else:
        out["hot2w"] = np.nan
        out["hot4w"] = np.nan
        out["hot_weeks_ytd"] = np.nan
        out["temp_mean_2w"] = np.nan
        out["temp_mean_4w"] = np.nan

    return out


def _add_prior_year_month_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for source_col, prefix in MONTHLY_BASELINE_SPECS.items():
        baseline_col = f"{prefix}_cell_month_hist_med"
        delta_col = f"{prefix}_vs_cell_month_hist_med"

        if source_col not in out.columns:
            out[baseline_col] = np.nan
            out[delta_col] = np.nan
            continue

        yearly = (
            out[["cell_id", "month", "year", source_col]]
            .copy()
            .assign(source_value=lambda d: _safe_num(d[source_col]))
            .dropna(subset=["cell_id", "month", "year", "source_value"])
            .groupby(["cell_id", "month", "year"], as_index=False)["source_value"]
            .median()
            .sort_values(["cell_id", "month", "year"])
        )

        if yearly.empty:
            out[baseline_col] = np.nan
            out[delta_col] = np.nan
            continue

        yearly[baseline_col] = yearly.groupby(["cell_id", "month"])["source_value"].transform(
            lambda s: s.shift(1).expanding().median()
        )

        out = out.merge(yearly[["cell_id", "month", "year", baseline_col]], on=["cell_id", "month", "year"], how="left")
        out[delta_col] = _safe_num(out[source_col]) - out[baseline_col]

    return out


def _add_reference_month_baselines(df: pd.DataFrame, history_reference: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ref = _ensure_time_columns(history_reference)

    for source_col, prefix in MONTHLY_BASELINE_SPECS.items():
        baseline_col = f"{prefix}_cell_month_hist_med"
        delta_col = f"{prefix}_vs_cell_month_hist_med"

        if source_col not in out.columns or source_col not in ref.columns:
            out[baseline_col] = np.nan
            out[delta_col] = np.nan
            continue

        monthly = (
            ref[["cell_id", "month", source_col]]
            .copy()
            .assign(source_value=lambda d: _safe_num(d[source_col]))
            .dropna(subset=["cell_id", "month", "source_value"])
            .groupby(["cell_id", "month"], as_index=False)["source_value"]
            .median()
            .rename(columns={"source_value": baseline_col})
        )

        out = out.merge(monthly, on=["cell_id", "month"], how="left")
        out[delta_col] = _safe_num(out[source_col]) - out[baseline_col]

    return out


def build_temporal_features(df: pd.DataFrame, history_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    out = _ensure_time_columns(df)
    out = _add_within_year_history_features(out)

    if history_reference is None:
        out = _add_prior_year_month_baselines(out)
    else:
        out = _add_reference_month_baselines(out, history_reference)

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_temporal_features(df)
