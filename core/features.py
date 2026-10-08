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


def _rolling_by_group(
    series: pd.Series,
    groupers: list[pd.Series],
    window: int,
    aggregation: str,
) -> pd.Series:
    rolling = series.groupby(groupers, sort=False).rolling(window, min_periods=1)
    values = getattr(rolling, aggregation)()
    return values.reset_index(level=list(range(len(groupers))), drop=True).sort_index()


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

    cell_year_keys = ["year", "cell_id"]
    cell_year_grp = out.groupby(cell_year_keys, sort=False)
    out["gap_days_from_previous"] = cell_year_grp["week_start"].diff().dt.days
    # Overlapping Earth Engine batches remain usable observations. Reset only when a
    # reporting start moves forward by more than one weekly cadence.
    sequence_break = out["gap_days_from_previous"].gt(8)
    out["week_sequence_id"] = sequence_break.groupby(
        [out["year"], out["cell_id"]], sort=False
    ).cumsum().astype("int32")
    sequence_keys = ["year", "cell_id", "week_sequence_id"]
    out["sequence_week_index"] = out.groupby(sequence_keys, sort=False).cumcount().add(1).astype("int16")
    grp = out.groupby(sequence_keys, sort=False)
    rolling_groupers = [out[column] for column in sequence_keys]

    if "NDMI_med" in out.columns:
        out["ndmi_peak_ytd"] = grp["NDMI_med"].cummax()
        out["ndmi_drop_ytd"] = (out["ndmi_peak_ytd"] - out["NDMI_med"]).clip(lower=0)
        out["ndmi_mean_2w"] = _rolling_by_group(out["NDMI_med"], rolling_groupers, 2, "mean")
        out["ndmi_mean_3w"] = _rolling_by_group(out["NDMI_med"], rolling_groupers, 3, "mean")
        out["ndmi_mean_4w"] = _rolling_by_group(out["NDMI_med"], rolling_groupers, 4, "mean")
    else:
        out["ndmi_peak_ytd"] = np.nan
        out["ndmi_drop_ytd"] = np.nan
        out["ndmi_mean_2w"] = np.nan
        out["ndmi_mean_3w"] = np.nan
        out["ndmi_mean_4w"] = np.nan

    if "Rain_mm_7d" in out.columns:
        out["Rain2w"] = _rolling_by_group(out["Rain_mm_7d"], rolling_groupers, 2, "sum")
        out["Rain3w"] = _rolling_by_group(out["Rain_mm_7d"], rolling_groupers, 3, "sum")
        out["Rain4w"] = _rolling_by_group(out["Rain_mm_7d"], rolling_groupers, 4, "sum")
        out["Rain6w"] = _rolling_by_group(out["Rain_mm_7d"], rolling_groupers, 6, "sum")
        dry_week = out["Rain_mm_7d"].lt(5).astype(int)
        out["dry_weeks_3w"] = _rolling_by_group(dry_week, rolling_groupers, 3, "sum")
        out["dry_weeks_6w"] = _rolling_by_group(dry_week, rolling_groupers, 6, "sum")
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
        drying_week = out["dNDMI"].lt(0).astype(int)
        out["drying3w"] = _rolling_by_group(drying_week, rolling_groupers, 3, "sum")
        out["drying6w"] = _rolling_by_group(drying_week, rolling_groupers, 6, "sum")
        out["drying_weeks_ytd"] = drying_week.groupby(rolling_groupers, sort=False).cumsum()
        out["dndmi_mean_2w"] = _rolling_by_group(out["dNDMI"], rolling_groupers, 2, "mean")
        out["dndmi_mean_4w"] = _rolling_by_group(out["dNDMI"], rolling_groupers, 4, "mean")
    else:
        out["drying3w"] = np.nan
        out["drying6w"] = np.nan
        out["drying_weeks_ytd"] = np.nan
        out["dndmi_mean_2w"] = np.nan
        out["dndmi_mean_4w"] = np.nan

    if "Temp_C_7d" in out.columns:
        hot_week = out["Temp_C_7d"].ge(28).astype(int)
        out["hot2w"] = _rolling_by_group(hot_week, rolling_groupers, 2, "sum")
        out["hot4w"] = _rolling_by_group(hot_week, rolling_groupers, 4, "sum")
        out["hot_weeks_ytd"] = hot_week.groupby(rolling_groupers, sort=False).cumsum()
        out["temp_mean_2w"] = _rolling_by_group(out["Temp_C_7d"], rolling_groupers, 2, "mean")
        out["temp_mean_4w"] = _rolling_by_group(out["Temp_C_7d"], rolling_groupers, 4, "mean")
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


def build_reference_monthly_baselines(history_reference: pd.DataFrame) -> pd.DataFrame:
    ref = _ensure_time_columns(history_reference)
    keys = ref[["cell_id", "month"]].dropna().drop_duplicates().copy()

    for source_col, prefix in MONTHLY_BASELINE_SPECS.items():
        baseline_col = f"{prefix}_cell_month_hist_med"
        if source_col not in ref.columns:
            keys[baseline_col] = np.nan
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
        keys = keys.merge(monthly, on=["cell_id", "month"], how="left")

    return keys


def _add_precomputed_month_baselines(df: pd.DataFrame, monthly_baselines: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    baselines = monthly_baselines.copy()
    if "cell_id" in baselines.columns:
        baselines["cell_id"] = baselines["cell_id"].astype(str)
    if "month" in baselines.columns:
        baselines["month"] = pd.to_numeric(baselines["month"], errors="coerce")

    expected_baseline_columns = [
        f"{prefix}_cell_month_hist_med"
        for prefix in MONTHLY_BASELINE_SPECS.values()
    ]
    for column in expected_baseline_columns:
        if column not in baselines.columns:
            baselines[column] = np.nan

    merge_columns = ["cell_id", "month", *expected_baseline_columns]
    out = out.merge(baselines[merge_columns].drop_duplicates(["cell_id", "month"]), on=["cell_id", "month"], how="left")

    for source_col, prefix in MONTHLY_BASELINE_SPECS.items():
        baseline_col = f"{prefix}_cell_month_hist_med"
        delta_col = f"{prefix}_vs_cell_month_hist_med"
        out[delta_col] = _safe_num(out[source_col]) - out[baseline_col] if source_col in out.columns else np.nan

    return out


def _add_reference_month_baselines(df: pd.DataFrame, history_reference: pd.DataFrame) -> pd.DataFrame:
    monthly = build_reference_monthly_baselines(history_reference)
    return _add_precomputed_month_baselines(df, monthly)


def build_temporal_features(
    df: pd.DataFrame,
    history_reference: pd.DataFrame | None = None,
    history_monthly_baselines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = _ensure_time_columns(df)
    out = _add_within_year_history_features(out)

    if history_monthly_baselines is not None:
        out = _add_precomputed_month_baselines(out, history_monthly_baselines)
    elif history_reference is None:
        out = _add_prior_year_month_baselines(out)
    else:
        out = _add_reference_month_baselines(out, history_reference)

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_temporal_features(df)
