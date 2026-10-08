from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_WEEKLY_COLUMNS = {"cell_id", "week_start", "week_end", "year"}
NODATA_VALUES = (-9999, -9999.0, -999)
NUMERIC_COLUMNS = {
    "year", "month", "doy", "lat", "lon", "latitude", "longitude",
    "NDVI_med", "NDMI_med", "dNDMI", "VV_med", "dVV", "Rain_mm_7d",
    "Temp_C_7d", "Wind_ms_7d", "WetScore", "veg_flag", "valid_s2", "valid_s1",
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    count: int = 1


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def messages(self, severity: str | None = None) -> list[str]:
        return [
            issue.message
            for issue in self.issues
            if severity is None or issue.severity == severity
        ]


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def normalize_cell_id(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else format(float(value), "g")

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0"):
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            pass
    return text


def normalize_weekly_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(column).strip() for column in out.columns]

    aliases = {
        "CellID": "cell_id",
        "cell": "cell_id",
        "weekStart": "week_start",
        "weekEnd": "week_end",
    }
    for source, target in aliases.items():
        if source in out.columns and target not in out.columns:
            out[target] = out[source]

    if "cell_id" in out.columns:
        out["cell_id"] = out["cell_id"].map(normalize_cell_id).astype("string")

    for column in ("week_start", "week_end"):
        if column in out.columns:
            out[column] = _to_dt(out[column])

    if "year" not in out.columns and "week_start" in out.columns:
        out["year"] = out["week_start"].dt.year
    if "month" not in out.columns and "week_start" in out.columns:
        out["month"] = out["week_start"].dt.month
    if "doy" not in out.columns and "week_start" in out.columns:
        out["doy"] = out["week_start"].dt.dayofyear

    for column in NUMERIC_COLUMNS.intersection(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(NODATA_VALUES, np.nan)

    return out


def validate_weekly_df(
    df: pd.DataFrame,
    *,
    expected_cell_count: int | None = None,
    require_contiguous_weeks: bool = False,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    missing = sorted(REQUIRED_WEEKLY_COLUMNS - set(df.columns))
    if missing:
        issues.append(ValidationIssue(
            severity="error",
            code="missing_columns",
            message=f"Missing required weekly columns: {', '.join(missing)}",
            count=len(missing),
        ))
        return ValidationReport(tuple(issues))

    invalid_key_rows = df[["cell_id", "week_start", "week_end"]].isna().any(axis=1)
    if invalid_key_rows.any():
        count = int(invalid_key_rows.sum())
        issues.append(ValidationIssue("error", "invalid_keys", f"{count:,} rows have a missing cell ID or week date.", count))

    valid_dates = df["week_start"].notna() & df["week_end"].notna()
    invalid_duration = valid_dates & ((df["week_end"] - df["week_start"]).dt.days != 7)
    if invalid_duration.any():
        count = int(invalid_duration.sum())
        issues.append(ValidationIssue("error", "invalid_week_duration", f"{count:,} rows do not use a seven-day reporting interval.", count))

    duplicates = df.duplicated(["cell_id", "week_start"], keep=False)
    if duplicates.any():
        count = int(duplicates.sum())
        issues.append(ValidationIssue("warning", "duplicate_cell_weeks", f"{count:,} rows overlap on cell ID and week; the newest source will be used.", count))

    numeric_year = pd.to_numeric(df["year"], errors="coerce")
    year_mismatch = df["week_start"].notna() & numeric_year.notna() & (numeric_year != df["week_start"].dt.year)
    if year_mismatch.any():
        count = int(year_mismatch.sum())
        issues.append(ValidationIssue("warning", "year_mismatch", f"{count:,} rows have a year that does not match week_start.", count))

    if expected_cell_count is not None and expected_cell_count > 0:
        counts = df.dropna(subset=["week_start", "cell_id"]).groupby("week_start")["cell_id"].nunique()
        incomplete = counts[counts != int(expected_cell_count)]
        if not incomplete.empty:
            issues.append(ValidationIssue(
                "warning", "unexpected_cell_count",
                f"{len(incomplete):,} reporting weeks contain a different number of cells than the expected {int(expected_cell_count):,}.",
                int(len(incomplete)),
            ))

    if require_contiguous_weeks:
        weeks = pd.Series(df["week_start"].dropna().drop_duplicates().sort_values().to_numpy())
        if len(weeks) > 1:
            irregular = weeks.diff().dt.days.iloc[1:]
            irregular = irregular[irregular > 8]
            if not irregular.empty:
                issues.append(ValidationIssue("warning", "week_gaps", f"{len(irregular):,} gaps in the weekly reporting sequence indicate missing or overlapping reports.", int(len(irregular))))

    return ValidationReport(tuple(issues))


def _read_exports(files: Iterable[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in files:
        frame = normalize_weekly_df(pd.read_csv(path, low_memory=False))
        frame["source_file"] = path.name
        frame["_src"] = path.name
        frame["_src_mtime_ns"] = int(path.stat().st_mtime_ns)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def load_exports(folder: Path, pattern: str = "*.csv") -> pd.DataFrame:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs in {folder}")

    out = _read_exports(files)
    report = validate_weekly_df(out)
    if report.errors:
        raise ValueError("; ".join(report.messages("error")))
    return out


def deduplicate_cell_weeks(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not {"cell_id", "week_start"}.issubset(df.columns):
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    sort_columns = ["week_start", "cell_id"]
    for column in ("_src_mtime_ns", "source_file"):
        if column in out.columns:
            sort_columns.append(column)
    out = out.sort_values(sort_columns, kind="stable")
    out = out.drop_duplicates(["cell_id", "week_start"], keep="last")
    final_sort = [column for column in ("year", "cell_id", "week_start") if column in out.columns]
    return out.sort_values(final_sort, kind="stable").reset_index(drop=True)


def assemble_live_season(
    folder: Path,
    *,
    pattern: str = "*.csv",
    expected_cell_count: int | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    files = sorted(folder.glob(pattern), key=lambda path: (path.stat().st_mtime_ns, path.name))
    if not files:
        return pd.DataFrame(), ValidationReport((ValidationIssue("error", "no_exports", f"No CSVs in {folder}"),))

    combined = _read_exports(files)
    report = validate_weekly_df(combined, expected_cell_count=expected_cell_count, require_contiguous_weeks=True)
    if report.errors:
        return combined, report

    combined = combined.dropna(subset=["cell_id", "week_start"]).copy()
    combined = deduplicate_cell_weeks(combined)
    return combined, report


def select_week(df: pd.DataFrame, week_start) -> pd.DataFrame:
    if df is None or df.empty or "week_start" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else None)
    selected = pd.Timestamp(week_start)
    return df.loc[pd.to_datetime(df["week_start"], errors="coerce").eq(selected)].copy()


def select_latest_week(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "week_start" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else None)
    weeks = pd.to_datetime(df["week_start"], errors="coerce").dropna()
    return select_week(df, weeks.max()) if not weeks.empty else pd.DataFrame(columns=df.columns)


def select_previous_week(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "week_start" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else None)
    weeks = pd.to_datetime(df["week_start"], errors="coerce").dropna().drop_duplicates().sort_values()
    return select_week(df, weeks.iloc[-2]) if len(weeks) >= 2 else pd.DataFrame(columns=df.columns)
