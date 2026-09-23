from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_ANY = {"cell_id", "week_start", "week_end", "year"}

def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def normalize_weekly_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # normalize key columns
    if "CellID" in df.columns and "cell_id" not in df.columns:
        df["cell_id"] = df["CellID"]
    if "cell" in df.columns and "cell_id" not in df.columns:
        df["cell_id"] = df["cell"]

    if "weekStart" in df.columns and "week_start" not in df.columns:
        df["week_start"] = df["weekStart"]
    if "weekEnd" in df.columns and "week_end" not in df.columns:
        df["week_end"] = df["weekEnd"]

    # types
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str)

    # dates
    if "week_start" in df.columns:
        df["week_start"] = _to_dt(df["week_start"])
    if "week_end" in df.columns:
        df["week_end"] = _to_dt(df["week_end"])

    # year sometimes missing -> derive
    if "year" not in df.columns and "week_start" in df.columns:
        df["year"] = df["week_start"].dt.year

    # replace -9999 with NaN for numeric columns
    for c in df.columns:
        if df[c].dtype.kind in "if":
            df[c] = df[c].replace([-9999, -9999.0, -999], np.nan)

    return df

def load_exports(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {folder}")

    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d = normalize_weekly_df(d)
        d["_src"] = f.name
        parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    missing = REQUIRED_ANY - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns in exports: {missing}")
    return out