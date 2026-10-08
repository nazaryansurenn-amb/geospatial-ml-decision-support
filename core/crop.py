from __future__ import annotations

import numpy as np
import pandas as pd


CROP_ANNUAL = "ANNUAL"
CROP_PERENNIAL = "PERENNIAL"
CROP_NON_AGRI = "NON_AGRI"


def classify_crop_year(
    df_year: pd.DataFrame,
    ndvi_green_thr: float = 0.30,
    perennial_green_weeks: int = 10,
    perennial_p25: float = 0.25,
    nonagri_p90_max: float = 0.20,
    nonagri_green_weeks_max: int = 2,
) -> pd.DataFrame:
    df = df_year.copy()
    df["cell_id"] = df["cell_id"].astype(str)

    ndvi = pd.to_numeric(df.get("NDVI_med"), errors="coerce")
    if "valid_s2" in df.columns:
        valid = pd.to_numeric(df["valid_s2"], errors="coerce").fillna(0).astype(int)
    else:
        valid = pd.Series(1, index=df.index, dtype=int)
    ok = valid.eq(1) & ndvi.notna()
    observations = df.loc[ok, ["cell_id"]].copy()
    observations["NDVI_med"] = ndvi.loc[ok]

    all_cells = df[["cell_id"]].drop_duplicates().copy()
    if observations.empty:
        all_cells["crop_type"] = CROP_ANNUAL
        return all_cells

    grouped = observations.groupby("cell_id")["NDVI_med"]
    stats = grouped.agg(
        NDVI_p25=lambda values: float(np.nanpercentile(values, 25)),
        NDVI_p90=lambda values: float(np.nanpercentile(values, 90)),
        GREEN_WEEKS=lambda values: int((pd.to_numeric(values, errors="coerce") > ndvi_green_thr).sum()),
    ).reset_index()

    is_nonagri = (
        stats["GREEN_WEEKS"].le(nonagri_green_weeks_max)
        & stats["NDVI_p90"].le(nonagri_p90_max)
    )
    is_perennial = (
        stats["GREEN_WEEKS"].ge(perennial_green_weeks)
        & stats["NDVI_p25"].ge(perennial_p25)
    )
    stats["crop_type"] = np.select(
        [is_nonagri, is_perennial],
        [CROP_NON_AGRI, CROP_PERENNIAL],
        default=CROP_ANNUAL,
    )
    return all_cells.merge(stats[["cell_id", "crop_type"]], on="cell_id", how="left").fillna({"crop_type": CROP_ANNUAL})


def build_stable_crop_map(
    history: pd.DataFrame,
    ndvi_green_thr: float = 0.30,
    perennial_green_weeks: int = 10,
    perennial_p25: float = 0.25,
) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["cell_id", "crop_type", "crop_years", "crop_stability"])

    df = history.copy()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["week_start"], errors="coerce").dt.year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["cell_id", "year"])

    yearly_maps: list[pd.DataFrame] = []
    for year, year_df in df.groupby("year", sort=True):
        classified = classify_crop_year(
            year_df,
            ndvi_green_thr=ndvi_green_thr,
            perennial_green_weeks=perennial_green_weeks,
            perennial_p25=perennial_p25,
        )
        classified["year"] = int(year)
        yearly_maps.append(classified)

    if not yearly_maps:
        return pd.DataFrame(columns=["cell_id", "crop_type", "crop_years", "crop_stability"])

    votes = pd.concat(yearly_maps, ignore_index=True)

    def choose_class(values: pd.Series) -> str:
        counts = values.value_counts()
        highest = int(counts.max())
        tied = set(counts[counts == highest].index.astype(str))
        for conservative_choice in (CROP_ANNUAL, CROP_PERENNIAL, CROP_NON_AGRI):
            if conservative_choice in tied:
                return conservative_choice
        return CROP_ANNUAL

    result = votes.groupby("cell_id").agg(
        crop_type=("crop_type", choose_class),
        crop_years=("year", "nunique"),
    ).reset_index()
    vote_counts = votes.groupby(["cell_id", "crop_type"]).size().rename("votes").reset_index()
    result = result.merge(vote_counts, on=["cell_id", "crop_type"], how="left")
    result["crop_stability"] = result["votes"] / result["crop_years"].clip(lower=1)
    return result.drop(columns="votes")
