import pandas as pd
import pytest

from core.io import load_exports, normalize_weekly_df


def test_normalize_weekly_df_maps_columns_and_derives_year():
    df = pd.DataFrame(
        {
            "CellID": ["1,1"],
            "weekStart": ["2026-04-01"],
            "weekEnd": ["2026-04-07"],
            "NDVI_med": [0.4],
        }
    )

    out = normalize_weekly_df(df)

    assert "cell_id" in out.columns
    assert "week_start" in out.columns
    assert "week_end" in out.columns
    assert "year" in out.columns
    assert int(out.loc[0, "year"]) == 2026


def test_normalize_weekly_df_replaces_nodata_markers():
    df = pd.DataFrame({"cell_id": ["1,1"], "NDVI_med": [-9999.0], "year": [2026]})
    out = normalize_weekly_df(df)
    assert pd.isna(out.loc[0, "NDVI_med"])


def test_load_exports_raises_when_required_columns_missing(tmp_path):
    bad = pd.DataFrame({"cell_id": ["1,1"]})
    bad.to_csv(tmp_path / "bad.csv", index=False)

    with pytest.raises(ValueError):
        load_exports(tmp_path)
