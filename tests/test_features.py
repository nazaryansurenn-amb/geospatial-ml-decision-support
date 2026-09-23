import pandas as pd

from core.features import build_temporal_features


def test_build_temporal_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "cell_id": ["1,1", "1,1", "1,1"],
            "week_start": ["2026-04-01", "2026-04-08", "2026-04-15"],
            "week_end": ["2026-04-07", "2026-04-14", "2026-04-21"],
            "year": [2026, 2026, 2026],
            "NDMI_med": [0.4, 0.3, 0.2],
            "NDVI_med": [0.6, 0.6, 0.5],
            "Rain_mm_7d": [1.0, 2.0, 3.0],
            "Temp_C_7d": [25.0, 27.0, 29.0],
            "dNDMI": [-0.01, -0.02, -0.03],
        }
    )
    out = build_temporal_features(df)

    for col in [
        "month",
        "doy",
        "ndmi_peak_ytd",
        "Rain2w",
        "drying3w",
        "temp_mean_2w",
        "ndmi_cell_month_hist_med",
        "ndmi_vs_cell_month_hist_med",
    ]:
        assert col in out.columns

    assert out["month"].notna().all()
    assert out["doy"].notna().all()
