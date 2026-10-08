import pandas as pd

from core.baselines import persistence_forecast, trend_rule_forecast
from core.rules import CALIBRATION_GRID, calibrate_rule_thresholds


def cell_week(cell, start, ndmi, trend, raw, cls, rain=0.0, anomaly=0.0):
    return {"cell_id": cell, "week_start": pd.Timestamp(start), "NDVI_med": 0.5, "NDMI_med": ndmi,
            "dNDMI": trend, "dndmi_mean_2w": trend, "ndmi_vs_cell_month_hist_med": anomaly,
            "Rain_mm_7d": rain, "rule_class_raw": raw, "rule_class": cls}


def test_persistence_repeats_this_week():
    f = pd.DataFrame([cell_week("1,1", "2025-07-01", 0.2, 0.0, "WAIT", "WAIT"),
                      cell_week("2,2", "2025-07-01", 0.05, -0.02, "NOW", "NOW")])
    assert persistence_forecast(f).tolist() == ["WAIT", "NOW"]


def test_trend_rule_projects_one_week_ahead():
    f = pd.DataFrame([
        cell_week("1,1", "2025-07-01", 0.08, -0.03, "SOON", "SOON"),             # dry and falling fast -> NOW
        cell_week("2,2", "2025-07-01", 0.08, -0.03, "WAIT", "WAIT"),             # same, but not dry yet -> SOON
        cell_week("3,3", "2025-07-01", 0.08, -0.03, "SOON", "SOON", rain=20.0),  # rain relief -> SOON
        cell_week("4,4", "2025-07-01", 0.40, 0.02, "WAIT", "WAIT"),              # moist and recovering -> WAIT
    ])
    assert trend_rule_forecast(f).tolist() == ["NOW", "SOON", "SOON", "WAIT"]


def test_calibration_returns_a_setting_from_the_grid():
    rows = []
    for cell in range(30):
        for week, ndmi in enumerate([0.30, 0.20, 0.12, 0.08, 0.05, 0.03]):
            rows.append({"cell_id": f"{cell},0", "week_start": pd.Timestamp("2025-06-02") + pd.Timedelta(weeks=week),
                         "NDVI_med": 0.5, "NDMI_med": ndmi + cell * 0.004, "ndmi_mean_2w": ndmi + cell * 0.004,
                         "dNDMI": -0.04, "dndmi_mean_2w": -0.04, "ndmi_vs_cell_month_hist_med": 0.0,
                         "Rain_mm_7d": 0.0, "Rain2w": 0.0})
    thresholds, grid = calibrate_rule_thresholds(pd.DataFrame(rows))
    assert thresholds["ndmi_very_dry"] in CALIBRATION_GRID["ndmi_very_dry"]
    assert thresholds["drying_max_trend"] in CALIBRATION_GRID["drying_max_trend"]
    assert thresholds["below_normal_max_anomaly"] in CALIBRATION_GRID["below_normal_max_anomaly"]
    assert len(grid) == 11 * 3 * 4 * 3
