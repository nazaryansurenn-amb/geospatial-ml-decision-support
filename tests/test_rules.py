import pandas as pd
import pytest

from core.next_week import FORECAST_FEATURES, build_forecast_features, following_week_values
from core.rules import assign_rule_classes, rule_priority_score


def report(cell, start, ndmi_2w, trend, rain_7d=0.0, rain_14d=0.0, anomaly=0.0, ndvi=0.5):
    return {
        "cell_id": cell,
        "week_start": pd.Timestamp(start),
        "NDVI_med": ndvi,
        "NDMI_med": ndmi_2w,
        "ndmi_mean_2w": ndmi_2w,
        "dndmi_mean_2w": trend,
        "dNDMI": trend,
        "ndmi_vs_cell_month_hist_med": anomaly,
        "Rain_mm_7d": rain_7d,
        "Rain2w": rain_14d,
    }


def classes(rows):
    result = assign_rule_classes(pd.DataFrame(rows))
    return dict(zip(zip(result["cell_id"], result["week_start"].dt.strftime("%m-%d")), result["rule_class"]))


def test_now_needs_dry_previous_week():
    got = classes([report("1,1", "2025-07-01", 0.05, -0.02), report("1,1", "2025-07-08", 0.05, -0.02)])
    assert got[("1,1", "07-01")] == "SOON"
    assert got[("1,1", "07-08")] == "NOW"


def test_now_after_wait_week_is_soon():
    got = classes([report("1,1", "2025-07-01", 0.30, 0.02), report("1,1", "2025-07-08", 0.05, -0.02)])
    assert got[("1,1", "07-01")] == "WAIT"
    assert got[("1,1", "07-08")] == "SOON"


@pytest.mark.parametrize("rain_7d, rain_14d", [(20.0, 20.0), (5.0, 30.0)])
def test_rain_relief_blocks_now(rain_7d, rain_14d):
    got = classes([
        report("1,1", "2025-07-01", 0.05, -0.02),
        report("1,1", "2025-07-08", 0.05, -0.02, rain_7d=rain_7d, rain_14d=rain_14d),
    ])
    assert got[("1,1", "07-08")] == "SOON"


def test_month_start_week_finds_previous_report_ten_days_earlier():
    # Monthly batches: 05-29 overlaps 06-01, so 06-01's previous week is 05-22.
    got = classes([
        report("1,1", "2025-05-22", 0.12, 0.0),
        report("1,1", "2025-05-29", 0.05, -0.02),
        report("1,1", "2025-06-01", 0.05, -0.02),
    ])
    assert got[("1,1", "05-22")] == "SOON"
    assert got[("1,1", "05-29")] == "NOW"
    assert got[("1,1", "06-01")] == "NOW"


def test_below_normal_counts_without_drying():
    got = classes([
        report("1,1", "2025-07-01", 0.14, 0.01, anomaly=-0.06),
        report("1,1", "2025-07-08", 0.10, 0.01, anomaly=-0.06),
        report("2,2", "2025-07-01", 0.14, 0.01),
        report("2,2", "2025-07-08", 0.10, 0.01),
    ])
    assert got[("1,1", "07-08")] == "NOW"
    assert got[("2,2", "07-08")] == "SOON"  # very dry but neither drying nor below normal


def test_slight_decline_is_not_drying():
    got = classes([report("1,1", "2025-07-01", 0.05, -0.01), report("1,1", "2025-07-08", 0.05, -0.01)])
    assert got[("1,1", "07-08")] == "SOON"


def test_drying_below_normal_is_soon_even_when_moist():
    got = classes([report("1,1", "2025-07-08", 0.20, -0.02, anomaly=-0.06)])
    assert got[("1,1", "07-08")] == "SOON"


def test_unvegetated_cells_get_no_class():
    got = classes([report("1,1", "2025-07-08", 0.05, -0.02, ndvi=0.2)])
    assert got[("1,1", "07-08")] is None


def test_priority_score_keeps_class_bands_and_ranks_dryness():
    frame = assign_rule_classes(pd.DataFrame([
        report("1,1", "2025-07-01", 0.05, -0.02), report("1,1", "2025-07-08", 0.05, -0.02),
        report("2,2", "2025-07-01", 0.00, -0.02), report("2,2", "2025-07-08", -0.05, -0.02),
        report("3,3", "2025-07-08", 0.40, 0.02),
    ]))
    score = dict(zip(zip(frame["cell_id"], frame["week_start"].dt.strftime("%m-%d")), rule_priority_score(frame)))
    assert 0.8 <= score[("1,1", "07-08")] < score[("2,2", "07-08")] <= 1.0
    assert 0.5 <= score[("1,1", "07-01")] <= 0.75
    assert 0.0 <= score[("3,3", "07-08")] <= 0.3


def test_forecast_features_and_next_week_target():
    rows = [report(cell, start, 0.05, -0.02) for cell in ("10,10", "11,10") for start in ("2025-07-01", "2025-07-08")]
    features = build_forecast_features(assign_rule_classes(pd.DataFrame(rows)))
    assert set(FORECAST_FEATURES) <= set(features.columns)
    latest = features[features["week_start"] == pd.Timestamp("2025-07-08")]
    assert (latest["nb_count"] == 1).all() and (latest["nb_is_now"] == 1.0).all()
    target = following_week_values(features, features["rule_class"])
    assert target[features["week_start"] == pd.Timestamp("2025-07-01")].tolist() == ["NOW", "NOW"]
    assert target[features["week_start"] == pd.Timestamp("2025-07-08")].isna().all()
