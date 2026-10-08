from __future__ import annotations

import numpy as np
import pandas as pd

from core.rules import classify_week_rules

# Two forecasts of next week's class that need no training. The ML forecast is only
# worth using if it beats both on years it has never seen.


def persistence_forecast(features: pd.DataFrame) -> np.ndarray:
    """No-change baseline: next week's class is this week's class."""
    return features["rule_class"].to_numpy()


def trend_rule_forecast(features: pd.DataFrame, params: dict | None = None) -> np.ndarray:
    """Rule-based trend forecast: extend each cell's 2-week NDMI trend by one week,
    assume this week's rain repeats, and apply the class rules to that projected week
    (NOW still needs SOON/NOW this week)."""
    trend = features["dndmi_mean_2w"].fillna(features["dNDMI"]).fillna(0)
    ndmi_next = features["NDMI_med"] + trend
    projected = pd.DataFrame(
        {
            "cell_id": features["cell_id"],
            "week_start": features["week_start"],
            "NDVI_med": features["NDVI_med"],
            "NDMI_med": ndmi_next,
            "ndmi_mean_2w": (features["NDMI_med"] + ndmi_next) / 2,
            "dndmi_mean_2w": trend,
            "dNDMI": trend,
            "ndmi_vs_cell_month_hist_med": features["ndmi_vs_cell_month_hist_med"] + trend,
            "Rain_mm_7d": features["Rain_mm_7d"],
            "Rain2w": 2 * features["Rain_mm_7d"],
        },
        index=features.index,
    )
    raw = classify_week_rules(projected, params).to_numpy()
    already_dry = features["rule_class_raw"].isin(["SOON", "NOW"]).to_numpy()
    return np.where((raw == "NOW") & ~already_dry, "SOON", raw)
