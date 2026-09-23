# core/labels.py
from __future__ import annotations

import numpy as np
import pandas as pd


def make_teacher_labels(
    df: pd.DataFrame,
    NDVI_min: float = 0.30,
    NDMI_soft_max: float = 0.25,     # soft because NDMI can reflect canopy moisture
    WetScore_max: float = 0.65,
    rain7d_max: float = 8.0,
    dVV_dry_max: float = -0.02,      # IMPORTANT: negative = drying (tune later)
    irr_event_forces_zero: bool = True,
    cols: dict | None = None,
) -> pd.Series:
    """
    Teacher label y_need (0/1).

    Need irrigation (1) when:
      - valid_s2 == 1  (if missing -> assume valid)
      - vegetation present (veg_flag==1 OR NDVI >= NDVI_min)
      - Rain_mm_7d < rain7d_max
      - WetScore < WetScore_max
      - NDMI <= NDMI_soft_max  (soft gate)
      - Radar condition (dVV):
          * If dVV exists:
              - if dVV is NaN -> do NOT block (fallback to optical+weather)
              - else require dVV <= dVV_dry_max (drying / no recent wetting)
          * If dVV missing -> do NOT block

      - If irr_event==1 and irr_event_forces_zero=True -> force 0
    """

    c = {
        "ndvi": "NDVI_med",
        "ndmi": "NDMI_med",
        "wet": "WetScore",
        "rain": "Rain_mm_7d",
        "valid": "valid_s2",
        "veg": "veg_flag",
        "irr": "irr_event",
        "dvv": "dVV",
    }
    if cols:
        c.update(cols)

    idx = df.index

    def num(col: str, default=np.nan) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(default, index=idx)

    ndvi = num(c["ndvi"])
    ndmi = num(c["ndmi"])
    wet = num(c["wet"])
    rain = num(c["rain"])
    dvv = num(c["dvv"])

    # valid_s2: if missing, assume valid
    if c["valid"] in df.columns:
        valid = num(c["valid"], default=0).fillna(0).astype(int)
    else:
        valid = pd.Series(1, index=idx, dtype=int)

    # vegetation gate: prefer veg_flag, fallback to NDVI
    if c["veg"] in df.columns:
        veg_raw = pd.to_numeric(df[c["veg"]], errors="coerce").fillna(0)
        # treat -9999 as 0
        veg_raw = veg_raw.where(veg_raw != -9999, 0)
        veg = (veg_raw == 1)
    else:
        veg = (ndvi >= NDVI_min)

    # -----------------------------
    # Radar dryness gate (SAFE)
    # -----------------------------
    # If dvv is missing as a column -> don't block
    # If dvv is NaN for a row -> don't block (fallback to WetScore + weather)
    if c["dvv"] in df.columns:
        radar_ok = dvv.isna() | (dvv <= dVV_dry_max)
    else:
        radar_ok = pd.Series(True, index=idx)

    # Teacher “need irrigation” label
    y = (
        (valid == 1)
        & veg.fillna(False)
        & (ndvi >= NDVI_min)
        & (rain < rain7d_max)
        & (wet < WetScore_max)
        & radar_ok
        & (ndmi <= NDMI_soft_max)
    ).astype(int)

    # If irrigation event already detected this week, force label 0
    if irr_event_forces_zero and c["irr"] in df.columns:
        irr = pd.to_numeric(df[c["irr"]], errors="coerce").fillna(0).astype(int)
        y = y.where(irr != 1, 0)

    return pd.Series(y, index=idx, name="y_need")