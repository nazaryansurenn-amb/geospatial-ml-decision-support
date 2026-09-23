from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ACTION_WAIT = "WAIT"
ACTION_SOON = "IRRIGATE_SOON"
ACTION_NOW  = "IRRIGATE_NOW"

def pick_latest_live_file(live_dir: Path) -> Path | None:
    files = sorted(live_dir.glob("live_*.csv"))
    if not files:
        return None

    # Prefer newest by week_start inside file
    best = None
    best_dt = None
    for f in files:
        try:
            df = pd.read_csv(f, nrows=50)
            if "week_start" in df.columns:
                ws = pd.to_datetime(df["week_start"], errors="coerce").dropna()
                if not ws.empty:
                    dt = ws.max().to_pydatetime()
                    if best_dt is None or dt > best_dt:
                        best_dt = dt
                        best = f
        except Exception:
            pass

    return best or files[-1]

def to_actions(
    p: np.ndarray,
    T_decision: float,
    T_now: float,
) -> list[str]:
    out = []
    for x in p:
        if x >= T_now:
            out.append(ACTION_NOW)
        elif x >= T_decision:
            out.append(ACTION_SOON)
        else:
            out.append(ACTION_WAIT)
    return out

def adjust_actions_with_forecast(
    actions: list[str],
    rain_fc_7d: float,
    temp_fc_7d: float,
    wind_fc_7d: float,
    rain_ok_mm: float = 10.0,
    hot_c: float = 30.0,
    windy: float = 25.0,  # Open-Meteo often km/h
) -> list[str]:
    """
    Downshift if rain is enough; upshift if hot+windy stress.
    """
    def down(a):
        if a == ACTION_NOW:
            return ACTION_SOON
        if a == ACTION_SOON:
            return ACTION_WAIT
        return ACTION_WAIT

    def up(a):
        if a == ACTION_WAIT:
            return ACTION_SOON
        if a == ACTION_SOON:
            return ACTION_NOW
        return ACTION_NOW

    out = actions[:]

    if rain_fc_7d >= rain_ok_mm:
        out = [down(a) for a in out]

    if (temp_fc_7d >= hot_c) and (wind_fc_7d >= windy):
        out = [up(a) for a in out]

    return out