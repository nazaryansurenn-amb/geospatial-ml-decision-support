from __future__ import annotations
import requests

def fetch_openmeteo_7d(lat: float, lon: float) -> dict:
    """
    Returns dict with:
      rain_fc_7d (mm), temp_fc_7d (C mean of max), wind_fc_7d (km/h mean of max)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,windspeed_10m_max",
        "forecast_days": 7,
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})

    rain = daily.get("precipitation_sum", []) or []
    tmax = daily.get("temperature_2m_max", []) or []
    wmax = daily.get("windspeed_10m_max", []) or []

    rain_sum = float(sum(rain)) if rain else 0.0
    temp_mean = float(sum(tmax) / len(tmax)) if tmax else 0.0
    wind_mean = float(sum(wmax) / len(wmax)) if wmax else 0.0

    # Open-Meteo windspeed_10m_max is provided in km/h by default.
    # Keep values in km/h so action thresholds remain consistent.
    return {
        "rain_fc_7d": rain_sum,
        "temp_fc_7d": temp_mean,
        "wind_fc_7d": wind_mean,
    }