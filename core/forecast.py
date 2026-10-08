from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARIABLES = (
    "precipitation_sum",
    "precipitation_probability_max",
    "temperature_2m_max",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
)


@dataclass(frozen=True)
class ForecastPoint:
    point_id: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class PointForecast:
    point_id: str
    latitude: float
    longitude: float
    rain_mm_7d: float
    precipitation_probability_max: float
    temperature_max_mean_c: float
    wind_max_kmh: float
    et0_mm_7d: float
    rain_minus_et0_mm: float


def build_sampling_grid(
    coordinates: pd.DataFrame,
    *,
    rows: int = 3,
    columns: int = 3,
) -> list[ForecastPoint]:
    if coordinates is None or coordinates.empty:
        return []

    lat_column = next((column for column in ("lat", "latitude", "center_lat") if column in coordinates.columns), None)
    lon_column = next((column for column in ("lon", "longitude", "center_lon") if column in coordinates.columns), None)
    if lat_column is None or lon_column is None:
        raise ValueError("Coordinates must contain latitude and longitude columns.")

    latitudes = pd.to_numeric(coordinates[lat_column], errors="coerce").dropna()
    longitudes = pd.to_numeric(coordinates[lon_column], errors="coerce").dropna()
    if latitudes.empty or longitudes.empty:
        return []

    lat_grid = np.linspace(float(latitudes.min()), float(latitudes.max()), max(1, int(rows)))
    lon_grid = np.linspace(float(longitudes.min()), float(longitudes.max()), max(1, int(columns)))
    return [
        ForecastPoint(f"r{row_index + 1}c{column_index + 1}", float(latitude), float(longitude))
        for row_index, latitude in enumerate(lat_grid)
        for column_index, longitude in enumerate(lon_grid)
    ]


def _numbers(values: Iterable | None) -> list[float]:
    if values is None:
        return []
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return [float(value) for value in numeric]


def _sum(values: Iterable | None) -> float:
    numeric = _numbers(values)
    return float(sum(numeric)) if numeric else 0.0


def _mean(values: Iterable | None) -> float:
    numeric = _numbers(values)
    return float(np.mean(numeric)) if numeric else 0.0


def _max(values: Iterable | None) -> float:
    numeric = _numbers(values)
    return float(max(numeric)) if numeric else 0.0


def summarize_point_forecast(payload: dict, point: ForecastPoint) -> PointForecast:
    daily = payload.get("daily", {}) or {}
    rain = _sum(daily.get("precipitation_sum"))
    et0 = _sum(daily.get("et0_fao_evapotranspiration"))
    return PointForecast(
        point_id=point.point_id,
        latitude=float(payload.get("latitude", point.latitude)),
        longitude=float(payload.get("longitude", point.longitude)),
        rain_mm_7d=round(rain, 2),
        precipitation_probability_max=round(_max(daily.get("precipitation_probability_max")), 1),
        temperature_max_mean_c=round(_mean(daily.get("temperature_2m_max")), 2),
        wind_max_kmh=round(_max(daily.get("wind_speed_10m_max")), 2),
        et0_mm_7d=round(et0, 2),
        rain_minus_et0_mm=round(rain - et0, 2),
    )


def summarize_area_forecast(payload, points: list[ForecastPoint]) -> dict:
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(points):
        raise ValueError(f"Forecast returned {len(payloads)} locations for {len(points)} requested points.")

    point_summaries = [summarize_point_forecast(item, point) for item, point in zip(payloads, points)]
    frame = pd.DataFrame([asdict(item) for item in point_summaries])
    metric_columns = [
        "rain_mm_7d",
        "precipitation_probability_max",
        "temperature_max_mean_c",
        "wind_max_kmh",
        "et0_mm_7d",
        "rain_minus_et0_mm",
    ]
    area = {
        "point_count": int(len(frame)),
        "rain_mm_7d": round(float(frame["rain_mm_7d"].median()), 2),
        "rain_mm_7d_min": round(float(frame["rain_mm_7d"].min()), 2),
        "rain_mm_7d_max": round(float(frame["rain_mm_7d"].max()), 2),
        "precipitation_probability_max": round(float(frame["precipitation_probability_max"].max()), 1),
        "temperature_max_mean_c": round(float(frame["temperature_max_mean_c"].median()), 2),
        "wind_max_kmh": round(float(frame["wind_max_kmh"].max()), 2),
        "et0_mm_7d": round(float(frame["et0_mm_7d"].median()), 2),
        "rain_minus_et0_mm": round(float(frame["rain_minus_et0_mm"].median()), 2),
    }
    area["ranges"] = {
        column: {
            "min": round(float(frame[column].min()), 2),
            "max": round(float(frame[column].max()), 2),
        }
        for column in metric_columns
    }
    return {
        "provider": "Open-Meteo",
        "forecast_days": 7,
        "area": area,
        "points": [asdict(item) for item in point_summaries],
    }


def fetch_area_forecast(
    points: list[ForecastPoint],
    *,
    timeout: int = 20,
    session=None,
) -> dict:
    if not points:
        raise ValueError("At least one forecast point is required.")
    if session is None:
        import requests

        session = requests
    params = {
        "latitude": ",".join(f"{point.latitude:.6f}" for point in points),
        "longitude": ",".join(f"{point.longitude:.6f}" for point in points),
        "daily": ",".join(DAILY_VARIABLES),
        "forecast_days": 7,
        "timezone": "Asia/Yerevan",
    }
    response = session.get(FORECAST_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return summarize_area_forecast(response.json(), points)


def fetch_openmeteo_7d(lat: float, lon: float) -> dict:
    point = ForecastPoint("center", float(lat), float(lon))
    forecast = fetch_area_forecast([point])
    area = forecast["area"]
    return {
        "rain_fc_7d": area["rain_mm_7d"],
        "temp_fc_7d": area["temperature_max_mean_c"],
        "wind_fc_7d": area["wind_max_kmh"],
        "et0_fc_7d": area["et0_mm_7d"],
        "rain_minus_et0_fc_7d": area["rain_minus_et0_mm"],
    }
