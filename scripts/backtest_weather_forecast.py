"""Exploratory: does a weather forecast improve the next-week forecast?

Replays the test year week by week with two models trained on earlier years:
  A  the app's inputs only (no weather forecast)
  B  the same inputs plus next week's rain, ET0, maximum temperature and rain minus ET0.
B is trained on what the weather actually did (Open-Meteo ERA5 archive). In the test year it
is given the forecast that was issued at the end of each report week (Open-Meteo Previous Runs
API: day i of the next week comes from the run made i+1 days earlier), and, as an upper limit,
the weather that actually happened. Weather is taken at the app's 3 x 3 grid of points.

This uses the app's rule thresholds and crop map (not the strict setting of evaluate_strict.py).
The Previous Runs archive starts in January 2024, so test years before 2024 are not possible.

Usage:
    python scripts/backtest_weather_forecast.py                 # test 2025, train 2022-2024
    python scripts/backtest_weather_forecast.py --cache-dir reports/weather_cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

from core.forecast import build_sampling_grid
from core.io import normalize_cell_id
from core.next_week import FORECAST_FEATURES, MODEL_PARAMS, label_next_week_pairs
from core.runtime_artifacts import artifact_paths, historical_features_path

WEATHER = ["wx_rain7", "wx_et0_7", "wx_tmax", "wx_balance"]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Yerevan"


def get_json(url: str, params: dict, cache: Path) -> object:
    if cache.exists():
        return json.loads(cache.read_bytes())
    response = requests.get(url, params=params, timeout=300)
    response.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(response.content)
    return response.json()


def daily_actual(location: dict, years: list[int], cache_dir: Path) -> pd.DataFrame:
    parts = []
    for year in years:
        data = get_json(ARCHIVE_URL, {**location, "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max",
                                      "start_date": f"{year}-03-25", "end_date": f"{year}-10-15"},
                        cache_dir / f"wx_actual_{year}.json")
        for point, block in enumerate(data if isinstance(data, list) else [data]):
            d = block["daily"]
            parts.append(pd.DataFrame({"point": point, "date": pd.to_datetime(d["time"]), "rain": d["precipitation_sum"],
                                       "et0": d["et0_fao_evapotranspiration"], "tmax": d["temperature_2m_max"]}))
    return pd.concat(parts, ignore_index=True)


def daily_forecast(location: dict, year: int, cache_dir: Path) -> dict[int, pd.DataFrame]:
    by_lead = {}
    for lead in range(1, 8):
        variables = f"precipitation_previous_day{lead},et0_fao_evapotranspiration_previous_day{lead},temperature_2m_previous_day{lead}"
        data = get_json(PREVIOUS_RUNS_URL, {**location, "hourly": variables, "start_date": f"{year}-03-25", "end_date": f"{year}-10-15"},
                        cache_dir / f"wx_forecast_{year}_lead{lead}.json")
        parts = []
        for point, block in enumerate(data if isinstance(data, list) else [data]):
            h = block["hourly"]
            hourly = pd.DataFrame({"time": pd.to_datetime(h["time"]), "rain": h[f"precipitation_previous_day{lead}"],
                                   "et0": h[f"et0_fao_evapotranspiration_previous_day{lead}"],
                                   "temp": h[f"temperature_2m_previous_day{lead}"]})
            hourly["date"] = hourly["time"].dt.normalize()
            day = hourly.groupby("date").agg(rain=("rain", lambda s: s.sum(min_count=18)),
                                             et0=("et0", lambda s: s.sum(min_count=18)), tmax=("temp", "max")).reset_index()
            parts.append(day.assign(point=point))
        by_lead[lead] = pd.concat(parts, ignore_index=True)
    return by_lead


def next_week_weather(points: np.ndarray, starts: pd.Series, daily_for_offset) -> pd.DataFrame:
    """Rain/ET0 sums and mean maximum temperature over the 7 days starting at each start date."""
    base = pd.DataFrame({"point": points, "start": starts.to_numpy()})
    days = []
    for offset in range(7):
        merged = base.assign(date=base["start"] + pd.Timedelta(days=offset)).merge(
            daily_for_offset(offset)[["point", "date", "rain", "et0", "tmax"]], on=["point", "date"], how="left")
        days.append(merged[["rain", "et0", "tmax"]].to_numpy(dtype=float))
    arr = np.stack(days)
    rain, et0 = np.nansum(arr[:, :, 0], 0), np.nansum(arr[:, :, 1], 0)
    incomplete = np.isnan(arr[:, :, 0]).sum(0) > 1
    rain[incomplete], et0[incomplete] = np.nan, np.nan
    return pd.DataFrame({"wx_rain7": rain, "wx_et0_7": et0, "wx_tmax": np.nanmean(arr[:, :, 2], 0), "wx_balance": rain - et0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--train-years", type=int, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reports" / "weather_cache")
    args = parser.parse_args()
    train_years = args.train_years or list(range(2022, args.test_year))
    runtime = args.data_dir / "runtime"

    coords = pd.read_csv(args.data_dir / "Echmiadzin_Cell_Coordinates.csv")
    coords["cell_id"] = coords["cell_id"].map(normalize_cell_id)
    grid = build_sampling_grid(coords, rows=3, columns=3)
    lat, lon = np.array([p.latitude for p in grid]), np.array([p.longitude for p in grid])
    dist = (coords["lat"].to_numpy()[:, None] - lat) ** 2 + ((coords["lon"].to_numpy()[:, None] - lon) * np.cos(np.radians(lat.mean()))) ** 2
    nearest = pd.Series(dist.argmin(1), index=coords["cell_id"])
    location = {"latitude": ",".join(f"{v:.4f}" for v in lat), "longitude": ",".join(f"{v:.4f}" for v in lon), "timezone": TIMEZONE}

    crop_map = pd.read_parquet(artifact_paths(runtime)["stable_crop_map"])
    crop_map["cell_id"] = crop_map["cell_id"].map(normalize_cell_id)
    pairs = pd.concat([label_next_week_pairs(pd.read_parquet(historical_features_path(runtime, y)), crop_map).assign(year=y)
                       for y in [*train_years, args.test_year]], ignore_index=True)
    pairs["point"] = pairs["cell_id"].map(nearest).fillna(4).astype(int).to_numpy()
    pairs["next_start"] = pd.to_datetime(pairs["week_start"]) + pd.Timedelta(days=7)   # forecast issued as the report week ends

    actual = daily_actual(location, [*train_years, args.test_year], args.cache_dir)
    pairs[WEATHER] = next_week_weather(pairs["point"].to_numpy(), pairs["next_start"], lambda offset: actual).to_numpy()
    test = pairs[pairs["year"] == args.test_year].reset_index(drop=True)
    forecast = daily_forecast(location, args.test_year, args.cache_dir)
    forecast_weather = next_week_weather(test["point"].to_numpy(), test["next_start"], lambda offset: forecast[offset + 1])

    train = pairs[pairs["year"].isin(train_years)]
    model_a = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(train[FORECAST_FEATURES], train["target_next_class"])
    model_b = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(train[FORECAST_FEATURES + WEATHER], train["target_next_class"])
    with_forecast = test[FORECAST_FEATURES].assign(**{c: forecast_weather[c].to_numpy() for c in WEATHER})

    y = test["target_next_class"].to_numpy()
    results = pd.DataFrame({
        "no_change": test["rule_class"].to_numpy(),
        "A_no_weather": model_a.predict(test[FORECAST_FEATURES]),
        "B_weather_forecast": model_b.predict(with_forecast[FORECAST_FEATURES + WEATHER]),
        "B_actual_weather": model_b.predict(test[FORECAST_FEATURES + WEATHER]),
    })
    month = pd.to_datetime(test["week_start"]).dt.month.to_numpy()
    print(f"test {args.test_year}: {len(test):,} cell-weeks; trained on {train_years}")
    print(f"{'method':22s} {'season':>8s} " + " ".join(f"{'m' + str(m):>7s}" for m in sorted(set(month))) + "   (skill vs no-change by month)")
    for col in results:
        correct = results[col].to_numpy() == y
        line = f"{col:22s} {correct.mean():8.1%} "
        for m in sorted(set(month)):
            sel = month == m
            base_err = (results['no_change'].to_numpy()[sel] != y[sel]).mean()
            line += f"{1 - (~correct[sel]).mean() / base_err:7.2f} "
        print(line)
    weekly = pd.DataFrame({"week": test["week_start"], "fc": forecast_weather["wx_rain7"], "act": test["wx_rain7"],
                           "fc_et0": forecast_weather["wx_et0_7"], "act_et0": test["wx_et0_7"]}).groupby("week").mean()
    print(f"weekly rain forecast vs actual: correlation {weekly['fc'].corr(weekly['act']):.2f}, "
          f"mean {weekly['fc'].mean():.1f} vs {weekly['act'].mean():.1f} mm | ET0: correlation {weekly['fc_et0'].corr(weekly['act_et0']):.2f}, "
          f"mean {weekly['fc_et0'].mean():.1f} vs {weekly['act_et0'].mean():.1f} mm")


if __name__ == "__main__":
    main()
