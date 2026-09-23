# Geospatial ML Decision Support

[![CI](https://github.com/nazaryansurenn-amb/geospatial-ml-decision-support/actions/workflows/ci.yml/badge.svg)](https://github.com/nazaryansurenn-amb/geospatial-ml-decision-support/actions/workflows/ci.yml)

A spatiotemporal classification pipeline with a map-based decision tool:

- Sentinel-2 satellite indices and weather are aggregated per 250 m grid cell and per week.
- Rolling-window and seasonal-baseline features are built from them.
- A RandomForest assigns each cell one of three priority classes.

There is no ground truth, so the model trains on labels produced by a documented rule (weak
supervision). The evaluation section states what that does and does not show.

Case study: weekly irrigation priorities (**WAIT** / **SOON** / **NOW**) for the Echmiadzin Water
User Association, Armavir region, Armenia.

Methods: geospatial grid construction and reprojection · time-series feature engineering · weak
supervision · class-balanced RandomForest · per-year held-out evaluation · Streamlit and Folium.

Code only. The Association's data, boundary and trained model are not included (see [Data](#data)).

## Pipeline

| Step | What it does | Code |
|---|---|---|
| Grid | 250 m square cells built in UTM zone 38N (`EPSG:32638`) around each centroid, then reprojected to WGS84 | `core/grid.py` |
| Load | Reads weekly per-cell CSV exports, normalises column-name variants (`CellID`/`cell_id`, `weekStart`/`week_start`) and replaces `-9999` with NaN | `core/io.py` |
| Crop type | Classifies each cell per season from its NDVI: `NON_AGRI`, `PERENNIAL` or `ANNUAL` | training script |
| Features | 36 numeric features plus crop type | `core/features.py` |
| Labels | Rule-based need score, mapped to three classes | training script |
| Model | RandomForest classifier, WAIT / SOON / NOW | training script |
| App | Scores the selected or latest week and draws the map | `app.py` |

### Input columns

| Column | Meaning |
|---|---|
| `cell_id`, `week_start`, `week_end`, `year` | Required keys (`year` is derived from `week_start` if missing) |
| `NDVI_med` | Median NDVI in the cell for the week |
| `NDMI_med`, `dNDMI` | Median NDMI and its week-on-week change |
| `Rain_mm_7d`, `Temp_C_7d` | 7-day rainfall and temperature |
| `valid_s2` | 1 if the Sentinel-2 observation is cloud-free |
| `veg_flag` | 1 if vegetation is present |

### Crop type

A green week is a valid observation with NDVI > 0.30.

| Class | Condition |
|---|---|
| `NON_AGRI` | ≤ 2 green weeks and NDVI 90th percentile ≤ 0.20 |
| `PERENNIAL` | ≥ 10 green weeks and NDVI 25th percentile ≥ 0.25 |
| `ANNUAL` | otherwise |

### Features

- Rolling 2-, 3-, 4- and 6-week windows of NDMI, NDMI change, rainfall and temperature
- Counts of dry, hot and drying weeks over recent windows and year to date
- Year-to-date NDMI drop from its peak, and cumulative rainfall
- The cell's median NDMI, NDVI, rainfall and temperature for the same calendar month in earlier
  years, and the current value's difference from it. These are empty for the first season and
  are filled by median imputation.
- Month and day of year

### Labels

There is no field record of when a cell needed water, so the training target is computed by a rule.
The need score is a weighted sum of three terms, each scaled to 0–1:

| Term | Scaled between | Weight (annual) | Weight (perennial) |
|---|---|---|---|
| NDMI dryness | 0.18 → 0.06 (annual), 0.20 → 0.05 (perennial) | 0.45 | 0.55 |
| NDMI decline rate (−dNDMI) | 0 → 0.04 per week | 0.35 | 0.30 |
| 7-day temperature | 18 → 32 °C | 0.20 | 0.15 |

The score is set to 0 when 7-day rainfall exceeds 15 mm, when there is no vegetation, or when the
cell is `NON_AGRI`. Classes: **NOW** ≥ 0.66, **SOON** ≥ 0.33, otherwise **WAIT**. `WetScore` and
Sentinel-1 radar (`dVV`) are left out of both the target and the features.

### Model

- scikit-learn `RandomForestClassifier`: 300 trees, `min_samples_leaf=2`,
  `class_weight="balanced_subsample"`
- Median imputation for numeric features, one-hot encoding for crop type
- Trained on 2021–2025, on rows with `veg_flag = 1`
- Stratified 80/20 random split (`random_state=42`)
- In the app, each cell gets the predicted class and a priority score of 0.5·P(SOON) + P(NOW)

## Application

- **Historical view:** pick a season and week from the weekly exports.
- **Live view:** the latest file in `data/live_exports/`, with features computed against the
  historical exports.
- **Map:** Folium, on Esri satellite imagery, with the WUA boundary. Cells are coloured by class.
  Cells without a valid observation are hidden unless "Show low-confidence cells" is on.
- **Tabs:** overall, annual crops and perennial crops.
- **Comparisons:** week-on-week changes, spatial clusters of urgent cells, and the same time of
  year in previous seasons (±14 days).
- **Optional analyst panel:** OpenAI-based briefing and per-cell explanations. Needs
  `OPENAI_API_KEY`. The rest of the app works without it.

## Evaluation

Last training run: 1,435,014 cell-weeks loaded, 609,593 training rows after the `veg_flag = 1`
filter. Held-out accuracy is about 0.99 per year, for example 0.9916 on 26,726 rows for 2021
(WAIT 12,716 / SOON 10,701 / NOW 3,309).

This number measures how well the model reproduces the labelling rule. It does not measure
agronomic correctness:

- The labels are computed from inputs the model also sees, so near-perfect agreement is expected.
- The split is random by row, so neighbouring weeks of the same cell fall on both sides of it.
  Holding out whole seasons or whole areas would be a stricter test.
- No field record of irrigation outcomes exists to validate against.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `train_need_ml_2024_2025_stress_only_fixed.py` | Trains the model the app uses (2021–2025) |
| `train_need_ml.py` | Earlier single-season (2024) trainer. Writes to the same model path, so running it replaces the current model. |
| `core/io.py`, `core/grid.py`, `core/features.py` | Loading, grid construction, features |
| `core/model.py`, `core/labels.py`, `core/live.py`, `core/forecast.py` | Alternative pipeline, not used by the app or the current trainer (see [Status](#status)) |
| `scripts/smoke_check.py` | Loads the model and scores 20 rows of the latest live export |
| `tests/` | Unit tests for `core/io.py`, `core/features.py` and `core/live.py` |
| `water_debt_notification_platform/` | Standalone module for debt notices (see [Status](#status)) |

## Running

Python 3.12.

```bash
pip install -r requirements.txt
python train_need_ml_2024_2025_stress_only_fixed.py   # writes models/
streamlit run app.py                                  # or run.bat on Windows
pytest -q
python scripts/smoke_check.py
```

Expected data layout:

```text
data/
  weekly_exports/*.csv                  # one or more files per season, year in the file name
  live_exports/*.csv                    # latest week
  Echmiadzin_Cell_Coordinates.csv       # cell_id, lat, lon
boundary/Echmiadzin_wua_Boundary.shp    # optional
models/                                 # written by the training script
```

Optional environment variables (see `env.example`): `OPENAI_API_KEY` and `OPENAI_COPILOT_MODEL`.

## Data

The weekly and live exports, cell coordinates, boundary shapefile and trained model belong to the
Water User Association and are not published. `.gitignore` excludes `data/`, `boundary/`, `models/`
and the file types stored there.

## Status

- Prototype. Runs locally, with no deployment and no authentication.
- Built and calibrated for one Water User Association. The crop-type and label thresholds are
  specific to it, and use elsewhere is untested.
- `app.py` holds the UI and most of the analysis logic in one file.
- `core/model.py`, `core/labels.py`, `core/live.py` and `core/forecast.py` are an alternative
  design that is not connected to anything:
  - a binary XGBoost/RandomForest model with two probability thresholds
  - a label rule that uses `WetScore` and radar `dVV`
  - a 7-day Open-Meteo forecast adjustment
- `water_debt_notification_platform/` is not connected to the app. It maps debtor spreadsheets to
  a Haypost (Armenian Post) registered-letter export and simulates dispatch. The live Haypost API
  call is not implemented.

## Stack

Python 3.12 · Streamlit · pandas · NumPy · scikit-learn · GeoPandas · Shapely · pyproj · Folium ·
joblib · pytest · OpenAI (optional)
