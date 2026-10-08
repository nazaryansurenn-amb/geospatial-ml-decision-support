# Geospatial ML Decision Support

[![CI](https://github.com/nazaryansurenn-amb/geospatial-ml-decision-support/actions/workflows/ci.yml/badge.svg)](https://github.com/nazaryansurenn-amb/geospatial-ml-decision-support/actions/workflows/ci.yml)

Weekly irrigation priorities for every 250 m field cell, from Sentinel-2 satellite indices and
weather, with a machine-learning forecast of next week:

- **This week's priority** (WAIT / SOON / NOW) comes from transparent rules on canopy moisture
  (NDMI): its 2-week level and trend, the cell's own normal for the month, and recent rain.
- **Next week's priority** is forecast by a gradient-boosting model trained on 405,787 cell-weeks.
- The forecast is tested out of time, on seasons it never saw, against two baselines that need no
  training: "no change" (persistence) and a hand-written trend rule.

**Headline result.** In a strict test on 2025, where the rule thresholds, crop map and model were
all built from 2022–2024 only, the model forecast next week's class correctly for **85.0%** of
65,198 summer cell-weeks, against **77.8%** for "no change" (**32.5% fewer errors**) and **81.5%**
for the hand-written rule (**19.2% fewer errors**).

Case study: the Echmiadzin Water User Association, Armavir region, Armenia (9,762 cells, about
4,350 vegetated in a summer week). Code only: the Association's data, boundary and trained model
are not published (see [Data](#data)).

**Contents:** [Results at a glance](#results-at-a-glance) · [How it works](#how-it-works) ·
[Evaluation design](#evaluation-design) · [Detailed results](#detailed-results) ·
[Why not the previous random forest](#why-not-the-previous-random-forest) ·
[Limitations](#limitations) · [Application](#application) · [Repository layout](#repository-layout) ·
[Running](#running) · [Data](#data)

## Results at a glance

Share of next-week classes forecast correctly. Each test season was never seen in training.

| Test | Trained on | Months | ML forecast | Trend rule | No change | ML errors avoided vs no change | vs trend rule |
|---|---|---|---|---|---|---|---|
| **Strict 2025** | 2022–2024 | Jun–Aug | **85.0%** | 81.5% | 77.8% | **32.5%** | **19.2%** |
| Strict 2025 | 2022–2024 | Apr–Sep | 83.0% | 81.4% | 78.6% | 20.3% | 8.5% |
| Strict 2024 | 2022–2023 | Jun–Aug | 80.9% | 79.6% | 72.6% | 30.4% | 6.3% |
| Strict 2024 | 2022–2023 | Apr–Sep | 77.6% | 77.3% | 73.8% | 14.7% | 1.3% |
| Live 2026, app model | 2022–2025 | Jul–Aug | 84.5% | 83.9% | 78.8% | 26.8% | 3.8% |

"Errors avoided" = 1 − (model errors ÷ baseline errors). "Strict" means nothing from the test year
was used anywhere before forecasting (details below). The 2026 row uses the shipped model and the
app's thresholds, which were calibrated on 2022–2025.

What this shows:

- The model beats "no change" in every test, with 15–33% fewer errors.
- Against a well-designed rule the margin is smaller: 4–19% fewer errors in summer, and roughly a
  tie over a full season when the model has only two training years (strict 2024). Spring, and
  especially the start of dry spells in May, is where the rule is better.
- Margins are larger in summer than over a full season, and larger with three training years (2025)
  than with two (2024), although the two test years also differ in weather (2024 was a drought year).

## How it works

### Pipeline

| Step | What it does | Code |
|---|---|---|
| Exports | Earth Engine writes one row per 250 m cell per report week (April–September): `NDMI_med`, `dNDMI`, `NDVI_med`, `Rain_mm_7d`, `Temp_C_7d`, flags | not in this repo |
| Load | Normalises column names, turns `-9999` into missing values, validates 7-day weeks, removes duplicate cell-weeks where monthly batches overlap | `core/io.py` |
| Runtime files | Compact Parquet files per season, monthly baselines and a multi-year crop map | `core/runtime_artifacts.py`, `scripts/build_runtime_artifacts.py` |
| Features | Rolling 2–6 week averages and sums, drying/dry/hot week counts, drop from the season's peak, and the cell's normal for the month from earlier years only | `core/features.py` |
| Crop type | Per season from NDVI: `NON_AGRI`, `PERENNIAL` or `ANNUAL`, then the most common class over the years | `core/crop.py` |
| This week's class | Explicit rules, below | `core/rules.py` |
| Next week's class | Gradient-boosting forecast on 51 inputs | `core/next_week.py`, `scripts/train_next_week.py` |
| Baselines | No change, and the hand-written trend rule | `core/baselines.py` |
| App | Streamlit + Folium maps, next-week forecast page, optional AI briefings | `app.py` and the view modules |

### This week's class: the rules

Only vegetated cells (NDVI ≥ 0.30) get a class. NDMI values are 2-week averages, because a single
week is noisy.

| Condition | Test |
|---|---|
| Very dry | 2-week NDMI ≤ 0.11 |
| Dry | 2-week NDMI ≤ 0.15 |
| Drying | 2-week average weekly change ≤ −0.015 |
| Below normal | ≥ 0.05 below the cell's median NDMI for that month in earlier years |
| Rain relief | ≥ 15 mm in 7 days or ≥ 25 mm in 14 days |

- **NOW** = very dry and (drying or below normal), no rain relief, and the cell was already SOON or
  NOW in the previous report (2-week persistence: one noisy week cannot raise an alarm).
- **SOON** = dry without rain relief, or very dry despite rain, or drying while below normal.
- **WAIT** = everything else.

The four NDMI thresholds are chosen by `core.rules.calibrate_rule_thresholds`: a grid search over
396 settings that keeps those giving about 15% NOW and 35–42% SOON in a typical June–August week,
and picks the one whose classes are most stable from week to week. Run on 2022–2025 it returns
exactly the thresholds above (June–August: 14.0% NOW, 38.5% SOON; 75.1% of cells keep their class
from one week to the next).

"Previous report" and "next report" are matched by date (5–10 days), not by row order: the exports
come in monthly batches that restart on the 1st, so the last week of a month overlaps the first week
of the next.

### Next week's class: the forecast model

| | |
|---|---|
| Target | The cell's rule class in the report that starts 5–10 days later |
| Model | scikit-learn `HistGradientBoostingClassifier`: up to 500 rounds (one tree per class per round), learning rate 0.08, up to 63 leaves per tree, L2 regularisation 1.0, early stopping on 10% of the training data. Settings fixed in advance, never tuned on a test year |
| Training data | 2022–2025: 405,787 vegetated cell-weeks with a known next week (2021 has no earlier years for the monthly normal, so it serves only as history) |
| Output | Probabilities for WAIT, SOON and NOW; the forecast is the most likely class |
| Size | 5.4 MB (the previous random forest was 155 MB) |

The 51 inputs, all known at the end of the current report week:

| Group | Inputs | Examples |
|---|---|---|
| Taken directly from the exports | 7 | NDMI, weekly NDMI change, NDVI, 7-day rain, 7-day temperature, month, day of year |
| NDMI history | 9 | 2-, 3- and 4-week mean NDMI and change; drying weeks; drop from the season's peak |
| Rain history | 7 | 2-, 3-, 4- and 6-week rain; dry weeks; rain since the start of the year |
| Temperature history | 5 | 2- and 4-week mean; weeks at ≥ 28 °C |
| Cell's own normal | 8 | The cell's median NDMI, NDVI, rain and temperature for the month in earlier years, and today's difference from them |
| This week's rule state | 7 | Class, class before the persistence check, and the five rule conditions |
| Last week | 2 | Last report's class and NDMI |
| Crop type | 1 | Annual, perennial or non-agricultural |
| Neighbourhood | 5 | Share of the 8 neighbouring cells in NOW and SOON, their NDMI level and trend |

## Evaluation design

- **Out of time, not random rows.** Neighbouring weeks of one cell are nearly identical, so a
  random split lets a model recall the answer from the row next to it. Every test season here was
  absent from training.
- **Strict test** (`scripts/evaluate_strict.py`). For test year *Y*, everything is rebuilt from the
  years before *Y*: the rule thresholds (calibrated on those years only), the crop map and the
  model. Each forecast uses only data available at the end of its report week, and is compared with
  what happened next week, cell by cell (not just in total).
- **Two baselines that need no training**, scored on exactly the same rows:
  - *No change* (persistence): next week's class = this week's class.
  - *Trend rule*: extend each cell's 2-week NDMI trend one week, assume this week's rain repeats,
    and apply the same class rules (`core/baselines.py`). It shares the rules' logic and thresholds,
    so anything the model beats it by was learned from the data.
- **Metrics.** Accuracy and errors avoided; per-class precision, recall and F1; balanced accuracy
  and macro-F1 (so the rare NOW class counts as much as WAIT); Cohen's kappa; severe errors
  (NOW ↔ WAIT); a comparison at an equal number of NOW alarms; agreement when methods disagree;
  calibration and Brier skill score of the NOW probability; permutation importance by input group.

## Detailed results

All numbers in this section come from `python scripts/evaluate_strict.py --importance` (strict
2025, trained on 2022–2024) unless the heading says otherwise.

### Overall

| 2025 | Method | Correct | Wrong | Balanced accuracy | Macro-F1 | Cohen's kappa | Severe errors |
|---|---|---|---|---|---|---|---|
| Summer (Jun–Aug), 65,198 forecasts | **ML forecast** | **85.0%** | **9,767** | **82.4%** | **0.824** | **0.750** | **0.26%** |
| | Trend rule | 81.5% | 12,083 | 80.4% | 0.785 | 0.696 | 1.07% |
| | No change | 77.8% | 14,473 | 73.3% | 0.736 | 0.630 | 0.85% |
| Season (Apr–Sep), 124,000 forecasts | **ML forecast** | **83.0%** | **21,127** | 78.1% | **0.790** | **0.704** | **0.47%** |
| | Trend rule | 81.4% | 23,092 | **79.4%** | 0.777 | 0.683 | 0.79% |
| | No change | 78.6% | 26,523 | 72.1% | 0.732 | 0.630 | 0.59% |

Over the full season the trend rule has the higher balanced accuracy: it catches more NOW cases,
at the price of more false alarms (next tables). A kappa of 0.75 is "substantial agreement" on the
Landis–Koch scale; chance agreement given the class frequencies is 40%.

Weeks in which the ML forecast had more correct cells: 14 of 15 summer weeks against the trend
rule and 15 of 15 against no change; over the season, 21 of 28 and 24 of 28.

### By class (summer 2025)

| Class | Actual cases | ML precision | ML recall | ML F1 | Trend rule P / R / F1 | No change P / R / F1 |
|---|---|---|---|---|---|---|
| WAIT | 33,094 | 90.4% | 93.0% | 0.916 | 90.0% / 89.6% / 0.898 | 87.6% / 87.5% / 0.876 |
| SOON | 22,524 | 80.0% | 76.5% | 0.782 | 77.9% / 69.0% / 0.732 | 68.7% / 70.2% / 0.694 |
| NOW | 9,580 | 77.3% | 77.7% | 0.775 | 64.4% / 82.6% / 0.723 | 65.3% / 62.2% / 0.637 |

Precision: of the times a method said this class, how often it was right. Recall: of the cases that
really were this class, how many it forecast. Over the full season the ML's NOW precision and recall
are 74.5% and 68.4% (trend rule 64.2% and 79.1%).

### Confusion matrix (summer 2025)

Rows: what happened next week. Columns: what was forecast.

| ML forecast | WAIT | SOON | NOW |
|---|---|---|---|
| **WAIT** | **30,765** | 2,214 | 115 |
| **SOON** | 3,223 | **17,227** | 2,074 |
| **NOW** | 57 | 2,084 | **7,439** |

| Trend rule | WAIT | SOON | NOW |
|---|---|---|---|
| **WAIT** | **29,649** | 2,823 | 622 |
| **SOON** | 3,211 | **15,552** | 3,761 |
| **NOW** | 73 | 1,593 | **7,914** |

98% of the model's mistakes are one step apart (SOON vs WAIT or SOON vs NOW); only 172 forecasts
(0.26%) confuse a calm field with an urgent one. The trend rule catches 475 more real NOW cells
(7,914 vs 7,439) because it raises 2,669 more NOW alarms (12,297 vs 9,628, against 9,580 real NOW
cases), and its false NOW alarms on calm fields are five times the model's (622 vs 115).

### Same number of NOW alarms

Ranking cells by the model's NOW probability and taking as many as the trend rule flags:

| Test | NOW alarms | Trend rule correct | ML correct |
|---|---|---|---|
| Summer 2025 | 12,297 | 7,914 | **8,311 (+5.0%)** |
| Season 2025 | 18,536 | 11,902 | **11,997 (+0.8%)** |
| Season 2024 (strict, 2 training years) | 25,500 | 18,242 | **18,957 (+3.9%)** |

The model's lower NOW recall is a choice of threshold, not a lack of signal: at the rule's alarm
budget it finds at least as many real NOW cells.

### When the methods disagree

| 2025 | Disagreeing forecasts | ML right | The other right |
|---|---|---|---|
| Summer, ML vs no change | 9,968 | **72.0%** | 24.8% |
| Summer, ML vs trend rule | 6,927 | **64.2%** | 30.7% |
| Season, ML vs no change | 19,470 | **61.8%** | 34.1% |
| Season, ML vs trend rule | 15,715 | **53.8%** | 41.3% |

When the model forecasts a change that "no change" does not, it is right about three times as
often in summer.

### By month (season 2025)

| Month | ML | Trend rule | No change | ML errors avoided vs rule | vs no change |
|---|---|---|---|---|---|
| April | 81.0% | 78.1% | 81.6% | +13.2% | −3.4% |
| May | 75.6% | 79.5% | 74.5% | **−19.1%** | +4.2% |
| June | 83.0% | 79.7% | 75.7% | +16.3% | +30.0% |
| July | 84.4% | 80.2% | 76.5% | +21.4% | +33.6% |
| August | 87.8% | 84.7% | 81.4% | +20.4% | +34.7% |
| September | 88.3% | 86.7% | 85.4% | +12.0% | +19.4% |

The model earns its place in June–August, the main irrigation season. In April it is level with
"no change"; in May, when dry spells start suddenly, the trend rule is better.

### Can the NOW probability be trusted?

| Model said (average) | 1% | 18% | 40% | 60% | 81% | 95% |
|---|---|---|---|---|---|---|
| Became NOW, season 2025 | 2% | 21% | 40% | 55% | 74% | 91% |
| Became NOW, season 2024 (strict) | 4% | 33% | 55% | 73% | 87% | 96% |

Brier skill score of the NOW probability against always forecasting the average NOW rate: 0.54
(season 2025), 0.61 (summer 2025), 0.56 (season 2024). The ranking is reliable, but calibration
drifts with the season: slightly over-confident in 2025, under-confident in the 2024 drought year.

### What the model relies on

Accuracy drop on season 2025 when one group of inputs is randomly shuffled:

| Input group | Inputs | Accuracy drop |
|---|---|---|
| NDMI level and multi-week trend | 11 | −36.0 points |
| Cell's own monthly normal | 8 | −5.1 |
| This week's rule state | 7 | −2.9 |
| Vegetation, crop type, calendar | 4 | −1.9 |
| Weather so far | 14 | −1.2 |
| Last week | 2 | −0.2 |
| Neighbouring cells | 5 | −0.1 |

The model has learned how each cell's moisture signal moves relative to its own normal. Weather
history barely helps because it is nearly the same for every cell in a given week, and the
neighbourhood inputs add nothing; both are candidates for removal.

### Second strict year: 2024

`python scripts/evaluate_strict.py --test-year 2024 --train-years 2022 2023`. 2024 was a drought
year, and the model has only two training seasons (170,667 cell-weeks).

| 2024 | ML | Trend rule | No change | Kappa (ML / rule / no change) | Weeks ML beats rule |
|---|---|---|---|---|---|
| Summer (Jun–Aug) | 80.9% | 79.6% | 72.6% | 0.706 / 0.691 / 0.580 | 9 of 13 |
| Season (Apr–Sep) | 77.6% | 77.3% | 73.8% | 0.643 / 0.647 / 0.586 | 14 of 26 |

With two training years the model still clearly beats "no change" but only ties the trend rule over
the season, and in May it is worse than both (73.9% against 80.0% and 79.6%).

### The shipped model, with the app's thresholds

`python scripts/train_next_week.py` scores each test year with a model trained on earlier years
only, then trains the shipped model on 2022–2025 and scores the 2026 live season (July–August).
Here the rule thresholds are the app's (calibrated on 2022–2025), so this is out-of-time but not
strict.

| Test | ML | Trend rule | No change | NOW precision / recall (ML) | Macro-F1 (ML / rule) |
|---|---|---|---|---|---|
| 2024 (trained 2022–23) | 79.5% | 79.3% | 75.3% | 86.1% / 59.1% | 0.766 / 0.768 |
| 2025 (trained 2022–24) | 83.9% | 82.1% | 79.0% | 72.6% / 68.5% | 0.799 / 0.782 |
| 2026 live (trained 2022–25) | 84.5% | 83.9% | 78.8% | 87.1% / 54.9% | 0.803 / 0.816 |

In 2024 and 2026 the trend rule's macro-F1 is equal or higher, because it catches more NOW cells.
The model's NOW forecasts are more precise; its threshold could be lowered where a missed NOW costs
more than a false alarm.

### Exploratory: adding a weather forecast

`python scripts/backtest_weather_forecast.py` replays 2025 week by week. Model B also gets next
week's rain, ET0, maximum temperature and rain minus ET0. It is trained on what the weather actually
did (Open-Meteo ERA5 archive) and, in 2025, given the forecast that was issued at the end of each
report week (Open-Meteo Previous Runs archive, 1–7 days ahead). App thresholds; trained on 2022–2024.

| 2025 | Season correct | Skill vs no change: Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|
| A: no weather forecast | 83.9% | 0.07 | 0.07 | 0.33 | 0.36 | 0.35 | 0.18 |
| B: + the forecast issued that week | 84.4% | 0.14 | 0.13 | 0.31 | 0.37 | 0.37 | 0.21 |
| B: + the actual weather (upper limit) | 84.7% | 0.21 | 0.20 | 0.31 | 0.37 | 0.35 | 0.13 |

A real forecast captures most of what perfect weather knowledge could add (+0.5 of +0.8 points),
almost all of it in spring, when rain falls and matters; summers are nearly rainless. The archived
ET0 forecast was excellent (weekly correlation 0.97 with what happened); the rain forecast was
moderate (0.69, and on average 4.9 mm against 6.9 mm actual per week). This is one test year, so
the weather inputs are not in the shipped model.

## Why not the previous random forest

The previous version of this project trained a 300-tree random forest that reported about 99%
held-out accuracy. That number measured how well it reproduced its own labels: the target was this
week's class, computed by a rule from the same NDMI and weather values the model received, and the
split was random by row. Before it was removed, a single decision tree 12 levels deep reproduced
its predictions on 99.1% of 2025 rows (4 levels: 91%), so the forest had learned the labelling
rule, which the rule computes directly anyway.

This version keeps a rule for what rules are good at, a transparent class for this week, and uses
machine learning for something nobody knows when the forecast is made: next week's class. The
model is judged only against baselines, on seasons it never saw.

## Limitations

- **The target is a proxy.** The model forecasts the rule-based class, a satellite signal of canopy
  water stress, not measured irrigation demand. No field record has been used to validate it.
- **Modest margin over a good rule.** The gain over the trend rule is 4–19% fewer errors in summer
  and 1–10% over a full season, and the rule is better in May. All results come from one Water User
  Association and three test seasons.
- **Calibration drifts by season**, and the NOW decision threshold has not been tuned.
- **Correlated forecasts.** Cells in the same week share weather, and neighbouring cells are
  similar, so 65,198 forecasts are far fewer independent observations. Week-level and season-level
  comparisons are reported for that reason.
- **Same fields, later years.** Transfer to another region has not been tested.
- **Strict tests recalibrate the thresholds** from their own training years, so the class
  definition differs slightly between the strict tests and the app.

## Application

Streamlit, Folium maps on satellite imagery, in English, Armenian and Russian.

| View | What it shows |
|---|---|
| Irrigation priority map | This week's classes for any season and week (2021–2025 history, 2026 live) as a continuous priority surface |
| Next-week forecast | Forecast counts with the change from this week, a probability map, the class-change table, the cells most likely to become NOW, and the model's validation |
| Field briefing & AI | Operational summary, priority zones, the 7-day Open-Meteo weather outlook (rain, ET0), and optional AI briefings |
| Season history | Week-by-week animation of a season |
| Weekly changes | Where priority improved or worsened from one week to the next |

The optional AI briefings call an OpenAI model with structured context only (class counts, zones,
weather, the next-week forecast) and curated FAO / Earth-observation notes from
`data/knowledge/eo_irrigation_knowledge.json`. They explain the classes but never change them.
`PUBLIC_APP_MODE=1` hides expert controls, raw tables and model details.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point, data loading, rule classes, AI context |
| `next_week_forecast_page.py` | Next-week forecast view |
| `pressure_surface_*.py`, `pressure_change_timelapse_demo.py` | Priority map, season history and weekly change views |
| `core/rules.py` | Class rules, date-based week matching, threshold calibration |
| `core/next_week.py` | Forecast inputs, training pairs, model settings, prediction |
| `core/baselines.py` | No-change and trend-rule baselines |
| `core/features.py`, `core/crop.py`, `core/io.py`, `core/grid.py`, `core/runtime_artifacts.py` | Features, crop type, loading, grid, compact runtime files |
| `core/ai.py`, `core/knowledge.py`, `core/i18n.py`, `core/forecast.py` | AI briefings, knowledge retrieval, translations, 7-day weather outlook |
| `core/live.py` | Earlier forecast-adjustment helper, not used by the app |
| `scripts/build_runtime_artifacts.py` | Weekly CSV exports → `data/runtime/` |
| `scripts/train_next_week.py` | Trains the shipped model; scores 2024, 2025 and the live season against both baselines |
| `scripts/evaluate_strict.py` | Strict out-of-time evaluation; the source of [Detailed results](#detailed-results) |
| `scripts/backtest_weather_forecast.py` | Exploratory weather-forecast backtest |
| `scripts/smoke_check.py` | End-to-end check on real data |
| `tests/` | Rules, baselines, calibration, features, loading |
| `data/knowledge/` | Curated FAO / EO interpretation notes (the only data in this repo) |
| `water_debt_notification_platform/` | Separate module for debt notices, not connected to the app |

## Running

Python 3.12. On Windows, `run.bat` creates `.venv`, installs `requirements.txt` on first use and
opens the app at http://localhost:8502. Otherwise:

```bash
pip install -r requirements.txt
python scripts/build_runtime_artifacts.py      # data/weekly_exports/*.csv -> data/runtime/
python scripts/train_next_week.py              # writes models/ and the validation above
python scripts/evaluate_strict.py --importance # strict 2025 test -> reports/strict_2025.json
python scripts/smoke_check.py                  # rules + one forecast on the latest live week
streamlit run app.py
pytest -q
```

Expected local data layout (not in this repo):

```text
data/
  weekly_exports/*.csv                    # Earth Engine exports, one row per cell per report week
  live_exports/*.csv                      # current season
  runtime/                                # written by scripts/build_runtime_artifacts.py
  Echmiadzin_Cell_Coordinates.csv         # cell_id, lat, lon
  knowledge/eo_irrigation_knowledge.json  # in this repo
boundary/Echmiadzin_wua_Boundary.shp
models/next_week_forecast.joblib          # written by scripts/train_next_week.py
```

Optional environment variables (see `env.example`): `OPENAI_API_KEY`, `OPENAI_COPILOT_MODEL`,
`AI_CACHE_TTL_SECONDS`, `AI_WEEKLY_CACHE_TTL_SECONDS`, `PUBLIC_APP_MODE`, `PUBLIC_LANGUAGES`.

## Data

The weekly and live exports, cell coordinates, boundary shapefile, runtime files, trained models
and evaluation outputs belong to the Water User Association or are computed from its data, and are
not published. `.gitignore` excludes `data/` (except `data/knowledge/`), `boundary/`, `models/`,
`reports/` and the file types stored there.

## Stack

Python 3.12 · scikit-learn · pandas · NumPy · GeoPandas · Shapely · pyproj · rasterio · SciPy ·
Streamlit · Folium · Open-Meteo · OpenAI (optional) · pytest · ruff · GitHub Actions
