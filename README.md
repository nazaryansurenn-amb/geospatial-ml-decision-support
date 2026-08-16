# Irrigation Need Prediction — Water User Association Decision Support

A Streamlit tool that predicts, week by week and cell by cell, where a Water User Association should
irrigate next — and puts the answer on a map so a field crew knows where to go first.

Built for the Echmiadzin Water User Association in Armenia's Armavir region, on five seasons of
operational and satellite data.

---

## The problem

A Water User Association distributes irrigation water across thousands of hectares with a small field
crew and no way to see the whole system at once. In practice, decisions about where to send water are
made from experience, phone calls, and whichever complaint arrived most recently.

The question the crew needs answered is not *"what is the soil moisture index of this polygon"*. It is
**"where do we go first this week?"** That is a prioritisation problem, not a measurement problem, and
the distinction drove every design decision below.

---

## How it works

### 1. The grid

The service area is divided into **250 m × 250 m cells**. Cell centroids arrive as lat/lon; the tool
projects them to UTM (`EPSG:32638`, the Echmiadzin zone), builds a square polygon around each centre
in metres, and projects back to WGS84 for mapping.

Projecting before building the squares matters: constructing a "square" in degrees produces cells that
are not square and whose area varies with latitude.

*`core/grid.py`*

### 2. Weekly observations

Each cell gets a weekly record combining earth-observation indices and weather:

| Signal | Meaning |
|---|---|
| `NDVI_med` | vegetation vigour — is there a crop here at all |
| `NDMI_med` | canopy/soil moisture |
| `dNDMI` | week-on-week moisture change |
| `dVV` | Sentinel-1 radar backscatter change — negative means drying |
| `WetScore` | composite wetness indicator |
| `Rain_mm_7d`, `Temp_C_7d`, `Wind_ms_7d` | trailing 7-day weather |
| `valid_s2`, `veg_flag`, `irr_event` | cloud validity, vegetation presence, known irrigation event |

The loader tolerates the column-naming drift that real exports accumulate — `CellID` / `cell_id`,
`weekStart` / `week_start` — because operational data is not tidy and a tool that breaks on a renamed
column will not be used twice.

*`core/io.py`*

### 3. Teacher labels

There is no ground-truth register of "this cell needed water on this date". So the training signal is
built from an explicit agronomic rule, and the rule is written down rather than buried:

A cell **needs irrigation** when the Sentinel-2 observation is valid, vegetation is present
(`veg_flag` or `NDVI ≥ 0.30`), trailing rainfall is under **8 mm/7d**, `WetScore` is under **0.65**,
`NDMI ≤ 0.25`, and radar shows drying (`dVV ≤ −0.02`). A recorded irrigation event forces the label
to zero.

Missing radar does not block a label — the rule falls back to optical plus weather rather than
discarding the cell. Every threshold is a named, tunable parameter.

*`core/labels.py`*

### 4. Features

On top of the raw weekly signals, the pipeline builds **within-season history per cell**: cumulative
NDMI peak to date, drop from that peak, 2- and 3-week rolling means, and monthly baselines for NDMI,
NDVI, rainfall and temperature.

The reasoning is that absolute moisture is less informative than *moisture relative to where this
particular cell was three weeks ago and where it usually is in this month*. A cell at NDMI 0.20 that
has been falling for three weeks is a different case from one at 0.20 that has been flat all season.

*`core/features.py`*

### 5. Model

**XGBoost** when available (400 trees, depth 6, learning rate 0.05, subsample and column-sample 0.9,
L2 = 1.0), falling back automatically to a **class-balanced RandomForest** (500 trees) when it is not.
Trained on **2021–2025**: 1,435,014 cell-weeks loaded, 609,593 rows after filtering to the growing
season and valid observations.

The binary need-probability is converted to a three-class action by two thresholds carried with the
model, `T_decision` and `T_now`:

```
p < T_decision   ->  WAIT
T_decision..T_now ->  IRRIGATE_SOON
p >= T_now       ->  IRRIGATE_NOW
```

Model, feature list and both thresholds travel together as a single `ModelBundle`, so inference cannot
silently drift from training.

*`core/model.py`, `train_need_ml.py`*

### 6. Forecast and live run

The live path picks the newest `live_*.csv` export, joins it to the cell grid, and calls
[Open-Meteo](https://open-meteo.com/) for a 7-day precipitation, temperature and wind outlook per
location — so a cell that is dry today but has 20 mm of rain coming is not sent a crew.

Output renders as a Folium map coloured by action.

*`core/live.py`, `core/forecast.py`*

### 7. Debt notification module

A separate workflow that turns overdue water accounts into registered-letter exports for **Haypost**
(Armenian Post), producing the recipient, community, debt amount and period, notice text and tracking
fields in the format the postal service expects.

Included because it is part of the same operational reality: a Water User Association that cannot
collect cannot maintain the network it is being asked to optimise.

*`water_debt_notification_platform/`*

---

## About the accuracy figures

Per-year test accuracy sits around **0.99** (2021: 0.9916 on 26,726 held-out rows, across WAIT 12,716
/ SOON 10,701 / NOW 3,309).

**That number should not be read as "99% correct about reality", and I would rather say so than let a
reader assume it.** The labels are teacher labels generated by the rule in `core/labels.py` from the
same signals the model sees. So the model is largely learning to reproduce a deterministic function,
and high agreement is the expected result, not evidence of agronomic truth.

What the metric *does* establish: the learned model reproduces the expert rule faithfully while
generalising across cells, seasons and missing data — which is what allows the rule to be replaced by
a probability, and the probability by a graded three-class priority instead of a hard yes/no.

Validating against actual field outcomes would require an irrigation log the Association does not
currently keep. That is the honest ceiling on this work, and closing it is a data-collection problem
rather than a modelling one.

---

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application — map, filters, analyst panel |
| `core/io.py` | Loading and normalising weekly and live exports |
| `core/grid.py` | 250 m grid construction and reprojection |
| `core/labels.py` | Agronomic teacher-label rule |
| `core/features.py` | Temporal and baseline feature engineering |
| `core/model.py` | Model bundle, training, thresholds, evaluation |
| `core/forecast.py` | Open-Meteo 7-day forecast client |
| `core/live.py` | Latest-export selection and action assignment |
| `train_need_ml.py` | Training entry point |
| `train_need_ml_2024_2025_stress_only_fixed.py` | Retraining variant restricted to water-stress seasons |
| `scripts/smoke_check.py` | Integrity check: model + data + one prediction pass |
| `tests/` | Unit tests for features, IO and live handling |
| `water_debt_notification_platform/` | Debt-notice workflow and Haypost export |

## Stack

Python 3.12 · Streamlit · pandas · NumPy · scikit-learn · XGBoost (optional) · GeoPandas · Shapely ·
pyproj · Folium · joblib · pytest · OpenAI (optional, analyst panel only)

## Running it

```bash
pip install -r requirements.txt

run.bat                        # launcher auto-detects an interpreter and starts Streamlit
python scripts/smoke_check.py  # integrity check
pytest -q                      # unit tests
```

Optional environment variables — see `env.example`:

- `OPENAI_API_KEY` — enables the analyst copilot panel. The tool is fully functional without it.
- `OPENAI_COPILOT_MODEL` — overrides the default copilot model.

---

## Data is deliberately not published

This repository is **code only**. The Association's weekly and live exports, the field boundary
shapefile, the trained model artifacts and the training snapshot are all excluded — they are a real
organisation's operating record, and publishing them is not mine to do. `.gitignore` enforces it.

To run against your own data you need weekly per-cell records with the columns listed in section 2 and
a cell-centroid CSV of `cell_id, lat, lon`.

## Honest status

- **Advanced prototype, unit tested.** Not a deployed product.
- Local Streamlit app: no server deployment, no authentication, no multi-tenancy.
- Trained and validated on a **single** Water User Association. Transfer to others is untested, and
  the thresholds in `core/labels.py` are calibrated to this service area.
- Accuracy is measured against teacher labels, not field outcomes — see above.
- `app.py` is a monolith and should be decomposed; `core/` is where the reusable logic already lives.
- The training/inference feature contract is carried by `ModelBundle` but not otherwise enforced.

## Why it is built this way

The interesting part of this project is not the gradient-boosted tree. Anyone can fit one.

The judgement is in what the model is asked to predict and how the answer is presented. A probability
surface is useless to a field crew; three actions on a map are not. The features are built from what a
Water User Association actually records week to week, not from what would be ideal to have. Missing
radar degrades the decision instead of dropping the cell. And the output is scoped to a decision
somebody will make on Monday morning.

That framing came from nine years working inside Armenia's irrigation system before writing any of
this code.
