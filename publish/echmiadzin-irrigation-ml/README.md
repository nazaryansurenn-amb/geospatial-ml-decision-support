# Irrigation Need Prediction — Water User Association Decision Support

A Streamlit tool that predicts, cell by cell, where a Water User Association should irrigate next —
and puts the answer on a map so a field crew knows where to go first.

Built for the Echmiadzin Water User Association in Armenia's Armavir region.

## The problem it solves

A Water User Association distributes irrigation water across thousands of hectares with a small
field crew and no way to see the whole system at once. Decisions about where to send water are made
from experience, phone calls and whichever complaint arrived most recently.

The question the crew actually needs answered is not "what is the soil moisture index of this
polygon". It is **"where do we go first this week?"** — which is a prioritisation problem, not a
measurement problem. That framing drove every design decision here.

## What it does

- Ingests historical weekly exports and the latest live export from the Association's own operational records.
- Builds temporal features per grid cell from operational data and earth-observation indices.
- Applies a trained scikit-learn model that classifies each cell as **WAIT**, **SOON** or **NOW**.
- Renders a prioritised map for field operations rather than a table of probabilities.
- Optionally runs an analyst copilot panel for plain-language summaries (requires an OpenAI key; the tool works fully without it).

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `core/` | Data IO, feature engineering, grid handling, labelling, forecasting, model interface |
| `train_need_ml.py` | Model training |
| `train_need_ml_2024_2025_stress_only_fixed.py` | Retraining variant restricted to water-stress seasons |
| `scripts/smoke_check.py` | Integrity check — model, data and one prediction pass |
| `tests/` | Unit tests for feature engineering, IO and live-data handling |
| `water_debt_notification_platform/` | Debt-notice workflow module, including Haypost (Armenian Post) integration |

## Running it

Python 3.12 recommended. Install from `requirements.txt`, then:

```
run.bat                        # launcher auto-detects a usable interpreter and starts Streamlit
python scripts/smoke_check.py  # integrity check
pytest -q                      # unit tests
```

Optional environment variables — see `env.example`:

- `OPENAI_API_KEY` — enables the copilot panel
- `OPENAI_COPILOT_MODEL` — overrides the default copilot model

## Data is deliberately not published

This repository contains **code only**. The Water User Association's operational exports, the field
boundary shapefile, the trained model artifacts and the training snapshot are all excluded, because
they are a real organisation's operating record and not mine to publish. The tool runs against
exports in the Association's own format.

`.gitignore` enforces this. If you clone it, you will need to supply your own data in the expected
shape.

## Honest status

- **Advanced prototype, unit tested** — not a deployed product.
- Runs as a local Streamlit app. There is no server deployment, no auth and no multi-tenancy.
- Trained and validated on a **single** Water User Association. Generalisation to others is untested.
- `app.py` is a monolith and should be decomposed; the `core/` package is where the reusable logic already lives.
- The model contract between training and inference is maintained by hand, not enforced.

## Why the design is the way it is

The interesting part of this project is not the RandomForest. Anyone can fit one.

The judgement is in what the model is asked to predict and how the answer is presented. A
probability surface is useless to a field crew; a three-class priority on a map is actionable. The
features are built from what a Water User Association actually records week to week, not from what
would be ideal to have. And the output is scoped to a decision someone will make on Monday morning.

That framing came from nine years working inside Armenia's irrigation system before writing any of
this code.
