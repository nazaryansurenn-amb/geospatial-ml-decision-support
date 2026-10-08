"""Strict out-of-time evaluation of the next-week forecast.

Everything the forecast depends on is rebuilt from the years before the test year only:
the rule thresholds (core.rules.calibrate_rule_thresholds), the crop map and the model.
The test year is then replayed week by week, each forecast made from data available at the
end of its report week, and compared with what actually happened next week. Two baselines
that need no training are scored on the same rows:

  - no-change (persistence): next week's class = this week's class
  - rule-based trend forecast: core.baselines.trend_rule_forecast

Usage:
    python scripts/evaluate_strict.py                    # test 2025, train 2022-2024
    python scripts/evaluate_strict.py --test-year 2024 --train-years 2022 2023
    python scripts/evaluate_strict.py --importance        # add permutation importance by input group

Prints the tables and writes reports/strict_<test year>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

import core.rules as rules
from core.baselines import persistence_forecast, trend_rule_forecast
from core.crop import build_stable_crop_map
from core.io import normalize_cell_id
from core.next_week import FORECAST_FEATURES, MODEL_PARAMS, label_next_week_pairs
from core.runtime_artifacts import DEFAULT_CROP_THRESHOLDS, artifact_paths, historical_features_path

LABELS = ["WAIT", "SOON", "NOW"]
METHODS = {"ml": "ML forecast", "trend_rule": "Rule-based trend forecast", "no_change": "No-change baseline"}
SUMMER = (6, 7, 8)
CALIBRATION_BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
INPUT_GROUPS = {
    "NDMI level and multi-week trend": ["NDMI_med", "dNDMI", "ndmi_mean_2w", "ndmi_mean_3w", "ndmi_mean_4w",
                                        "dndmi_mean_2w", "dndmi_mean_4w", "ndmi_drop_ytd", "drying3w", "drying6w",
                                        "drying_weeks_ytd"],
    "Cell's own monthly normal": [c for c in FORECAST_FEATURES if "cell_month_hist" in c],
    "This week's rule state": ["cur_code", "raw_code", "c_very_dry", "c_dry", "c_drying", "c_below_normal", "c_rain_relief"],
    "Last week": ["prev_code", "ndmi_prev"],
    "Vegetation, crop, calendar": ["NDVI_med", "crop_code", "month", "doy"],
    "Neighbouring cells": [c for c in FORECAST_FEATURES if c.startswith("nb_")],
    "Weather so far": ["Rain_mm_7d", "Rain2w", "Rain3w", "Rain4w", "Rain6w", "dry_weeks_3w", "dry_weeks_6w",
                       "rain_cum_ytd", "Temp_C_7d", "temp_mean_2w", "temp_mean_4w", "hot2w", "hot4w", "hot_weeks_ytd"],
}
STARTED = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - STARTED:5.0f}s] {message}", flush=True)


def load_year(runtime_dir: Path, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(historical_features_path(runtime_dir, year))
    frame["cell_id"] = frame["cell_id"].map(normalize_cell_id)
    frame["year"] = year
    return frame


def errors_avoided(model_wrong: int, other_wrong: int) -> float:
    return 1 - model_wrong / other_wrong if other_wrong else 0.0


def score(d: pd.DataFrame) -> dict:
    y = d["actual"].to_numpy()
    out = {"forecasts": int(len(d)), "weeks": int(d["week_start"].nunique()), "methods": {}}
    for key in METHODS:
        pred = d[key].to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
        cm = pd.crosstab(pd.Series(y, name="actual"), pd.Series(pred, name="forecast")).reindex(index=LABELS, columns=LABELS, fill_value=0)
        out["methods"][key] = {
            "accuracy": accuracy_score(y, pred),
            "wrong": int((pred != y).sum()),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=LABELS, average="macro"),
            "weighted_f1": f1_score(y, pred, labels=LABELS, average="weighted"),
            "cohen_kappa": cohen_kappa_score(y, pred),
            "severe_errors": int(cm.loc["NOW", "WAIT"] + cm.loc["WAIT", "NOW"]),
            "per_class": {c: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "actual": int(support[i]),
                              "forecast": int((pred == c).sum())} for i, c in enumerate(LABELS)},
            "confusion_matrix": cm.to_numpy().tolist(),
        }
    m = out["methods"]
    out["ml_fewer_errors_vs_no_change"] = errors_avoided(m["ml"]["wrong"], m["no_change"]["wrong"])
    out["ml_fewer_errors_vs_trend_rule"] = errors_avoided(m["ml"]["wrong"], m["trend_rule"]["wrong"])

    weekly = d.groupby("week_start").apply(lambda g: pd.Series({k: (g[k] == g["actual"]).sum() for k in METHODS}))
    out["weeks_ml_beats_trend_rule"] = int((weekly["ml"] > weekly["trend_rule"]).sum())
    out["weeks_ml_beats_no_change"] = int((weekly["ml"] > weekly["no_change"]).sum())

    # Same number of NOW alarms as the rule: the rule's NOW cells vs the ML's most likely ones.
    budget = int((d["trend_rule"] == "NOW").sum())
    is_now = (d["actual"] == "NOW").to_numpy()
    out["equal_alarms"] = {
        "alarms": budget,
        "trend_rule_correct": int((is_now & (d["trend_rule"] == "NOW").to_numpy()).sum()),
        "ml_correct": int(is_now[np.argsort(-d["p_now"].to_numpy())[:budget]].sum()),
    }

    # When the ML disagrees with a baseline, who is right?
    out["disagreements"] = {}
    for other in ("no_change", "trend_rule"):
        dis = d["ml"] != d[other]
        out["disagreements"][other] = {
            "cases": int(dis.sum()),
            "ml_right": float((d.loc[dis, "ml"] == d.loc[dis, "actual"]).mean()) if dis.any() else 0.0,
            "other_right": float((d.loc[dis, other] == d.loc[dis, "actual"]).mean()) if dis.any() else 0.0,
        }

    out["by_month"] = {}
    for month, g in d.groupby(pd.to_datetime(d["week_start"]).dt.month):
        acc = {k: float((g[k] == g["actual"]).mean()) for k in METHODS}
        wrong = {k: int((g[k] != g["actual"]).sum()) for k in METHODS}
        out["by_month"][int(month)] = {"forecasts": int(len(g)), **{f"{k}_accuracy": v for k, v in acc.items()},
                                       "ml_fewer_errors_vs_no_change": errors_avoided(wrong["ml"], wrong["no_change"]),
                                       "ml_fewer_errors_vs_trend_rule": errors_avoided(wrong["ml"], wrong["trend_rule"])}

    base_rate = float(is_now.mean())
    out["now_calibration"] = {
        "bins": [],
        "brier_ml": brier_score_loss(is_now, d["p_now"]),
        "brier_base_rate": base_rate * (1 - base_rate),
    }
    out["now_calibration"]["brier_skill"] = 1 - out["now_calibration"]["brier_ml"] / out["now_calibration"]["brier_base_rate"]
    for lo, hi in CALIBRATION_BINS:
        sel = (d["p_now"] >= lo) & (d["p_now"] < hi)
        if sel.any():
            out["now_calibration"]["bins"].append({"from": lo, "to": min(hi, 1.0), "forecasts": int(sel.sum()),
                                                   "mean_probability": float(d.loc[sel, "p_now"].mean()),
                                                   "became_now": float(is_now[sel.to_numpy()].mean())})
    return out


def permutation_importance(model, test: pd.DataFrame, repeats: int = 3) -> dict:
    X = test[FORECAST_FEATURES]
    y = test["target_next_class"].to_numpy()
    classes = np.array(model.classes_)
    base = accuracy_score(y, classes[model.predict_proba(X).argmax(1)])
    rng = np.random.default_rng(0)
    out = {}
    for name, columns in INPUT_GROUPS.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[columns] = shuffled[columns].to_numpy()[rng.permutation(len(shuffled))]
            drops.append(base - accuracy_score(y, classes[model.predict_proba(shuffled).argmax(1)]))
        out[name] = {"inputs": len(columns), "accuracy_drop": float(np.mean(drops))}
    return out


def print_report(name: str, r: dict) -> None:
    m = r["methods"]
    print(f"\n=== {name}: {r['forecasts']:,} forecasts, {r['weeks']} weeks ===")
    print(f"{'method':28s} {'accuracy':>9s} {'wrong':>8s} {'bal.acc':>8s} {'macroF1':>8s} {'kappa':>7s} {'severe':>7s}")
    for key, label in METHODS.items():
        s = m[key]
        print(f"{label:28s} {s['accuracy']:9.1%} {s['wrong']:8,} {s['balanced_accuracy']:8.1%} {s['macro_f1']:8.3f} "
              f"{s['cohen_kappa']:7.3f} {s['severe_errors'] / r['forecasts']:7.2%}")
    print(f"ML fewer errors: {r['ml_fewer_errors_vs_no_change']:.1%} vs no-change, {r['ml_fewer_errors_vs_trend_rule']:.1%} vs trend rule; "
          f"weeks won: {r['weeks_ml_beats_trend_rule']}/{r['weeks']} vs rule, {r['weeks_ml_beats_no_change']}/{r['weeks']} vs no-change")
    e = r["equal_alarms"]
    print(f"Equal NOW alarms ({e['alarms']:,}): trend rule correct {e['trend_rule_correct']:,}, ML correct {e['ml_correct']:,}")
    for key, label in METHODS.items():
        pc = m[key]["per_class"]
        print(f"  {label:28s} " + " | ".join(f"{c} P {pc[c]['precision']:.1%} R {pc[c]['recall']:.1%} F1 {pc[c]['f1']:.3f}" for c in LABELS))
    print("  ML confusion matrix (rows actual WAIT/SOON/NOW, columns forecast):", m["ml"]["confusion_matrix"])
    for other, dv in r["disagreements"].items():
        print(f"  ML vs {other}: disagree on {dv['cases']:,}; ML right {dv['ml_right']:.1%}, {other} right {dv['other_right']:.1%}")
    for month, v in r["by_month"].items():
        print(f"  month {month:2d}: ML {v['ml_accuracy']:.1%} | trend rule {v['trend_rule_accuracy']:.1%} | no-change {v['no_change_accuracy']:.1%} "
              f"| ML fewer errors vs rule {v['ml_fewer_errors_vs_trend_rule']:+.1%}, vs no-change {v['ml_fewer_errors_vs_no_change']:+.1%}")
    c = r["now_calibration"]
    print(f"  NOW Brier skill score {c['brier_skill']:.2f}; calibration: " +
          ", ".join(f"{b['mean_probability']:.0%}->{b['became_now']:.0%}" for b in c["bins"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--train-years", type=int, nargs="+", help="default: 2022 .. test year - 1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--importance", action="store_true", help="also compute permutation importance by input group")
    args = parser.parse_args()
    train_years = args.train_years or list(range(2022, args.test_year))
    runtime_dir = args.data_dir / "runtime"

    # 1. thresholds from the training years only
    frames = {year: load_year(runtime_dir, year) for year in [*train_years, args.test_year]}
    thresholds, _ = rules.calibrate_rule_thresholds(pd.concat([frames[y] for y in train_years], ignore_index=True))
    app_thresholds = {k: rules.DEFAULT_RULE_PARAMS[k] for k in thresholds}
    rules.DEFAULT_RULE_PARAMS.update(thresholds)   # labels, inputs and the trend rule below all use these
    log(f"thresholds from {train_years}: {thresholds} (app uses {app_thresholds})")

    # 2. crop map from the years before the test year
    reference = pd.read_parquet(artifact_paths(runtime_dir)["historical_reference"])
    reference = reference[pd.to_numeric(reference["year"], errors="coerce") < args.test_year]
    crop_map = build_stable_crop_map(reference, **DEFAULT_CROP_THRESHOLDS)
    crop_map["cell_id"] = crop_map["cell_id"].map(normalize_cell_id)

    # 3. training pairs, model, test predictions
    pairs = {y: label_next_week_pairs(frames[y], crop_map).assign(year=y) for y in frames}
    train = pd.concat([pairs[y] for y in train_years], ignore_index=True)
    test = pairs[args.test_year].copy()
    model = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(train[FORECAST_FEATURES], train["target_next_class"])
    log(f"model trained on {len(train):,} cell-weeks ({model.n_iter_} rounds); test {args.test_year}: {len(test):,} cell-weeks")

    proba = model.predict_proba(test[FORECAST_FEATURES])
    classes = list(model.classes_)
    test["ml"] = np.array(classes)[proba.argmax(1)]
    test["p_now"] = proba[:, classes.index("NOW")]
    test["trend_rule"] = trend_rule_forecast(test)
    test["no_change"] = persistence_forecast(test)
    test["actual"] = test["target_next_class"]

    summer = test[pd.to_datetime(test["week_start"]).dt.month.isin(SUMMER)]
    report = {
        "test_year": args.test_year,
        "train_years": train_years,
        "thresholds_from_train_years": thresholds,
        "train_cell_weeks": int(len(train)),
        "season": score(test),
        "summer_jun_aug": score(summer),
    }
    if args.importance:
        report["input_importance_season"] = permutation_importance(model, test)
        log("permutation importance done")

    print_report(f"{args.test_year} season", report["season"])
    print_report(f"{args.test_year} summer (Jun-Aug)", report["summer_jun_aug"])
    if args.importance:
        print("\nAccuracy drop when one input group is shuffled:")
        for name, v in sorted(report["input_importance_season"].items(), key=lambda kv: -kv[1]["accuracy_drop"]):
            print(f"  {name:34s} {v['inputs']:2d} inputs  -{100 * v['accuracy_drop']:.1f} points")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"strict_{args.test_year}.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log(f"saved {out}")


if __name__ == "__main__":
    main()
