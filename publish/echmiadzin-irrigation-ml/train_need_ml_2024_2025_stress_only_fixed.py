from __future__ import annotations
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from core.features import build_temporal_features
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EXPORTS_DIR = DATA_DIR / "weekly_exports"
COORDS_CSV = DATA_DIR / "Echmiadzin_Cell_Coordinates.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_YEARS = [2021, 2022, 2023, 2024, 2025]


def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def parse_year_from_name(path: Path):
    m = re.search(r"(20\d{2})", path.name)
    return int(m.group(1)) if m else None


def load_exports_multiyear(exports_dir: Path, years: list[int]) -> pd.DataFrame:
    files = sorted(exports_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {exports_dir}")

    dfs = []
    matched = []
    for f in files:
        year = parse_year_from_name(f)
        if year not in years:
            continue
        df = pd.read_csv(f)
        df["source_file"] = f.name
        if "year" not in df.columns:
            df["year"] = year
        dfs.append(df)
        matched.append(f.name)

    if not dfs:
        raise ValueError(f"No training CSV files found for years: {years}")

    out = pd.concat(dfs, ignore_index=True)
    out["cell_id"] = out["cell_id"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    print("Matched files:")
    for name in matched:
        print(" -", name)
    return out


def compute_crop_type_apr_sep(
    df_year: pd.DataFrame,
    ndvi_green_thr: float = 0.30,
    perennial_green_weeks: int = 10,
    perennial_p25: float = 0.25,
    nonagri_p90_max: float = 0.20,
    nonagri_green_weeks_max: int = 2,
) -> pd.DataFrame:
    df = df_year.copy()
    df["cell_id"] = df["cell_id"].astype(str)

    ndvi = _safe_num(df.get("NDVI_med"))
    valid = (
        _safe_num(df.get("valid_s2")).fillna(0).astype(int)
        if "valid_s2" in df.columns
        else pd.Series(1, index=df.index, dtype=int)
    )

    ok = (valid == 1) & ndvi.notna()
    df = df.loc[ok, ["cell_id", "NDVI_med"]].copy()

    if df.empty:
        out = df_year[["cell_id"]].drop_duplicates().copy()
        out["crop_type"] = "ANNUAL"
        return out

    def p25(x):
        return float(np.nanpercentile(x, 25)) if len(x) else np.nan

    def p90(x):
        return float(np.nanpercentile(x, 90)) if len(x) else np.nan

    g = df.groupby("cell_id")["NDVI_med"]
    stats = pd.DataFrame({
        "cell_id": g.size().index.astype(str),
        "NDVI_p25": g.apply(p25).values,
        "NDVI_p90": g.apply(p90).values,
        "GREEN_WEEKS": g.apply(
            lambda s: int((pd.to_numeric(s, errors="coerce") > ndvi_green_thr).sum())
        ).values,
    })

    is_nonagri = (
        (stats["GREEN_WEEKS"] <= nonagri_green_weeks_max)
        & (stats["NDVI_p90"] <= nonagri_p90_max)
    )
    is_perennial = (
        (stats["GREEN_WEEKS"] >= perennial_green_weeks)
        & (stats["NDVI_p25"] >= perennial_p25)
    )

    stats["crop_type"] = np.where(
        is_nonagri, "NON_AGRI", np.where(is_perennial, "PERENNIAL", "ANNUAL")
    )
    return stats[["cell_id", "crop_type"]].copy()


def compute_need_teacher(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cell_id"] = out["cell_id"].astype(str)
    if "crop_type" not in out.columns:
        out["crop_type"] = "ANNUAL"
    out["crop_type"] = out["crop_type"].fillna("ANNUAL")

    rain_heavy_mm = 15.0
    ndmi_wet = 0.20
    ndmi_dry = 0.05
    dndmi_speed = 0.04
    temp_lo = 18.0
    temp_hi = 32.0

    rain = _safe_num(out.get("Rain_mm_7d")).fillna(0)
    ndvi = _safe_num(out.get("NDVI_med"))
    ndmi = _safe_num(out.get("NDMI_med"))
    dndmi = _safe_num(out.get("dNDMI"))
    temp = _safe_num(out.get("Temp_C_7d")).fillna(temp_lo)

    if "veg_flag" in out.columns:
        veg_flag = _safe_num(out["veg_flag"]).fillna(0)
        veg_ok = veg_flag == 1
    else:
        veg_ok = ndvi >= 0.30

    out["veg_ok"] = veg_ok.astype("Int64")
    heavy_rain = rain > rain_heavy_mm

    def scale_inv(x, wetv, dryv):
        denom = (wetv - dryv) if (wetv - dryv) != 0 else 1.0
        s = (wetv - pd.to_numeric(x, errors="coerce")) / denom
        return s.clip(0, 1)

    def scale_pos(x, lo, hi):
        denom = (hi - lo) if (hi - lo) != 0 else 1.0
        s = (pd.to_numeric(x, errors="coerce") - lo) / denom
        return s.clip(0, 1)

    annual_s1 = scale_inv(ndmi, 0.18, 0.06)
    annual_s2 = scale_pos((-dndmi).fillna(0), 0.0, dndmi_speed)
    annual_s3 = scale_pos(temp, temp_lo, temp_hi)
    need_annual = (0.45 * annual_s1 + 0.35 * annual_s2 + 0.20 * annual_s3).clip(0, 1)

    per_s1 = scale_inv(ndmi, ndmi_wet, ndmi_dry)
    per_s2 = scale_pos((-dndmi).fillna(0), 0.0, dndmi_speed)
    per_s3 = scale_pos(temp, temp_lo, temp_hi)
    need_per = (0.55 * per_s1 + 0.30 * per_s2 + 0.15 * per_s3).clip(0, 1)

    need_annual = need_annual.where(~heavy_rain, 0.0).where(veg_ok, 0.0)
    need_per = need_per.where(~heavy_rain, 0.0).where(veg_ok, 0.0)

    is_per = out["crop_type"].astype(str) == "PERENNIAL"
    is_non = out["crop_type"].astype(str) == "NON_AGRI"

    need = need_annual.where(~is_per, need_per)
    need = need.where(~is_non, 0.0)

    out["need_prob_teacher"] = need.astype(float).clip(0, 1)
    out["need_class_teacher"] = np.where(
        out["need_prob_teacher"] >= 0.66,
        "NOW",
        np.where(out["need_prob_teacher"] >= 0.33, "SOON", "WAIT")
    )
    return out


def add_coordinates_if_available(df: pd.DataFrame, coords_csv: Path) -> pd.DataFrame:
    if not coords_csv.exists():
        return df
    coords = pd.read_csv(coords_csv)
    if "cell_id" not in coords.columns:
        return df
    coords["cell_id"] = coords["cell_id"].astype(str)
    use_cols = ["cell_id"] + [c for c in ["lon", "lat"] if c in coords.columns]
    coords = coords[use_cols].drop_duplicates("cell_id")
    return df.merge(coords, on="cell_id", how="left", suffixes=("", "_coord"))


def counts_by_year(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "year" not in df.columns:
        return {}
    years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    return {str(year): int(count) for year, count in years.value_counts().sort_index().items()}


def effective_years_from_df(df: pd.DataFrame) -> list[int]:
    if df.empty or "year" not in df.columns:
        return []
    return sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique().tolist())


def build_yearly_eval_report(df: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series) -> dict[str, dict]:
    report: dict[str, dict] = {}
    if df.empty:
        return report

    labels = ["WAIT", "SOON", "NOW"]
    years = effective_years_from_df(df)
    for year in years:
        mask = pd.to_numeric(df["year"], errors="coerce").astype("Int64") == year
        part = df.loc[mask].copy()
        if part.empty:
            continue

        part_true = y_true.loc[part.index].astype(str)
        part_pred = y_pred.loc[part.index].astype(str)
        cm = confusion_matrix(part_true, part_pred, labels=labels)

        report[str(year)] = {
            "rows": int(len(part)),
            "accuracy": float(accuracy_score(part_true, part_pred)),
            "true_class_counts": {str(k): int(v) for k, v in part_true.value_counts().to_dict().items()},
            "pred_class_counts": {str(k): int(v) for k, v in part_pred.value_counts().to_dict().items()},
            "confusion_matrix": {
                "labels": labels,
                "values": cm.tolist(),
            },
        }

    return report


def main():
    print("Loading exports...")
    df_all = load_exports_multiyear(EXPORTS_DIR, TRAIN_YEARS)
    print(f"Rows loaded total: {len(df_all):,}")
    print("\\nRows by year:")
    print(df_all["year"].value_counts(dropna=False).sort_index())
    loaded_rows_by_year = counts_by_year(df_all)

    if "week_start" not in df_all.columns:
        raise ValueError("Expected column 'week_start' in exports")

    df_all["week_start"] = pd.to_datetime(df_all["week_start"], errors="coerce")
    df_all = df_all.dropna(subset=["week_start", "cell_id"]).copy()
    df_all["cell_id"] = df_all["cell_id"].astype(str)
    cleaned_rows_by_year = counts_by_year(df_all)

    print("\\nBuilding crop map by year...")
    crop_maps = []
    for year in sorted(df_all["year"].dropna().astype(int).unique()):
        df_y = df_all[df_all["year"].astype(int) == year].copy()
        cm = compute_crop_type_apr_sep(df_y)
        cm["year"] = year
        crop_maps.append(cm)
        print(f"\\nYear {year} crop counts:")
        print(cm["crop_type"].value_counts())

    crop_map_year = pd.concat(crop_maps, ignore_index=True)

    print("\\nAdding history features...")
    df_all = build_temporal_features(df_all)

    print("Computing teacher labels (stress only, no WetScore/dVV in target)...")
    df_all = df_all.merge(crop_map_year, on=["cell_id", "year"], how="left")
    # normalize any duplicate crop_type columns after merge
    crop_candidates = [c for c in df_all.columns if c.startswith("crop_type")]
    if "crop_type" not in df_all.columns:
        if crop_candidates:
            base = crop_candidates[0]
            df_all["crop_type"] = df_all[base]
            for c in crop_candidates[1:]:
                df_all["crop_type"] = df_all["crop_type"].fillna(df_all[c])
        else:
            df_all["crop_type"] = "ANNUAL"
    df_all["crop_type"] = df_all["crop_type"].fillna("ANNUAL")

    df_all = compute_need_teacher(df_all)

    df_all = add_coordinates_if_available(df_all, COORDS_CSV)

    feature_cols_num = [
        "NDVI_med",
        "NDMI_med",
        "dNDMI",
        "ndmi_drop_ytd",
        "ndmi_mean_2w",
        "ndmi_mean_3w",
        "ndmi_mean_4w",
        "dndmi_mean_2w",
        "dndmi_mean_4w",
        "Rain_mm_7d",
        "Rain2w",
        "Rain3w",
        "Rain4w",
        "Rain6w",
        "dry_weeks_3w",
        "dry_weeks_6w",
        "rain_cum_ytd",
        "Temp_C_7d",
        "temp_mean_2w",
        "temp_mean_4w",
        "hot2w",
        "hot4w",
        "hot_weeks_ytd",
        "drying3w",
        "drying6w",
        "drying_weeks_ytd",
        "ndmi_cell_month_hist_med",
        "ndmi_vs_cell_month_hist_med",
        "ndvi_cell_month_hist_med",
        "ndvi_vs_cell_month_hist_med",
        "rain_cell_month_hist_med",
        "rain_vs_cell_month_hist_med",
        "temp_cell_month_hist_med",
        "temp_vs_cell_month_hist_med",
        "month",
        "doy",
    ]
    feature_cols_cat = ["crop_type"]

    for col in feature_cols_num:
        if col not in df_all.columns:
            df_all[col] = np.nan
    if "crop_type" not in df_all.columns:
        df_all["crop_type"] = "ANNUAL"

    if "veg_flag" in df_all.columns:
        train_df = df_all[_safe_num(df_all["veg_flag"]).fillna(0) == 1].copy()
    else:
        train_df = df_all.copy()

    train_df = train_df.dropna(subset=["need_class_teacher"]).copy()
    filtered_rows_by_year = counts_by_year(train_df)
    effective_training_years = effective_years_from_df(train_df)

    print("\\nTraining class counts:")
    print(train_df["need_class_teacher"].value_counts())

    X = train_df[feature_cols_num + feature_cols_cat].copy()
    y = train_df["need_class_teacher"].astype(str)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_num),
            ("cat", categorical_transformer, feature_cols_cat),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    train_idx, test_idx = train_test_split(
        train_df.index, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    train_rows_by_year = counts_by_year(train_df.loc[train_idx])
    test_rows_by_year = counts_by_year(train_df.loc[test_idx])

    print("\\nTraining RandomForest...")
    clf.fit(X_train, y_train)

    y_pred = pd.Series(clf.predict(X_test), index=X_test.index).astype(str)
    acc = accuracy_score(y_test, y_pred)

    print(f"\\nAccuracy: {acc:.4f}\\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    labels = ["WAIT", "SOON", "NOW"]
    cmx = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cmx)

    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    importances = clf.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    print("\\nTop 20 feature importances:")
    print(fi.head(20).to_string(index=False))

    yearly_eval_report = build_yearly_eval_report(train_df.loc[test_idx], y_test, y_pred)
    print("\\nPer-year held-out accuracy:")
    for year, year_report in yearly_eval_report.items():
        print(f" - {year}: rows={year_report['rows']:,}, accuracy={year_report['accuracy']:.4f}")

    model_path = MODELS_DIR / "need_rf_pipeline.joblib"
    meta_path = MODELS_DIR / "need_rf_metadata.json"
    fi_path = MODELS_DIR / "need_rf_feature_importance_stress_only.csv"
    snap_path = MODELS_DIR / "need_training_snapshot_stress_only.parquet"
    yearly_eval_path = MODELS_DIR / "need_rf_yearly_evaluation.json"

    joblib.dump(clf, model_path)
    fi.to_csv(fi_path, index=False)
    train_df.to_parquet(snap_path, index=False)
    yearly_eval_path.write_text(json.dumps(yearly_eval_report, indent=2), encoding="utf-8")

    metadata = {
        "train_years": TRAIN_YEARS,
        "loaded_years": effective_years_from_df(df_all),
        "effective_training_years": effective_training_years,
        "n_rows_total": int(len(df_all)),
        "n_rows_train": int(len(train_df)),
        "rows_by_year": {
            "loaded": loaded_rows_by_year,
            "after_basic_cleaning": cleaned_rows_by_year,
            "after_training_filters": filtered_rows_by_year,
            "train_split": train_rows_by_year,
            "test_split": test_rows_by_year,
        },
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "accuracy": float(acc),
        "classes": labels,
        "model_path": str(model_path),
        "feature_importance_path": str(fi_path),
        "training_snapshot_path": str(snap_path),
        "yearly_evaluation_path": str(yearly_eval_path),
        "note": "stress-only model; excludes WetScore/dVV from predictors and target",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\\nSaved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved feature importances: {fi_path}")
    print(f"Saved labeled snapshot: {snap_path}")
    print(f"Saved yearly evaluation: {yearly_eval_path}")
    print("\\nDone.")


if __name__ == "__main__":
    main()
