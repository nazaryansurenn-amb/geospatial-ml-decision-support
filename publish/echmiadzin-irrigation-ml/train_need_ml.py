from __future__ import annotations
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Echmiadzin — Train ML to predict irrigation NEED directly
#
# What this script does:
#   1) loads all weekly export CSVs from data/weekly_exports
#   2) rebuilds crop_type (ANNUAL / PERENNIAL / NON_AGRI)
#   3) rebuilds rule-based need_prob_v2 and 3 classes
#   4) trains RandomForest on those NEED classes directly
#   5) saves model + feature importance files into models/
#
# Outputs:
#   models/need_rf_pipeline.joblib
#   models/need_rf_metadata.json
#   models/need_rf_feature_importance.csv
# ============================================================

ROOT = Path(__file__).parent
EXPORTS_DIR = ROOT / "data" / "weekly_exports"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Parameters (same logic as your app)
# -----------------------------
YEAR = 2024

# crop type
NDVI_GREEN_THR = 0.30
PERENNIAL_GREEN_WEEKS = 10
PERENNIAL_P25 = 0.25
NONAGRI_P90_MAX = 0.20
NONAGRI_GREEN_WEEKS_MAX = 2

# need v2
RAIN_HEAVY_MM = 15.0
WET_RECENT_THR = 0.65
NDMI_WET = 0.20
NDMI_DRY = 0.05
DNDMI_SPEED = 0.04
TEMP_LO = 18.0
TEMP_HI = 32.0

# class thresholds
T_SOON = 0.33
T_NOW = 0.66

# model
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 300
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 3


# -----------------------------
# Load / normalize CSVs
# -----------------------------
def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def normalize_weekly_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "CellID" in df.columns and "cell_id" not in df.columns:
        df["cell_id"] = df["CellID"]
    if "cell" in df.columns and "cell_id" not in df.columns:
        df["cell_id"] = df["cell"]
    if "weekStart" in df.columns and "week_start" not in df.columns:
        df["week_start"] = df["weekStart"]
    if "weekEnd" in df.columns and "week_end" not in df.columns:
        df["week_end"] = df["weekEnd"]

    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str)
    if "week_start" in df.columns:
        df["week_start"] = _to_dt(df["week_start"])
    if "week_end" in df.columns:
        df["week_end"] = _to_dt(df["week_end"])
    if "year" not in df.columns and "week_start" in df.columns:
        df["year"] = df["week_start"].dt.year

    # replace export nodata markers
    for c in df.columns:
        if df[c].dtype.kind in "iuf":
            df[c] = df[c].replace([-9999, -9999.0, -999, -999.0], np.nan)

    return df


def load_exports(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {folder}")

    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d = normalize_weekly_df(d)
        d["_src"] = f.name
        parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["cell_id", "week_start"])
    return out


# -----------------------------
# Same crop type logic as app
# -----------------------------
def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def compute_crop_type_apr_sep(
    df_year: pd.DataFrame,
    ndvi_green_thr: float = NDVI_GREEN_THR,
    perennial_green_weeks: int = PERENNIAL_GREEN_WEEKS,
    perennial_p25: float = PERENNIAL_P25,
    nonagri_p90_max: float = NONAGRI_P90_MAX,
    nonagri_green_weeks_max: int = NONAGRI_GREEN_WEEKS_MAX,
) -> pd.DataFrame:
    df = df_year.copy()
    df["cell_id"] = df["cell_id"].astype(str)

    ndvi = _safe_num(df.get("NDVI_med"))
    valid = _safe_num(df.get("valid_s2")).fillna(0).astype(int) if "valid_s2" in df.columns else pd.Series(1, index=df.index, dtype=int)

    ok = (valid == 1) & ndvi.notna()
    df = df.loc[ok, ["cell_id", "NDVI_med"]].copy()

    if df.empty:
        out = df_year[["cell_id"]].drop_duplicates().copy()
        out["crop_type"] = "ANNUAL"
        return out

    def p25(x): return float(np.nanpercentile(x, 25)) if len(x) else np.nan
    def p90(x): return float(np.nanpercentile(x, 90)) if len(x) else np.nan

    g = df.groupby("cell_id")["NDVI_med"]
    stats = pd.DataFrame({
        "cell_id": g.size().index.astype(str),
        "NDVI_p25": g.apply(p25).values,
        "NDVI_p90": g.apply(p90).values,
        "GREEN_WEEKS": g.apply(lambda s: int((pd.to_numeric(s, errors="coerce") > ndvi_green_thr).sum())).values,
    })

    is_nonagri = (stats["GREEN_WEEKS"] <= nonagri_green_weeks_max) & (stats["NDVI_p90"] <= nonagri_p90_max)
    is_perennial = (stats["GREEN_WEEKS"] >= perennial_green_weeks) & (stats["NDVI_p25"] >= perennial_p25)

    stats["crop_type"] = np.where(is_nonagri, "NON_AGRI", np.where(is_perennial, "PERENNIAL", "ANNUAL"))
    return stats[["cell_id", "crop_type"]].copy()


# -----------------------------
# Same need logic as app
# -----------------------------
def compute_need_v2(
    df_week: pd.DataFrame,
    crop_map: pd.DataFrame,
    rain_heavy_mm: float = RAIN_HEAVY_MM,
    wet_recent_thr: float = WET_RECENT_THR,
    ndmi_wet: float = NDMI_WET,
    ndmi_dry: float = NDMI_DRY,
    dndmi_speed: float = DNDMI_SPEED,
    temp_lo: float = TEMP_LO,
    temp_hi: float = TEMP_HI,
) -> pd.DataFrame:
    out = df_week.copy()
    out["cell_id"] = out["cell_id"].astype(str)

    cm = crop_map.copy()
    cm["cell_id"] = cm["cell_id"].astype(str)
    out = out.merge(cm, on="cell_id", how="left")
    out["crop_type"] = out["crop_type"].fillna("ANNUAL")

    rain = _safe_num(out.get("Rain_mm_7d")).fillna(0)
    wet = _safe_num(out.get("WetScore")).fillna(0.50)
    ndmi = _safe_num(out.get("NDMI_med"))
    dndmi = _safe_num(out.get("dNDMI"))
    temp = _safe_num(out.get("Temp_C_7d")).fillna(temp_lo)

    if "veg_flag" in out.columns:
        veg_flag = _safe_num(out["veg_flag"]).fillna(0)
        veg_ok = (veg_flag == 1)
    else:
        ndvi = _safe_num(out.get("NDVI_med"))
        veg_ok = (ndvi >= 0.30)

    out["veg_ok"] = veg_ok.astype("Int64")
    heavy_rain = rain > rain_heavy_mm

    # annual: higher need when no recent wetting
    need_annual = (1.0 - wet).clip(0, 1)

    # perennial: NDMI + drying + temp stress
    def scale_inv(x, wetv, dryv):
        denom = (wetv - dryv) if (wetv - dryv) != 0 else 1.0
        s = (wetv - pd.to_numeric(x, errors="coerce")) / denom
        return s.clip(0, 1)

    def scale_pos(x, lo, hi):
        denom = (hi - lo) if (hi - lo) != 0 else 1.0
        s = (pd.to_numeric(x, errors="coerce") - lo) / denom
        return s.clip(0, 1)

    s1 = scale_inv(ndmi, ndmi_wet, ndmi_dry)
    s2 = scale_pos((-dndmi).fillna(0), 0.0, dndmi_speed)
    s3 = scale_pos(temp, temp_lo, temp_hi)

    need_per = (0.55 * s1 + 0.30 * s2 + 0.15 * s3).clip(0, 1)

    need_annual = need_annual.where(~heavy_rain, 0.0).where(veg_ok, 0.0)
    need_per = need_per.where(~heavy_rain, 0.0).where(veg_ok, 0.0)

    recent_wet = wet >= wet_recent_thr
    need_annual = need_annual.where(~recent_wet, need_annual * 0.25)
    need_per = need_per.where(~recent_wet, need_per * 0.40)

    is_per = out["crop_type"].astype(str) == "PERENNIAL"
    is_non = out["crop_type"].astype(str) == "NON_AGRI"

    need = need_annual.where(~is_per, need_per)
    need = need.where(~is_non, 0.0)

    out["need_prob_v2"] = need.astype(float).clip(0, 1)
    out["need_class_v2"] = np.where(
        out["need_prob_v2"] >= T_NOW,
        "NOW",
        np.where(out["need_prob_v2"] >= T_SOON, "SOON", "WAIT")
    )
    return out


# -----------------------------
# Train
# -----------------------------
def main():
    print("Loading exports...")
    exports = load_exports(EXPORTS_DIR)
    exports["year"] = pd.to_numeric(exports["year"], errors="coerce")
    exports = exports[exports["year"] == YEAR].copy()
    if exports.empty:
        raise ValueError(f"No rows found for YEAR={YEAR}")

    print(f"Rows loaded for {YEAR}: {len(exports):,}")

    print("Building crop map...")
    crop_map = compute_crop_type_apr_sep(exports)
    print(crop_map["crop_type"].value_counts(dropna=False).to_string())

    print("Computing need labels...")
    labeled = compute_need_v2(exports, crop_map)

    # Keep only actionable vegetation rows for first ML
    train_df = labeled.copy()
    train_df = train_df[train_df["veg_ok"] == 1].copy()
    train_df = train_df[train_df["crop_type"].isin(["ANNUAL", "PERENNIAL"])].copy()

    # Features for ML
    feature_cols_num = [
        "NDVI_med",
        "NDMI_med",
        "dNDMI",
        "WetScore",
        "Rain_mm_7d",
        "Temp_C_7d",
        "dVV",
        "doy",
        "month",
    ]
    feature_cols_cat = ["crop_type"]

    # Keep only columns that exist
    feature_cols_num = [c for c in feature_cols_num if c in train_df.columns]
    feature_cols_cat = [c for c in feature_cols_cat if c in train_df.columns]

    needed = feature_cols_num + feature_cols_cat + ["need_class_v2", "need_prob_v2", "cell_id", "week_start"]
    train_df = train_df[needed].copy()

    # Drop rows with missing target
    train_df = train_df.dropna(subset=["need_class_v2"])
    if train_df.empty:
        raise ValueError("No training rows left after filtering.")

    print("Training class counts:")
    print(train_df["need_class_v2"].value_counts(dropna=False).to_string())

    X = train_df[feature_cols_num + feature_cols_cat].copy()
    y = train_df["need_class_v2"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), feature_cols_num),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), feature_cols_cat),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", clf),
    ])

    print("Training RandomForest...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, pred, labels=["WAIT", "SOON", "NOW"]))

    # Feature importances
    prep = model.named_steps["prep"]
    rf = model.named_steps["rf"]
    feature_names = prep.get_feature_names_out()
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 feature importances:")
    print(fi.head(20).to_string(index=False))

    model_path = MODELS_DIR / "need_rf_pipeline.joblib"
    meta_path = MODELS_DIR / "need_rf_metadata.json"
    fi_path = MODELS_DIR / "need_rf_feature_importance.csv"
    labeled_path = MODELS_DIR / "need_training_snapshot.parquet"

    joblib.dump(model, model_path)
    fi.to_csv(fi_path, index=False)
    train_df.to_parquet(labeled_path, index=False)

    meta = {
        "year": YEAR,
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "class_thresholds": {"soon": T_SOON, "now": T_NOW},
        "crop_type_logic": {
            "ndvi_green_thr": NDVI_GREEN_THR,
            "perennial_green_weeks": PERENNIAL_GREEN_WEEKS,
            "perennial_p25": PERENNIAL_P25,
        },
        "need_logic": {
            "rain_heavy_mm": RAIN_HEAVY_MM,
            "wet_recent_thr": WET_RECENT_THR,
            "ndmi_wet": NDMI_WET,
            "ndmi_dry": NDMI_DRY,
            "dndmi_speed": DNDMI_SPEED,
            "temp_lo": TEMP_LO,
            "temp_hi": TEMP_HI,
        },
        "accuracy": float(acc),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_counts": y.value_counts().to_dict(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved feature importances: {fi_path}")
    print(f"Saved labeled snapshot: {labeled_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
