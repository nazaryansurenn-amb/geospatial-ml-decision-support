from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.features import build_temporal_features
from core.io import normalize_weekly_df

MODEL_PATH = ROOT / "models" / "need_rf_pipeline.joblib"
META_PATH = ROOT / "models" / "need_rf_metadata.json"
LIVE_DIR = ROOT / "data" / "live_exports"


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"[FAIL] Model missing: {MODEL_PATH}")
        return 1

    live_files = sorted(LIVE_DIR.glob("*.csv"))
    if not live_files:
        print(f"[FAIL] No live files found in: {LIVE_DIR}")
        return 1

    latest = live_files[-1]
    df = pd.read_csv(latest, low_memory=False)
    df = normalize_weekly_df(df)
    df = build_temporal_features(df)

    model = joblib.load(MODEL_PATH)
    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    feature_cols_num = meta.get("feature_cols_num", [])
    feature_cols_cat = meta.get("feature_cols_cat", [])
    needed = feature_cols_num + feature_cols_cat

    if not needed:
        print("[FAIL] Metadata has no feature columns.")
        return 1

    sample = df.head(20).copy()
    for col in feature_cols_num:
        if col not in sample.columns:
            sample[col] = np.nan
        sample[col] = pd.to_numeric(sample[col], errors="coerce")
    for col in feature_cols_cat:
        if col not in sample.columns:
            sample[col] = "ANNUAL"
        sample[col] = sample[col].astype(str)

    preds = model.predict(sample[needed])
    print(f"[OK] Smoke check passed on file: {latest.name}")
    print(f"[OK] Rows scored: {len(sample)}")
    print(f"[OK] Pred classes sample: {pd.Series(preds).value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
