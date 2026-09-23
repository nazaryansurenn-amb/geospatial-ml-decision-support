from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import f1_score, precision_recall_fscore_support

@dataclass
class ModelBundle:
    model: object
    features: list[str]
    T_decision: float
    T_now: float

def _try_make_model():
    # Prefer XGBoost if available
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            eval_metric="logloss",
        ), "xgboost"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ), "random_forest"

def choose_threshold_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):  # step 0.01
        y_hat = (p >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t

def train_and_eval(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    label_col: str = "y_need",
) -> tuple[ModelBundle, dict]:
    model, model_name = _try_make_model()

    X_train = df_train[features].to_numpy()
    y_train = df_train[label_col].to_numpy().astype(int)

    X_test = df_test[features].to_numpy()
    y_test = df_test[label_col].to_numpy().astype(int)

    model.fit(X_train, y_train)

    # probabilities
    if hasattr(model, "predict_proba"):
        p_test = model.predict_proba(X_test)[:, 1]
    else:
        # fallback
        p_test = model.predict(X_test).astype(float)

    T_decision = choose_threshold_f1(y_test, p_test)

    # NOW threshold = 80th percentile among predicted-need probabilities
    need_mask = p_test >= T_decision
    if need_mask.any():
        T_now = float(np.quantile(p_test[need_mask], 0.80))
    else:
        T_now = float(min(0.9, T_decision + 0.2))

    y_hat = (p_test >= T_decision).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_hat, average="binary", zero_division=0)

    report = {
        "model": model_name,
        "T_decision": float(T_decision),
        "T_now": float(T_now),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "test_size": int(len(df_test)),
        "train_size": int(len(df_train)),
    }

    bundle = ModelBundle(model=model, features=features, T_decision=T_decision, T_now=T_now)
    return bundle, report

def save_bundle(bundle: ModelBundle, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(bundle.model, out_dir / "need_model.joblib")
    with open(out_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "features": bundle.features,
                "T_decision": bundle.T_decision,
                "T_now": bundle.T_now,
            },
            f,
            indent=2,
        )

def load_bundle(model_dir: Path) -> ModelBundle:
    model = load(model_dir / "need_model.joblib")
    with open(model_dir / "schema.json", "r", encoding="utf-8") as f:
        s = json.load(f)
    return ModelBundle(model=model, features=s["features"], T_decision=s["T_decision"], T_now=s["T_now"])