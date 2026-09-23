from __future__ import annotations
import os
from pathlib import Path
import json
import warnings
import html
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import Element
from streamlit_folium import st_folium
from sklearn import __version__ as sklearn_version
from shapely.geometry import Point

from core.features import build_temporal_features
from core.io import load_exports
from core.grid import load_coords, build_grid, load_boundary

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_TITLE = "Irrigation Intelligence Dashboard"
APP_SUBTITLE = "Satellite-informed water demand monitoring and field priorities"
st.set_page_config(page_title=APP_TITLE, layout="wide")

ROOT = Path(__file__).parent


def load_local_env(env_path: Path):
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_local_env(ROOT / ".env")

DATA_DIR = ROOT / "data"
EXPORTS_DIR = DATA_DIR / "weekly_exports"
LIVE_EXPORTS_DIR = DATA_DIR / "live_exports"
COORDS_CSV = DATA_DIR / "Echmiadzin_Cell_Coordinates.csv"
BOUNDARY_SHP = ROOT / "boundary" / "Echmiadzin_wua_Boundary.shp"

MODEL_PATH = ROOT / "models" / "need_rf_pipeline.joblib"
META_PATH = ROOT / "models" / "need_rf_metadata.json"
YEARLY_EVAL_PATH = ROOT / "models" / "need_rf_yearly_evaluation.json"
FEATURE_IMPORTANCE_PATH = ROOT / "models" / "need_rf_feature_importance_stress_only.csv"

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri"
DEFAULT_COPILOT_MODEL = os.getenv("OPENAI_COPILOT_MODEL", "gpt-5.4-mini")
COPILOT_SYSTEM_PROMPT = """
You are an irrigation analysis copilot for Echmiadzin.
Use only the structured data provided to you.
Write concise, evidence-based operational analysis for an irrigation analyst.
Focus on current irrigation need, notable patterns, historical context, model context, spatial cluster context, and uncertainty.
Do not invent facts, locations, or causes that are not present in the input.
If the signals are mixed or limited, say so plainly.
Use previous-years context and live-vs-previous-live context when they are provided.
Prefer short markdown headings and compact bullets.
When asked for a weekly summary, structure it as:
## Current Situation
## History And Change
## Priority Cells
## Uncertainty
When asked about a selected cell, explain the predicted class, strongest evidence, historical context, and caution.
When asked a user question, answer only from the supplied context and say clearly if the context is insufficient.
When spatial clusters are provided, discuss urgent zones before isolated cells.
Always follow the project guidance context if it is provided.
""".strip()

COPILOT_GUIDANCE = {
    "invalid_data_rules": [
        "-9999 means invalid or missing earth observation data.",
        "Cells with invalid core inputs must not be treated as trustworthy irrigation evidence.",
        "Hidden or invalid cells must not be described as urgent targets.",
    ],
    "crop_type_rules": [
        "In live mode, perennial classification comes from historical years rather than only the current live file.",
        "Crop type is a stability layer; irrigation need is still decided from live model features.",
        "Annual and perennial sections should be interpreted separately when possible.",
    ],
    "need_interpretation_rules": [
        "WAIT, SOON, and NOW come from the trained ML model.",
        "Live irrigation need should be interpreted from current NDMI, dNDMI, rain, temperature, and other model features.",
        "Prioritize spatial clusters over isolated cells when giving operational recommendations.",
    ],
    "earth_observation_notes": [
        "NDMI and NDVI are observational indicators, not direct proof of irrigation events.",
        "Recent rain and temperature should be interpreted together with vegetation and moisture signals.",
        "Historical comparison is useful for judging whether the current week is abnormal for the same seasonal period.",
    ],
}

GRAY = "#BDBDBD"
Y_LIGHT = (255, 249, 196)
Y_DARK = (249, 168, 37)
R_LIGHT = (255, 205, 210)
R_DARK = (183, 28, 28)

UNIFIED_STOPS = [
    (0.00, (255, 249, 196)),
    (0.50, (255, 167, 38)),
    (1.00, (183, 28, 28)),
]


def apply_dashboard_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #17212f;
        }
        section[data-testid="stSidebar"] {
            background: #102235;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        section[data-testid="stSidebar"] * {
            color: #eef6ff;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label {
            color: #d7e4f2;
        }
        .main .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        .app-hero {
            background: linear-gradient(120deg, #12395a 0%, #1f6f78 58%, #6b8f45 100%);
            border-radius: 8px;
            padding: 1.35rem 1.55rem;
            margin-bottom: 1rem;
            color: white;
            box-shadow: 0 14px 34px rgba(25, 54, 83, 0.18);
        }
        .app-hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 760;
            letter-spacing: 0;
        }
        .app-hero p {
            margin: 0.45rem 0 0;
            color: rgba(255,255,255,0.86);
            font-size: 1rem;
        }
        .section-kicker {
            color: #5b6b7d;
            font-size: 0.84rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-bottom: -0.25rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e4e9f0;
            border-left: 4px solid #1f6f78;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 22px rgba(24, 39, 57, 0.06);
        }
        div[data-testid="stMetricLabel"] p {
            color: #5c6d7f;
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
            color: #17212f;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e4e9f0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(24, 39, 57, 0.05);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #dde6ef;
            border-radius: 8px 8px 0 0;
            padding: 0.45rem 0.85rem;
        }
        .stButton > button {
            border-radius: 7px;
            border: 1px solid #1f6f78;
            background: #1f6f78;
            color: white;
            font-weight: 700;
        }
        .stButton > button:hover {
            border-color: #174f57;
            background: #174f57;
            color: white;
        }
        .ai-studio {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 88% 10%, rgba(246, 178, 70, 0.28), transparent 30%),
                linear-gradient(135deg, #0f1f33 0%, #153f50 52%, #1d6c65 100%);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 8px;
            padding: 1.25rem;
            margin: 1.1rem 0 0.9rem;
            color: #ffffff;
            box-shadow: 0 18px 44px rgba(17, 41, 61, 0.22);
        }
        .ai-studio:before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent 36%, rgba(255,255,255,0.08));
            pointer-events: none;
        }
        .ai-studio > * {
            position: relative;
            z-index: 1;
        }
        .ai-kicker {
            color: #a8f0df;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .ai-studio h2 {
            margin: 0;
            font-size: 1.55rem;
            line-height: 1.2;
            letter-spacing: 0;
        }
        .ai-studio p {
            margin: 0.45rem 0 0;
            color: rgba(255,255,255,0.82);
        }
        .ai-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1rem;
        }
        .ai-stat {
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 0.75rem 0.8rem;
            backdrop-filter: blur(8px);
        }
        .ai-stat span {
            display: block;
            color: rgba(255,255,255,0.68);
            font-size: 0.74rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 0.06em;
        }
        .ai-stat strong {
            display: block;
            margin-top: 0.2rem;
            color: #ffffff;
            font-size: 1.05rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .ai-result-title {
            margin-top: 0.9rem;
            padding: 0.65rem 0.8rem;
            border-radius: 8px;
            background: #ffffff;
            border-left: 4px solid #f0a23a;
            color: #17212f;
            font-weight: 800;
            box-shadow: 0 8px 22px rgba(24, 39, 57, 0.06);
        }
        @media (max-width: 900px) {
            .ai-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_header():
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_studio_header(section_title, selected_cell_id, display_week_df, spatial_cluster_summary):
    zones = len(spatial_cluster_summary.get("clusters", [])) if spatial_cluster_summary else 0
    if not display_week_df.empty and "need_prob_ml" in display_week_df.columns:
        top_priority = _safe_num(display_week_df["need_prob_ml"]).max()
        top_priority_text = f"{top_priority:.2f}" if pd.notna(top_priority) else "n/a"
    else:
        top_priority_text = "n/a"
    focus_text = selected_cell_id if selected_cell_id is not None else "Not selected"
    st.markdown(
        f"""
        <div class="ai-studio">
            <div class="ai-kicker">AI Insight Studio</div>
            <h2>Executive field briefing</h2>
            <p>{html.escape(str(section_title))}</p>
            <div class="ai-grid">
                <div class="ai-stat"><span>View</span><strong>{html.escape(str(section_title))}</strong></div>
                <div class="ai-stat"><span>Focus cell</span><strong>{html.escape(str(focus_text))}</strong></div>
                <div class="ai-stat"><span>Priority zones</span><strong>{zones:,}</strong></div>
                <div class="ai-stat"><span>Top score</span><strong>{html.escape(top_priority_text)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def _geometry_center_point(geometries):
    centers = []
    weights = []
    for geom in geometries:
        if geom is None:
            continue
        try:
            if geom.is_empty:
                continue
            centroid = geom.centroid
            if centroid is None or centroid.is_empty:
                continue
            weight = abs(float(getattr(geom, "area", 0.0) or 0.0))
            if not np.isfinite(weight) or weight <= 0:
                weight = 1.0
            centers.append((float(centroid.x), float(centroid.y)))
            weights.append(weight)
        except Exception:
            continue
    if not centers:
        return None
    total_weight = float(sum(weights))
    if total_weight <= 0:
        total_weight = float(len(centers))
        weights = [1.0] * len(centers)
    x = sum(cx * weight for (cx, _), weight in zip(centers, weights)) / total_weight
    y = sum(cy * weight for (_, cy), weight in zip(centers, weights)) / total_weight
    return Point(x, y)


def _normalize_cell_id_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f"{parts[0]},{parts[1]}"

    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 10:
        return f"{digits[:5]},{digits[5:]}"
    return s


def _normalize_cell_id_series(series: pd.Series) -> pd.Series:
    return series.apply(_normalize_cell_id_value)


def _rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _lerp(a, b, t: float):
    return a + (b - a) * t


def _ramp(light_rgb, dark_rgb, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return (
        _lerp(light_rgb[0], dark_rgb[0], t),
        _lerp(light_rgb[1], dark_rgb[1], t),
        _lerp(light_rgb[2], dark_rgb[2], t),
    )


def _multi_ramp(stops, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        if p0 <= t <= p1:
            local = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return (
                _lerp(c0[0], c1[0], local),
                _lerp(c0[1], c1[1], local),
                _lerp(c0[2], c1[2], local),
            )
    return stops[-1][1]


@st.cache_resource
def load_ml_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = None
    sklearn_warning = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(MODEL_PATH)
    for w in caught:
        if "InconsistentVersionWarning" in getattr(w.category, "__name__", ""):
            sklearn_warning = str(w.message)
            break

    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    # Avoid carrying machine-specific absolute paths from training metadata.
    for key in ("model_path", "feature_importance_path", "training_snapshot_path", "yearly_evaluation_path"):
        val = meta.get(key)
        if isinstance(val, str) and (":\\" in val or val.startswith("/")):
            meta[key] = Path(val).name
    if sklearn_warning:
        meta["model_runtime_warning"] = (
            f"Model/runtime sklearn mismatch detected. Runtime sklearn={sklearn_version}. "
            f"Details: {sklearn_warning}"
        )
    return model, meta


@st.cache_data(show_spinner=False)
def load_yearly_evaluation():
    if not YEARLY_EVAL_PATH.exists():
        return {}
    try:
        return json.loads(YEARLY_EVAL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_top_feature_importance(top_n: int = 10):
    if not FEATURE_IMPORTANCE_PATH.exists():
        return []
    try:
        fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    except Exception:
        return []
    if fi.empty or "feature" not in fi.columns or "importance" not in fi.columns:
        return []
    fi = fi.sort_values("importance", ascending=False).head(int(top_n)).copy()
    return [
        {
            "feature": str(row["feature"]),
            "importance": round(float(row["importance"]), 6),
        }
        for _, row in fi.iterrows()
    ]


@st.cache_data(show_spinner=False)
def load_spatial():
    coords = load_coords(COORDS_CSV)
    grid = build_grid(coords)
    boundary = load_boundary(BOUNDARY_SHP)
    return grid, boundary


def _live_export_files():
    if not LIVE_EXPORTS_DIR.exists():
        return []
    return sorted(
        LIVE_EXPORTS_DIR.glob("*.csv"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )


def live_exports_signature():
    signature = []
    for path in _live_export_files():
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def format_live_signature_item(signature_item):
    if not signature_item:
        return None
    name, mtime_ns, size = signature_item
    updated = datetime.fromtimestamp(int(mtime_ns) / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")
    return f"{name} | updated {updated} | {int(size):,} bytes"


@st.cache_data(show_spinner="Loading historical exports...")
def load_all_exports_cached():
    exports = load_exports(EXPORTS_DIR)
    exports = exports.copy()
    exports["week_start"] = pd.to_datetime(exports["week_start"], errors="coerce")
    if "week_end" in exports.columns:
        exports["week_end"] = pd.to_datetime(exports["week_end"], errors="coerce")
    exports = exports.dropna(subset=["week_start", "cell_id"])
    exports["cell_id"] = _normalize_cell_id_series(exports["cell_id"])
    exports = exports.dropna(subset=["cell_id"])
    if "year" not in exports.columns:
        exports["year"] = exports["week_start"].dt.year
    if "doy" not in exports.columns:
        exports["doy"] = exports["week_start"].dt.dayofyear
    if "month" not in exports.columns:
        exports["month"] = exports["week_start"].dt.month
    return exports


@st.cache_data(show_spinner="Loading latest live export...")
def load_live_latest_cached(live_signature=None):
    files = _live_export_files()
    if not files:
        return pd.DataFrame()

    df = pd.read_csv(files[0]).copy()
    df["source_file"] = files[0].name
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df = df.dropna(subset=["cell_id"])
    df["cell_id"] = _normalize_cell_id_series(df["cell_id"])
    df = df.dropna(subset=["cell_id"])

    if "year" not in df.columns:
        if "week_start" in df.columns and df["week_start"].notna().any():
            df["year"] = df["week_start"].dt.year
        else:
            df["year"] = 2026
    if "doy" not in df.columns and "week_start" in df.columns:
        df["doy"] = df["week_start"].dt.dayofyear
    if "month" not in df.columns and "week_start" in df.columns:
        df["month"] = df["week_start"].dt.month
    return df


@st.cache_data(show_spinner="Loading previous live export...")
def load_live_previous_cached(live_signature=None):
    files = _live_export_files()
    if len(files) < 2:
        return pd.DataFrame()

    df = pd.read_csv(files[1]).copy()
    df["source_file"] = files[1].name
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df = df.dropna(subset=["cell_id"])
    df["cell_id"] = _normalize_cell_id_series(df["cell_id"])
    df = df.dropna(subset=["cell_id"])
    if "year" not in df.columns:
        if "week_start" in df.columns and df["week_start"].notna().any():
            df["year"] = df["week_start"].dt.year
        else:
            df["year"] = 2026
    if "doy" not in df.columns and "week_start" in df.columns:
        df["doy"] = df["week_start"].dt.dayofyear
    if "month" not in df.columns and "week_start" in df.columns:
        df["month"] = df["week_start"].dt.month
    return df


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
    valid = _safe_num(df.get("valid_s2")).fillna(0).astype(int) if "valid_s2" in df.columns else pd.Series(1, index=df.index, dtype=int)
    ok = (valid == 1) & ndvi.notna() & (ndvi != -9999)
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


@st.cache_data(show_spinner="Preparing selected historical year...")
def prepare_year_data_cached(year: int):
    exports = load_all_exports_cached()
    featured = build_temporal_features(exports)
    return featured[featured["year"].astype(int) == int(year)].copy()


@st.cache_data(show_spinner="Preparing live features and history context...")
def prepare_live_data_cached(history_years: tuple[int, ...], live_signature=None):
    exports = load_all_exports_cached()
    if history_years:
        exports = exports[exports["year"].astype(int).isin([int(year) for year in history_years])].copy()
    df_live = load_live_latest_cached(live_signature)
    if df_live is None or df_live.empty:
        return pd.DataFrame()
    return build_temporal_features(df_live, history_reference=exports)


@st.cache_data(show_spinner="Building crop map...")
def build_crop_map_cached(df_year: pd.DataFrame, ndvi_green_thr: float, perennial_green_weeks: int, perennial_p25: float):
    return compute_crop_type_apr_sep(
        df_year=df_year,
        ndvi_green_thr=float(ndvi_green_thr),
        perennial_green_weeks=int(perennial_green_weeks),
        perennial_p25=float(perennial_p25),
    )


@st.cache_data(show_spinner="Building historical perennial map for live mode...")
def build_live_crop_map_cached(history_years: tuple[int, ...], ndvi_green_thr: float, perennial_green_weeks: int, perennial_p25: float):
    exports = load_all_exports_cached()
    if exports is None or exports.empty:
        return pd.DataFrame(columns=["cell_id", "crop_type"])
    if history_years:
        exports = exports[exports["year"].astype(int).isin([int(year) for year in history_years])].copy()
    if exports.empty:
        return pd.DataFrame(columns=["cell_id", "crop_type"])
    return compute_crop_type_apr_sep(
        df_year=exports,
        ndvi_green_thr=float(ndvi_green_thr),
        perennial_green_weeks=int(perennial_green_weeks),
        perennial_p25=float(perennial_p25),
    )


def add_ml_predictions(df_week: pd.DataFrame, crop_map: pd.DataFrame, model, meta: dict):
    out = df_week.copy()
    out["cell_id"] = out["cell_id"].astype(str)

    cm = crop_map.copy()
    cm["cell_id"] = cm["cell_id"].astype(str)
    if "crop_type" in out.columns:
        out = out.drop(columns=["crop_type"])
    out = out.merge(cm, on="cell_id", how="left")
    out["crop_type"] = out["crop_type"].fillna("ANNUAL")

    feature_cols_num = meta.get("feature_cols_num", [
        "NDVI_med", "NDMI_med", "dNDMI", "ndmi_drop_ytd", "ndmi_mean_2w", "ndmi_mean_3w",
        "ndmi_mean_4w", "dndmi_mean_2w", "dndmi_mean_4w", "Rain_mm_7d", "Rain2w", "Rain3w",
        "Rain4w", "Rain6w", "dry_weeks_3w", "dry_weeks_6w", "rain_cum_ytd", "Temp_C_7d",
        "temp_mean_2w", "temp_mean_4w", "hot2w", "hot4w", "hot_weeks_ytd", "drying3w",
        "drying6w", "drying_weeks_ytd", "ndmi_cell_month_hist_med", "ndmi_vs_cell_month_hist_med",
        "ndvi_cell_month_hist_med", "ndvi_vs_cell_month_hist_med", "rain_cell_month_hist_med",
        "rain_vs_cell_month_hist_med", "temp_cell_month_hist_med", "temp_vs_cell_month_hist_med",
        "month", "doy"
    ])
    feature_cols_cat = meta.get("feature_cols_cat", ["crop_type"])

    for col in feature_cols_num:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = _safe_num(out[col])
    for col in feature_cols_cat:
        if col not in out.columns:
            out[col] = "ANNUAL"
        out[col] = out[col].astype(str)

    features = out[feature_cols_num + feature_cols_cat].copy()
    pred = model.predict(features)
    out["need_class_ml"] = pd.Series(pred, index=out.index).astype(str)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        classes = [str(c) for c in model.classes_]
        proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=out.index)
        out = pd.concat([out, proba_df], axis=1)
        out["need_prob_ml"] = 0.0
        if "p_SOON" in out.columns:
            out["need_prob_ml"] += 0.5 * out["p_SOON"].fillna(0)
        if "p_NOW" in out.columns:
            out["need_prob_ml"] += 1.0 * out["p_NOW"].fillna(0)
        out["need_prob_ml"] = out["need_prob_ml"].clip(0, 1)
    else:
        out["need_prob_ml"] = np.where(out["need_class_ml"].eq("NOW"), 1.0, np.where(out["need_class_ml"].eq("SOON"), 0.5, 0.0))
    return out


def add_boundary(m, boundary_gdf, label_col="name", show_labels=True):
    if boundary_gdf is None or boundary_gdf.empty:
        return
    def style_fn(_):
        return {"color": "#00acc1", "weight": 2, "fillOpacity": 0}

    tooltip = folium.GeoJsonTooltip(fields=[label_col]) if label_col in boundary_gdf.columns else None
    folium.GeoJson(boundary_gdf, name="WUA Boundary", style_function=style_fn, tooltip=tooltip).add_to(m)

    if show_labels and label_col in boundary_gdf.columns:
        for _, r in boundary_gdf.iterrows():
            c = r.geometry.centroid
            folium.Marker(
                location=(c.y, c.x),
                icon=folium.DivIcon(html=f"<div style='font-size:13px; font-weight:700; color:#00796b;'>{r[label_col]}</div>"),
            ).add_to(m)


WAIT_COLOR = "#43a047"
SOON_COLOR = "#fdd835"
NOW_COLOR = "#e53935"


def _valid_display_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    mask = pd.Series(True, index=df.index)
    required_numeric = [c for c in ["NDVI_med", "NDMI_med", "dNDMI"] if c in df.columns]
    for col in required_numeric:
        vals = _safe_num(df[col])
        mask &= vals.notna() & (vals != -9999)

    if "veg_flag" in df.columns:
        mask &= _safe_num(df["veg_flag"]).eq(1)
    if "valid_s2" in df.columns:
        mask &= _safe_num(df["valid_s2"]).eq(1)
    if "need_class_ml" in df.columns:
        mask &= df["need_class_ml"].astype(str).isin(["WAIT", "SOON", "NOW"])
    if "cell_id" in df.columns:
        mask &= df["cell_id"].notna()
    return mask.fillna(False)


def filter_display_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)
    return df.loc[_valid_display_mask(df)].copy()


def add_legend(m):
    html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: rgba(255,255,255,0.92); padding: 10px 12px; border-radius: 10px; border: 1px solid #ccc; box-shadow: 0 2px 10px rgba(0,0,0,0.2); width: 240px;">
      <div style="font-weight:700;margin-bottom:8px;font-size:14px;">ML Need Class</div>
      <div style="display:flex;align-items:center;font-size:12px;margin-bottom:6px;"><div style="width:14px;height:14px;background:{WAIT_COLOR};margin-right:8px;border:1px solid #444;"></div><div>WAIT</div></div>
      <div style="display:flex;align-items:center;font-size:12px;margin-bottom:6px;"><div style="width:14px;height:14px;background:{SOON_COLOR};margin-right:8px;border:1px solid #444;"></div><div>SOON</div></div>
      <div style="display:flex;align-items:center;font-size:12px;margin-bottom:8px;"><div style="width:14px;height:14px;background:{NOW_COLOR};margin-right:8px;border:1px solid #444;"></div><div>NOW</div></div>
      <div style="display:flex;align-items:center;font-size:12px;"><div style="width:14px;height:14px;background:{GRAY};margin-right:8px;border:1px solid #444;"></div><div>Invalid or hidden cells</div></div>
    </div>
    """
    m.get_root().html.add_child(Element(html))


def render_map(grid, boundary, week_df, show_boundary_labels: bool, show_gray_cells: bool, map_key: str = "irrigation_map"):
    if week_df is None or week_df.empty:
        st.warning("No rows to render.")
        return

    merged = grid.copy()
    keep = ["cell_id"]
    for c in ["crop_type", "veg_flag", "valid_s2", "need_prob_ml", "need_class_ml", "NDVI_med", "NDMI_med", "dNDMI", "Rain_mm_7d", "Temp_C_7d", "ndmi_drop_ytd", "Rain2w", "Rain3w"]:
        if c in week_df.columns:
            keep.append(c)

    w = week_df[keep].copy().groupby("cell_id", as_index=False).first()
    if "crop_type" in merged.columns and "crop_type" in w.columns:
        merged = merged.drop(columns=["crop_type"])
    merged = merged.merge(w, on="cell_id", how="left")
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    merged["_display_valid"] = _valid_display_mask(merged)

    if not show_gray_cells:
        merged = merged.loc[merged["_display_valid"]].copy()
    if merged.empty:
        st.warning("No cells match the current map filters.")
        return

    center_geometries = boundary.geometry if boundary is not None and not boundary.empty else merged.geometry
    cen = _geometry_center_point(center_geometries)
    if cen is None:
        st.warning("Map center could not be calculated from the available geometries.")
        return
    center = (float(cen.y), float(cen.x))

    m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles=None)
    folium.TileLayer(tiles=ESRI_TILES, attr=ESRI_ATTR, name="ESRI Satellite").add_to(m)
    add_boundary(m, boundary, label_col="name", show_labels=show_boundary_labels)
    add_legend(m)

    def style_fn(feat):
        props = feat.get("properties", {})
        if not bool(props.get("_display_valid", False)):
            return {"fillColor": GRAY, "color": GRAY, "weight": 0.15, "fillOpacity": 0.18}
        need_class = str(props.get("need_class_ml", "")).upper()
        color = {"WAIT": WAIT_COLOR, "SOON": SOON_COLOR, "NOW": NOW_COLOR}.get(need_class, GRAY)
        return {"fillColor": color, "color": color, "weight": 0.15, "fillOpacity": 0.60}

    tooltip_fields = ["cell_id"]
    tooltip_aliases = ["cell_id:"]
    for c, a in [("crop_type", "crop_type:"), ("need_class_ml", "need_class_ml:"), ("need_prob_ml", "need_prob_ml:"), ("veg_flag", "veg_flag:"), ("NDVI_med", "NDVI:"), ("NDMI_med", "NDMI:"), ("dNDMI", "dNDMI:"), ("ndmi_drop_ytd", "ndmi_drop_ytd:"), ("Rain_mm_7d", "Rain7d:"), ("Rain2w", "Rain2w:"), ("Rain3w", "Rain3w:"), ("Temp_C_7d", "Temp7d:")]:
        if c in merged.columns:
            tooltip_fields.append(c)
            tooltip_aliases.append(a)

    tooltip = None
    if not merged.empty and tooltip_fields:
        tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True)

    folium.GeoJson(merged, name="250m Grid", style_function=style_fn, tooltip=tooltip).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(
        m,
        width=None,
        height=720,
        key=map_key,
        returned_objects=[],
    )


def _round_or_none(value, digits: int = 4):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _sanitize_for_ai(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_ai(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_ai(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_ai(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        value = float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"-9999", "-9999.0", "nan", "NaN", "None", ""}:
            return None
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        if float(value) == -9999.0:
            return None
        return value
    return value


def build_model_context(meta: dict, yearly_eval: dict, top_features: list[dict]) -> dict:
    eval_summary = {}
    for year, metrics in sorted(yearly_eval.items(), key=lambda kv: kv[0]):
        if not isinstance(metrics, dict):
            continue
        eval_summary[str(year)] = {
            "rows": int(metrics.get("rows", 0)),
            "accuracy": _round_or_none(metrics.get("accuracy")),
            "true_class_counts": metrics.get("true_class_counts", {}),
            "pred_class_counts": metrics.get("pred_class_counts", {}),
        }

    return {
        "train_years": meta.get("train_years", []),
        "effective_training_years": meta.get("effective_training_years", []),
        "overall_accuracy": _round_or_none(meta.get("accuracy")),
        "n_rows_total": int(meta.get("n_rows_total", 0)) if meta.get("n_rows_total") is not None else None,
        "n_rows_train": int(meta.get("n_rows_train", 0)) if meta.get("n_rows_train") is not None else None,
        "classes": meta.get("classes", []),
        "rows_by_year_after_training_filters": meta.get("rows_by_year", {}).get("after_training_filters", {}),
        "yearly_evaluation": eval_summary,
        "top_feature_importance": top_features,
        "note": meta.get("note"),
    }


def build_live_comparison_summary(current_df: pd.DataFrame, previous_df: pd.DataFrame | None) -> dict | None:
    if current_df is None or current_df.empty or previous_df is None or previous_df.empty:
        return None

    current = current_df.copy()
    previous = previous_df.copy()
    for df in (current, previous):
        if "cell_id" in df.columns:
            df["cell_id"] = df["cell_id"].astype(str)

    current_counts = {str(k): int(v) for k, v in current.get("need_class_ml", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()}
    previous_counts = {str(k): int(v) for k, v in previous.get("need_class_ml", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()}

    class_count_delta = {}
    for label in sorted(set(current_counts) | set(previous_counts)):
        class_count_delta[label] = int(current_counts.get(label, 0) - previous_counts.get(label, 0))

    merged = current[["cell_id", "need_prob_ml", "need_class_ml"]].merge(
        previous[["cell_id", "need_prob_ml", "need_class_ml"]],
        on="cell_id",
        how="inner",
        suffixes=("_current", "_previous"),
    )
    probability_shift = []
    if not merged.empty:
        merged["need_prob_delta"] = _safe_num(merged["need_prob_ml_current"]).fillna(0) - _safe_num(merged["need_prob_ml_previous"]).fillna(0)
        probability_shift = (
            merged.sort_values("need_prob_delta", ascending=False)
            .head(10)
            .replace({np.nan: None})
            .to_dict(orient="records")
        )

    current_source = str(current["source_file"].iloc[0]) if "source_file" in current.columns and not current.empty else None
    previous_source = str(previous["source_file"].iloc[0]) if "source_file" in previous.columns and not previous.empty else None

    return {
        "current_live_file": current_source,
        "previous_live_file": previous_source,
        "current_mean_need_probability": _round_or_none(_safe_num(current.get("need_prob_ml")).fillna(0).mean()),
        "previous_mean_need_probability": _round_or_none(_safe_num(previous.get("need_prob_ml")).fillna(0).mean()),
        "mean_need_probability_delta": _round_or_none(
            _safe_num(current.get("need_prob_ml")).fillna(0).mean() - _safe_num(previous.get("need_prob_ml")).fillna(0).mean()
        ),
        "current_class_counts": current_counts,
        "previous_class_counts": previous_counts,
        "class_count_delta": class_count_delta,
        "largest_probability_increases": probability_shift,
    }


def build_spatial_cluster_summary(
    grid,
    week_df: pd.DataFrame,
    boundary=None,
    urgent_probability_threshold: float = 0.5,
    max_clusters: int = 8,
) -> dict | None:
    if grid is None or week_df is None or week_df.empty:
        return None

    base = grid[["cell_id", "geometry"]].copy()
    base["cell_id"] = base["cell_id"].astype(str)

    cols = [c for c in ["cell_id", "crop_type", "need_class_ml", "need_prob_ml", "veg_flag"] if c in week_df.columns]
    urgent = week_df[cols].copy()
    urgent["cell_id"] = urgent["cell_id"].astype(str)
    merged = base.merge(urgent, on="cell_id", how="inner")
    merged["need_prob_ml"] = _safe_num(merged.get("need_prob_ml")).fillna(0)
    merged = merged[
        (merged["need_prob_ml"] >= float(urgent_probability_threshold))
        | (merged.get("need_class_ml", pd.Series(index=merged.index, dtype=object)).astype(str) == "NOW")
    ].copy()
    if merged.empty:
        return None
    merged = merged.reset_index(drop=True)

    try:
        spatial_index = merged.sindex
    except Exception:
        spatial_index = None

    components = []
    visited = set()
    index_list = list(merged.index)
    for idx in index_list:
        if idx in visited:
            continue
        stack = [idx]
        component = []
        while stack:
            current_idx = stack.pop()
            if current_idx in visited:
                continue
            visited.add(current_idx)
            component.append(current_idx)
            geom = merged.at[current_idx, "geometry"]
            if geom is None or geom.is_empty:
                continue
            if spatial_index is not None:
                candidate_ids = list(spatial_index.intersection(geom.bounds))
            else:
                candidate_ids = index_list
            for candidate_idx in candidate_ids:
                if candidate_idx == current_idx or candidate_idx in visited:
                    continue
                candidate_geom = merged.at[candidate_idx, "geometry"]
                if candidate_geom is None or candidate_geom.is_empty:
                    continue
                try:
                    is_neighbor = geom.touches(candidate_geom) or geom.intersects(candidate_geom)
                except Exception:
                    is_neighbor = False
                if is_neighbor:
                    stack.append(candidate_idx)
        components.append(component)

    cluster_records = []
    for cluster_number, component in enumerate(components, start=1):
        cluster = merged.loc[component].copy()
        centroid = _geometry_center_point(cluster.geometry)
        class_counts = {str(k): int(v) for k, v in cluster["need_class_ml"].astype(str).value_counts(dropna=False).to_dict().items()} if "need_class_ml" in cluster.columns else {}
        crop_counts = {str(k): int(v) for k, v in cluster["crop_type"].astype(str).value_counts(dropna=False).to_dict().items()} if "crop_type" in cluster.columns else {}
        top_cells = (
            cluster.sort_values("need_prob_ml", ascending=False)[["cell_id", "need_class_ml", "need_prob_ml"]]
            .head(5)
            .replace({np.nan: None})
            .to_dict(orient="records")
        )

        boundary_name = None
        if boundary is not None and hasattr(boundary, "empty") and not boundary.empty and centroid is not None:
            try:
                match = boundary[boundary.geometry.contains(centroid)]
                if match.empty:
                    match = boundary[boundary.geometry.intersects(centroid)]
                if not match.empty and "name" in match.columns:
                    boundary_name = str(match.iloc[0]["name"])
            except Exception:
                boundary_name = None

        dominant_class = max(class_counts.items(), key=lambda item: item[1])[0] if class_counts else None
        dominant_crop = max(crop_counts.items(), key=lambda item: item[1])[0] if crop_counts else None
        cluster_records.append(
            {
                "cluster_id": f"C{cluster_number}",
                "boundary_name": boundary_name,
                "cell_count": int(len(cluster)),
                "dominant_class": dominant_class,
                "dominant_crop_type": dominant_crop,
                "mean_need_probability": _round_or_none(cluster["need_prob_ml"].mean()),
                "max_need_probability": _round_or_none(cluster["need_prob_ml"].max()),
                "center_lat": _round_or_none(centroid.y, 5) if centroid is not None else None,
                "center_lon": _round_or_none(centroid.x, 5) if centroid is not None else None,
                "class_counts": class_counts,
                "crop_counts": crop_counts,
                "top_cells": top_cells,
            }
        )

    cluster_records = sorted(
        cluster_records,
        key=lambda rec: (
            -float(rec["max_need_probability"] or 0),
            -int(rec["cell_count"]),
        ),
    )
    cluster_records = cluster_records[: int(max_clusters)]
    return {
        "urgent_probability_threshold": float(urgent_probability_threshold),
        "cluster_count": int(len(cluster_records)),
        "clusters": cluster_records,
    }


def build_previous_years_summary(df_week: pd.DataFrame, reference_exports: pd.DataFrame | None) -> dict | None:
    if df_week is None or df_week.empty or reference_exports is None or reference_exports.empty:
        return None

    current = df_week.copy()
    reference = reference_exports.copy()

    if "week_start" in current.columns:
        current["week_start"] = pd.to_datetime(current["week_start"], errors="coerce")
    if "week_start" in reference.columns:
        reference["week_start"] = pd.to_datetime(reference["week_start"], errors="coerce")
    if "year" not in current.columns and "week_start" in current.columns:
        current["year"] = current["week_start"].dt.year
    if "year" not in reference.columns and "week_start" in reference.columns:
        reference["year"] = reference["week_start"].dt.year
    if "month" not in current.columns and "week_start" in current.columns:
        current["month"] = current["week_start"].dt.month
    if "month" not in reference.columns and "week_start" in reference.columns:
        reference["month"] = reference["week_start"].dt.month
    if "doy" not in current.columns and "week_start" in current.columns:
        current["doy"] = current["week_start"].dt.dayofyear
    if "doy" not in reference.columns and "week_start" in reference.columns:
        reference["doy"] = reference["week_start"].dt.dayofyear

    current_month = int(current["month"].dropna().iloc[0]) if "month" in current.columns and current["month"].notna().any() else None
    current_doy = int(current["doy"].dropna().median()) if "doy" in current.columns and current["doy"].notna().any() else None

    monthly_reference = reference.copy()
    if current_month is not None and "month" in monthly_reference.columns:
        monthly_reference = monthly_reference[monthly_reference["month"].astype("Int64") == current_month].copy()

    window_reference = monthly_reference.copy()
    if current_doy is not None and "doy" in window_reference.columns:
        window_reference = window_reference[_safe_num(window_reference["doy"]).sub(current_doy).abs() <= 14].copy()
        if window_reference.empty:
            window_reference = monthly_reference

    if window_reference.empty:
        return None

    metric_cols = [c for c in ["NDVI_med", "NDMI_med", "Rain_mm_7d", "Temp_C_7d", "dNDMI"] if c in current.columns and c in window_reference.columns]
    current_means = {col: _round_or_none(_safe_num(current[col]).mean()) for col in metric_cols}
    reference_medians = {col: _round_or_none(_safe_num(window_reference[col]).median()) for col in metric_cols}
    deltas = {}
    for col in metric_cols:
        cur_val = current_means.get(col)
        ref_val = reference_medians.get(col)
        deltas[col] = None if cur_val is None or ref_val is None else round(cur_val - ref_val, 4)

    year_counts = {}
    if "year" in window_reference.columns:
        year_counts = {str(int(k)): int(v) for k, v in window_reference["year"].dropna().astype(int).value_counts().sort_index().to_dict().items()}

    per_year_feature_means = []
    if "year" in window_reference.columns and metric_cols:
        grouped = window_reference.groupby(window_reference["year"].astype(int))
        for year, group in grouped:
            record = {"year": int(year), "rows": int(len(group))}
            for col in metric_cols:
                record[col] = _round_or_none(_safe_num(group[col]).mean())
            per_year_feature_means.append(record)

    return {
        "reference_years": sorted([int(y) for y in window_reference["year"].dropna().astype(int).unique().tolist()]) if "year" in window_reference.columns else [],
        "reference_period_month": current_month,
        "reference_period_doy_center": current_doy,
        "reference_rows": int(len(window_reference)),
        "reference_rows_by_year": year_counts,
        "current_feature_means": current_means,
        "historical_feature_medians": reference_medians,
        "current_minus_historical": deltas,
        "per_year_feature_means": per_year_feature_means,
    }


def build_copilot_context(
    df_week: pd.DataFrame,
    title_suffix: str,
    mode: str,
    live_label: str | None = None,
    previous_years_summary: dict | None = None,
    model_context: dict | None = None,
    ui_context: dict | None = None,
    live_comparison_summary: dict | None = None,
    spatial_cluster_summary: dict | None = None,
) -> dict:
    if df_week is None or df_week.empty:
        return {
            "mode": mode,
            "title_suffix": title_suffix,
            "row_count": 0,
            "message": "No rows available for the selected view.",
        }

    df = df_week.copy()
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str)

    week_start = None
    week_end = None
    if "week_start" in df.columns and df["week_start"].notna().any():
        week_start = pd.Timestamp(df["week_start"].dropna().min()).date().isoformat()
    if "week_end" in df.columns and df["week_end"].notna().any():
        week_end = pd.Timestamp(df["week_end"].dropna().max()).date().isoformat()

    need_counts = {str(k): int(v) for k, v in df.get("need_class_ml", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()}
    crop_counts = {str(k): int(v) for k, v in df.get("crop_type", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()}
    veg_ready = int((_safe_num(df.get("veg_flag")).fillna(0) == 1).sum()) if "veg_flag" in df.columns else None

    top_cols = [c for c in [
        "cell_id", "crop_type", "need_class_ml", "need_prob_ml", "NDVI_med", "NDMI_med",
        "dNDMI", "Rain_mm_7d", "Temp_C_7d", "ndmi_drop_ytd",
        "ndmi_vs_cell_month_hist_med", "rain_vs_cell_month_hist_med"
    ] if c in df.columns]
    top_cells = (
        df.sort_values("need_prob_ml", ascending=False)
        .loc[:, top_cols]
        .head(12)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    anomaly_cols = [c for c in [
        "cell_id", "crop_type", "need_class_ml", "need_prob_ml",
        "ndmi_vs_cell_month_hist_med", "rain_vs_cell_month_hist_med", "temp_vs_cell_month_hist_med"
    ] if c in df.columns]
    anomaly_cells = []
    if "ndmi_vs_cell_month_hist_med" in df.columns:
        anomaly_cells = (
            df.sort_values(["ndmi_vs_cell_month_hist_med", "need_prob_ml"], ascending=[True, False])
            .loc[:, anomaly_cols]
            .head(8)
            .replace({np.nan: None})
            .to_dict(orient="records")
        )

    field_priority_brief = []
    if spatial_cluster_summary and spatial_cluster_summary.get("clusters"):
        for cluster in spatial_cluster_summary.get("clusters", [])[:3]:
            field_priority_brief.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "boundary_name": cluster.get("boundary_name"),
                    "cell_count": cluster.get("cell_count"),
                    "dominant_class": cluster.get("dominant_class"),
                    "mean_need_probability": cluster.get("mean_need_probability"),
                    "max_need_probability": cluster.get("max_need_probability"),
                    "top_cells": cluster.get("top_cells", [])[:3],
                }
            )

    context = {
        "mode": mode,
        "title_suffix": title_suffix,
        "live_source_file": live_label,
        "project_guidance": COPILOT_GUIDANCE,
        "row_count": int(len(df)),
        "week_start": week_start,
        "week_end": week_end,
        "need_class_counts": need_counts,
        "crop_type_counts": crop_counts,
        "veg_ready_cells": veg_ready,
        "mean_need_probability": round(float(_safe_num(df.get("need_prob_ml")).fillna(0).mean()), 4) if "need_prob_ml" in df.columns else None,
        "max_need_probability": round(float(_safe_num(df.get("need_prob_ml")).fillna(0).max()), 4) if "need_prob_ml" in df.columns else None,
        "top_priority_cells": top_cells,
        "lowest_ndmi_anomaly_cells": anomaly_cells,
        "previous_years_summary": previous_years_summary,
        "model_context": model_context,
        "ui_context": ui_context,
        "live_comparison_summary": live_comparison_summary,
        "spatial_cluster_summary": spatial_cluster_summary,
        "context_availability": {
            "has_previous_years_summary": previous_years_summary is not None,
            "has_live_comparison_summary": live_comparison_summary is not None,
            "has_spatial_cluster_summary": bool(spatial_cluster_summary and spatial_cluster_summary.get("clusters")),
        },
        "field_priority_brief": field_priority_brief,
    }
    return _sanitize_for_ai(context)


def build_selected_cell_context(
    df_week: pd.DataFrame,
    selected_cell_id: str,
    reference_exports: pd.DataFrame | None,
    previous_live_df: pd.DataFrame | None = None,
) -> dict | None:
    if df_week is None or df_week.empty or not selected_cell_id:
        return None

    current = df_week.copy()
    current["cell_id"] = current["cell_id"].astype(str)
    row = current[current["cell_id"] == str(selected_cell_id)].copy()
    if row.empty:
        return None
    row = row.iloc[0]

    rank = None
    if "need_prob_ml" in current.columns:
        ranked = current.sort_values("need_prob_ml", ascending=False).reset_index(drop=True)
        matches = ranked.index[ranked["cell_id"] == str(selected_cell_id)].tolist()
        if matches:
            rank = int(matches[0] + 1)

    cell_columns = [c for c in [
        "cell_id", "crop_type", "need_class_ml", "need_prob_ml", "NDVI_med", "NDMI_med", "dNDMI",
        "ndmi_drop_ytd", "Rain_mm_7d", "Rain2w", "Rain3w", "Temp_C_7d",
        "ndmi_vs_cell_month_hist_med", "rain_vs_cell_month_hist_med", "temp_vs_cell_month_hist_med"
    ] if c in current.columns]
    cell_snapshot = {col: (None if pd.isna(row[col]) else row[col]) for col in cell_columns}

    historical_cell_summary = None
    if reference_exports is not None and not reference_exports.empty and "cell_id" in reference_exports.columns:
        hist = reference_exports.copy()
        hist["cell_id"] = hist["cell_id"].astype(str)
        hist = hist[hist["cell_id"] == str(selected_cell_id)].copy()
        if not hist.empty:
            for date_col in ["week_start", "week_end"]:
                if date_col in hist.columns:
                    hist[date_col] = pd.to_datetime(hist[date_col], errors="coerce")
            if "year" not in hist.columns and "week_start" in hist.columns:
                hist["year"] = hist["week_start"].dt.year
            metric_cols = [c for c in ["NDVI_med", "NDMI_med", "Rain_mm_7d", "Temp_C_7d", "dNDMI"] if c in hist.columns]
            historical_cell_summary = {
                "rows": int(len(hist)),
                "years": sorted([int(y) for y in hist["year"].dropna().astype(int).unique().tolist()]) if "year" in hist.columns else [],
                "metric_medians": {col: _round_or_none(_safe_num(hist[col]).median()) for col in metric_cols},
                "metric_means_by_year": [],
            }
            if "year" in hist.columns:
                grouped = hist.groupby(hist["year"].astype(int))
                for year, group in grouped:
                    rec = {"year": int(year), "rows": int(len(group))}
                    for col in metric_cols:
                        rec[col] = _round_or_none(_safe_num(group[col]).mean())
                    historical_cell_summary["metric_means_by_year"].append(rec)

    previous_live_snapshot = None
    if previous_live_df is not None and not previous_live_df.empty and "cell_id" in previous_live_df.columns:
        prev = previous_live_df.copy()
        prev["cell_id"] = prev["cell_id"].astype(str)
        prev_row = prev[prev["cell_id"] == str(selected_cell_id)].copy()
        if not prev_row.empty:
            prev_row = prev_row.iloc[0]
            prev_cols = [c for c in [
                "cell_id", "need_class_ml", "need_prob_ml", "NDVI_med", "NDMI_med", "dNDMI",
                "Rain_mm_7d", "Temp_C_7d", "source_file"
            ] if c in prev.index]
            previous_live_snapshot = {col: (None if pd.isna(prev_row[col]) else prev_row[col]) for col in prev_cols}

    return _sanitize_for_ai({
        "selected_cell_id": str(selected_cell_id),
        "current_rank_by_need_probability": rank,
        "current_cell_snapshot": cell_snapshot,
        "historical_cell_summary": historical_cell_summary,
        "previous_live_snapshot": previous_live_snapshot,
    })


def generate_copilot_response(context: dict, task_prompt: str, max_output_tokens: int = 500) -> tuple[str | None, str | None]:
    if OpenAI is None:
        return None, "The `openai` Python package is not installed in this environment."

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "Set `OPENAI_API_KEY` to enable the AI copilot panel."

    try:
        clean_context = _sanitize_for_ai(context)
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=DEFAULT_COPILOT_MODEL,
            instructions=COPILOT_SYSTEM_PROMPT,
            input=[
                {"role": "user", "content": task_prompt},
                {"role": "user", "content": json.dumps(clean_context, ensure_ascii=True, indent=2)},
            ],
            max_output_tokens=int(max_output_tokens),
        )
    except Exception as exc:
        return None, f"Copilot request failed: {exc}"

    summary = getattr(response, "output_text", None)
    if isinstance(summary, str) and summary.strip():
        return summary.strip(), None
    return None, "The AI copilot returned an empty response."


def generate_copilot_summary(context: dict) -> tuple[str | None, str | None]:
    return generate_copilot_response(
        context=context,
        task_prompt=(
            "Create a field-priority briefing for this irrigation view. "
            "Use exactly these markdown sections: "
            "## Inspect First, ## Why Now, ## What Changed, ## Uncertainty. "
            "Start with the top 3 zones or cells to inspect first. "
            "If previous_years_summary exists, use it explicitly. "
            "If live_comparison_summary exists, quantify what changed from the previous live file. "
            "Do not say those contexts are missing unless their values are null or empty in the supplied payload. "
            "Keep the briefing practical and decisive."
        ),
        max_output_tokens=650,
    )


def main_with_valid_display():
    apply_dashboard_theme()
    render_dashboard_header()

    try:
        grid, boundary = load_spatial()
        ml_model, ml_meta = load_ml_artifacts()
    except Exception as e:
        st.error(str(e))
        return

    exports = load_all_exports_cached()
    live_signature = live_exports_signature()
    live_df = load_live_latest_cached(live_signature)
    live_prev_df = load_live_previous_cached(live_signature)
    yearly_eval = load_yearly_evaluation()
    top_features = load_top_feature_importance(10)

    with st.sidebar:
        st.header("Decision Controls")
        mode = st.radio("Data view", options=["historical", "live"], index=0, format_func=lambda x: "Historical baseline" if x == "historical" else "Current monitoring")
        ndvi_green_thr = st.number_input("Green cover threshold", value=0.30, step=0.01, format="%.2f")
        perennial_green_weeks = st.slider("Perennial green weeks", 6, 20, 10, 1)
        perennial_p25 = st.number_input("Perennial NDVI p25 threshold", value=0.25, step=0.01, format="%.2f")
        st.divider()
        show_gray_cells = st.checkbox("Show low-confidence cells", value=False)
        show_boundary_labels = st.checkbox("Show area labels", value=True)

    live_label = None
    reference_exports = pd.DataFrame()
    df_prev_live_scored = pd.DataFrame()

    if mode == "historical":
        if exports is None or exports.empty:
            st.error(f"No exports found in: {EXPORTS_DIR}")
            return

        years = sorted(exports["year"].dropna().astype(int).unique().tolist())
        sel_year = st.selectbox("Year", years, index=0 if 2024 not in years else years.index(2024))
        df_year = prepare_year_data_cached(int(sel_year))
        week_rows = df_year[["week_start", "week_end"]].dropna(subset=["week_start"]).drop_duplicates().sort_values("week_start")
        if week_rows.empty:
            st.error("No weeks found for selected year.")
            return

        crop_map = build_crop_map_cached(df_year, float(ndvi_green_thr), int(perennial_green_weeks), float(perennial_p25))
        options = []
        for _, r in week_rows.iterrows():
            ws = pd.Timestamp(r["week_start"]).date()
            we = pd.Timestamp(r["week_end"]).date() if pd.notna(r.get("week_end")) else ws
            options.append((r["week_start"], f"{ws} -> {we}"))

        sel_week_start = st.selectbox(
            "Week (by start date)",
            options=[o[0] for o in options],
            format_func=lambda x: dict(options).get(x, str(x)),
            index=max(0, len(options) - 1),
        )
        df_week = df_year[df_year["week_start"] == sel_week_start].copy()
        reference_exports = exports[exports["year"].astype(int) < int(sel_year)].copy()
        title_suffix = f"Historical - {sel_year}"
    else:
        if live_df is None or live_df.empty:
            st.error(f"No live CSV found in: {LIVE_EXPORTS_DIR}")
            return
        history_years = tuple(int(year) for year in ml_meta.get("effective_training_years", []))
        df_live = prepare_live_data_cached(history_years, live_signature)
        crop_map = build_live_crop_map_cached(history_years, float(ndvi_green_thr), int(perennial_green_weeks), float(perennial_p25))
        df_week = df_live.copy()
        if exports is not None and not exports.empty and history_years:
            reference_exports = exports[exports["year"].astype(int).isin(history_years)].copy()
        if live_prev_df is not None and not live_prev_df.empty:
            prev_live_temporal = build_temporal_features(live_prev_df.copy(), history_reference=exports)
            prev_crop_map = crop_map
            df_prev_live_scored = add_ml_predictions(prev_live_temporal.copy(), prev_crop_map, ml_model, ml_meta)
        live_label = str(df_live["source_file"].iloc[0]) if "source_file" in df_live.columns else "Last_week.csv"
        if "week_start" in df_live.columns and df_live["week_start"].notna().any():
            ws = pd.Timestamp(df_live["week_start"].dropna().iloc[0]).date()
            st.info(f"Current source: {live_label} - week starting {ws}")
        else:
            st.info(f"Current source: {live_label}")
        live_file_status = format_live_signature_item(live_signature[0] if live_signature else None)
        if live_file_status:
            st.caption(f"Live file loaded: {live_file_status} | rows: {len(df_live):,}")
        title_suffix = "Current monitoring"

    df_week = add_ml_predictions(df_week, crop_map, ml_model, ml_meta)
    display_prev_live_df = filter_display_rows(df_prev_live_scored) if not df_prev_live_scored.empty else pd.DataFrame()

    def render_live_or_historical_section(section_df: pd.DataFrame, section_title: str, section_key: str, previous_section_df: pd.DataFrame | None = None):
        display_week_df = filter_display_rows(section_df)

        st.markdown('<div class="section-kicker">Operational Summary</div>', unsafe_allow_html=True)
        st.subheader(section_title)
        ct_counts = display_week_df["crop_type"].value_counts(dropna=False).to_dict() if not display_week_df.empty else {}
        vc = _safe_num(display_week_df.get("veg_flag")).value_counts(dropna=False).to_dict() if "veg_flag" in display_week_df.columns else {}
        nc = display_week_df["need_class_ml"].value_counts(dropna=False).to_dict() if not display_week_df.empty else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Perennial cells", f"{int(ct_counts.get('PERENNIAL', 0)):,}")
        c2.metric("Annual cells", f"{int(ct_counts.get('ANNUAL', 0)):,}")
        c3.metric("Valid vegetation", f"{int(vc.get(1, 0)):,}")
        c4.metric("NOW / SOON / WAIT", f"{int(nc.get('NOW', 0)):,} / {int(nc.get('SOON', 0)):,} / {int(nc.get('WAIT', 0)):,}")

        top_cols = [c for c in ["cell_id", "crop_type", "need_class_ml", "need_prob_ml", "NDVI_med", "NDMI_med", "dNDMI", "ndmi_drop_ytd", "Rain_mm_7d", "Rain2w", "Rain3w", "Temp_C_7d"] if c in section_df.columns]
        if display_week_df.empty:
            st.warning("No valid cells remain after removing rows with -9999 or invalid vegetation flags.")
        else:
            st.markdown('<div class="section-kicker">Highest Priority Cells</div>', unsafe_allow_html=True)
            st.dataframe(display_week_df.sort_values("need_prob_ml", ascending=False)[top_cols].head(20), use_container_width=True)

        ui_context = {
            "mode": mode,
            "title_suffix": section_title,
            "ndvi_green_threshold": float(ndvi_green_thr),
            "perennial_green_weeks_threshold": int(perennial_green_weeks),
            "perennial_ndvi_p25_threshold": float(perennial_p25),
            "show_gray_cells": bool(show_gray_cells),
            "show_boundary_labels": bool(show_boundary_labels),
            "section_key": section_key,
        }
        model_context = build_model_context(ml_meta, yearly_eval, top_features)
        previous_years_summary = build_previous_years_summary(display_week_df, reference_exports)
        live_comparison_summary = build_live_comparison_summary(display_week_df, previous_section_df if previous_section_df is not None and not previous_section_df.empty else None)
        spatial_cluster_summary = build_spatial_cluster_summary(
            grid=grid,
            week_df=display_week_df,
            boundary=boundary,
        )
        copilot_context = build_copilot_context(
            df_week=display_week_df,
            title_suffix=section_title,
            mode=mode,
            live_label=live_label,
            previous_years_summary=previous_years_summary,
            model_context=model_context,
            ui_context=ui_context,
            live_comparison_summary=live_comparison_summary,
            spatial_cluster_summary=spatial_cluster_summary,
        )

        focus_cell_options = []
        if "cell_id" in display_week_df.columns:
            focus_cell_options = display_week_df.sort_values("need_prob_ml", ascending=False)["cell_id"].astype(str).drop_duplicates().tolist()
        selected_cell_id = None
        selected_cell_context = None
        if focus_cell_options:
            selected_cell_id = st.selectbox("Focus cell", options=focus_cell_options, index=0, key=f"focus_cell_{section_key}")
            selected_cell_context = build_selected_cell_context(
                df_week=display_week_df,
                selected_cell_id=selected_cell_id,
                reference_exports=reference_exports,
                previous_live_df=previous_section_df if previous_section_df is not None and not previous_section_df.empty else None,
            )

        state_prefix = f"copilot_{section_key}"
        context_key = json.dumps(
            {
                "mode": mode,
                "title_suffix": section_title,
                "live_label": live_label,
                "row_count": int(len(display_week_df)),
                "week_start": copilot_context.get("week_start"),
                "week_end": copilot_context.get("week_end"),
                "selected_cell_id": selected_cell_id,
            },
            sort_keys=True,
        )
        if st.session_state.get(f"{state_prefix}_context_key") != context_key:
            st.session_state[f"{state_prefix}_context_key"] = context_key
            st.session_state[f"{state_prefix}_summary_text"] = None
            st.session_state[f"{state_prefix}_summary_error"] = None
            st.session_state[f"{state_prefix}_cell_text"] = None
            st.session_state[f"{state_prefix}_cell_error"] = None
            st.session_state[f"{state_prefix}_answer_text"] = None
            st.session_state[f"{state_prefix}_answer_error"] = None

        render_ai_studio_header(section_title, selected_cell_id, display_week_df, spatial_cluster_summary)
        summary_col, explain_col, model_col = st.columns([1.1, 1.1, 1.4])
        with summary_col:
            if st.button("Create executive briefing", use_container_width=True, key=f"summary_btn_{section_key}"):
                summary_text, summary_error = generate_copilot_summary(copilot_context)
                st.session_state[f"{state_prefix}_summary_text"] = summary_text
                st.session_state[f"{state_prefix}_summary_error"] = summary_error
        with explain_col:
            if st.button("Explain selected cell", use_container_width=True, disabled=selected_cell_context is None, key=f"explain_btn_{section_key}"):
                explain_text, explain_error = generate_copilot_response(
                    context={
                        "copilot_context": copilot_context,
                        "focus_cell_context": selected_cell_context,
                    },
                    task_prompt=(
                        "Explain the selected focus cell. "
                        "Use short markdown headings. "
                        "Cover predicted class, strongest evidence, historical context, previous-live context if available, and caution."
                    ),
                    max_output_tokens=550,
                )
                st.session_state[f"{state_prefix}_cell_text"] = explain_text
                st.session_state[f"{state_prefix}_cell_error"] = explain_error
        with model_col:
            st.markdown(
                f"""
                <div class="ai-result-title">
                    Model ready: {html.escape(DEFAULT_COPILOT_MODEL)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        ask_col, ask_button_col = st.columns([3, 1])
        with ask_col:
            user_question = st.text_input("Direct question", value="What changed from the previous monitoring week?", key=f"question_{section_key}")
        with ask_button_col:
            st.write("")
            st.write("")
            if st.button("Ask AI", key=f"ask_btn_{section_key}", use_container_width=True):
                answer_text, answer_error = generate_copilot_response(
                    context={
                        "copilot_context": copilot_context,
                        "focus_cell_context": selected_cell_context,
                        "user_question": user_question,
                    },
                    task_prompt=f"Answer this user question from the supplied irrigation context: {user_question}",
                    max_output_tokens=500,
                )
                st.session_state[f"{state_prefix}_answer_text"] = answer_text
                st.session_state[f"{state_prefix}_answer_error"] = answer_error

        if st.session_state.get(f"{state_prefix}_summary_error"):
            st.warning(st.session_state[f"{state_prefix}_summary_error"])
        elif st.session_state.get(f"{state_prefix}_summary_text"):
            st.markdown('<div class="ai-result-title">Executive briefing</div>', unsafe_allow_html=True)
            st.markdown(st.session_state[f"{state_prefix}_summary_text"])

        if st.session_state.get(f"{state_prefix}_cell_error"):
            st.warning(st.session_state[f"{state_prefix}_cell_error"])
        elif st.session_state.get(f"{state_prefix}_cell_text"):
            st.markdown('<div class="ai-result-title">Selected cell analysis</div>', unsafe_allow_html=True)
            st.markdown(st.session_state[f"{state_prefix}_cell_text"])

        if st.session_state.get(f"{state_prefix}_answer_error"):
            st.warning(st.session_state[f"{state_prefix}_answer_error"])
        elif st.session_state.get(f"{state_prefix}_answer_text"):
            st.markdown('<div class="ai-result-title">AI answer</div>', unsafe_allow_html=True)
            st.markdown(st.session_state[f"{state_prefix}_answer_text"])

        with st.expander("Copilot context snapshot"):
            st.json(copilot_context)

        if selected_cell_context is not None:
            with st.expander("Focus cell context"):
                st.json(selected_cell_context)

        if spatial_cluster_summary and spatial_cluster_summary.get("clusters"):
            st.subheader("Priority Zones")
            cluster_rows = []
            for cluster in spatial_cluster_summary.get("clusters", []):
                cluster_rows.append(
                    {
                        "cluster_id": cluster.get("cluster_id"),
                        "boundary_name": cluster.get("boundary_name"),
                        "cell_count": cluster.get("cell_count"),
                        "dominant_class": cluster.get("dominant_class"),
                        "dominant_crop_type": cluster.get("dominant_crop_type"),
                        "mean_need_probability": cluster.get("mean_need_probability"),
                        "max_need_probability": cluster.get("max_need_probability"),
                        "center_lat": cluster.get("center_lat"),
                        "center_lon": cluster.get("center_lon"),
                    }
                )
            st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)

        render_map(
            grid=grid,
            boundary=boundary,
            week_df=section_df,
            show_boundary_labels=show_boundary_labels,
            show_gray_cells=show_gray_cells,
            map_key=f"irrigation_map_{section_key}",
        )

    if mode == "live":
        st.header("Current Monitoring")
        overall_tab, annual_tab, perennial_tab = st.tabs(["Overall", "Annual crops", "Perennial crops"])

        with overall_tab:
            render_live_or_historical_section(
                section_df=df_week,
                section_title="Current overview",
                section_key="live_overall",
                previous_section_df=display_prev_live_df,
            )

        with annual_tab:
            annual_df = df_week[df_week.get("crop_type", "").astype(str).str.upper() == "ANNUAL"].copy()
            prev_annual_df = display_prev_live_df[display_prev_live_df.get("crop_type", "").astype(str).str.upper() == "ANNUAL"].copy() if not display_prev_live_df.empty else pd.DataFrame()
            render_live_or_historical_section(
                section_df=annual_df,
                section_title="Annual crop priority",
                section_key="live_annual",
                previous_section_df=prev_annual_df,
            )

        with perennial_tab:
            perennial_df = df_week[df_week.get("crop_type", "").astype(str).str.upper() == "PERENNIAL"].copy()
            prev_perennial_df = display_prev_live_df[display_prev_live_df.get("crop_type", "").astype(str).str.upper() == "PERENNIAL"].copy() if not display_prev_live_df.empty else pd.DataFrame()
            render_live_or_historical_section(
                section_df=perennial_df,
                section_title="Perennial crop priority",
                section_key="live_perennial",
                previous_section_df=prev_perennial_df,
            )
    else:
        render_live_or_historical_section(
            section_df=df_week,
            section_title=title_suffix,
            section_key="historical_main",
            previous_section_df=display_prev_live_df,
        )


main = main_with_valid_display


if __name__ == "__main__":
    main_with_valid_display()
