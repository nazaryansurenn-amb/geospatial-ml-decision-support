from __future__ import annotations
import os
from pathlib import Path
import json
import html
import sys


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

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import Element
from streamlit_folium import st_folium
from shapely.geometry import Point

from core.ai import (
    COPILOT_GUIDANCE as AI_COPILOT_GUIDANCE,
    DEFAULT_COPILOT_MODEL as AI_COPILOT_MODEL,
    generate_copilot_response as ai_generate_copilot_response,
    generate_copilot_summary as ai_generate_copilot_summary,
    generate_next_week_outlook as ai_generate_next_week_outlook,
)
from core.crop import build_stable_crop_map, classify_crop_year
from core.features import build_reference_monthly_baselines, build_temporal_features
from core.forecast import build_sampling_grid, fetch_area_forecast
from core.i18n import language_options, translate
from core.knowledge import knowledge_sources as ai_knowledge_sources
from core.io import (
    assemble_live_season,
    load_exports,
    select_latest_week,
    select_previous_week,
)
from core.grid import load_coords, build_grid, load_boundary
from core.next_week import build_forecast_features, load_forecast_bundle, predict_next_week
from core.rules import CLASS_ORDER, DEFAULT_RULE_PARAMS, RULE_VERSION, assign_rule_classes, rule_priority_score
from core.runtime_artifacts import (
    DEFAULT_CROP_THRESHOLDS,
    REFERENCE_COLUMNS,
    artifact_paths,
    historical_crop_map_path,
    historical_features_path,
    load_runtime_manifest,
)

APP_TITLE = "Echmiadzin Irrigation Monitor"
APP_SUBTITLE = "Weekly satellite monitoring of vegetation condition and irrigation priorities"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Standalone demo modules import `app` for the production data pipeline. When
# Streamlit executes this file as its entry point, expose the running module
# under that stable name so embedded pages reuse it instead of executing twice.
sys.modules.setdefault("app", sys.modules[__name__])

PRODUCT_VIEWS = (
    "pressure_map",
    "next_week_forecast",
    "operations_ai",
    "season_timelapse",
    "weekly_change",
)
PRODUCT_VIEW_LABEL_KEYS = {
    "pressure_map": "tool_pressure_map",
    "next_week_forecast": "tool_next_week_forecast",
    "operations_ai": "tool_operations_ai",
    "season_timelapse": "tool_season_timelapse",
    "weekly_change": "tool_weekly_change",
}

DATA_DIR = ROOT / "data"
EXPORTS_DIR = DATA_DIR / "weekly_exports"
LIVE_EXPORTS_DIR = DATA_DIR / "live_exports"
RUNTIME_DIR = DATA_DIR / "runtime"
COORDS_CSV = DATA_DIR / "Echmiadzin_Cell_Coordinates.csv"
BOUNDARY_SHP = ROOT / "boundary" / "Echmiadzin_wua_Boundary.shp"

MODELS_DIR = ROOT / "models"

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri"
CARTO_LIGHT_TILES = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "OpenStreetMap contributors, CARTO"
SHOW_INTERNAL_DIAGNOSTICS = os.getenv("SHOW_INTERNAL_DIAGNOSTICS", "0") == "1"
PUBLIC_APP_MODE = os.getenv("PUBLIC_APP_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
PUBLIC_LANGUAGE_CODES = tuple(
    code.strip().lower()
    for code in os.getenv("PUBLIC_LANGUAGES", "en,ru").split(",")
    if code.strip()
)
GRAY = "#BDBDBD"


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
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #f8fbfe;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #20323e !important;
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
            background: linear-gradient(135deg, #0f1f33 0%, #153f50 52%, #1d6c65 100%);
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
            color: #ffffff !important;
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
            line-height: 1.2;
            min-height: 2.4em;
            white-space: normal;
            overflow-wrap: anywhere;
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


def render_dashboard_header(language: str = "en"):
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{html.escape(translate("app_title", language))}</h1>
            <p>{html.escape(translate("app_subtitle", language))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_studio_header(section_title, selected_cell_id, display_week_df, spatial_cluster_summary, language: str = "en"):
    high_priority_cells = 0
    if not display_week_df.empty and "need_class_ml" in display_week_df.columns:
        high_priority_cells = int(display_week_df["need_class_ml"].astype(str).str.upper().eq("NOW").sum())
    if not display_week_df.empty and "need_prob_ml" in display_week_df.columns:
        top_priority = _safe_num(display_week_df["need_prob_ml"]).max()
        top_priority_text = f"{top_priority:.2f}" if pd.notna(top_priority) else "n/a"
    else:
        top_priority_text = "n/a"
    if PUBLIC_APP_MODE:
        focus_label = translate("valid_vegetation", language)
        focus_text = f"{int((_safe_num(display_week_df.get('veg_flag')).fillna(0) == 1).sum()):,}" if "veg_flag" in display_week_df.columns else "0"
    else:
        focus_label = translate("focus_cell", language)
        focus_text = selected_cell_id if selected_cell_id is not None else translate("not_selected", language)
    st.markdown(
        f"""
        <div class="ai-studio">
            <div class="ai-kicker">{html.escape(translate("ai_studio", language))}</div>
            <h2>{html.escape(translate("executive_field_briefing", language))}</h2>
            <p>{html.escape(str(section_title))}</p>
            <div class="ai-grid">
                <div class="ai-stat"><span>{html.escape(translate("view", language))}</span><strong>{html.escape(str(section_title))}</strong></div>
                <div class="ai-stat"><span>{html.escape(focus_label)}</span><strong>{html.escape(str(focus_text))}</strong></div>
                <div class="ai-stat"><span>{html.escape(translate("high_cells", language))}</span><strong>{high_priority_cells:,}</strong></div>
                <div class="ai-stat"><span>{html.escape(translate("top_score", language))}</span><strong>{html.escape(top_priority_text)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def render_approved_pressure_surface(week_df: pd.DataFrame, boundary, language: str, map_key: str) -> None:
    import pressure_surface_demo as surface_demo
    from pressure_surface_vegetation_demo import build_vegetation_surface

    if week_df is None or week_df.empty:
        st.info(translate("map_no_rows", language))
        return
    try:
        surface = build_vegetation_surface(week_df, boundary)
    except Exception:
        st.info(translate("map_no_matches", language))
        return

    show_grid = False
    if not PUBLIC_APP_MODE:
        show_grid = st.toggle(
            surface_demo.t(language, "show_grid"),
            value=False,
            key=f"approved_grid_{map_key}",
        )
    irrigation_map = surface_demo.build_map(surface, boundary, language, show_grid)
    st_folium(
        irrigation_map,
        width=None,
        height=760,
        key=f"approved_pressure_{map_key}_{language}_{show_grid}",
        returned_objects=[],
    )


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


@st.cache_resource
def load_forecast_artifacts():
    """Next-week forecast model and its training metadata, or None when not trained yet."""
    return load_forecast_bundle(MODELS_DIR)


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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_area_forecast_cached():
    coordinates = load_coords(COORDS_CSV)
    points = build_sampling_grid(coordinates, rows=3, columns=3)
    return fetch_area_forecast(points)


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


def _prepare_runtime_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("week_start", "week_end"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if "cell_id" in out.columns:
        out["cell_id"] = _normalize_cell_id_series(out["cell_id"])
    return out


@st.cache_data(show_spinner=False)
def load_runtime_manifest_cached():
    return load_runtime_manifest(RUNTIME_DIR)


@st.cache_data(show_spinner=False)
def load_historical_years_cached() -> tuple[int, ...]:
    manifest = load_runtime_manifest_cached()
    years = tuple(int(year) for year in manifest.get("years", []))
    if years:
        return years
    exports = load_all_exports_cached()
    return tuple(sorted(exports["year"].dropna().astype(int).unique().tolist()))


@st.cache_data(show_spinner="Loading compact historical context...")
def load_historical_reference_cached(history_years: tuple[int, ...] = ()) -> pd.DataFrame:
    reference_path = artifact_paths(RUNTIME_DIR)["historical_reference"]
    if reference_path.exists():
        reference = _prepare_runtime_frame(pd.read_parquet(reference_path))
    else:
        exports = load_all_exports_cached()
        reference = exports[[column for column in REFERENCE_COLUMNS if column in exports.columns]].copy()
    if history_years and "year" in reference.columns:
        allowed_years = {int(year) for year in history_years}
        reference = reference[pd.to_numeric(reference["year"], errors="coerce").isin(allowed_years)].copy()
    return reference


@st.cache_data(show_spinner=False)
def load_monthly_baselines_cached(history_years: tuple[int, ...]) -> pd.DataFrame:
    manifest = load_runtime_manifest_cached()
    manifest_years = tuple(int(year) for year in manifest.get("years", []))
    baseline_path = artifact_paths(RUNTIME_DIR)["monthly_baselines"]
    if baseline_path.exists() and tuple(sorted(history_years)) == tuple(sorted(manifest_years)):
        return _prepare_runtime_frame(pd.read_parquet(baseline_path))
    reference = load_historical_reference_cached(history_years)
    return build_reference_monthly_baselines(reference)


@st.cache_data(show_spinner="Loading live season...")
def load_live_season_cached(live_signature=None):
    season, report = assemble_live_season(LIVE_EXPORTS_DIR)
    if report.errors:
        raise ValueError("; ".join(report.messages("error")))
    return season, report


@st.cache_data(show_spinner="Preparing selected historical year...")
def prepare_year_data_cached(year: int):
    # Rules run on the whole period: NOW needs the cell's previous week.
    runtime_path = historical_features_path(RUNTIME_DIR, year)
    if runtime_path.exists():
        return assign_rule_classes(_prepare_runtime_frame(pd.read_parquet(runtime_path)))
    exports = load_all_exports_cached()
    featured = build_temporal_features(exports)
    return assign_rule_classes(featured[featured["year"].astype(int) == int(year)].copy())


@st.cache_data(show_spinner="Preparing live features and history context...")
def prepare_live_data_cached(history_years: tuple[int, ...], live_signature=None):
    live_season, _ = load_live_season_cached(live_signature)
    if live_season is None or live_season.empty:
        return pd.DataFrame()
    baselines = load_monthly_baselines_cached(history_years)
    return assign_rule_classes(build_temporal_features(live_season, history_monthly_baselines=baselines))


@st.cache_data(show_spinner="Preparing the next-week forecast...")
def load_next_week_forecast_cached(
    history_years: tuple[int, ...],
    live_signature=None,
    ndvi_green_thr: float = 0.30,
    perennial_green_weeks: int = 10,
    perennial_p25: float = 0.25,
) -> pd.DataFrame:
    """Forecast for the week after the latest live report, one row per vegetated cell."""
    bundle = load_forecast_artifacts()
    if bundle is None:
        return pd.DataFrame()
    season = prepare_live_data_cached(history_years, live_signature)
    if season.empty:
        return pd.DataFrame()
    crop_map = build_live_crop_map_cached(history_years, ndvi_green_thr, perennial_green_weeks, perennial_p25)
    features = build_forecast_features(season, crop_map)
    latest = features[features["week_start"] == features["week_start"].max()]
    return predict_next_week(latest, bundle)


def _uses_default_crop_thresholds(ndvi_green_thr: float, perennial_green_weeks: int, perennial_p25: float) -> bool:
    return (
        np.isclose(float(ndvi_green_thr), DEFAULT_CROP_THRESHOLDS["ndvi_green_thr"])
        and int(perennial_green_weeks) == DEFAULT_CROP_THRESHOLDS["perennial_green_weeks"]
        and np.isclose(float(perennial_p25), DEFAULT_CROP_THRESHOLDS["perennial_p25"])
    )


@st.cache_data(show_spinner="Building crop map...")
def build_crop_map_cached(df_year: pd.DataFrame, ndvi_green_thr: float, perennial_green_weeks: int, perennial_p25: float):
    years = pd.to_numeric(df_year.get("year"), errors="coerce").dropna().astype(int).unique().tolist()
    if len(years) == 1 and _uses_default_crop_thresholds(ndvi_green_thr, perennial_green_weeks, perennial_p25):
        runtime_path = historical_crop_map_path(RUNTIME_DIR, years[0])
        if runtime_path.exists():
            return _prepare_runtime_frame(pd.read_parquet(runtime_path))
    return classify_crop_year(
        df_year,
        ndvi_green_thr=float(ndvi_green_thr),
        perennial_green_weeks=int(perennial_green_weeks),
        perennial_p25=float(perennial_p25),
    )


@st.cache_data(show_spinner="Building historical perennial map for live mode...")
def build_live_crop_map_cached(history_years: tuple[int, ...], ndvi_green_thr: float, perennial_green_weeks: int, perennial_p25: float):
    manifest = load_runtime_manifest_cached()
    manifest_years = tuple(int(year) for year in manifest.get("years", []))
    stable_path = artifact_paths(RUNTIME_DIR)["stable_crop_map"]
    if (
        stable_path.exists()
        and tuple(sorted(history_years)) == tuple(sorted(manifest_years))
        and _uses_default_crop_thresholds(ndvi_green_thr, perennial_green_weeks, perennial_p25)
    ):
        return _prepare_runtime_frame(pd.read_parquet(stable_path))
    reference = load_historical_reference_cached(history_years)
    if reference is None or reference.empty:
        return pd.DataFrame(columns=["cell_id", "crop_type"])
    return build_stable_crop_map(
        history=reference,
        ndvi_green_thr=float(ndvi_green_thr),
        perennial_green_weeks=int(perennial_green_weeks),
        perennial_p25=float(perennial_p25),
    )


def add_priority_predictions(df_week: pd.DataFrame, crop_map: pd.DataFrame):
    out = df_week.copy()
    out["cell_id"] = out["cell_id"].astype(str)

    cm = crop_map.copy()
    cm["cell_id"] = cm["cell_id"].astype(str)
    if "crop_type" in out.columns:
        out = out.drop(columns=["crop_type"])
    out = out.merge(cm, on="cell_id", how="left")
    out["crop_type"] = out["crop_type"].fillna("ANNUAL")

    # Frames from prepare_*_data_cached already carry rule classes for the whole period.
    # Classifying a single week here cannot see the previous week, so NOW falls back to SOON.
    if "rule_class" not in out.columns:
        out = assign_rule_classes(out)

    # need_class_ml / need_prob_ml / p_* keep their historical names for the map, zone and
    # AI code; they now carry the rule classes (core/rules.py) instead of a model output.
    out["need_class_ml"] = out["rule_class"].astype(object)
    for class_name in CLASS_ORDER:
        out[f"p_{class_name}"] = out["rule_class"].eq(class_name).astype(float)
    out["need_prob_ml"] = rule_priority_score(out)
    return out


def add_boundary(m, boundary_gdf, label_col="name", show_labels=True, layer_name="WUA Boundary", show_layer=True):
    if boundary_gdf is None or boundary_gdf.empty:
        return
    def style_fn(_):
        return {"color": "#247a83", "weight": 1.0, "opacity": 0.62, "fillOpacity": 0}

    tooltip = folium.GeoJsonTooltip(fields=[label_col]) if label_col in boundary_gdf.columns else None
    folium.GeoJson(
        boundary_gdf,
        name=layer_name,
        style_function=style_fn,
        tooltip=tooltip,
        show=show_layer,
    ).add_to(m)

    if show_labels and label_col in boundary_gdf.columns:
        for _, r in boundary_gdf.iterrows():
            c = r.geometry.centroid
            folium.Marker(
                location=(c.y, c.x),
                icon=folium.DivIcon(html=f"<div style='font-size:11px; font-weight:650; color:#16616d; white-space:nowrap; text-shadow:0 1px 2px #fff;'>{html.escape(str(r[label_col]))}</div>"),
            ).add_to(m)


WAIT_COLOR = "#3c9874"
SOON_COLOR = "#f2a93b"
NOW_COLOR = "#cf3f36"


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


def add_legend(m, palette_mode: str, language: str = "en"):
    legend_html = f"""
    <div style="position:fixed;bottom:24px;left:20px;z-index:9999;background:rgba(255,255,255,0.96);padding:10px 12px;border-radius:6px;border:1px solid #d7dee5;box-shadow:0 3px 12px rgba(20,40,60,0.14);min-width:176px;">
      <div style="font-weight:700;margin-bottom:7px;font-size:12px;color:#243746;">{html.escape(translate("legend_title", language))}</div>
      <div style="display:flex;align-items:center;font-size:11px;margin-bottom:5px;color:#34495e;"><div style="width:12px;height:12px;background:{NOW_COLOR};margin-right:7px;border-radius:2px;"></div><div>{html.escape(translate("class_now", language))}</div></div>
      <div style="display:flex;align-items:center;font-size:11px;margin-bottom:5px;color:#34495e;"><div style="width:12px;height:12px;background:{SOON_COLOR};margin-right:7px;border-radius:2px;"></div><div>{html.escape(translate("class_soon", language))}</div></div>
      <div style="display:flex;align-items:center;font-size:11px;color:#34495e;"><div style="width:12px;height:12px;background:{WAIT_COLOR};margin-right:7px;border-radius:2px;"></div><div>{html.escape(translate("class_wait", language))}</div></div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))


def build_area_priority_overview(merged, boundary, language: str = "en"):
    if boundary is None or boundary.empty or "name" not in boundary.columns or merged.empty:
        return gpd.GeoDataFrame()

    valid_cells = merged.loc[merged["_display_valid"]].copy()
    if valid_cells.empty:
        return gpd.GeoDataFrame()

    points = valid_cells[["cell_id", "need_class_ml", "need_prob_ml", "geometry"]].copy()
    points["geometry"] = points.geometry.map(lambda geometry: geometry.centroid if geometry is not None else None)
    areas = boundary[["name", "geometry"]].copy()
    joined = gpd.sjoin(points, areas, how="inner", predicate="within")
    if joined.empty:
        return gpd.GeoDataFrame()
    joined = joined.sort_values(["cell_id", "name"]).drop_duplicates("cell_id", keep="first")
    joined["need_class_ml"] = joined["need_class_ml"].astype(str).str.upper()
    joined["need_prob_ml"] = _safe_num(joined["need_prob_ml"])
    joined["_high"] = joined["need_class_ml"].eq("NOW").astype(int)
    joined["_soon"] = joined["need_class_ml"].eq("SOON").astype(int)
    joined["_routine"] = joined["need_class_ml"].eq("WAIT").astype(int)

    summary = joined.groupby("name", as_index=False).agg(
        _cell_count=("cell_id", "nunique"),
        _high_cells=("_high", "sum"),
        _soon_cells=("_soon", "sum"),
        _routine_cells=("_routine", "sum"),
        _mean_priority=("need_prob_ml", "mean"),
    )
    count_columns = ["_high_cells", "_soon_cells", "_routine_cells"]
    dominant_indices = summary[count_columns].to_numpy().argmax(axis=1)
    summary["_area_class"] = np.array(["NOW", "SOON", "WAIT"])[dominant_indices]
    summary["_area_name"] = summary["name"].astype(str)
    summary["_area_priority"] = summary["_area_class"].map({
        "NOW": translate("class_now", language),
        "SOON": translate("class_soon", language),
        "WAIT": translate("class_wait", language),
    })
    summary["_mean_priority"] = summary["_mean_priority"].round(2)
    return areas.merge(summary, on="name", how="left")


def render_map(
    grid,
    boundary,
    week_df,
    show_boundary_labels: bool,
    zero_to_gray_eps: float,
    palette_mode: str,
    show_gray_cells: bool,
    spatial_cluster_summary: dict | None = None,
    map_key: str = "irrigation_map",
    language: str = "en",
):
    if week_df is None or week_df.empty:
        st.warning(translate("map_no_rows", language))
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
        st.warning(translate("map_no_matches", language))
        return

    class_labels = {
        "WAIT": translate("class_wait", language),
        "SOON": translate("class_soon", language),
        "NOW": translate("class_now", language),
    }
    crop_labels = {
        "ANNUAL": translate("annual", language),
        "PERENNIAL": translate("perennial", language),
        "NON_AGRI": translate("non_agri", language),
    }
    if "need_class_ml" in merged.columns:
        merged["_priority_label"] = merged["need_class_ml"].astype(str).str.upper().map(class_labels)
    if "crop_type" in merged.columns:
        merged["_crop_label"] = merged["crop_type"].astype(str).str.upper().map(crop_labels)

    center_geometries = boundary.geometry if boundary is not None and not boundary.empty else merged.geometry
    cen = _geometry_center_point(center_geometries)
    if cen is None:
        st.warning(translate("map_no_center", language))
        return
    center = (float(cen.y), float(cen.x))

    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles=None, prefer_canvas=True)
    folium.TileLayer(
        tiles=CARTO_LIGHT_TILES,
        attr=CARTO_ATTR,
        name=translate("map_clean", language),
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles=ESRI_TILES,
        attr=ESRI_ATTR,
        name=translate("map_satellite", language),
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)
    add_legend(m, palette_mode, language)

    def detailed_style_fn(feat):
        props = feat.get("properties", {})
        if not bool(props.get("_display_valid", False)):
            return {"fillColor": GRAY, "color": "#ffffff", "weight": 0.2, "fillOpacity": 0.12}
        need_class = str(props.get("need_class_ml", "")).upper()
        color = {"WAIT": WAIT_COLOR, "SOON": SOON_COLOR, "NOW": NOW_COLOR}.get(need_class, GRAY)
        opacity = {"WAIT": 0.28, "SOON": 0.46, "NOW": 0.62}.get(need_class, 0.12)
        return {"fillColor": color, "color": "#ffffff", "weight": 0.25, "opacity": 0.45, "fillOpacity": opacity}

    def priority_style_fn(feat):
        need_class = str(feat.get("properties", {}).get("need_class_ml", "")).upper()
        color = NOW_COLOR if need_class == "NOW" else SOON_COLOR
        return {
            "fillColor": color,
            "color": color,
            "weight": 0,
            "fillOpacity": 0.72 if need_class == "NOW" else 0.48,
        }

    def highlight_fn(_):
        return {"color": "#ffffff", "weight": 1.8, "opacity": 1, "fillOpacity": 0.78}

    area_overview = build_area_priority_overview(merged, boundary, language)
    if not area_overview.empty:
        def area_style_fn(feat):
            area_class = str(feat.get("properties", {}).get("_area_class", "")).upper()
            color = {"WAIT": WAIT_COLOR, "SOON": SOON_COLOR, "NOW": NOW_COLOR}.get(area_class, GRAY)
            has_result = area_class in {"WAIT", "SOON", "NOW"}
            return {
                "fillColor": color,
                "color": "#ffffff",
                "weight": 1.2,
                "opacity": 0.95,
                "fillOpacity": 0.62 if has_result else 0.16,
            }

        folium.GeoJson(
            area_overview,
            name=translate("map_area_overview", language),
            style_function=area_style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["_area_name", "_area_priority", "_high_cells", "_soon_cells", "_routine_cells", "_mean_priority"],
                aliases=[
                    f"{translate('area', language)}:",
                    f"{translate('priority_class', language)}:",
                    f"{translate('high_cells', language)}:",
                    f"{translate('soon_cells', language)}:",
                    f"{translate('routine_cells', language)}:",
                    f"{translate('mean_score', language)}:",
                ],
                sticky=True,
            ),
            show=True,
        ).add_to(m)

    tooltip_fields = ["cell_id"]
    tooltip_aliases = [f"{translate('cell_id', language)}:"]
    for c, a in [("_crop_label", f"{translate('crop_type', language)}:"), ("_priority_label", f"{translate('priority_class', language)}:"), ("need_prob_ml", f"{translate('priority_score', language)}:"), ("NDVI_med", "NDVI:"), ("NDMI_med", "NDMI:"), ("dNDMI", "dNDMI:"), ("Rain_mm_7d", "Rain 7d:"), ("Rain2w", "Rain 2w:"), ("Rain3w", "Rain 3w:"), ("Temp_C_7d", "Temp 7d:")]:
        if c in merged.columns:
            tooltip_fields.append(c)
            tooltip_aliases.append(a)

    def make_tooltip():
        if merged.empty or not tooltip_fields:
            return None
        return folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True)

    need_probability = _safe_num(merged.get("need_prob_ml", pd.Series(index=merged.index, dtype=float))).fillna(0)
    priority_rows = merged[
        merged["_display_valid"]
        & merged.get("need_class_ml", pd.Series(index=merged.index, dtype=object)).astype(str).isin(["NOW", "SOON"])
        & need_probability.gt(float(zero_to_gray_eps))
    ].copy()
    high_priority_rows = priority_rows[priority_rows["need_class_ml"].astype(str).eq("NOW")].copy()
    attention_rows = priority_rows[priority_rows["need_class_ml"].astype(str).eq("SOON")].copy()
    if not high_priority_rows.empty:
        folium.GeoJson(
            high_priority_rows,
            name=translate("map_high_priority", language),
            style_function=priority_style_fn,
            highlight_function=highlight_fn,
            tooltip=make_tooltip(),
            show=False,
        ).add_to(m)
    if not attention_rows.empty:
        folium.GeoJson(
            attention_rows,
            name=translate("map_attention_soon", language),
            style_function=priority_style_fn,
            highlight_function=highlight_fn,
            tooltip=make_tooltip(),
            show=False,
        ).add_to(m)

    folium.GeoJson(
        merged,
        name=translate("map_cell_detail", language),
        style_function=detailed_style_fn,
        highlight_function=highlight_fn,
        tooltip=make_tooltip(),
        show=False,
    ).add_to(m)

    if spatial_cluster_summary and spatial_cluster_summary.get("clusters"):
        marker_layer = folium.FeatureGroup(name=translate("map_zone_markers", language), show=False)
        for rank, cluster in enumerate(spatial_cluster_summary["clusters"], start=1):
            lat = cluster.get("center_lat")
            lon = cluster.get("center_lon")
            if lat is None or lon is None:
                continue
            cluster_class = str(cluster.get("dominant_class", "SOON")).upper()
            marker_color = NOW_COLOR if cluster_class == "NOW" else SOON_COLOR
            area_name = cluster.get("boundary_name") or translate("not_selected", language)
            marker_html = (
                f"<div style='width:26px;height:26px;border-radius:50%;background:{marker_color};"
                "border:2px solid #fff;box-shadow:0 2px 7px rgba(20,40,60,.35);"
                "display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:750;'>"
                f"{rank}</div>"
            )
            tooltip_html = (
                f"<strong>{html.escape(str(area_name))}</strong><br>"
                f"{html.escape(translate('cells', language))}: {int(cluster.get('cell_count') or 0)}<br>"
                f"{html.escape(translate('priority_score', language))}: {float(cluster.get('max_need_probability') or 0):.2f}"
            )
            folium.Marker(
                location=(float(lat), float(lon)),
                icon=folium.DivIcon(html=marker_html, icon_size=(26, 26), icon_anchor=(13, 13)),
                tooltip=folium.Tooltip(tooltip_html),
            ).add_to(marker_layer)
        marker_layer.add_to(m)

    add_boundary(
        m,
        boundary,
        label_col="name",
        show_labels=show_boundary_labels,
        layer_name=translate("map_boundary", language),
        show_layer=False,
    )

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    st_folium(
        m,
        width=None,
        height=640,
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


def build_method_context(forecast_metadata: dict | None) -> dict:
    forecast = None
    if forecast_metadata:
        forecast = {
            "target": forecast_metadata.get("target"),
            "train_years": forecast_metadata.get("train_years", []),
            "uses_weather_forecast": forecast_metadata.get("uses_weather_forecast"),
            "baseline": forecast_metadata.get("baseline"),
            "validation": forecast_metadata.get("validation", {}),
        }
    return {
        "priority_method": "explicit rules on vegetated cells (NDVI >= 0.30)",
        "rule_version": RULE_VERSION,
        "rule_parameters": DEFAULT_RULE_PARAMS,
        "rule_summary": {
            "NOW": "2-week NDMI very dry, drying or below the cell's normal for the month, no rain relief, and SOON/NOW in the previous week",
            "SOON": "2-week NDMI dry without rain relief, very dry despite rain, or drying while below normal",
            "WAIT": "all other vegetated cells",
        },
        "next_week_forecast_model": forecast,
    }


def build_next_week_forecast_summary(forecast: pd.DataFrame | None, current_display: pd.DataFrame | None) -> dict | None:
    if forecast is None or forecast.empty:
        return None
    rows = forecast.copy()
    if current_display is not None and not current_display.empty and "cell_id" in current_display.columns:
        rows = rows[rows["cell_id"].astype(str).isin(set(current_display["cell_id"].astype(str)))]
    if rows.empty:
        return None

    basis_start = pd.to_datetime(rows["week_start"], errors="coerce").max()
    transitions = rows.groupby(["rule_class", "next_class"]).size()
    escalating = (
        rows[rows["rule_class"].ne("NOW")]
        .sort_values("p_next_NOW", ascending=False)
        .head(10)
    )
    return {
        "basis_week_start": basis_start.date().isoformat() if pd.notna(basis_start) else None,
        "forecast_week_start": (basis_start + pd.Timedelta(days=7)).date().isoformat() if pd.notna(basis_start) else None,
        "current_class_counts": {name: int(rows["rule_class"].eq(name).sum()) for name in CLASS_ORDER},
        "forecast_class_counts": {name: int(rows["next_class"].eq(name).sum()) for name in CLASS_ORDER},
        "transitions": {f"{a}->{b}": int(n) for (a, b), n in transitions.items() if a != b},
        "cells_most_likely_to_become_high_priority": [
            {"cell_id": str(r.cell_id), "current_class": r.rule_class, "probability_now_next_week": round(float(r.p_next_NOW), 3)}
            for r in escalating.itertuples(index=False)
        ],
        "note": "Model forecast of next week's rule class from this week's observations. It is not an observation.",
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
    next_week_forecast: dict | None = None,
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
        "project_guidance": AI_COPILOT_GUIDANCE,
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
        "next_week_forecast": next_week_forecast,
        "context_availability": {
            "has_previous_years_summary": previous_years_summary is not None,
            "has_live_comparison_summary": live_comparison_summary is not None,
            "has_spatial_cluster_summary": bool(spatial_cluster_summary and spatial_cluster_summary.get("clusters")),
            "has_next_week_forecast": next_week_forecast is not None,
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


def main():
    apply_dashboard_theme()
    language_choices = language_options(PUBLIC_LANGUAGE_CODES if PUBLIC_APP_MODE else None)
    if not language_choices:
        language_choices = language_options(("en",))
    language_name = st.sidebar.selectbox(
        "Language / Язык" if PUBLIC_APP_MODE else "Language / Լեզու / Язык",
        options=list(language_choices),
        index=0,
        key="interface_language",
    )
    language = language_choices[language_name]
    product_view = st.sidebar.radio(
        translate("tool_menu", language),
        options=PRODUCT_VIEWS,
        index=0,
        format_func=lambda value: translate(PRODUCT_VIEW_LABEL_KEYS[value], language),
        key="product_view",
    )

    if product_view == "pressure_map":
        from pressure_surface_vegetation_demo import render_page

        render_page(language)
        return
    if product_view == "next_week_forecast":
        from next_week_forecast_page import render_page

        render_page(language)
        return
    if product_view == "season_timelapse":
        from pressure_surface_timelapse_demo import render_page

        render_page(language)
        return
    if product_view == "weekly_change":
        from pressure_change_timelapse_demo import render_page

        render_page(language)
        return

    render_dashboard_header(language)

    try:
        grid, boundary = load_spatial()
    except Exception as e:
        st.error(str(e))
        return
    forecast_bundle = load_forecast_artifacts()

    live_signature = live_exports_signature()
    try:
        live_season_raw, live_report = load_live_season_cached(live_signature)
    except Exception:
        live_season_raw, live_report = pd.DataFrame(), None

    if PUBLIC_APP_MODE:
        mode = "live" if not live_season_raw.empty else "historical"
        ndvi_green_thr = 0.30
        perennial_green_weeks = 10
        perennial_p25 = 0.25
        palette_mode = "unified"
        show_gray_cells = False
        show_boundary_labels = False
        zero_to_gray_eps = 0.02
    else:
        with st.sidebar:
            st.header(translate("decision_controls", language))
            mode = st.radio(
                translate("data_view", language),
                options=["historical", "live"],
                index=1 if not live_season_raw.empty else 0,
                format_func=lambda value: translate("historical_baseline", language) if value == "historical" else translate("current_monitoring", language),
            )
            ndvi_green_thr = st.number_input(translate("green_cover_threshold", language), value=0.30, step=0.01, format="%.2f")
            perennial_green_weeks = st.slider(translate("perennial_green_weeks", language), 6, 20, 10, 1)
            perennial_p25 = st.number_input(translate("perennial_ndvi_threshold", language), value=0.25, step=0.01, format="%.2f")
            st.divider()
            palette_mode = st.radio(
                translate("map_palette", language),
                options=["unified", "split"],
                index=0,
                format_func=lambda value: translate("unified_priority_scale", language) if value == "unified" else translate("crop_specific_colors", language),
            )
            show_gray_cells = st.checkbox(translate("show_hidden_cells", language), value=False)
            show_boundary_labels = st.checkbox(translate("area_labels", language), value=False)
            zero_to_gray_eps = st.slider(translate("low_priority_cutoff", language), 0.0, 0.30, 0.02, 0.01)
            if SHOW_INTERNAL_DIAGNOSTICS and live_report is not None:
                for issue in live_report.warnings:
                    st.warning(translate("data_warning", language, message=issue.message))

    live_label = None
    reference_exports = pd.DataFrame()
    df_prev_live_scored = pd.DataFrame()
    weather_outlook = None
    next_week_forecast_df = pd.DataFrame()

    if mode == "historical":
        years = list(load_historical_years_cached())
        if not years:
            st.error(translate("no_historical_exports", language))
            return

        sel_year = st.selectbox(translate("year", language), years, index=0 if 2024 not in years else years.index(2024))
        df_year = prepare_year_data_cached(int(sel_year))
        week_rows = df_year[["week_start", "week_end"]].dropna(subset=["week_start"]).drop_duplicates().sort_values("week_start")
        if week_rows.empty:
            st.error(translate("no_weeks", language))
            return

        crop_map = build_crop_map_cached(df_year, float(ndvi_green_thr), int(perennial_green_weeks), float(perennial_p25))
        options = []
        for _, r in week_rows.iterrows():
            ws = pd.Timestamp(r["week_start"]).date()
            we = pd.Timestamp(r["week_end"]).date() if pd.notna(r.get("week_end")) else ws
            options.append((r["week_start"], f"{ws} -> {we}"))

        sel_week_start = st.selectbox(
            translate("week", language),
            options=[o[0] for o in options],
            format_func=lambda x: dict(options).get(x, str(x)),
            index=max(0, len(options) - 1),
        )
        df_week = df_year[df_year["week_start"] == sel_week_start].copy()
        prior_years = tuple(year for year in years if int(year) < int(sel_year))
        reference_exports = load_historical_reference_cached(prior_years) if prior_years else pd.DataFrame()
        title_suffix = f"{translate('historical_baseline', language)} - {sel_year}"
    else:
        if live_season_raw is None or live_season_raw.empty:
            st.error(translate("no_live_exports", language))
            return
        history_years = tuple(int(year) for year in load_historical_years_cached())
        df_live_season = prepare_live_data_cached(history_years, live_signature)
        crop_map = build_live_crop_map_cached(history_years, float(ndvi_green_thr), int(perennial_green_weeks), float(perennial_p25))
        df_week = select_latest_week(df_live_season)
        if history_years:
            reference_exports = load_historical_reference_cached(history_years)
        prev_live_temporal = select_previous_week(df_live_season)
        if not prev_live_temporal.empty:
            df_prev_live_scored = add_priority_predictions(prev_live_temporal, crop_map)
        source_names = sorted(df_week.get("source_file", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        live_label = ", ".join(source_names) if source_names else "weekly export"
        if PUBLIC_APP_MODE and "week_start" in df_week.columns and df_week["week_start"].notna().any():
            ws = pd.Timestamp(df_week["week_start"].dropna().min()).date()
            week_end = pd.to_datetime(df_week.get("week_end"), errors="coerce").dropna()
            we = week_end.max().date() if not week_end.empty else ws
            st.info(f"{translate('reporting_period', language)}: {ws} → {we}")
        elif "week_start" in df_week.columns and df_week["week_start"].notna().any():
            ws = pd.Timestamp(df_week["week_start"].dropna().iloc[0]).date()
            st.info(f"{translate('live_source', language)}: {live_label} - {ws}")
        elif not PUBLIC_APP_MODE:
            st.info(f"{translate('live_source', language)}: {live_label}")
        if not PUBLIC_APP_MODE:
            live_week_count = int(pd.to_datetime(df_live_season["week_start"], errors="coerce").nunique())
            live_rows_key = "live_rows_one" if live_week_count == 1 else "live_rows"
            st.caption(translate(live_rows_key, language, weeks=live_week_count, rows=len(df_live_season)))
        try:
            weather_outlook = load_area_forecast_cached()
        except Exception:
            weather_outlook = None
        try:
            next_week_forecast_df = load_next_week_forecast_cached(
                history_years, live_signature, float(ndvi_green_thr), int(perennial_green_weeks), float(perennial_p25)
            )
        except Exception:
            next_week_forecast_df = pd.DataFrame()
        title_suffix = translate("current_monitoring", language)

    df_week = add_priority_predictions(df_week, crop_map)
    display_prev_live_df = filter_display_rows(df_prev_live_scored) if not df_prev_live_scored.empty else pd.DataFrame()

    def render_live_or_historical_section(
        section_df: pd.DataFrame,
        section_title: str,
        section_key: str,
        previous_section_df: pd.DataFrame | None = None,
        show_pressure_map: bool = False,
    ):
        display_week_df = filter_display_rows(section_df)

        st.markdown(f'<div class="section-kicker">{html.escape(translate("operational_summary", language))}</div>', unsafe_allow_html=True)
        st.subheader(section_title)
        ct_counts = display_week_df["crop_type"].value_counts(dropna=False).to_dict() if not display_week_df.empty else {}
        vc = _safe_num(display_week_df.get("veg_flag")).value_counts(dropna=False).to_dict() if "veg_flag" in display_week_df.columns else {}
        nc = display_week_df["need_class_ml"].value_counts(dropna=False).to_dict() if not display_week_df.empty else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(translate("perennial_cells", language), f"{int(ct_counts.get('PERENNIAL', 0)):,}")
        c2.metric(translate("annual_cells", language), f"{int(ct_counts.get('ANNUAL', 0)):,}")
        c3.metric(translate("valid_vegetation", language), f"{int(vc.get(1, 0)):,}")
        c4.metric(translate("priority_counts", language), f"{int(nc.get('NOW', 0)):,} / {int(nc.get('SOON', 0)):,} / {int(nc.get('WAIT', 0)):,}")

        top_cols = [c for c in ["cell_id", "crop_type", "need_class_ml", "need_prob_ml", "NDVI_med", "NDMI_med", "dNDMI", "ndmi_drop_ytd", "Rain_mm_7d", "Rain2w", "Rain3w", "Temp_C_7d"] if c in section_df.columns]
        if display_week_df.empty:
            st.warning(translate("no_public_results", language))
        elif not PUBLIC_APP_MODE:
            st.markdown(f'<div class="section-kicker">{html.escape(translate("highest_priority_cells", language))}</div>', unsafe_allow_html=True)
            priority_table = display_week_df.sort_values("need_prob_ml", ascending=False)[top_cols].head(20).copy()
            if "crop_type" in priority_table.columns:
                priority_table["crop_type"] = priority_table["crop_type"].replace({
                    "ANNUAL": translate("annual", language),
                    "PERENNIAL": translate("perennial", language),
                    "NON_AGRI": translate("non_agri", language),
                })
            if "need_class_ml" in priority_table.columns:
                priority_table["need_class_ml"] = priority_table["need_class_ml"].replace({
                    "WAIT": translate("class_wait", language),
                    "SOON": translate("class_soon", language),
                    "NOW": translate("class_now", language),
                })
            priority_table = priority_table.rename(columns={
                "cell_id": translate("cell_id", language),
                "crop_type": translate("crop_type", language),
                "need_class_ml": translate("priority_class", language),
                "need_prob_ml": translate("priority_score", language),
            })
            st.dataframe(priority_table, use_container_width=True)

        ui_context = {"mode": mode, "title_suffix": section_title, "section_key": section_key}
        if not PUBLIC_APP_MODE:
            ui_context.update({
                "ndvi_green_threshold": float(ndvi_green_thr),
                "perennial_green_weeks_threshold": int(perennial_green_weeks),
                "perennial_ndvi_p25_threshold": float(perennial_p25),
                "palette_mode": palette_mode,
                "show_gray_cells": bool(show_gray_cells),
                "show_boundary_labels": bool(show_boundary_labels),
                "zero_to_gray_need_threshold": float(zero_to_gray_eps),
            })
        model_context = None if PUBLIC_APP_MODE else build_method_context((forecast_bundle or {}).get("metadata"))
        previous_years_summary = build_previous_years_summary(display_week_df, reference_exports)
        live_comparison_summary = build_live_comparison_summary(display_week_df, previous_section_df if previous_section_df is not None and not previous_section_df.empty else None)
        if PUBLIC_APP_MODE and live_comparison_summary:
            live_comparison_summary = dict(live_comparison_summary)
            live_comparison_summary.pop("current_live_file", None)
            live_comparison_summary.pop("previous_live_file", None)
        spatial_cluster_summary = build_spatial_cluster_summary(
            grid=grid,
            week_df=display_week_df,
            boundary=boundary,
            urgent_probability_threshold=max(0.5, float(zero_to_gray_eps)),
        )
        copilot_context = build_copilot_context(
            df_week=display_week_df,
            title_suffix=section_title,
            mode=mode,
            live_label=None if PUBLIC_APP_MODE else live_label,
            previous_years_summary=previous_years_summary,
            model_context=model_context,
            ui_context=ui_context,
            live_comparison_summary=live_comparison_summary,
            spatial_cluster_summary=spatial_cluster_summary,
            next_week_forecast=build_next_week_forecast_summary(next_week_forecast_df, display_week_df) if mode == "live" else None,
        )
        copilot_context["weather_forecast_7d"] = weather_outlook
        copilot_context["response_language"] = language

        focus_cell_options = []
        if "cell_id" in display_week_df.columns:
            focus_cell_options = display_week_df.sort_values("need_prob_ml", ascending=False)["cell_id"].astype(str).drop_duplicates().tolist()
        selected_cell_id = None
        selected_cell_context = None
        if focus_cell_options and not PUBLIC_APP_MODE:
            selected_cell_id = st.selectbox(translate("focus_cell", language), options=focus_cell_options, index=0, key=f"focus_cell_{section_key}")
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
                "language": language,
                "weather_area": (weather_outlook or {}).get("area") if isinstance(weather_outlook, dict) else None,
            },
            sort_keys=True,
        )
        if st.session_state.get(f"{state_prefix}_context_key") != context_key:
            st.session_state[f"{state_prefix}_context_key"] = context_key
            st.session_state[f"{state_prefix}_summary_text"] = None
            st.session_state[f"{state_prefix}_summary_error"] = None
            st.session_state[f"{state_prefix}_outlook_text"] = None
            st.session_state[f"{state_prefix}_outlook_error"] = None
            st.session_state[f"{state_prefix}_cell_text"] = None
            st.session_state[f"{state_prefix}_cell_error"] = None
            st.session_state[f"{state_prefix}_answer_text"] = None
            st.session_state[f"{state_prefix}_answer_error"] = None

        render_ai_studio_header(section_title, selected_cell_id, display_week_df, spatial_cluster_summary, language)
        action_columns = st.columns(2 if PUBLIC_APP_MODE else 3)
        summary_col, outlook_col = action_columns[:2]
        with summary_col:
            if st.button(translate("create_briefing", language), use_container_width=True, key=f"summary_btn_{section_key}"):
                summary_text, summary_error = ai_generate_copilot_summary(copilot_context, language=language)
                st.session_state[f"{state_prefix}_summary_text"] = summary_text
                st.session_state[f"{state_prefix}_summary_error"] = summary_error
        with outlook_col:
            if st.button(translate("next_week_outlook", language), use_container_width=True, key=f"outlook_btn_{section_key}"):
                outlook_text, outlook_error = ai_generate_next_week_outlook(copilot_context, language=language)
                st.session_state[f"{state_prefix}_outlook_text"] = outlook_text
                st.session_state[f"{state_prefix}_outlook_error"] = outlook_error
        if not PUBLIC_APP_MODE:
            explain_col = action_columns[2]
            with explain_col:
                if st.button(translate("explain_cell", language), use_container_width=True, disabled=selected_cell_context is None, key=f"explain_btn_{section_key}"):
                    explain_text, explain_error = ai_generate_copilot_response(
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
                        language=language,
                        kind="cell_explanation",
                    )
                    st.session_state[f"{state_prefix}_cell_text"] = explain_text
                    st.session_state[f"{state_prefix}_cell_error"] = explain_error

        st.caption(
            translate("grounded_analysis", language)
            if PUBLIC_APP_MODE
            else translate("model_ready", language, model=AI_COPILOT_MODEL)
        )
        with st.expander(translate("knowledge_basis", language)):
            st.caption(translate("knowledge_basis_caption", language))
            for source in ai_knowledge_sources():
                st.markdown(f"- [{source['title']}]({source['url']})")

        if not PUBLIC_APP_MODE:
            ask_col, ask_button_col = st.columns([3, 1])
            with ask_col:
                user_question = st.text_input(translate("direct_question", language), value=translate("default_question", language), key=f"question_{section_key}")
            with ask_button_col:
                st.write("")
                st.write("")
                if st.button(translate("ask_ai", language), key=f"ask_btn_{section_key}", use_container_width=True):
                    answer_text, answer_error = ai_generate_copilot_response(
                        context={
                            "copilot_context": copilot_context,
                            "focus_cell_context": selected_cell_context,
                            "user_question": user_question,
                        },
                        task_prompt=f"Answer this user question from the supplied irrigation context: {user_question}",
                        max_output_tokens=500,
                        language=language,
                        kind="direct_question",
                    )
                    st.session_state[f"{state_prefix}_answer_text"] = answer_text
                    st.session_state[f"{state_prefix}_answer_error"] = answer_error

        if st.session_state.get(f"{state_prefix}_summary_error"):
            st.warning(st.session_state[f"{state_prefix}_summary_error"])
        elif st.session_state.get(f"{state_prefix}_summary_text"):
            st.markdown(f'<div class="ai-result-title">{html.escape(translate("briefing_result", language))}</div>', unsafe_allow_html=True)
            st.markdown(st.session_state[f"{state_prefix}_summary_text"])

        if st.session_state.get(f"{state_prefix}_outlook_error"):
            st.warning(st.session_state[f"{state_prefix}_outlook_error"])
        elif st.session_state.get(f"{state_prefix}_outlook_text"):
            st.markdown(f'<div class="ai-result-title">{html.escape(translate("outlook_result", language))}</div>', unsafe_allow_html=True)
            st.markdown(st.session_state[f"{state_prefix}_outlook_text"])

        if st.session_state.get(f"{state_prefix}_cell_error"):
            st.warning(st.session_state[f"{state_prefix}_cell_error"])
        elif st.session_state.get(f"{state_prefix}_cell_text"):
            st.markdown(f'<div class="ai-result-title">{html.escape(translate("cell_result", language))}</div>', unsafe_allow_html=True)
            st.markdown(st.session_state[f"{state_prefix}_cell_text"])

        if st.session_state.get(f"{state_prefix}_answer_error"):
            st.warning(st.session_state[f"{state_prefix}_answer_error"])
        elif st.session_state.get(f"{state_prefix}_answer_text"):
            st.markdown(f'<div class="ai-result-title">{html.escape(translate("answer_result", language))}</div>', unsafe_allow_html=True)
            st.markdown(st.session_state[f"{state_prefix}_answer_text"])

        if SHOW_INTERNAL_DIAGNOSTICS:
            with st.expander(translate("context_snapshot", language)):
                st.json(copilot_context)
            if selected_cell_context is not None:
                with st.expander(translate("focus_context", language)):
                    st.json(selected_cell_context)

        if show_pressure_map:
            st.subheader(translate("tool_pressure_map", language))
            render_approved_pressure_surface(
                display_week_df,
                boundary,
                language,
                map_key=f"operations_{section_key}",
            )

    if mode == "live":
        if weather_outlook and weather_outlook.get("area"):
            area_weather = weather_outlook["area"]
            st.subheader(translate("weather_outlook", language))
            w1, w2, w3, w4, w5 = st.columns(5)
            w1.metric(translate("forecast_rain", language), f"{area_weather['rain_mm_7d']:.1f} mm")
            w2.metric(translate("forecast_et0", language), f"{area_weather['et0_mm_7d']:.1f} mm")
            w3.metric(translate("forecast_balance", language), f"{area_weather['rain_minus_et0_mm']:+.1f} mm")
            w4.metric(translate("forecast_temperature", language), f"{area_weather['temperature_max_mean_c']:.1f} °C")
            w5.metric(translate("forecast_wind", language), f"{area_weather['wind_max_kmh']:.1f} km/h")
            st.caption(f"{translate('forecast_points', language)}: {area_weather['point_count']}")
        else:
            st.info(translate("forecast_unavailable", language))

        st.header(translate("current_monitoring", language))
        if PUBLIC_APP_MODE:
            render_live_or_historical_section(
                section_df=df_week,
                section_title=translate("current_overview", language),
                section_key="live_overall",
                previous_section_df=display_prev_live_df,
                show_pressure_map=True,
            )
        else:
            overall_tab, annual_tab, perennial_tab = st.tabs([
                translate("overall", language),
                translate("annual_crops", language),
                translate("perennial_crops", language),
            ])

            with overall_tab:
                render_live_or_historical_section(
                    section_df=df_week,
                    section_title=translate("current_overview", language),
                    section_key="live_overall",
                    previous_section_df=display_prev_live_df,
                    show_pressure_map=True,
                )

            with annual_tab:
                annual_df = df_week[df_week.get("crop_type", "").astype(str).str.upper() == "ANNUAL"].copy()
                prev_annual_df = display_prev_live_df[display_prev_live_df.get("crop_type", "").astype(str).str.upper() == "ANNUAL"].copy() if not display_prev_live_df.empty else pd.DataFrame()
                render_live_or_historical_section(
                    section_df=annual_df,
                    section_title=translate("annual_priority", language),
                    section_key="live_annual",
                    previous_section_df=prev_annual_df,
                )

            with perennial_tab:
                perennial_df = df_week[df_week.get("crop_type", "").astype(str).str.upper() == "PERENNIAL"].copy()
                prev_perennial_df = display_prev_live_df[display_prev_live_df.get("crop_type", "").astype(str).str.upper() == "PERENNIAL"].copy() if not display_prev_live_df.empty else pd.DataFrame()
                render_live_or_historical_section(
                    section_df=perennial_df,
                    section_title=translate("perennial_priority", language),
                    section_key="live_perennial",
                    previous_section_df=prev_perennial_df,
                )
    else:
        render_live_or_historical_section(
            section_df=df_week,
            section_title=title_suffix,
            section_key="historical_main",
            previous_section_df=display_prev_live_df,
            show_pressure_map=True,
        )


if __name__ == "__main__":
    main()
