from __future__ import annotations

import html

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import streamlit as st
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
from streamlit_folium import st_folium

import app as production


TEXT = {
    "en": {
        "title": "Weekly irrigation priority surface",
        "subtitle": "A continuous view of the green, yellow, orange, and red priority mix.",
        "period": "Reporting period",
        "all_cells": "Model cells",
        "now": "High priority",
        "soon": "Watch closely",
        "wait": "Routine monitoring",
        "show_grid": "Show 250 m cell outlines",
        "surface": "Probability surface",
        "satellite": "Esri satellite imagery",
        "street": "Street map",
        "boundary": "WUA outline",
        "dominant": "Dominant grade",
        "pressure": "Irrigation priority",
        "now_share": "Red probability",
        "soon_share": "Yellow probability",
        "wait_share": "Green probability",
        "low": "Lower priority",
        "high": "Higher priority",
        "no_data": "The latest weekly probability surface could not be prepared.",
    },
    "hy": {
        "title": "Այս շաբաթվա ոռոգման առաջնահերթությունների քարտեզը",
        "subtitle": "Կանաչ, դեղին, նարնջագույն և կարմիր առաջնահերթությունների շարունակական պատկերը։",
        "period": "Հաշվետու ժամանակահատված",
        "all_cells": "Մոդելի բջիջներ",
        "now": "Բարձր առաջնահերթություն",
        "soon": "Հետևել առաջիկայում",
        "wait": "Շարունակել սովորական հետևումը",
        "show_grid": "Ցույց տալ 250 մ բջիջների սահմանները",
        "surface": "Հավանականությունների խառը պատկեր",
        "satellite": "Esri արբանյակային պատկեր",
        "street": "Փողոցային քարտեզ",
        "boundary": "ՋՕԸ-ի ընդհանուր սահմանը",
        "dominant": "Գերակշռող աստիճան",
        "pressure": "Ոռոգման առաջնահերթություն",
        "now_share": "Կարմիրի հավանականություն",
        "soon_share": "Դեղինի հավանականություն",
        "wait_share": "Կանաչի հավանականություն",
        "low": "Ավելի ցածր առաջնահերթություն",
        "high": "Ավելի բարձր առաջնահերթություն",
        "no_data": "Չհաջողվեց պատրաստել վերջին շաբաթվա հավանականությունների քարտեզը։",
    },
    "ru": {
        "title": "Недельная карта приоритетов орошения",
        "subtitle": "Непрерывная карта зелёного, жёлтого, оранжевого и красного уровней приоритета.",
        "period": "Отчётный период",
        "all_cells": "Ячейки модели",
        "now": "Высокий приоритет",
        "soon": "Наблюдать внимательно",
        "wait": "Обычное наблюдение",
        "show_grid": "Показывать границы ячеек 250 м",
        "surface": "Поверхность приоритетов",
        "satellite": "Спутниковые снимки Esri",
        "street": "Карта улиц",
        "boundary": "Граница АВП",
        "dominant": "Преобладающий уровень",
        "pressure": "Приоритет орошения",
        "now_share": "Вероятность красного уровня",
        "soon_share": "Вероятность жёлтого уровня",
        "wait_share": "Вероятность зелёного уровня",
        "low": "Ниже приоритет",
        "high": "Выше приоритет",
        "no_data": "Не удалось подготовить карту приоритетов за последнюю неделю.",
    },
}

GRID_CRS = "EPSG:3857"
GRID_M = 250
SMOOTH_SIGMA_CELLS = 1.0
SMOOTH_RADIUS_CELLS = 2.0

CLASS_ORDER = ("WAIT", "SOON", "NOW")
PROBABILITY_COLUMNS = ("p_WAIT", "p_SOON", "p_NOW")
CLASS_COLORS = {
    "WAIT": "#2f8f6b",
    "SOON": "#efbf32",
    "NOW": "#c9342f",
}
PRESSURE_COLORMAP = LinearColormap(
    colors=["#2f8f6b", "#9fbd45", "#efc23a", "#eb812f", "#c9342f"],
    index=[0.0, 0.28, 0.5, 0.74, 1.0],
    vmin=0.0,
    vmax=1.0,
)
COLOR_STOPS = np.array([0.0, 0.28, 0.5, 0.74, 1.0])
COLOR_RGB = np.array(
    [
        [47, 143, 107],
        [159, 189, 69],
        [239, 194, 58],
        [235, 129, 47],
        [201, 52, 47],
    ],
    dtype=float,
)


def t(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["en"])[key]


def _cell_coordinates(cell_id: object) -> tuple[int, int] | None:
    parts = str(cell_id).split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None


def _cell_polygon(cell_id: object):
    coordinate = _cell_coordinates(cell_id)
    if coordinate is None:
        return None
    x, y = coordinate
    return box(
        x * GRID_M,
        y * GRID_M,
        (x + 1) * GRID_M,
        (y + 1) * GRID_M,
    )


def ensure_class_probabilities(scored: pd.DataFrame) -> pd.DataFrame:
    result = scored.copy()
    for class_name, column in zip(CLASS_ORDER, PROBABILITY_COLUMNS):
        if column not in result.columns:
            result[column] = result["need_class_ml"].astype(str).str.upper().eq(class_name).astype(float)
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).clip(0, 1)

    probabilities = result[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    totals = probabilities.sum(axis=1)
    missing = totals <= 0
    if missing.any():
        classes = result["need_class_ml"].astype(str).str.upper().to_numpy()
        probabilities[missing] = np.column_stack(
            [classes[missing] == class_name for class_name in CLASS_ORDER]
        ).astype(float)
        totals = probabilities.sum(axis=1)
    probabilities = np.divide(
        probabilities,
        totals[:, None],
        out=np.full_like(probabilities, 1 / len(CLASS_ORDER)),
        where=totals[:, None] > 0,
    )
    result.loc[:, list(PROBABILITY_COLUMNS)] = probabilities
    return result


def smooth_class_probabilities(
    scored: pd.DataFrame,
    *,
    sigma_cells: float = SMOOTH_SIGMA_CELLS,
    radius_cells: float = SMOOTH_RADIUS_CELLS,
) -> pd.DataFrame:
    result = ensure_class_probabilities(scored)
    coordinates = result["cell_id"].map(_cell_coordinates)
    if coordinates.isna().any():
        bad_id = result.loc[coordinates.isna(), "cell_id"].iloc[0]
        raise ValueError(f"Cannot decode grid coordinates from cell_id {bad_id!r}")

    x_values = np.array([coordinate[0] for coordinate in coordinates], dtype=int)
    y_values = np.array([coordinate[1] for coordinate in coordinates], dtype=int)
    min_x, max_x = int(x_values.min()), int(x_values.max())
    min_y, max_y = int(y_values.min()), int(y_values.max())
    x_index = x_values - min_x
    y_index = y_values - min_y
    shape = (max_y - min_y + 1, max_x - min_x + 1)

    support = np.zeros(shape, dtype=float)
    support[y_index, x_index] = 1.0
    smoothed_support = gaussian_filter(
        support,
        sigma=float(sigma_cells),
        mode="constant",
        cval=0,
        truncate=float(radius_cells) / float(sigma_cells),
    )

    smoothed_columns = []
    for column in PROBABILITY_COLUMNS:
        values = np.zeros(shape, dtype=float)
        values[y_index, x_index] = result[column].to_numpy(dtype=float)
        weighted = gaussian_filter(
            values,
            sigma=float(sigma_cells),
            mode="constant",
            cval=0,
            truncate=float(radius_cells) / float(sigma_cells),
        )
        smoothed = np.divide(
            weighted,
            smoothed_support,
            out=np.zeros_like(weighted),
            where=smoothed_support > 0,
        )
        smoothed_columns.append(smoothed[y_index, x_index])

    probabilities = np.column_stack(smoothed_columns)
    totals = probabilities.sum(axis=1)
    probabilities = np.divide(
        probabilities,
        totals[:, None],
        out=np.full_like(probabilities, 1 / len(CLASS_ORDER)),
        where=totals[:, None] > 0,
    )
    for index, class_name in enumerate(CLASS_ORDER):
        result[f"smooth_p_{class_name}"] = probabilities[:, index]

    result["surface_pressure"] = (
        0.5 * result["smooth_p_SOON"] + result["smooth_p_NOW"]
    ).clip(0, 1)
    result["surface_class"] = np.array(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
    result["surface_color"] = result["surface_pressure"].map(pressure_color)
    return result


def pressure_color(value: float) -> str:
    return str(PRESSURE_COLORMAP(float(np.clip(value, 0, 1))))[:7]


def pressure_rgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0, 1)
    channels = [
        np.interp(clipped, COLOR_STOPS, COLOR_RGB[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def build_exact_surface_geometry(
    scored: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    surface = gpd.GeoDataFrame(
        scored.copy(),
        geometry=scored["cell_id"].map(_cell_polygon),
        crs=GRID_CRS,
    )
    if surface.geometry.isna().any():
        raise ValueError("One or more cell IDs could not be converted to grid polygons.")
    outline = shapely.union_all(boundary.to_crs(GRID_CRS).geometry.values)
    surface.geometry = surface.geometry.intersection(outline)
    surface = surface[~surface.geometry.is_empty].copy()
    return surface.to_crs("EPSG:4326")


def build_surface_image(
    surface: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    *,
    show_grid: bool,
    pixels_per_cell: int = 4,
) -> tuple[np.ndarray, list[list[float]]]:
    coordinates = surface["cell_id"].map(_cell_coordinates)
    x_values = np.array([coordinate[0] for coordinate in coordinates], dtype=int)
    y_values = np.array([coordinate[1] for coordinate in coordinates], dtype=int)
    min_x, max_x = int(x_values.min()), int(x_values.max())
    min_y, max_y = int(y_values.min()), int(y_values.max())

    width_cells = max_x - min_x + 1
    height_cells = max_y - min_y + 1
    rgba_cells = np.zeros((height_cells, width_cells, 4), dtype=np.uint8)
    row_index = max_y - y_values
    column_index = x_values - min_x
    rgba_cells[row_index, column_index, :3] = pressure_rgb(
        surface["surface_pressure"].to_numpy(dtype=float)
    )
    rgba_cells[row_index, column_index, 3] = 184

    rgba = np.repeat(
        np.repeat(rgba_cells, int(pixels_per_cell), axis=0),
        int(pixels_per_cell),
        axis=1,
    )
    min_easting = min_x * GRID_M
    max_northing = (max_y + 1) * GRID_M
    pixel_size = GRID_M / int(pixels_per_cell)
    transform = from_origin(min_easting, max_northing, pixel_size, pixel_size)
    outline = shapely.union_all(boundary.to_crs(GRID_CRS).geometry.values)
    boundary_mask = rasterize(
        [(outline, 1)],
        out_shape=rgba.shape[:2],
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    rgba[..., 3] = np.where(boundary_mask > 0, rgba[..., 3], 0)

    if show_grid:
        rows = np.arange(rgba.shape[0])[:, None] % int(pixels_per_cell) == 0
        columns = np.arange(rgba.shape[1])[None, :] % int(pixels_per_cell) == 0
        grid_lines = (rows | columns) & (rgba[..., 3] > 0)
        rgba[grid_lines, :3] = 255
        rgba[grid_lines, 3] = 120

    max_easting = (max_x + 1) * GRID_M
    min_northing = min_y * GRID_M
    to_wgs84 = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)
    west, south = to_wgs84.transform(min_easting, min_northing)
    east, north = to_wgs84.transform(max_easting, max_northing)
    return rgba, [[south, west], [north, east]]


@st.cache_data(show_spinner=False)
def load_current_scored_week(_signature):
    _, boundary = production.load_spatial()
    history_years = tuple(int(year) for year in production.load_historical_years_cached())

    live = production.prepare_live_data_cached(history_years, _signature)
    crop_map = production.build_live_crop_map_cached(history_years, 0.30, 10, 0.25)
    current = production.select_latest_week(live)
    scored = production.add_priority_predictions(current, crop_map)
    scored["cell_id"] = scored["cell_id"].astype(str)

    week_start = pd.to_datetime(scored.get("week_start"), errors="coerce").dropna()
    week_end = pd.to_datetime(scored.get("week_end"), errors="coerce").dropna()
    period = week_start.min().date().isoformat() if not week_start.empty else ""
    if not week_end.empty:
        period = f"{period} – {week_end.max().date().isoformat()}"
    return scored, boundary, period


@st.cache_data(show_spinner="Preparing the weekly probability surface...")
def load_surface_data(_signature):
    scored, boundary, period = load_current_scored_week(_signature)
    smoothed = smooth_class_probabilities(scored)
    surface = build_exact_surface_geometry(smoothed, boundary)
    return surface, boundary, period


def _add_pressure_legend(irrigation_map: folium.Map, language: str) -> None:
    title = html.escape(t(language, "pressure"))
    low = html.escape(t(language, "low"))
    high = html.escape(t(language, "high"))
    legend = MacroElement()
    legend._template = Template(
        "{% macro html(this, kwargs) %}"
        f"""
        <div style="position:fixed;bottom:58px;left:10px;z-index:9999;
          background:rgba(255,255,255,.96);padding:9px 11px;border-radius:6px;
          border:1px solid #d7dee2;box-shadow:0 3px 12px rgba(20,40,60,.14);
          width:220px;font-family:Inter,Arial,sans-serif;">
          <div style="font-weight:700;font-size:12px;color:#243746;margin-bottom:7px;">{title}</div>
          <div style="height:12px;border-radius:3px;background:linear-gradient(90deg,
            #2f8f6b 0%,#9fbd45 28%,#efc23a 50%,#eb812f 74%,#c9342f 100%);"></div>
          <div style="display:flex;justify-content:space-between;color:#5e7078;font-size:10px;margin-top:4px;">
            <span>{low}</span><span>{high}</span>
          </div>
        </div>
        """
        + "{% endmacro %}"
    )
    irrigation_map.get_root().add_child(legend)


def build_map(
    surface: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    language: str,
    show_grid: bool,
) -> folium.Map:
    outline = boundary.to_crs("EPSG:4326")
    bounds = outline.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    irrigation_map = folium.Map(
        location=center,
        zoom_start=11,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles=production.ESRI_TILES,
        attr=production.ESRI_ATTR,
        name=t(language, "satellite"),
        overlay=False,
        show=True,
    ).add_to(irrigation_map)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap contributors",
        name=t(language, "street"),
        overlay=False,
        show=False,
    ).add_to(irrigation_map)

    surface_image, image_bounds = build_surface_image(
        surface,
        boundary,
        show_grid=show_grid,
    )
    folium.raster_layers.ImageOverlay(
        image=surface_image,
        bounds=image_bounds,
        name=t(language, "surface"),
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=2,
    ).add_to(irrigation_map)
    folium.GeoJson(
        outline,
        name=t(language, "boundary"),
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "#385b66",
            "weight": 1.3,
            "opacity": 0.85,
            "fillOpacity": 0,
        },
        show=True,
    ).add_to(irrigation_map)

    irrigation_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]], padding=(12, 12))
    folium.LayerControl(collapsed=True, position="topright").add_to(irrigation_map)
    _add_pressure_legend(irrigation_map, language)
    return irrigation_map


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background:#f4f7f8; color:#152630; }
        .main .block-container { max-width:1480px; padding-top:1.2rem; padding-bottom:2rem; }
        header[data-testid="stHeader"] { background:rgba(244,247,248,.86); }
        .main div[role="radiogroup"] label p { color:#3f545d !important; font-weight:650; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label p { color:#d7e4f2 !important; font-weight:650; }
        .surface-kicker { color:#19705f; font-size:.78rem; font-weight:750; text-transform:uppercase; }
        .surface-title { color:#142a34; font-size:clamp(1.75rem,3vw,2.65rem); line-height:1.08; font-weight:760; margin:.35rem 0 .55rem; }
        .surface-subtitle { color:#52646e; font-size:1rem; line-height:1.55; max-width:900px; margin-bottom:.9rem; }
        .surface-period { color:#45606b; font-size:.86rem; margin:.1rem 0 1.1rem; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid #dce4e7; border-radius:6px; padding:.75rem .9rem; }
        div[data-testid="stMetricValue"] { color:#173a42; }
        @media (max-width:700px) {
          .main .block-container { padding-left:.8rem; padding-right:.8rem; }
          .surface-title { font-size:1.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()
    language_label = st.radio(
        "Language / Լեզու",
        options=["Հայերեն", "English"],
        horizontal=True,
        label_visibility="collapsed",
    )
    language = "hy" if language_label == "Հայերեն" else "en"

    try:
        surface, boundary, period = load_surface_data(production.live_exports_signature())
    except Exception as error:
        st.error(f"{t(language, 'no_data')} {error}")
        return

    st.markdown('<div class="surface-kicker">Echmiadzin WUA</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="surface-title">{html.escape(t(language, "title"))}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="surface-subtitle">{html.escape(t(language, "subtitle"))}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="surface-period"><strong>{html.escape(t(language, "period"))}:</strong> {html.escape(period)}</div>',
        unsafe_allow_html=True,
    )

    counts = surface["surface_class"].value_counts().to_dict()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t(language, "all_cells"), f"{len(surface):,}")
    m2.metric(t(language, "now"), f"{int(counts.get('NOW', 0)):,}")
    m3.metric(t(language, "soon"), f"{int(counts.get('SOON', 0)):,}")
    m4.metric(t(language, "wait"), f"{int(counts.get('WAIT', 0)):,}")

    show_grid = st.toggle(t(language, "show_grid"), value=False)
    irrigation_map = build_map(surface, boundary, language, show_grid)
    st_folium(
        irrigation_map,
        width=None,
        height=760,
        key=f"pressure_surface_{language}_{show_grid}",
        returned_objects=[],
    )


if __name__ == "__main__":
    main()
