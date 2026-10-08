from __future__ import annotations

import html

import folium
import numpy as np
import pandas as pd
import shapely
import streamlit as st
import streamlit.components.v1 as components
from branca.element import MacroElement, Template
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin

import app as production
import pressure_surface_demo as surface_demo
from pressure_surface_timelapse_demo import TimelineControl, chronological_week_groups
from pressure_surface_vegetation_demo import (
    available_years as available_surface_years,
    filter_vegetation_cells,
    load_period_frame,
)


DEFAULT_YEAR = 2025
STABLE_DELTA = 0.03
MIN_CHANGE_SCALE = 0.10

CHANGE_STOPS = np.array([-1.0, -0.58, -0.16, 0.0, 0.16, 0.58, 1.0])
CHANGE_RGB = np.array(
    [
        [33, 102, 172],
        [103, 169, 207],
        [209, 229, 240],
        [242, 242, 239],
        [254, 224, 144],
        [244, 109, 67],
        [178, 24, 43],
    ],
    dtype=float,
)

TEXT = {
    "en": {
        "title": "Week-to-week irrigation priority change",
        "subtitle": "Each frame compares irrigation priority with the previous weekly observation. Blue areas improved, light areas changed little, and yellow-to-red areas worsened. Only observations 5–10 days apart and areas with active vegetation in both reports are compared.",
        "year": "Season year",
        "weeks": "Comparable weeks",
        "season": "Change period",
        "average_cells": "Average compared cells",
        "compared": "Compared cells",
        "improved": "Improving",
        "stable": "Little change",
        "worsened": "Worsening",
        "legend": "Change from previous week",
        "no_years": "No historical or live reporting seasons are available.",
        "no_data": "The selected season's pressure-change time-lapse could not be prepared.",
    },
    "hy": {
        "title": "Ոռոգման առաջնահերթությունների փոփոխությունը՝ շաբաթ առ շաբաթ",
        "subtitle": "Յուրաքանչյուր կադրը համեմատվում է նախորդ շաբաթական դիտարկման հետ։ Կապույտ տարածքներում վիճակը բարելավվել է, բաց գույնը փոքր փոփոխություն է ցույց տալիս, իսկ դեղինից կարմիր գույները՝ վատթարացում։ Համեմատվում են միայն 5–10 օր տարբերությամբ հաշվետվությունները և այն բջիջները, որոնք երկուսում էլ բուսական ծածկ ունեն։",
        "year": "Սեզոնի տարի",
        "weeks": "Համեմատվող շաբաթներ",
        "season": "Փոփոխությունների շրջան",
        "average_cells": "Համեմատվող բջիջների միջինը",
        "compared": "Համեմատված բջիջներ",
        "improved": "Բարելավում",
        "stable": "Փոքր փոփոխություն",
        "worsened": "Վատթարացում",
        "legend": "Փոփոխությունը նախորդ շաբաթվա համեմատ",
        "no_years": "Պատմական կամ ընթացիկ հաշվետու սեզոններ չեն գտնվել։",
        "no_data": "Չհաջողվեց պատրաստել ընտրված սեզոնի շաբաթական փոփոխությունների շարժապատկերը։",
    },
    "ru": {
        "title": "Изменение приоритетов орошения по неделям",
        "subtitle": "Каждый кадр сравнивает приоритет орошения с предыдущим недельным наблюдением. Синие территории показывают улучшение, светлые — небольшие изменения, а жёлто-красные — ухудшение. Сравниваются только отчёты с интервалом 5–10 дней и территории с активной растительностью в обоих отчётах.",
        "year": "Год сезона",
        "weeks": "Сопоставимые недели",
        "season": "Период изменений",
        "average_cells": "Среднее число сопоставимых ячеек",
        "compared": "Сопоставимые ячейки",
        "improved": "Улучшение",
        "stable": "Небольшое изменение",
        "worsened": "Ухудшение",
        "legend": "Изменение с предыдущей недели",
        "no_years": "Нет доступных исторических или текущих сезонов.",
        "no_data": "Не удалось подготовить недельные изменения для выбранного сезона.",
    },
}


def t(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["en"])[key]


def compare_weekly_surfaces(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    required = ["cell_id", "surface_pressure"]
    if previous.empty or current.empty:
        return pd.DataFrame(columns=["cell_id", "previous_pressure", "current_pressure", "pressure_change"])
    previous_values = previous[required].rename(columns={"surface_pressure": "previous_pressure"})
    current_values = current[required].rename(columns={"surface_pressure": "current_pressure"})
    comparison = current_values.merge(previous_values, on="cell_id", how="inner")
    comparison["pressure_change"] = (
        pd.to_numeric(comparison["current_pressure"], errors="coerce")
        - pd.to_numeric(comparison["previous_pressure"], errors="coerce")
    )
    return comparison.dropna(subset=["pressure_change"]).copy()


def is_weekly_transition(previous_week, current_week, *, min_days: int = 5, max_days: int = 10) -> bool:
    if previous_week is None or current_week is None:
        return False
    elapsed_days = (pd.Timestamp(current_week).normalize() - pd.Timestamp(previous_week).normalize()).days
    return int(min_days) <= elapsed_days <= int(max_days)


def classify_pressure_change(change: pd.Series, stable_delta: float = STABLE_DELTA) -> pd.Series:
    values = pd.to_numeric(change, errors="coerce")
    return pd.Series(
        np.select(
            [values.lt(-float(stable_delta)), values.gt(float(stable_delta))],
            ["IMPROVED", "WORSENED"],
            default="STABLE",
        ),
        index=change.index,
    )


def change_rgb(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(np.asarray(values, dtype=float) / float(scale), -1, 1)
    channels = [
        np.interp(normalized, CHANGE_STOPS, CHANGE_RGB[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def build_change_image(
    comparison: pd.DataFrame,
    boundary,
    *,
    scale: float,
    pixels_per_cell: int = 4,
) -> tuple[np.ndarray, list[list[float]]]:
    coordinates = comparison["cell_id"].map(surface_demo._cell_coordinates)
    if coordinates.isna().any():
        raise ValueError("One or more comparison cell IDs could not be decoded.")
    x_values = np.array([coordinate[0] for coordinate in coordinates], dtype=int)
    y_values = np.array([coordinate[1] for coordinate in coordinates], dtype=int)
    min_x, max_x = int(x_values.min()), int(x_values.max())
    min_y, max_y = int(y_values.min()), int(y_values.max())

    width_cells = max_x - min_x + 1
    height_cells = max_y - min_y + 1
    rgba_cells = np.zeros((height_cells, width_cells, 4), dtype=np.uint8)
    row_index = max_y - y_values
    column_index = x_values - min_x
    changes = comparison["pressure_change"].to_numpy(dtype=float)
    rgba_cells[row_index, column_index, :3] = change_rgb(changes, scale)
    normalized_magnitude = np.clip(np.abs(changes) / float(scale), 0, 1)
    rgba_cells[row_index, column_index, 3] = (120 + 90 * normalized_magnitude).round().astype(np.uint8)

    rgba = np.repeat(
        np.repeat(rgba_cells, int(pixels_per_cell), axis=0),
        int(pixels_per_cell),
        axis=1,
    )
    min_easting = min_x * surface_demo.GRID_M
    max_northing = (max_y + 1) * surface_demo.GRID_M
    pixel_size = surface_demo.GRID_M / int(pixels_per_cell)
    transform = from_origin(min_easting, max_northing, pixel_size, pixel_size)
    outline = shapely.union_all(boundary.to_crs(surface_demo.GRID_CRS).geometry.values)
    boundary_mask = rasterize(
        [(outline, 1)],
        out_shape=rgba.shape[:2],
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    rgba[..., 3] = np.where(boundary_mask > 0, rgba[..., 3], 0)

    max_easting = (max_x + 1) * surface_demo.GRID_M
    min_northing = min_y * surface_demo.GRID_M
    to_wgs84 = Transformer.from_crs(surface_demo.GRID_CRS, "EPSG:4326", always_xy=True)
    west, south = to_wgs84.transform(min_easting, min_northing)
    east, north = to_wgs84.transform(max_easting, max_northing)
    return rgba, [[south, west], [north, east]]


def _period_metadata(week_start: pd.Timestamp, week: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    week_end = pd.to_datetime(week.get("week_end"), errors="coerce").dropna()
    start_label = week_start.date().isoformat()
    end_label = week_end.max().date().isoformat() if not week_end.empty else start_label
    classes = classify_pressure_change(comparison["pressure_change"])
    counts = classes.value_counts().to_dict()
    return {
        "period": f"{start_label} – {end_label}",
        "start": start_label,
        "end": end_label,
        "cells": int(len(comparison)),
        "IMPROVED": int(counts.get("IMPROVED", 0)),
        "STABLE": int(counts.get("STABLE", 0)),
        "WORSENED": int(counts.get("WORSENED", 0)),
    }


@st.cache_data(show_spinner="Preparing the selected season's weekly change frames...")
def load_change_frames(year: int = DEFAULT_YEAR, live_signature=None):
    _, boundary = production.load_spatial()
    if live_signature is None:
        live_signature = production.live_exports_signature()
    year_frame = load_period_frame(int(year), live_signature)
    if year_frame.empty:
        raise ValueError(f"No prepared data were found for {year}.")

    historical_years = set(int(value) for value in production.load_historical_years_cached())
    if int(year) in historical_years:
        crop_map = production.build_crop_map_cached(year_frame, 0.30, 10, 0.25)
    else:
        history_years = tuple(sorted(historical_years))
        crop_map = production.build_live_crop_map_cached(history_years, 0.30, 10, 0.25)
    scored_year = production.add_priority_predictions(year_frame, crop_map)

    comparisons = []
    previous_surface = pd.DataFrame()
    previous_week_start = None
    for week_start, scored_week in chronological_week_groups(scored_year):
        vegetation = filter_vegetation_cells(scored_week)
        current_surface = (
            surface_demo.smooth_class_probabilities(vegetation)
            if not vegetation.empty
            else pd.DataFrame(columns=["cell_id", "surface_pressure"])
        )
        comparison = (
            compare_weekly_surfaces(previous_surface, current_surface)
            if is_weekly_transition(previous_week_start, week_start)
            else pd.DataFrame(columns=["cell_id", "previous_pressure", "current_pressure", "pressure_change"])
        )
        if not comparison.empty:
            comparisons.append((week_start, scored_week, comparison))
        previous_surface = current_surface
        previous_week_start = week_start

    if not comparisons:
        raise ValueError(f"No consecutive vegetation weeks could be compared for {year}.")
    absolute_changes = np.concatenate(
        [comparison["pressure_change"].abs().to_numpy(dtype=float) for _, _, comparison in comparisons]
    )
    scale = max(MIN_CHANGE_SCALE, float(np.nanquantile(absolute_changes, 0.95)))

    frames = []
    for week_start, scored_week, comparison in comparisons:
        image, bounds = build_change_image(comparison, boundary, scale=scale)
        frames.append(
            {
                "image": image,
                "bounds": bounds,
                "metadata": _period_metadata(week_start, scored_week, comparison),
            }
        )
    return frames, boundary, scale


def _add_change_legend(irrigation_map: folium.Map, language: str) -> None:
    legend = MacroElement()
    legend._template = Template(
        "{% macro html(this, kwargs) %}"
        f"""
        <div style="position:fixed;bottom:58px;left:10px;z-index:9999;
          background:rgba(255,255,255,.96);padding:9px 11px;border-radius:6px;
          border:1px solid #d7dee2;box-shadow:0 3px 12px rgba(20,40,60,.14);
          width:245px;font-family:Inter,Arial,sans-serif;">
          <div style="font-weight:700;font-size:12px;color:#243746;margin-bottom:7px;">{html.escape(t(language, 'legend'))}</div>
          <div style="height:12px;border-radius:3px;background:linear-gradient(90deg,
            #2166ac 0%,#67a9cf 22%,#d1e5f0 40%,#f2f2ef 50%,#fee090 63%,#f46d43 82%,#b2182b 100%);"></div>
          <div style="display:flex;justify-content:space-between;color:#5e7078;font-size:10px;margin-top:4px;">
            <span>{html.escape(t(language, 'improved'))}</span>
            <span>{html.escape(t(language, 'stable'))}</span>
            <span>{html.escape(t(language, 'worsened'))}</span>
          </div>
        </div>
        """
        + "{% endmacro %}"
    )
    irrigation_map.get_root().add_child(legend)


def build_change_map(frames: list[dict], boundary, language: str) -> folium.Map:
    outline = boundary.to_crs("EPSG:4326")
    bounds = outline.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    irrigation_map = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(
        tiles=production.ESRI_TILES,
        attr=production.ESRI_ATTR,
        name=surface_demo.t(language, "satellite"),
        overlay=False,
        show=True,
    ).add_to(irrigation_map)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap contributors",
        name=surface_demo.t(language, "street"),
        overlay=False,
        show=False,
    ).add_to(irrigation_map)

    overlays = []
    for index, frame in enumerate(frames):
        overlay = folium.raster_layers.ImageOverlay(
            image=frame["image"],
            bounds=frame["bounds"],
            opacity=1.0 if index == 0 else 0.0,
            interactive=False,
            cross_origin=False,
            zindex=2,
            control=False,
            show=True,
        )
        overlay.add_to(irrigation_map)
        overlays.append(overlay)

    folium.GeoJson(
        outline,
        name=surface_demo.t(language, "boundary"),
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
    _add_change_legend(irrigation_map, language)
    irrigation_map.get_root().add_child(
        TimelineControl(
            [overlay.get_name() for overlay in overlays],
            [frame["metadata"] for frame in frames],
            language,
            count_items=[
                ("cells", t(language, "compared")),
                ("IMPROVED", t(language, "improved")),
                ("STABLE", t(language, "stable")),
                ("WORSENED", t(language, "worsened")),
            ],
        )
    )
    return irrigation_map


def render_page(language: str) -> None:
    surface_demo.apply_theme()
    st.markdown(
        f'<div class="surface-kicker">{html.escape(production.translate("organization_name", language))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="surface-title">{html.escape(t(language, "title"))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="surface-subtitle">{html.escape(t(language, "subtitle"))}</div>',
        unsafe_allow_html=True,
    )

    live_signature = production.live_exports_signature()
    years = available_surface_years(live_signature)
    if not years:
        st.error(t(language, "no_years"))
        return

    selected_year = st.selectbox(
        t(language, "year"),
        options=years,
        index=years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0,
    )

    try:
        frames, boundary, _ = load_change_frames(int(selected_year), live_signature)
    except Exception as error:
        st.error(f"{t(language, 'no_data')} {error}")
        return

    metadata = [frame["metadata"] for frame in frames]
    average_cells = round(sum(item["cells"] for item in metadata) / len(metadata))
    m1, m2, m3 = st.columns(3)
    m1.metric(t(language, "weeks"), f"{len(frames):,}")
    m2.metric(t(language, "season"), f"{metadata[0]['start']} → {metadata[-1]['end']}")
    m3.metric(t(language, "average_cells"), f"{average_cells:,}")

    irrigation_map = build_change_map(frames, boundary, language)
    components.html(irrigation_map.get_root().render(), height=780, scrolling=False)


def main() -> None:
    language_label = st.radio(
        "Language / Լեզու / Язык",
        options=["Հայերեն", "English", "Русский"],
        horizontal=True,
        label_visibility="collapsed",
    )
    language = {"Հայերեն": "hy", "English": "en", "Русский": "ru"}[language_label]
    render_page(language)


if __name__ == "__main__":
    main()
