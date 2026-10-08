from __future__ import annotations

import html
import json

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from branca.element import MacroElement, Template

import app as production
import pressure_surface_demo as surface_demo
from pressure_surface_vegetation_demo import (
    available_years as available_surface_years,
    filter_vegetation_cells,
    load_period_frame,
)


DEFAULT_YEAR = 2025
FRAME_INTERVAL_MS = 900

TEXT = {
    "en": {
        "title": "Season irrigation priority history",
        "subtitle": "Choose any available season and play its weekly observations. Every frame uses the same priority method, and areas without active vegetation remain empty.",
        "year": "Season year",
        "weeks": "Available weeks",
        "season": "Season covered",
        "average_cells": "Average vegetation cells",
        "play": "Play",
        "pause": "Pause",
        "previous": "Previous week",
        "next": "Next week",
        "now": "High priority",
        "soon": "Watch closely",
        "wait": "Routine monitoring",
        "vegetation": "Vegetation cells",
        "no_data": "The selected season's time-lapse could not be prepared.",
    },
    "hy": {
        "title": "Ոռոգման առաջնահերթությունների սեզոնային պատմություն",
        "subtitle": "Ընտրեք հասանելի սեզոններից մեկը և դիտեք շաբաթական փոփոխությունները։ Բոլոր կադրերը հաշվարկվում են նույն եղանակով, իսկ բուսական ծածկ չունեցող տարածքները մնում են դատարկ։",
        "year": "Սեզոնի տարի",
        "weeks": "Հասանելի շաբաթներ",
        "season": "Ընդգրկված շրջան",
        "average_cells": "Բուսական ծածկով բջիջների միջինը",
        "play": "Դիտել",
        "pause": "Դադար",
        "previous": "Նախորդ շաբաթ",
        "next": "Հաջորդ շաբաթ",
        "now": "Բարձր առաջնահերթություն",
        "soon": "Հետևել առաջիկայում",
        "wait": "Շարունակել սովորական հետևումը",
        "vegetation": "Բուսական ծածկով բջիջներ",
        "no_data": "Չհաջողվեց պատրաստել ընտրված սեզոնի շարժապատկերը։",
    },
    "ru": {
        "title": "История приоритетов орошения за сезон",
        "subtitle": "Выберите доступный сезон и просмотрите недельные наблюдения. Все кадры рассчитаны единым методом, а территории без активной растительности остаются пустыми.",
        "year": "Год сезона",
        "weeks": "Доступные недели",
        "season": "Охваченный период",
        "average_cells": "Среднее число ячеек с растительностью",
        "play": "Воспроизвести",
        "pause": "Пауза",
        "previous": "Предыдущая неделя",
        "next": "Следующая неделя",
        "now": "Высокий приоритет",
        "soon": "Наблюдать внимательно",
        "wait": "Обычное наблюдение",
        "vegetation": "Ячейки с растительностью",
        "no_data": "Не удалось подготовить историю выбранного сезона.",
    },
}


def t(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["en"])[key]


def chronological_week_groups(frame: pd.DataFrame):
    if frame.empty or "week_start" not in frame.columns:
        return
    ordered = frame.copy()
    ordered["week_start"] = pd.to_datetime(ordered["week_start"], errors="coerce")
    ordered = ordered.dropna(subset=["week_start"]).sort_values("week_start")
    for week_start, week in ordered.groupby("week_start", sort=True):
        yield pd.Timestamp(week_start), week.copy()


def frame_metadata(week_start: pd.Timestamp, week: pd.DataFrame, surface: pd.DataFrame) -> dict:
    week_end = pd.to_datetime(week.get("week_end"), errors="coerce").dropna()
    start_label = pd.Timestamp(week_start).date().isoformat()
    end_label = week_end.max().date().isoformat() if not week_end.empty else start_label
    counts = surface["surface_class"].value_counts().to_dict()
    return {
        "period": f"{start_label} – {end_label}",
        "start": start_label,
        "end": end_label,
        "cells": int(len(surface)),
        "NOW": int(counts.get("NOW", 0)),
        "SOON": int(counts.get("SOON", 0)),
        "WAIT": int(counts.get("WAIT", 0)),
    }


def empty_surface_frame(boundary) -> tuple[np.ndarray, list[list[float]]]:
    west, south, east, north = boundary.to_crs("EPSG:4326").total_bounds
    return np.zeros((2, 2, 4), dtype=np.uint8), [[south, west], [north, east]]


@st.cache_data(show_spinner="Preparing the selected season's weekly map frames...")
def load_timelapse_frames(year: int = DEFAULT_YEAR, live_signature=None):
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

    frames = []
    for week_start, scored_week in chronological_week_groups(scored_year):
        vegetation = filter_vegetation_cells(scored_week)
        if vegetation.empty:
            smoothed = pd.DataFrame(columns=["surface_class"])
            image, bounds = empty_surface_frame(boundary)
        else:
            smoothed = surface_demo.smooth_class_probabilities(vegetation)
            image, bounds = surface_demo.build_surface_image(
                smoothed,
                boundary,
                show_grid=False,
                pixels_per_cell=4,
            )
        frames.append(
            {
                "image": image,
                "bounds": bounds,
                "metadata": frame_metadata(week_start, scored_week, smoothed),
            }
        )

    if not frames:
        raise ValueError(f"No vegetation frames could be prepared for {year}.")
    return frames, boundary


class TimelineControl(MacroElement):
    def __init__(
        self,
        layer_names: list[str],
        metadata: list[dict],
        language: str,
        count_items: list[tuple[str, str]] | None = None,
    ):
        super().__init__()
        self._name = "IrrigationTimeline"
        control_id = self.get_name()
        layers_js = "[" + ",".join(layer_names) + "]"
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        labels_json = json.dumps(
            {
                "play": t(language, "play"),
                "pause": t(language, "pause"),
                "previous": t(language, "previous"),
                "next": t(language, "next"),
                "vegetation": t(language, "vegetation"),
                "now": t(language, "now"),
                "soon": t(language, "soon"),
                "wait": t(language, "wait"),
            },
            ensure_ascii=False,
        )
        if count_items is None:
            count_items = [
                ("cells", t(language, "vegetation")),
                ("NOW", t(language, "now")),
                ("SOON", t(language, "soon")),
                ("WAIT", t(language, "wait")),
            ]
        count_items_json = json.dumps(count_items, ensure_ascii=False)
        self._template = Template(
            "{% macro html(this, kwargs) %}"
            f"""
            <div id="{control_id}" class="irrigation-timeline">
              <div class="timeline-heading">
                <strong id="{control_id}_period"></strong>
                <span id="{control_id}_position"></span>
              </div>
              <div class="timeline-actions">
                <button id="{control_id}_previous" type="button" title="{html.escape(t(language, 'previous'))}">&#9664;</button>
                <button id="{control_id}_play" class="timeline-play" type="button" title="{html.escape(t(language, 'play'))}">&#9654;</button>
                <button id="{control_id}_next" type="button" title="{html.escape(t(language, 'next'))}">&#9654;</button>
                <input id="{control_id}_range" type="range" min="0" max="{len(metadata) - 1}" value="0" step="1" aria-label="Timeline week">
              </div>
              <div id="{control_id}_counts" class="timeline-counts"></div>
            </div>
            <style>
              .irrigation-timeline {{
                position:fixed; left:50%; bottom:18px; transform:translateX(-50%);
                z-index:9999; width:min(580px,calc(100% - 24px)); box-sizing:border-box;
                padding:10px 12px 9px; border:1px solid #d4dde0; border-radius:6px;
                background:rgba(255,255,255,.96); box-shadow:0 4px 18px rgba(20,40,55,.2);
                color:#213740; font-family:Inter,Arial,sans-serif;
              }}
              .timeline-heading {{ display:flex; justify-content:space-between; gap:12px; font-size:12px; margin-bottom:8px; }}
              .timeline-heading span {{ color:#62747b; white-space:nowrap; }}
              .timeline-actions {{ display:grid; grid-template-columns:32px 38px 32px 1fr; gap:6px; align-items:center; }}
              .timeline-actions button {{
                width:32px; height:30px; border:1px solid #c8d4d8; border-radius:4px;
                background:#f6f9fa; color:#294953; cursor:pointer; font-size:12px;
              }}
              .timeline-actions .timeline-play {{ width:38px; color:#fff; background:#19705f; border-color:#19705f; }}
              .timeline-actions input {{ width:100%; accent-color:#19705f; }}
              .timeline-counts {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:7px; color:#536970; font-size:10px; }}
              .timeline-counts b {{ color:#243b43; }}
              @media (max-width:620px) {{
                .timeline-counts {{ gap:7px; }}
                .timeline-heading {{ font-size:11px; }}
              }}
            </style>
            """
            + "{% endmacro %}"
            + "{% macro script(this, kwargs) %}"
            f"""
            const timelineLayers = {layers_js};
            const timelineData = {metadata_json};
            const timelineLabels = {labels_json};
            const timelineCountItems = {count_items_json};
            const timelineRange = document.getElementById('{control_id}_range');
            const timelinePlay = document.getElementById('{control_id}_play');
            let timelineIndex = 0;
            let timelineTimer = null;

            function setTimelineFrame(index) {{
              timelineIndex = (index + timelineLayers.length) % timelineLayers.length;
              timelineLayers.forEach((layer, layerIndex) => layer.setOpacity(layerIndex === timelineIndex ? 1 : 0));
              const item = timelineData[timelineIndex];
              timelineRange.value = timelineIndex;
              document.getElementById('{control_id}_period').textContent = item.period;
              document.getElementById('{control_id}_position').textContent = `${{timelineIndex + 1}} / ${{timelineData.length}}`;
              document.getElementById('{control_id}_counts').innerHTML = timelineCountItems
                .map(([key, label]) => `<span>${{label}}: <b>${{Number(item[key] || 0).toLocaleString()}}</b></span>`)
                .join('');
            }}

            function stopTimeline() {{
              if (timelineTimer !== null) window.clearInterval(timelineTimer);
              timelineTimer = null;
              timelinePlay.innerHTML = '&#9654;';
              timelinePlay.title = timelineLabels.play;
            }}

            function toggleTimeline() {{
              if (timelineTimer !== null) {{ stopTimeline(); return; }}
              timelinePlay.innerHTML = '&#10074;&#10074;';
              timelinePlay.title = timelineLabels.pause;
              timelineTimer = window.setInterval(() => setTimelineFrame(timelineIndex + 1), {FRAME_INTERVAL_MS});
            }}

            timelinePlay.addEventListener('click', toggleTimeline);
            document.getElementById('{control_id}_previous').addEventListener('click', () => {{ stopTimeline(); setTimelineFrame(timelineIndex - 1); }});
            document.getElementById('{control_id}_next').addEventListener('click', () => {{ stopTimeline(); setTimelineFrame(timelineIndex + 1); }});
            timelineRange.addEventListener('input', event => {{ stopTimeline(); setTimelineFrame(Number(event.target.value)); }});
            setTimelineFrame(0);
            """
            + "{% endmacro %}"
        )


def build_timelapse_map(frames: list[dict], boundary, language: str) -> folium.Map:
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
    surface_demo._add_pressure_legend(irrigation_map, language)
    irrigation_map.get_root().add_child(
        TimelineControl(
            [overlay.get_name() for overlay in overlays],
            [frame["metadata"] for frame in frames],
            language,
        )
    )
    return irrigation_map


def render_page(language: str) -> None:
    surface_demo.apply_theme()
    live_signature = production.live_exports_signature()
    years = available_surface_years(live_signature)
    if not years:
        st.error(t(language, "no_data"))
        return

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
    selected_year = st.selectbox(
        t(language, "year"),
        options=years,
        index=years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0,
        key="playback_year",
    )

    try:
        frames, boundary = load_timelapse_frames(int(selected_year), live_signature)
    except Exception as error:
        st.error(f"{t(language, 'no_data')} {error}")
        return

    metadata = [frame["metadata"] for frame in frames]
    average_cells = round(sum(item["cells"] for item in metadata) / len(metadata))
    first_start = metadata[0]["start"]
    last_end = metadata[-1]["end"]
    m1, m2, m3 = st.columns(3)
    m1.metric(t(language, "weeks"), f"{len(frames):,}")
    m2.metric(t(language, "season"), f"{first_start} → {last_end}")
    m3.metric(t(language, "average_cells"), f"{average_cells:,}")

    irrigation_map = build_timelapse_map(frames, boundary, language)
    components.html(
        irrigation_map.get_root().render(),
        height=780,
        scrolling=False,
    )


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
