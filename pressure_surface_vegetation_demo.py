from __future__ import annotations

import html

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import app as production
import pressure_surface_demo as surface_demo


TEXT = {
    "en": {
        "title": "Weekly irrigation priority",
        "subtitle": "Choose a year and week to compare irrigation priorities over time. Only areas with active vegetation are shown.",
        "year": "Year",
        "week": "Week",
        "cells": "Vegetation cells",
        "no_years": "No historical or live reporting periods are available.",
        "no_weeks": "No weekly reports are available for the selected year.",
        "no_data": "The selected weekly priority map could not be prepared.",
    },
    "hy": {
        "title": "Ոռոգման շաբաթական առաջնահերթությունները",
        "subtitle": "Ընտրեք տարին և շաբաթը՝ ժամանակի ընթացքում ոռոգման առաջնահերթությունները համեմատելու համար։ Ցուցադրվում են միայն տվյալ շաբաթ բուսական ծածկ ունեցող տարածքները։",
        "year": "Տարի",
        "week": "Շաբաթ",
        "cells": "Բուսական ծածկով բջիջներ",
        "no_years": "Պատմական կամ ընթացիկ հաշվետու ժամանակահատվածներ չեն գտնվել։",
        "no_weeks": "Ընտրված տարվա համար շաբաթական հաշվետվություններ չեն գտնվել։",
        "no_data": "Չհաջողվեց պատրաստել ընտրված շաբաթվա քարտեզը։",
    },
    "ru": {
        "title": "Недельные приоритеты орошения",
        "subtitle": "Выберите год и неделю, чтобы сравнить приоритеты орошения во времени. Показаны только территории с активной растительностью.",
        "year": "Год",
        "week": "Неделя",
        "cells": "Ячейки с растительностью",
        "no_years": "Нет доступных исторических или текущих отчётных периодов.",
        "no_weeks": "Для выбранного года нет недельных отчётов.",
        "no_data": "Не удалось подготовить карту приоритетов за выбранную неделю.",
    },
}


def t(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["en"])[key]


def filter_vegetation_cells(scored: pd.DataFrame) -> pd.DataFrame:
    if "veg_flag" not in scored.columns:
        raise ValueError("The weekly dataset does not contain veg_flag.")
    vegetation = pd.to_numeric(scored["veg_flag"], errors="coerce").eq(1)
    return scored.loc[vegetation].copy()


def build_vegetation_surface(scored: pd.DataFrame, boundary):
    vegetation = filter_vegetation_cells(scored)
    if vegetation.empty:
        raise ValueError("No vegetation cells were found for the selected period.")
    smoothed = surface_demo.smooth_class_probabilities(vegetation)
    return surface_demo.build_exact_surface_geometry(smoothed, boundary)


def available_week_options(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    if frame.empty or "week_start" not in frame.columns:
        return ()
    weeks = frame[[column for column in ("week_start", "week_end") if column in frame.columns]].copy()
    weeks["week_start"] = pd.to_datetime(weeks["week_start"], errors="coerce")
    if "week_end" not in weeks.columns:
        weeks["week_end"] = weeks["week_start"]
    weeks["week_end"] = pd.to_datetime(weeks["week_end"], errors="coerce")
    weeks = weeks.dropna(subset=["week_start"]).drop_duplicates().sort_values("week_start", ascending=False)

    options = []
    for row in weeks.itertuples(index=False):
        start = pd.Timestamp(row.week_start).date().isoformat()
        end = pd.Timestamp(row.week_end).date().isoformat() if pd.notna(row.week_end) else start
        options.append((start, f"{start} → {end}"))
    return tuple(options)


def select_period_week(frame: pd.DataFrame, week_start: str) -> pd.DataFrame:
    if frame.empty or "week_start" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    dates = pd.to_datetime(frame["week_start"], errors="coerce").dt.normalize()
    selected = pd.Timestamp(week_start).normalize()
    return frame.loc[dates.eq(selected)].copy()


def _historical_years() -> tuple[int, ...]:
    return tuple(int(year) for year in production.load_historical_years_cached())


def _live_history_years() -> tuple[int, ...]:
    return _historical_years()


@st.cache_data(show_spinner=False)
def available_years(live_signature) -> tuple[int, ...]:
    years = set(_historical_years())
    try:
        live_season, _ = production.load_live_season_cached(live_signature)
        live_dates = pd.to_datetime(live_season.get("week_start"), errors="coerce")
        years.update(live_dates.dropna().dt.year.astype(int).tolist())
    except Exception:
        pass
    return tuple(sorted(years, reverse=True))


def load_period_frame(year: int, live_signature) -> pd.DataFrame:
    if int(year) in set(_historical_years()):
        return production.prepare_year_data_cached(int(year))

    live = production.prepare_live_data_cached(_live_history_years(), live_signature)
    live_years = pd.to_datetime(live.get("week_start"), errors="coerce").dt.year
    return live.loc[live_years.eq(int(year))].copy()


@st.cache_data(show_spinner="Preparing the selected vegetation-cell probability surface...")
def load_vegetation_surface(year: int, week_start: str, live_signature):
    _, boundary = production.load_spatial()
    year_frame = load_period_frame(int(year), live_signature)
    selected_week = select_period_week(year_frame, week_start)
    if selected_week.empty:
        raise ValueError(f"No rows were found for week {week_start}.")

    if int(year) in set(_historical_years()):
        crop_map = production.build_crop_map_cached(year_frame, 0.30, 10, 0.25)
    else:
        history_years = _live_history_years()
        crop_map = production.build_live_crop_map_cached(history_years, 0.30, 10, 0.25)

    scored = production.add_priority_predictions(selected_week, crop_map)
    surface = build_vegetation_surface(scored, boundary)
    week_end = pd.to_datetime(selected_week.get("week_end"), errors="coerce").dropna()
    period = week_start
    if not week_end.empty:
        period = f"{period} – {week_end.max().date().isoformat()}"
    return surface, boundary, period


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
    years = available_years(live_signature)
    if not years:
        st.error(t(language, "no_years"))
        return

    year_column, week_column = st.columns([1, 2])
    selected_year = year_column.selectbox(t(language, "year"), options=years, index=0)
    try:
        period_frame = load_period_frame(int(selected_year), live_signature)
        week_options = available_week_options(period_frame)
    except Exception as error:
        st.error(f"{t(language, 'no_data')} {error}")
        return
    if not week_options:
        st.error(t(language, "no_weeks"))
        return

    week_labels = dict(week_options)
    selected_week = week_column.selectbox(
        t(language, "week"),
        options=[value for value, _ in week_options],
        format_func=lambda value: week_labels[value],
        index=0,
    )

    try:
        surface, boundary, period = load_vegetation_surface(
            int(selected_year),
            selected_week,
            live_signature,
        )
    except Exception as error:
        st.error(f"{t(language, 'no_data')} {error}")
        return

    st.markdown(
        (
            f'<div class="surface-period"><strong>{html.escape(surface_demo.t(language, "period"))}:'
            f'</strong> {html.escape(period)}</div>'
        ),
        unsafe_allow_html=True,
    )

    counts = surface["surface_class"].value_counts().to_dict()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t(language, "cells"), f"{len(surface):,}")
    m2.metric(surface_demo.t(language, "now"), f"{int(counts.get('NOW', 0)):,}")
    m3.metric(surface_demo.t(language, "soon"), f"{int(counts.get('SOON', 0)):,}")
    m4.metric(surface_demo.t(language, "wait"), f"{int(counts.get('WAIT', 0)):,}")

    show_grid = False if production.PUBLIC_APP_MODE else st.toggle(surface_demo.t(language, "show_grid"), value=False)
    irrigation_map = surface_demo.build_map(
        surface,
        boundary,
        language,
        show_grid,
    )
    st_folium(
        irrigation_map,
        width=None,
        height=760,
        key=f"vegetation_pressure_surface_{language}_{selected_year}_{selected_week}_{show_grid}",
        returned_objects=[],
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
