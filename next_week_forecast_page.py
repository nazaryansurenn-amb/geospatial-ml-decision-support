from __future__ import annotations

from datetime import date
import html

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import app as production
import pressure_surface_demo as surface_demo
from core.rules import CLASS_ORDER


TEXT = {
    "en": {
        "title": "Next-week irrigation priority forecast",
        "subtitle": "Expected priority for the week after the latest satellite report, forecast from this week's observations. A forecast is an estimate, not an observation: check fields before acting.",
        "basis": "Based on observations for",
        "forecast_for": "Forecast for",
        "cells": "Vegetation cells",
        "stale": "This forecast covers a week that has already passed; a newer satellite report is not available yet.",
        "changes_title": "Expected changes",
        "changes_caption": "Cells by this week's class (rows) and forecast class for next week (columns).",
        "this_week": "This week",
        "next_week": "Next week",
        "escalating_title": "Cells most likely to become high priority",
        "cell_id": "Cell",
        "crop_type": "Crop type",
        "probability_now": "Chance of high priority next week",
        "reliability_title": "How reliable is this forecast?",
        "reliability_line": "Tested on the {season} season, which the model never saw: the forecast class was right for {model:.0%} of cells, against {persistence:.0%} when assuming next week stays the same.",
        "reliability_now": "High-priority forecasts were correct {precision:.0%} of the time and caught {recall:.0%} of next week's high-priority cells.",
        "no_model": "The next-week forecast model has not been trained yet. Run scripts/train_next_week.py.",
        "no_data": "The next-week forecast is not available.",
    },
    "hy": {
        "title": "Ոռոգման առաջնահերթությունների կանխատեսում հաջորդ շաբաթվա համար",
        "subtitle": "Վերջին արբանյակային հաշվետվությանը հաջորդող շաբաթվա սպասվող առաջնահերթությունը՝ կանխատեսված այս շաբաթվա դիտարկումներից։ Կանխատեսումը գնահատական է, ոչ թե դիտարկում․ գործելուց առաջ ստուգեք դաշտերը։",
        "basis": "Հիմնված է դիտարկումների վրա՝",
        "forecast_for": "Կանխատեսում՝",
        "cells": "Բուսական ծածկով բջիջներ",
        "stale": "Այս կանխատեսումը վերաբերում է արդեն անցած շաբաթվա․ ավելի նոր արբանյակային հաշվետվություն դեռ հասանելի չէ։",
        "changes_title": "Սպասվող փոփոխություններ",
        "changes_caption": "Բջիջներն ըստ այս շաբաթվա դասի (տողեր) և հաջորդ շաբաթվա կանխատեսված դասի (սյունակներ)։",
        "this_week": "Այս շաբաթ",
        "next_week": "Հաջորդ շաբաթ",
        "escalating_title": "Բարձր առաջնահերթության անցնելու ամենամեծ հավանականությամբ բջիջները",
        "cell_id": "Բջիջ",
        "crop_type": "Մշակաբույս",
        "probability_now": "Հաջորդ շաբաթ բարձր առաջնահերթության հավանականությունը",
        "reliability_title": "Որքանո՞վ է հուսալի այս կանխատեսումը",
        "reliability_line": "Ստուգվել է {season} սեզոնի վրա, որը մոդելը չէր տեսել․ կանխատեսված դասը ճիշտ է եղել բջիջների {model:.0%}-ի համար՝ {persistence:.0%}-ի դիմաց, եթե ենթադրենք, որ հաջորդ շաբաթը չի փոխվի։",
        "reliability_now": "Բարձր առաջնահերթության կանխատեսումները ճիշտ են եղել դեպքերի {precision:.0%}-ում և հայտնաբերել են հաջորդ շաբաթվա բարձր առաջնահերթությամբ բջիջների {recall:.0%}-ը։",
        "no_model": "Հաջորդ շաբաթվա կանխատեսման մոդելը դեռ ուսուցանված չէ։ Գործարկեք scripts/train_next_week.py։",
        "no_data": "Հաջորդ շաբաթվա կանխատեսումը հասանելի չէ։",
    },
    "ru": {
        "title": "Прогноз приоритетов орошения на следующую неделю",
        "subtitle": "Ожидаемый приоритет на неделю после последнего спутникового отчёта, рассчитанный по наблюдениям текущей недели. Прогноз — это оценка, а не наблюдение: проверяйте поля перед принятием решений.",
        "basis": "По наблюдениям за",
        "forecast_for": "Прогноз на",
        "cells": "Ячейки с растительностью",
        "stale": "Этот прогноз относится к уже прошедшей неделе; более новый спутниковый отчёт пока недоступен.",
        "changes_title": "Ожидаемые изменения",
        "changes_caption": "Ячейки по классу текущей недели (строки) и прогнозному классу на следующую неделю (столбцы).",
        "this_week": "Эта неделя",
        "next_week": "Следующая неделя",
        "escalating_title": "Ячейки с наибольшей вероятностью перехода в высокий приоритет",
        "cell_id": "Ячейка",
        "crop_type": "Тип культуры",
        "probability_now": "Вероятность высокого приоритета на следующей неделе",
        "reliability_title": "Насколько надёжен прогноз?",
        "reliability_line": "Проверено на сезоне {season}, который модель не видела: прогнозный класс был верным для {model:.0%} ячеек против {persistence:.0%}, если считать, что на следующей неделе ничего не изменится.",
        "reliability_now": "Прогнозы высокого приоритета оправдывались в {precision:.0%} случаев и выявили {recall:.0%} ячеек, получивших высокий приоритет на следующей неделе.",
        "no_model": "Модель прогноза на следующую неделю ещё не обучена. Запустите scripts/train_next_week.py.",
        "no_data": "Прогноз на следующую неделю недоступен.",
    },
}
CLASS_TEXT_KEYS = {"WAIT": "wait", "SOON": "soon", "NOW": "now"}


def t(language: str, key: str, **values) -> str:
    text = TEXT.get(language, TEXT["en"])[key]
    return text.format(**values) if values else text


@st.cache_data(show_spinner="Preparing the next-week forecast map...")
def load_forecast_view(live_signature):
    history_years = tuple(int(year) for year in production.load_historical_years_cached())
    forecast = production.load_next_week_forecast_cached(history_years, live_signature, 0.30, 10, 0.25)
    if forecast.empty:
        return forecast, None, None
    _, boundary = production.load_spatial()
    scored = forecast.rename(columns={f"p_next_{name}": f"p_{name}" for name in CLASS_ORDER})
    scored["need_class_ml"] = forecast["next_class"]
    smoothed = surface_demo.smooth_class_probabilities(scored)
    surface = surface_demo.build_exact_surface_geometry(smoothed, boundary)
    return forecast, surface, boundary


def _validation_for_display(metadata: dict) -> tuple[str, dict] | None:
    validation = metadata.get("validation", {}) if metadata else {}
    live_keys = sorted(key for key in validation if key.endswith("_live"))
    key = live_keys[-1] if live_keys else (sorted(validation)[-1] if validation else None)
    if key is None:
        return None
    return key.replace("_live", ""), validation[key]


def render_page(language: str) -> None:
    surface_demo.apply_theme()
    st.markdown(
        f'<div class="surface-kicker">{html.escape(production.translate("organization_name", language))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="surface-title">{html.escape(t(language, "title"))}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="surface-subtitle">{html.escape(t(language, "subtitle"))}</div>', unsafe_allow_html=True)

    bundle = production.load_forecast_artifacts()
    if bundle is None:
        st.info(t(language, "no_data") if production.PUBLIC_APP_MODE else t(language, "no_model"))
        return
    try:
        forecast, surface, boundary = load_forecast_view(production.live_exports_signature())
    except Exception:
        st.error(t(language, "no_data"))
        return
    if forecast.empty or surface is None:
        st.info(t(language, "no_data"))
        return

    basis_start = pd.to_datetime(forecast["week_start"]).max()
    basis_end = pd.to_datetime(forecast["week_end"]).max() if "week_end" in forecast.columns else basis_start + pd.Timedelta(days=7)
    next_start, next_end = basis_start + pd.Timedelta(days=7), basis_end + pd.Timedelta(days=7)
    st.markdown(
        (
            f'<div class="surface-period"><strong>{html.escape(t(language, "basis"))}</strong> '
            f"{basis_start.date().isoformat()} – {basis_end.date().isoformat()} · "
            f'<strong>{html.escape(t(language, "forecast_for"))}</strong> '
            f"{next_start.date().isoformat()} – {next_end.date().isoformat()}</div>"
        ),
        unsafe_allow_html=True,
    )
    if next_end.date() < date.today():
        st.warning(t(language, "stale"))

    current_counts = forecast["rule_class"].value_counts()
    next_counts = forecast["next_class"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t(language, "cells"), f"{len(forecast):,}")
    for column, name, colour in ((m2, "NOW", "inverse"), (m3, "SOON", "inverse"), (m4, "WAIT", "normal")):
        now_count, next_count = int(current_counts.get(name, 0)), int(next_counts.get(name, 0))
        column.metric(
            surface_demo.t(language, CLASS_TEXT_KEYS[name]),
            f"{next_count:,}",
            delta=f"{next_count - now_count:+,}" if next_count != now_count else None,
            delta_color=colour,
            help=f"{t(language, 'this_week')}: {now_count:,}",
        )

    show_grid = False if production.PUBLIC_APP_MODE else st.toggle(surface_demo.t(language, "show_grid"), value=False)
    st_folium(
        surface_demo.build_map(surface, boundary, language, show_grid),
        width=None,
        height=720,
        key=f"next_week_forecast_{language}_{show_grid}",
        returned_objects=[],
    )

    labels = {name: surface_demo.t(language, CLASS_TEXT_KEYS[name]) for name in CLASS_ORDER}
    st.subheader(t(language, "changes_title"))
    st.caption(t(language, "changes_caption"))
    transitions = (
        pd.crosstab(forecast["rule_class"], forecast["next_class"])
        .reindex(index=list(CLASS_ORDER), columns=list(CLASS_ORDER), fill_value=0)
        .rename(index=labels, columns=labels)
    )
    transitions.index.name = t(language, "this_week")
    transitions.columns.name = t(language, "next_week")
    st.dataframe(transitions, use_container_width=True)

    if not production.PUBLIC_APP_MODE:
        st.subheader(t(language, "escalating_title"))
        escalating = forecast[forecast["rule_class"].ne("NOW")].sort_values("p_next_NOW", ascending=False).head(15)
        st.dataframe(
            pd.DataFrame(
                {
                    t(language, "cell_id"): escalating["cell_id"].astype(str),
                    t(language, "this_week"): escalating["rule_class"].map(labels),
                    t(language, "crop_type"): escalating["crop_type"].map({
                        "ANNUAL": production.translate("annual", language),
                        "PERENNIAL": production.translate("perennial", language),
                        "NON_AGRI": production.translate("non_agri", language),
                    }),
                    t(language, "probability_now"): (escalating["p_next_NOW"] * 100).round(0).astype(int).astype(str) + "%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    validation = _validation_for_display(bundle.get("metadata", {}))
    if validation is not None:
        season, result = validation
        with st.expander(t(language, "reliability_title")):
            st.write(t(language, "reliability_line", season=season,
                       model=result["model"]["accuracy"], persistence=result["persistence"]["accuracy"]))
            st.write(t(language, "reliability_now", precision=result["model"]["now_precision"],
                       recall=result["model"]["now_recall"]))
