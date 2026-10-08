from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge" / "eo_irrigation_knowledge.json"


@dataclass(frozen=True)
class KnowledgeCard:
    card_id: str
    title: str
    topics: tuple[str, ...]
    keywords: tuple[str, ...]
    principles: tuple[str, ...]
    guardrails: tuple[str, ...]
    sources: tuple[dict[str, str], ...]


def _card_from_record(record: dict[str, Any]) -> KnowledgeCard:
    return KnowledgeCard(
        card_id=str(record["id"]),
        title=str(record["title"]),
        topics=tuple(str(value) for value in record.get("topics", [])),
        keywords=tuple(str(value).lower() for value in record.get("keywords", [])),
        principles=tuple(str(value) for value in record.get("principles", [])),
        guardrails=tuple(str(value) for value in record.get("guardrails", [])),
        sources=tuple(dict(value) for value in record.get("sources", [])),
    )


@lru_cache(maxsize=4)
def load_knowledge_catalog(path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> tuple[KnowledgeCard, ...]:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    cards = tuple(_card_from_record(record) for record in payload.get("cards", []))
    card_ids = [card.card_id for card in cards]
    if not cards:
        raise ValueError(f"Knowledge catalog has no cards: {catalog_path}")
    if len(card_ids) != len(set(card_ids)):
        raise ValueError(f"Knowledge catalog contains duplicate card IDs: {catalog_path}")
    return cards


def _base_context(context: dict[str, Any]) -> dict[str, Any]:
    nested = context.get("copilot_context")
    return nested if isinstance(nested, dict) else context


def _retrieval_text(context: dict[str, Any], task_prompt: str) -> str:
    base = _base_context(context)
    terms = [task_prompt]
    if base.get("weather_forecast_7d"):
        terms.append("weather forecast rain ET0 next week")
    if base.get("previous_years_summary"):
        terms.append("historical season anomaly NDVI NDMI")
    if base.get("live_comparison_summary"):
        terms.append("previous week temporal trend change")
    if base.get("spatial_cluster_summary"):
        terms.append("spatial zone cluster nearby cells")
    if context.get("focus_cell_context"):
        terms.append("selected cell crop type NDVI NDMI irrigation priority")
    if base.get("need_class_counts"):
        terms.append("priority WAIT SOON NOW public")
    return " ".join(terms).lower()


def retrieve_knowledge(
    context: dict[str, Any],
    task_prompt: str,
    *,
    limit: int = 6,
) -> tuple[KnowledgeCard, ...]:
    text = _retrieval_text(context, task_prompt)
    tokens = set(re.findall(r"[\w-]+", text, flags=re.UNICODE))
    scored: list[tuple[int, str, KnowledgeCard]] = []
    for card in load_knowledge_catalog():
        score = 0
        for keyword in card.keywords:
            if " " in keyword:
                score += 4 if keyword in text else 0
            elif keyword in tokens:
                score += 2
        for topic in card.topics:
            if topic.lower() in text:
                score += 2
        if card.card_id == "echmiadzin_decision_contract":
            score += 3
        scored.append((score, card.card_id, card))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [card for score, _, card in scored if score > 0][: max(1, int(limit))]
    if not any(card.card_id == "echmiadzin_decision_contract" for card in selected):
        contract = next(card for card in load_knowledge_catalog() if card.card_id == "echmiadzin_decision_contract")
        if len(selected) >= max(1, int(limit)):
            selected[-1] = contract
        else:
            selected.append(contract)
    return tuple(selected)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4)


def _direction(value: Any, *, tolerance: float = 0.0) -> str | None:
    number = _number(value)
    if number is None:
        return None
    if number > tolerance:
        return "above_or_increasing"
    if number < -tolerance:
        return "below_or_decreasing"
    return "approximately_unchanged"


def build_analytical_signals(context: dict[str, Any]) -> dict[str, Any]:
    base = _base_context(context)
    signals: dict[str, Any] = {}

    forecast = base.get("weather_forecast_7d") or {}
    area = forecast.get("area", forecast) if isinstance(forecast, dict) else {}
    if isinstance(area, dict) and area:
        rain = _number(area.get("rain_mm_7d", area.get("rain_fc_7d")))
        et0 = _number(area.get("et0_mm_7d", area.get("et0_fc_7d")))
        balance = _number(area.get("rain_minus_et0_mm", area.get("rain_minus_et0_fc_7d")))
        if balance is None and rain is not None and et0 is not None:
            balance = round(rain - et0, 4)
        signals["seven_day_weather_reference_balance"] = {
            "forecast_rain_mm": rain,
            "forecast_reference_et0_mm": et0,
            "rain_minus_reference_et0_mm": balance,
            "balance_direction": (
                "forecast_rain_below_reference_demand"
                if balance is not None and balance < -2
                else "forecast_rain_above_reference_demand"
                if balance is not None and balance > 2
                else "forecast_rain_near_reference_demand"
                if balance is not None
                else None
            ),
            "maximum_rain_probability_percent": _number(area.get("precipitation_probability_max")),
            "meaning": "This is a weather-pressure indicator, not a field irrigation amount.",
        }

    historical = base.get("previous_years_summary") or {}
    historical_deltas = historical.get("current_minus_historical") if isinstance(historical, dict) else None
    if isinstance(historical_deltas, dict) and historical_deltas:
        tolerances = {"NDVI_med": 0.01, "NDMI_med": 0.01, "dNDMI": 0.01, "Rain_mm_7d": 1.0, "Temp_C_7d": 0.5}
        signals["current_vs_same_season_history"] = {
            metric: {
                "difference": _number(value),
                "direction": _direction(value, tolerance=tolerances.get(metric, 0.0)),
            }
            for metric, value in historical_deltas.items()
            if _number(value) is not None
        }

    live_change = base.get("live_comparison_summary") or {}
    if isinstance(live_change, dict) and live_change:
        probability_delta = _number(live_change.get("mean_need_probability_delta"))
        signals["change_from_previous_live_week"] = {
            "mean_priority_probability_difference": probability_delta,
            "direction": _direction(probability_delta, tolerance=0.01),
            "class_count_differences": live_change.get("class_count_delta", {}),
            "largest_probability_increases": live_change.get("largest_probability_increases", [])[:5],
        }

    clusters = base.get("spatial_cluster_summary") or {}
    cluster_rows = clusters.get("clusters", []) if isinstance(clusters, dict) else []
    if cluster_rows:
        signals["priority_zone_evidence"] = {
            "zone_count": int(len(cluster_rows)),
            "top_zones": [
                {
                    "cluster_id": row.get("cluster_id"),
                    "area": row.get("boundary_name"),
                    "cell_count": row.get("cell_count"),
                    "dominant_ml_class": row.get("dominant_class"),
                    "mean_ml_probability": row.get("mean_need_probability"),
                    "max_ml_probability": row.get("max_need_probability"),
                }
                for row in cluster_rows[:3]
            ],
        }

    focus = context.get("focus_cell_context") or {}
    if isinstance(focus, dict) and focus:
        current = focus.get("current_cell_snapshot") or {}
        previous = focus.get("previous_live_snapshot") or {}
        current_probability = _number(current.get("need_prob_ml"))
        previous_probability = _number(previous.get("need_prob_ml"))
        change = (
            round(current_probability - previous_probability, 4)
            if current_probability is not None and previous_probability is not None
            else None
        )
        signals["selected_cell_evidence"] = {
            "cell_id": focus.get("selected_cell_id"),
            "rank_by_ml_probability": focus.get("current_rank_by_need_probability"),
            "ml_class": current.get("need_class_ml"),
            "ml_probability": current_probability,
            "change_from_previous_live_probability": change,
            "change_direction": _direction(change, tolerance=0.01),
            "crop_type_system_estimate": current.get("crop_type"),
            "ndvi": _number(current.get("NDVI_med")),
            "ndmi": _number(current.get("NDMI_med")),
            "ndmi_change": _number(current.get("dNDMI")),
        }

    return signals


def build_domain_grounding(
    context: dict[str, Any],
    task_prompt: str,
    *,
    language: str = "en",
    limit: int = 6,
) -> dict[str, Any]:
    cards = retrieve_knowledge(context, task_prompt, limit=limit)
    sources: dict[str, dict[str, str]] = {}
    for card in cards:
        for source in card.sources:
            sources[str(source["id"])] = dict(source)

    return {
        "catalog_version": "2026-08-29",
        "response_language": language,
        "analytical_signals": build_analytical_signals(context),
        "retrieved_principles": [
            {
                "id": card.card_id,
                "title": card.title,
                "principles": list(card.principles),
                "guardrails": list(card.guardrails),
                "source_ids": [source["id"] for source in card.sources],
            }
            for card in cards
        ],
        "approved_sources": list(sources.values()),
        "answer_contract": {
            "keep_separate": [
                "observed EO and weather inputs",
                "rule-based priority classes",
                "next-week class forecast",
                "AI interpretation",
                "future weather scenario",
            ],
            "never_claim": [
                "an exact irrigation volume without a calibrated root-zone water balance",
                "that NDVI or NDMI proves irrigation",
                "that the AI has changed a priority class",
                "that a forecast is an observed event",
            ],
        },
    }


def attach_domain_grounding(
    context: dict[str, Any],
    task_prompt: str,
    *,
    language: str = "en",
) -> dict[str, Any]:
    grounded = dict(context)
    grounded["domain_grounding"] = build_domain_grounding(context, task_prompt, language=language)
    return grounded


def knowledge_sources() -> tuple[dict[str, str], ...]:
    sources: dict[str, dict[str, str]] = {}
    for card in load_knowledge_catalog():
        for source in card.sources:
            sources[str(source["id"])] = dict(source)
    return tuple(sources.values())
