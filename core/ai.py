from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from threading import Lock
import time

from core.knowledge import attach_domain_grounding


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


DEFAULT_COPILOT_MODEL = os.getenv("OPENAI_COPILOT_MODEL", "gpt-5.4-mini")
AI_CACHE_TTL_SECONDS = int(os.getenv("AI_CACHE_TTL_SECONDS", str(12 * 60 * 60)))
AI_WEEKLY_CACHE_TTL_SECONDS = int(os.getenv("AI_WEEKLY_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))
AI_CACHE_MAX_ITEMS = int(os.getenv("AI_CACHE_MAX_ITEMS", "256"))

COPILOT_GUIDANCE = {
    "public_data_policy": [
        "Only cells with valid core observations are included in public analysis.",
        "Do not discuss omitted cells or internal validation labels in public text.",
    ],
    "crop_type_rules": [
        "Live crop type is a stable estimate built from separate historical-year classifications.",
        "Crop type provides context; it does not replace current-week irrigation priority evidence.",
    ],
    "priority_rules": [
        "WAIT, SOON, and NOW are rule-based priority classes from 2-week NDMI level and trend, the cell's own history, and recent rain.",
        "NOW requires dry conditions in two consecutive weeks, so it is a persistent signal rather than a one-week jump.",
        "NOW means high irrigation priority, not an automatic instruction to irrigate.",
        "The next-week forecast is a model estimate of next week's class; present it as a forecast, separate from this week's classes.",
        "Preserve the priority classes and the forecast; present AI interpretation separately.",
        "Prefer spatial priority zones over isolated cells when the evidence supports a cluster.",
    ],
    "evidence_rules": [
        "Interpret current NDMI, dNDMI, NDVI, rain, temperature, temporal features, and historical context together.",
        "Forecast rain and ET0 are supporting next-week evidence, not proof of future irrigation need.",
        "NDMI and NDVI are observational indicators, not direct proof of irrigation events.",
    ],
}

COPILOT_SYSTEM_PROMPT = """
You are an irrigation analysis assistant for the Echmiadzin Water Users Association.
Use only the supplied structured data. Never invent facts, locations, causes, or measurements.
Write concise, practical analysis for irrigation professionals and community reviewers.
The domain_grounding block contains retrieved scientific principles, deterministic analytical signals,
approved source links, and a response contract. Scientific principles explain how to interpret evidence;
they are not measurements from Echmiadzin. Prefer the supplied analytical signals over recalculating them.
Treat the rule-based priority classes, the next-week class forecast, and AI interpretation as separate evidence
layers. Do not overwrite or relabel the priority classes or the forecast.
The NOW class means high irrigation priority, not an automatic command to irrigate.
Use historical behavior, the current reporting week, change from the previous live week, spatial zones,
the next-week class forecast, and the next seven-day weather outlook when those fields are supplied.
Do not publish confidence labels, data-quality labels, or an "insufficient data" section.
The public context already excludes unusable cells. If evidence for a specific claim is absent, omit that claim.
Prefer compact markdown headings and short bullets. Avoid technical model jargon unless the user asks for it.
Write in the requested response language, using natural Eastern Armenian when the language is Armenian.
For Armenian, prefer clear everyday professional phrases over literal technical translation. Keep established
abbreviations such as NDVI, NDMI, ET0, and AI, then explain them in plain Armenian when needed.
For Russian, use clear professional language for irrigation and Earth observation practitioners. Keep established
abbreviations such as NDVI, NDMI, ET0, ДЗЗ, and ИИ, and avoid bureaucratic or overly literal phrasing.
When scientific interpretation materially supports the answer, finish with a short "Method sources" section
("Մեթոդական աղբյուրներ" in Armenian) containing one to three Markdown links selected only from approved_sources.
Never cite a source that was not supplied and never imply that a method source validates a specific field result.
""".strip()


@dataclass(frozen=True)
class CopilotResponse:
    text: str | None
    public_error: str | None
    model: str
    kind: str
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    internal_error: str | None = None


_CACHE: OrderedDict[str, tuple[float, CopilotResponse]] = OrderedDict()
_CACHE_LOCK = Lock()


def build_cache_key(
    context: dict,
    task_prompt: str,
    *,
    language: str,
    model: str = DEFAULT_COPILOT_MODEL,
    kind: str = "answer",
) -> str:
    payload = json.dumps(
        {
            "context": context,
            "task": task_prompt,
            "language": language,
            "model": model,
            "kind": kind,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_cached(key: str, *, ttl_seconds: int = AI_CACHE_TTL_SECONDS) -> CopilotResponse | None:
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is None:
            return None
        created_at, response = cached
        if now - created_at > int(ttl_seconds):
            _CACHE.pop(key, None)
            return None
        _CACHE.move_to_end(key)
        return CopilotResponse(
            text=response.text,
            public_error=response.public_error,
            model=response.model,
            kind=response.kind,
            cached=True,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            internal_error=response.internal_error,
        )


def _store_cached(key: str, response: CopilotResponse) -> None:
    if not response.text:
        return
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), response)
        _CACHE.move_to_end(key)
        while len(_CACHE) > AI_CACHE_MAX_ITEMS:
            _CACHE.popitem(last=False)


def _usage_value(response, name: str) -> int | None:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, None) if usage is not None else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def generate_copilot_result(
    context: dict,
    task_prompt: str,
    *,
    language: str = "en",
    kind: str = "answer",
    max_output_tokens: int = 500,
    model: str = DEFAULT_COPILOT_MODEL,
) -> CopilotResponse:
    public_unavailable = {
        "hy": "AI ամփոփումը ժամանակավորապես հասանելի չէ։",
        "ru": "Обзор ИИ временно недоступен.",
        "en": "The AI briefing is temporarily unavailable.",
    }.get(language, "The AI briefing is temporarily unavailable.")
    public_failed = {
        "hy": "Այս պահին չհաջողվեց պատրաստել AI ամփոփումը։",
        "ru": "Сейчас не удалось подготовить обзор ИИ.",
        "en": "The AI briefing could not be generated right now.",
    }.get(language, "The AI briefing could not be generated right now.")
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return CopilotResponse(None, public_unavailable, model, kind, internal_error="OpenAI client or API key is unavailable")

    grounded_context = attach_domain_grounding(context, task_prompt, language=language)
    cache_key = build_cache_key(grounded_context, task_prompt, language=language, model=model, kind=kind)
    cache_ttl = AI_WEEKLY_CACHE_TTL_SECONDS if kind in {"weekly_summary", "next_week_outlook"} else AI_CACHE_TTL_SECONDS
    cached = _get_cached(cache_key, ttl_seconds=cache_ttl)
    if cached is not None:
        return cached

    language_name = {
        "hy": "natural Eastern Armenian",
        "ru": "natural professional Russian",
        "en": "English",
    }.get(language, "English")
    request_text = json.dumps(
        {
            "response_language": language_name,
            "task": task_prompt,
            "context": grounded_context,
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.responses.create(
            model=model,
            instructions=COPILOT_SYSTEM_PROMPT,
            input=request_text,
            max_output_tokens=int(max_output_tokens),
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            return CopilotResponse(None, public_failed, model, kind, internal_error="OpenAI returned empty output")
        result = CopilotResponse(
            text=text,
            public_error=None,
            model=model,
            kind=kind,
            input_tokens=_usage_value(response, "input_tokens"),
            output_tokens=_usage_value(response, "output_tokens"),
        )
        _store_cached(cache_key, result)
        return result
    except Exception as exc:
        return CopilotResponse(None, public_failed, model, kind, internal_error=f"{type(exc).__name__}: {exc}")


def generate_copilot_response(
    context: dict,
    task_prompt: str,
    max_output_tokens: int = 500,
    *,
    language: str = "en",
    kind: str = "answer",
) -> tuple[str | None, str | None]:
    result = generate_copilot_result(
        context,
        task_prompt,
        language=language,
        kind=kind,
        max_output_tokens=max_output_tokens,
    )
    return result.text, result.public_error


def generate_copilot_summary(context: dict, *, language: str = "en") -> tuple[str | None, str | None]:
    if language == "hy":
        task = (
            "Պատրաստիր շաբաթական գործնական ամփոփում։ Ներկայացրու ընթացիկ պատկերը, "
            "փոփոխությունը նախորդ շաբաթվա և նախորդ տարիների համեմատ, առաջնահերթ գոտիները "
            "և առաջիկա յոթ օրվա եղանակի նշանակությունը։"
        )
    elif language == "ru":
        task = (
            "Подготовь краткий практический обзор за неделю. Опиши текущую ситуацию, изменения относительно "
            "предыдущей недели и исторического поведения, приоритетные территории и значение прогноза погоды "
            "на следующие семь дней."
        )
    else:
        task = (
            "Prepare a practical weekly briefing. Cover the current situation, change from the previous week "
            "and historical behavior, priority zones, and the implications of the next seven-day weather outlook."
        )
    return generate_copilot_response(context, task, max_output_tokens=650, language=language, kind="weekly_summary")


def generate_next_week_outlook(context: dict, *, language: str = "en") -> tuple[str | None, str | None]:
    if language == "hy":
        task = (
            "Պատրաստիր առաջիկա յոթ օրվա գործնական պատկեր։ Սկզբում ամփոփիր այս շաբաթվա դիտարկումները "
            "և մոդելի առաջնահերթությունները, ապա դրանցից առանձին ներկայացրու եղանակային սցենարը։ "
            "Նշիր՝ որ գոտիները կամ բջիջներն արժե առաջինը ստուգել, եթե կանխատեսումն իրականանա։ "
            "Մի տուր ոռոգման ավտոմատ հրահանգ կամ ջրի ճշգրիտ քանակ։"
        )
    elif language == "ru":
        task = (
            "Подготовь практический прогноз на следующие семь дней. Сначала опиши наблюдения текущей недели и "
            "приоритеты модели, затем отдельно представь погодный сценарий. Укажи, какие территории следует "
            "проверить в первую очередь, если прогноз подтвердится. Не давай автоматическую команду на полив "
            "или точную норму воды."
        )
    else:
        task = (
            "Build a practical outlook for the next seven days. Start with this week's observations and ML "
            "priorities, then present the weather scenario separately. Identify which zones or cells should be "
            "checked first if the forecast holds. Do not issue an automatic irrigation command or an exact water depth."
        )
    return generate_copilot_response(context, task, max_output_tokens=650, language=language, kind="next_week_outlook")
