from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any

try:
    from course_data_loader import normalize_course_code
except ImportError:  # pragma: no cover - package style import
    from .course_data_loader import normalize_course_code

try:
    from trace_utils import langsmith_trace
except ImportError:  # pragma: no cover - package style import
    from .trace_utils import langsmith_trace


ALLOWED_ACTIONS = {
    "recommend_schedule",
    "modify_schedule",
    "search_course_options",
    "check_graduation",
    "review_course",
    "explain_current_plan",
    "confirm_final",
    "help",
    "unknown",
}

ALLOWED_OPERATIONS = {
    "replan",
    "add_course",
    "remove_course",
    "replace_course",
    "add_more_ee",
    "add_general_education",
    "remove_category",
    "allow_category",
    "remove_day",
    "allow_day",
    "remove_teacher",
    "allow_teacher",
    "none",
}

VALID_DAY_CODES = {"M", "T", "W", "R", "F", "S", "U"}
DAY_KEYWORDS = {
    "M": ("星期一", "週一", "周一", "禮拜一", "monday", "mon"),
    "T": ("星期二", "週二", "周二", "禮拜二", "tuesday", "tue"),
    "W": ("星期三", "週三", "周三", "禮拜三", "wednesday", "wed"),
    "R": ("星期四", "週四", "周四", "禮拜四", "thursday", "thu", "thur"),
    "F": ("星期五", "週五", "周五", "禮拜五", "friday", "fri"),
    "S": ("星期六", "週六", "周六", "禮拜六", "saturday", "sat"),
    "U": ("星期日", "星期天", "週日", "週天", "周日", "周天", "禮拜日", "禮拜天", "sunday", "sun"),
}

DEFAULT_PREFERENCES = {
    "avoid_difficult_courses": False,
    "balance_with_other_electives": False,
    "prefer_theory_ee_courses": False,
    "preferred_days": [],
    "strict_preferred_days": False,
    "use_review_search": False,
    "review_prefer": "",
    "review_sources": [],
    "allow_live_ptt_review_ranking": False,
    "review_lookup_limit": None,
    "review_timeout": None,
    "review_max_results": None,
}

DEFAULT_INTENT = {
    "action": "unknown",
    "operation": "none",
    "course_names": [],
    "course_codes": [],
    "category": "",
    "teacher_names": [],
    "avoid_days": [],
    "allow_days": [],
    "exclude_time_slots": [],
    "allow_time_slots": [],
    "query_time_slots": [],
    "credit_min": None,
    "credit_max": None,
    "exact_credits": None,
    "preferences": deepcopy(DEFAULT_PREFERENCES),
    "is_persistent_constraint": False,
    "is_explicit_override": False,
    "confidence": 0.0,
    "needs_clarification": False,
    "clarification_question": "",
}

__version__ = "intent-parser-v3"

COURSE_CODE_RE = re.compile(r"\b[A-Za-z]{2,6}\s*[-_]?\s*0?\d{4}(?:\d{0,2})?\b")
PERIOD_ORDER = ["1", "2", "3", "4", "n", "5", "6", "7", "8", "9", "a", "b", "c", "d"]
PERIOD_INTERVALS = {
    "1": (8 * 60, 8 * 60 + 50),
    "2": (9 * 60, 9 * 60 + 50),
    "3": (10 * 60 + 10, 11 * 60),
    "4": (11 * 60 + 10, 12 * 60),
    "n": (12 * 60 + 10, 13 * 60),
    "5": (13 * 60 + 20, 14 * 60 + 10),
    "6": (14 * 60 + 20, 15 * 60 + 10),
    "7": (15 * 60 + 30, 16 * 60 + 20),
    "8": (16 * 60 + 30, 17 * 60 + 20),
    "9": (17 * 60 + 30, 18 * 60 + 20),
    "a": (18 * 60 + 30, 19 * 60 + 20),
    "b": (19 * 60 + 30, 20 * 60 + 20),
    "c": (20 * 60 + 30, 21 * 60 + 20),
    "d": (21 * 60 + 30, 22 * 60 + 20),
}
TIME_PREFIXES = ("早上", "上午", "中午", "下午", "晚上", "晚間", "夜間")
CHINESE_HOURS = {
    "零": 0,
    "一": 1,
    "兩": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _sort_periods(periods: set[str]) -> list[str]:
    return sorted(periods, key=lambda item: PERIOD_ORDER.index(item) if item in PERIOD_ORDER else 99)


def _parse_hour_token(token: str) -> int | None:
    token = str(token or "").strip()
    if not token:
        return None
    if token.isdigit():
        value = int(token)
        return value if 0 <= value <= 24 else None
    if token in CHINESE_HOURS:
        return CHINESE_HOURS[token]
    if token.startswith("十") and len(token) == 2:
        tail = CHINESE_HOURS.get(token[1:])
        return 10 + tail if tail is not None else None
    if "十" in token:
        head, _, tail = token.partition("十")
        head_value = CHINESE_HOURS.get(head, 1 if not head else None)
        tail_value = CHINESE_HOURS.get(tail, 0 if not tail else None)
        if head_value is not None and tail_value is not None:
            return head_value * 10 + tail_value
    return None


def _hour_to_minutes(hour: int, prefix: str = "", global_context: str = "") -> int:
    context = prefix or global_context
    if context in {"下午", "晚上", "晚間", "夜間"} and 1 <= hour <= 11:
        hour += 12
    elif context == "中午" and 1 <= hour <= 5:
        hour += 12
    return hour * 60


def _periods_overlapping(start_minute: int, end_minute: int) -> set[str]:
    if end_minute <= start_minute:
        end_minute += 12 * 60
    periods: set[str] = set()
    for period, (period_start, period_end) in PERIOD_INTERVALS.items():
        if start_minute <= period_start < end_minute:
            periods.add(period)
    return periods


def _global_time_context(text: str) -> str:
    for prefix in ("晚上", "晚間", "夜間", "下午", "中午", "上午", "早上"):
        if prefix in text:
            return prefix
    return ""


def _extract_general_time_periods(message: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(message or "").lower())
    english_compact = re.sub(r"[\s_-]+", "", str(message or "").lower())
    periods: set[str] = set()
    hour_token = r"\d{1,2}|十二|十一|十|九|八|七|六|五|四|三|兩|二|一"
    prefix_token = r"早上|上午|中午|下午|晚上|晚間|夜間"
    global_context = _global_time_context(compact)
    matched_range = False

    range_pattern = re.compile(
        rf"(?P<p1>{prefix_token})?(?P<h1>{hour_token})(?:[:：](?P<m1>\d{{2}})|點)?(?:到|至|~|-)(?P<p2>{prefix_token})?(?P<h2>{hour_token})(?:[:：](?P<m2>\d{{2}})|點)?"
    )
    for match in range_pattern.finditer(compact):
        raw = match.group(0)
        following = compact[match.end() : match.end() + 2]
        has_time_cue = any(prefix in raw for prefix in TIME_PREFIXES) or "點" in raw or ":" in raw or "：" in raw
        start_hour = _parse_hour_token(match.group("h1"))
        end_hour = _parse_hour_token(match.group("h2"))
        if start_hour is None or end_hour is None:
            continue
        if not has_time_cue and max(start_hour, end_hour) < 13:
            continue
        if "學" in following:
            continue
        start_minute = _hour_to_minutes(start_hour, match.group("p1") or "", global_context)
        end_minute = _hour_to_minutes(end_hour, match.group("p2") or match.group("p1") or "", global_context)
        start_minute += int(match.group("m1") or 0)
        end_minute += int(match.group("m2") or 0)
        periods.update(_periods_overlapping(start_minute, end_minute))
        matched_range = True

    if not matched_range:
        direct_pattern = re.compile(rf"(?P<p>{prefix_token})(?P<h>{hour_token})(?:[:：](?P<m>\d{{2}})|點)")
        for match in direct_pattern.finditer(compact):
            hour = _parse_hour_token(match.group("h"))
            if hour is None:
                continue
            start_minute = _hour_to_minutes(hour, match.group("p") or "", global_context) + int(match.group("m") or 0)
            periods.update(_periods_overlapping(start_minute, start_minute + 60))

    if not periods:
        if "早八" in compact or "earlymorning" in english_compact or (
            "early" in english_compact and "morning" in english_compact
        ):
            periods.update({"1", "2"})
        elif "早上" in compact or "上午" in compact or "morning" in english_compact:
            periods.update({"1", "2", "3", "4"})
        elif "中午" in compact or "noon" in english_compact:
            periods.add("n")
        elif "下午" in compact or "afternoon" in english_compact:
            periods.update({"5", "6", "7", "8", "9"})
        elif "晚上" in compact or "晚間" in compact or "夜間" in compact or "evening" in english_compact or "night" in english_compact:
            periods.update({"a", "b", "c", "d"})
    return periods


def _expand_periods_to_slots(periods: list[str] | set[str], day_codes: list[str] | None = None) -> list[str]:
    ordered_periods = _sort_periods(set(periods))
    days = day_codes or sorted(VALID_DAY_CODES)
    return [f"{day}{period}" for day in days for period in ordered_periods]


def _looks_like_slot_token(value: str) -> bool:
    return bool(re.fullmatch(r"[MTWRFSU][0-9A-Za-z]", value.strip(), flags=re.IGNORECASE))


def _mentions_avoid_early_morning(message: str) -> bool:
    lowered = str(message or "").lower()
    compact = re.sub(r"[\s_-]+", "", lowered)
    has_avoid = _contains_any(lowered, ("avoid", "no ", "not ", "don't", "do not", "不要", "避開", "排除"))
    has_early = "早八" in lowered or "earlymorning" in compact or ("early" in compact and "morning" in compact)
    return has_avoid and has_early


def _normalize_time_slot_values(
    values: list[str],
    user_message: str = "",
    include_message_fallback: bool = False,
    day_codes: list[str] | None = None,
) -> list[str]:
    normalized: list[str] = []
    expanded_days = day_codes or _mentioned_days(user_message) or sorted(VALID_DAY_CODES)
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if _looks_like_slot_token(text):
            normalized.append(text[:1].upper() + text[1:].lower())
            continue
        periods = _extract_requested_periods(text)
        if periods:
            normalized.extend(_expand_periods_to_slots(periods, expanded_days))

    if include_message_fallback and _mentions_avoid_early_morning(user_message):
        normalized.extend(_expand_periods_to_slots(["1", "2"], day_codes or sorted(VALID_DAY_CODES)))

    return list(dict.fromkeys(normalized))


def _has_general_time_range(message: str) -> bool:
    compact = re.sub(r"\s+", "", str(message or "").lower())
    hour_token = r"\d{1,2}|十二|十一|十|九|八|七|六|五|四|三|兩|二|一"
    prefix_token = r"早上|上午|中午|下午|晚上|晚間|夜間"
    range_pattern = re.compile(
        rf"(?P<p1>{prefix_token})?(?P<h1>{hour_token})(?:[:：](?P<m1>\d{{2}})|點)?(?:到|至|~|-)(?P<p2>{prefix_token})?(?P<h2>{hour_token})(?:[:：](?P<m2>\d{{2}})|點)?"
    )
    for match in range_pattern.finditer(compact):
        raw = match.group(0)
        following = compact[match.end() : match.end() + 2]
        has_time_cue = any(prefix in raw for prefix in TIME_PREFIXES) or "點" in raw or ":" in raw or "：" in raw
        start_hour = _parse_hour_token(match.group("h1"))
        end_hour = _parse_hour_token(match.group("h2"))
        if start_hour is None or end_hour is None:
            continue
        if not has_time_cue and max(start_hour, end_hour) < 13:
            continue
        if "學" in following:
            continue
        return True
    return False


def _blank_intent(**updates: Any) -> dict:
    intent = deepcopy(DEFAULT_INTENT)
    intent.update(updates)
    if "preferences" in updates:
        prefs = deepcopy(DEFAULT_PREFERENCES)
        prefs.update(updates.get("preferences") or {})
        intent["preferences"] = prefs
    return intent


def _json_object_from_text(text: str) -> dict:
    if not text:
        return {}
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return {}
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_days(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        upper = text.upper()
        if upper in VALID_DAY_CODES:
            normalized.append(upper)
            continue
        lowered = text.lower()
        for code, keywords in DAY_KEYWORDS.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                normalized.append(code)
    return list(dict.fromkeys(normalized))


def _mentioned_days(message: str) -> list[str]:
    lowered = str(message or "").lower()
    days = []
    for code, keywords in DAY_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            days.append(code)
    return days


def _extract_requested_periods(message: str) -> list[str]:
    text = str(message or "").lower()
    compact = re.sub(r"\s+", "", text)
    periods: set[str] = _extract_general_time_periods(message)
    matched_general_range = _has_general_time_range(message)

    range_patterns = (
        r"早八",
        r"8點到10點?",
        r"8點[~-]10點?",
        r"八點到十點?",
        r"八點[~-]十點?",
        r"八到十",
        r"(?<!\d)0?8[:：]00?[~-](?:10|十)[:：]?00?",
        r"(?<!\d)8[~-]10(?!\d)",
    )
    matched_range = False
    if any(re.search(pattern, compact) for pattern in range_patterns):
        periods.update({"1", "2"})
        matched_range = True

    ten_to_noon_patterns = (
        r"10點到12點?",
        r"10點[~-]12點?",
        r"10點到十二點?",
        r"10點[~-]十二點?",
        r"十點到十二點?",
        r"十點[~-]十二點?",
        r"十點到12點?",
        r"十點[~-]12點?",
        r"十到十二",
        r"(?<!\d)10[:：]00?[~-]12[:：]?00?",
        r"(?<!\d)10[~-]12(?!\d)",
    )
    if any(re.search(pattern, compact) for pattern in ten_to_noon_patterns):
        periods.update({"3", "4"})
        matched_range = True

    afternoon_one_to_three_patterns = (
        r"下午(?:1|一)點到(?:3|三)點?",
        r"下午(?:1|一)點[~-](?:3|三)點?",
        r"下午(?:1|一)到(?:3|三)",
        r"13點到15點?",
        r"13點[~-]15點?",
        r"(?<!\d)13[:：]00?[~-]15[:：]?00?",
        r"(?<!\d)13[~-]15(?!\d)",
    )
    if any(re.search(pattern, compact) for pattern in afternoon_one_to_three_patterns):
        periods.update({"5", "6"})
        matched_range = True

    if matched_range or matched_general_range:
        for match in re.finditer(r"第\s*([1-9a-dA-D])\s*節", text):
            period = match.group(1)
            periods.add(period.lower() if period.isalpha() else period)
        return _sort_periods(periods)

    if (
        re.search(r"(?<!\d)0?8[:：]00", compact)
        or re.search(r"(?<!\d)8點", compact)
        or "八點" in compact
        or "第1節" in compact
        or "第一節" in compact
    ):
        periods.add("1")

    if (
        re.search(r"(?<!\d)0?9[:：]00", compact)
        or re.search(r"(?<!\d)9點", compact)
        or "九點" in compact
        or "第2節" in compact
        or "第二節" in compact
    ):
        periods.add("2")

    if (
        re.search(r"(?<!\d)10[:：]00", compact)
        or re.search(r"(?<!\d)10點", compact)
        or "十點" in compact
        or "第3節" in compact
        or "第三節" in compact
    ):
        periods.add("3")

    if (
        re.search(r"(?<!\d)11[:：]00", compact)
        or re.search(r"(?<!\d)11點", compact)
        or "十一點" in compact
        or "第4節" in compact
        or "第四節" in compact
    ):
        periods.add("4")

    for match in re.finditer(r"第\s*([1-9a-dA-D])\s*節", text):
        period = match.group(1)
        periods.add(period.lower() if period.isalpha() else period)
    return _sort_periods(periods)


def _requested_time_slots_from_message(message: str, day_codes: list[str] | None = None) -> list[str]:
    periods = _extract_requested_periods(message)
    if not periods:
        return []
    days = list(day_codes or sorted(VALID_DAY_CODES))
    return [f"{day}{period}" for day in days for period in periods]


def _is_course_option_query(message: str) -> bool:
    lowered = str(message or "").lower()
    course_words = ("課", "通識", "選修", "必修", "外文", "體育", "ge", "gec", "course")
    has_query_words = _contains_any(
        lowered,
        (
            "有什麼課",
            "有哪些課",
            "有沒有",
            "什麼課",
            "哪些課",
            "可以選",
            "可選",
            "候選",
            "找課",
            "查課",
            "找",
            "查",
            "搜尋",
            "search",
            "course options",
        ),
    )
    if not has_query_words and (
        re.search(r"(?:有什麼|有哪些|有沒有|哪幾門|哪一些|什麼).*(?:課|通識|選修|必修|外文|體育|ge|gec)", lowered)
        or re.search(r"(?:課|通識|選修|必修|外文|體育|ge|gec).*(?:可以選|可選|推薦|候選)", lowered)
    ):
        has_query_words = True
    has_time = bool(_extract_requested_periods(message))
    mentions_course = _contains_any(lowered, course_words)
    return has_query_words and has_time and mentions_course


def _extract_course_codes(message: str) -> list[str]:
    codes = [normalize_course_code(match.group(0)) for match in COURSE_CODE_RE.finditer(message or "")]
    return list(dict.fromkeys([code for code in codes if code]))


def _clean_course_phrase(text: str) -> str:
    text = str(text or "")
    text = re.split(r"[，。！？!?;；\n]", text, maxsplit=1)[0]
    text = re.sub(r"^(但|可是|不過|然後|所以|我|幫我|請|請你|想要|想|要|把|將|這門|這堂|這個|那門|那堂|那個)\s*", "", text).strip()
    text = re.sub(r"(這門|這堂|這個|那門|那堂|那個)$", "", text).strip()
    text = re.sub(r"(幫我)?(換一門|換掉|替換|重新排|重排|排課|課表).*$", "", text).strip()
    text = re.sub(r"^(修|上|加|加入|加回|補回)\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip(" ：:。,.，")
    if text in {"課", "課程", "一門課", "一堂課"}:
        return ""
    return text


def _extract_after_patterns(message: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return _clean_course_phrase(match.group(1))
    return ""


def _course_name_from_state(state_snapshot: dict | None) -> str:
    state_snapshot = state_snapshot or {}
    for key in ("last_reviewed_course", "last_reviewed_course_name", "last_course_name"):
        value = str(state_snapshot.get(key) or "").strip()
        if value:
            return value
    current = state_snapshot.get("current_recommended_courses") or state_snapshot.get("current_schedule_courses") or []
    if isinstance(current, list) and current:
        last = current[-1]
        if isinstance(last, dict):
            return str(last.get("course_name_zh") or last.get("name") or last.get("code") or "").strip()
    return ""


def _resolve_pronoun_course(message: str, state_snapshot: dict | None) -> str:
    if _contains_any(message, ("這門", "這堂", "這個", "那門", "那堂", "那個", "剛剛那門", "剛剛那堂", "它", "他")):
        return _course_name_from_state(state_snapshot)
    return ""


def _extract_credit_intent(message: str) -> dict[str, int | None]:
    text = str(message or "")
    credit_unit = r"(?:學?分|credits?|units?)"
    range_match = re.search(rf"(\d{{1,2}})\s*(?:-|~|到|至|to)\s*(\d{{1,2}})\s*{credit_unit}", text, flags=re.IGNORECASE)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        return {"credit_min": min(low, high), "credit_max": max(low, high), "exact_credits": None}
    at_least = re.search(rf"(?:至少|最少|不少於|>=|at\s*least|min(?:imum)?)\s*(\d{{1,2}})\s*{credit_unit}", text, flags=re.IGNORECASE)
    at_most = re.search(rf"(?:最多|至多|不要超過|不超過|<=|at\s*most|max(?:imum)?|no\s*more\s*than)\s*(\d{{1,2}})\s*{credit_unit}", text, flags=re.IGNORECASE)
    exact = re.search(rf"(\d{{1,2}})\s*{credit_unit}", text, flags=re.IGNORECASE)
    if at_least:
        return {"credit_min": int(at_least.group(1)), "credit_max": None, "exact_credits": None}
    if at_most:
        return {"credit_min": None, "credit_max": int(at_most.group(1)), "exact_credits": None}
    if exact:
        return {"credit_min": None, "credit_max": None, "exact_credits": int(exact.group(1))}
    return {"credit_min": None, "credit_max": None, "exact_credits": None}


def _is_reduce_non_ee_request(message: str) -> bool:
    """Detect requests like 「我不要那麼多不是電機系的課」.

    This is not a request to remove a course literally named
    「那麼多不是電機系的課」. It means replace/reduce non-EE courses first.
    """
    lowered = str(message or "").lower()
    non_ee_terms = (
        "不是電機系",
        "非電機系",
        "不是電機",
        "非電機",
        "不是ee",
        "不是 ee",
        "non-ee",
        "non ee",
        "外系",
        "外系課",
        "其他系",
    )
    reduce_terms = ("不要那麼多", "不要這麼多", "太多", "少一點", "減少", "降低", "換成", "改成", "不要太多")
    return any(term in lowered for term in non_ee_terms) and any(term in lowered for term in reduce_terms)


def _rule_based_parse(user_message: str, state_snapshot: dict | None = None) -> dict:
    message = str(user_message or "").strip()
    lowered = message.lower()
    if not message:
        return _blank_intent(action="help", confidence=0.8)

    course_codes = _extract_course_codes(message)
    mentioned_days = _mentioned_days(message)

    if _contains_any(lowered, ("我決定好了", "決定好了", "就這樣", "final", "done", "confirm", "ok", "確定", "確定了", "確認最終")):
        return _blank_intent(action="confirm_final", confidence=0.95)

    if _contains_any(lowered, ("help", "幫助", "怎麼用", "可以做什麼")):
        return _blank_intent(action="help", confidence=0.85)

    if _contains_any(lowered, ("目前課表", "現在課表", "為什麼這樣排", "解釋課表", "current plan")):
        return _blank_intent(action="explain_current_plan", confidence=0.85)

    if _contains_any(lowered, ("畢業", "graduation")) and _contains_any(lowered, ("缺", "還缺", "缺什麼", "缺哪些", "門檻", "requirement")):
        return _blank_intent(action="check_graduation", operation="none", confidence=0.95)

    if _is_course_option_query(message):
        query_time_slots = _requested_time_slots_from_message(message, mentioned_days)
        sources = ["ptt_rag", "ptt"]
        if _contains_any(lowered, ("dcard", "網路", "網站", "web")) and "web" not in sources:
            sources.append("web")
        review_prefer = ""
        if _contains_any(lowered, ("甜", "甜度", "高分")):
            review_prefer = "sweetness"
        elif _contains_any(lowered, ("涼", "涼度", "輕鬆", "好過", "不要太硬", "不太硬", "比較輕鬆")):
            review_prefer = "coolness"
        prefer_ge = _contains_any(lowered, ("通識", "ge", "gec", "general education"))
        return _blank_intent(
            action="search_course_options",
            operation="none",
            query_time_slots=query_time_slots,
            preferences={
                "use_review_search": _contains_any(lowered, ("ptt", "dcard", "網路", "網站", "評語", "評價", "心得", "甜", "涼", "輕鬆", "好過", "不要太硬", "不太硬")),
                "review_sources": sources,
                "review_prefer": review_prefer,
                "avoid_difficult_courses": _contains_any(lowered, ("輕鬆", "好過", "不要太硬", "不太硬", "比較輕鬆")),
                "prefer_general_education_courses": prefer_ge,
            },
            confidence=0.94,
        )

    credit = _extract_credit_intent(message)
    has_credit_request = any(value is not None for value in credit.values())
    is_schedule_planning_request = has_credit_request or _contains_any(
        lowered,
        ("幫我排", "排課", "排一份", "課表", "選課", "推薦", "規劃", "schedule"),
    )
    if is_schedule_planning_request:
        exclude_time_slots = _requested_time_slots_from_message(message, mentioned_days)
        preferences = {}
        if _contains_any(lowered, ("輕鬆", "不要太硬", "太硬", "平衡", "涼一點", "好過")):
            preferences.update({"avoid_difficult_courses": True, "balance_with_other_electives": True})
        if _contains_any(lowered, ("少一點實驗", "少點實驗", "少一點實驗課", "少點實驗課")):
            preferences.update(
                {
                    "initial_lab_count": 0,
                    "reduce_lab_courses": True,
                    "balance_with_other_electives": True,
                }
            )
        if _contains_any(lowered, ("多一點通識", "多點通識", "多一些通識", "多放通識", "多排通識")):
            preferences.update(
                {
                    "prefer_general_education_courses": True,
                    "requested_general_education_count": 1,
                    "balance_with_other_electives": True,
                }
            )
        if _contains_any(lowered, ("ptt", "dcard", "網路", "網站", "評語", "評價", "心得", "甜", "涼")):
            sources = ["ptt_rag", "ptt"]
            if _contains_any(lowered, ("dcard", "網路", "網站", "web")) and "web" not in sources:
                sources.append("web")
            review_prefer = "sweetness" if _contains_any(lowered, ("甜", "甜度", "高分")) else "coolness"
            preferences.update(
                {
                    "use_review_search": True,
                    "review_sources": sources,
                    "review_prefer": review_prefer,
                    "allow_live_ptt_review_ranking": "ptt" in lowered,
                    "review_lookup_limit": 3,
                    "review_timeout": 3,
                    "review_max_results": 2,
                }
            )
        return _blank_intent(
            action="recommend_schedule",
            operation="none",
            exclude_time_slots=exclude_time_slots,
            avoid_days=[] if exclude_time_slots else mentioned_days,
            preferences=preferences,
            is_persistent_constraint=bool(exclude_time_slots or mentioned_days),
            confidence=0.96 if has_credit_request else 0.9,
            **credit,
        )

    # Handle day allow rules before removals.
    if mentioned_days and _contains_any(lowered, ("可以接受", "可以", "恢復", "允許", "不排除", "沒關係", "ok", "accept")) and not _contains_any(lowered, ("不要", "不想", "避開", "不上")):
        return _blank_intent(
            action="modify_schedule",
            operation="allow_day",
            allow_days=mentioned_days,
            is_explicit_override=True,
            confidence=0.92,
        )

    remove_day_terms = ("不要", "不想", "避開", "不上", "不要有", "去掉", "移除", "刪掉", "拿掉", "退掉", "avoid", "remove", "drop")
    if mentioned_days and _contains_any(lowered, remove_day_terms):
        exclude_time_slots = _requested_time_slots_from_message(message, mentioned_days)
        return _blank_intent(
            action="modify_schedule",
            operation="remove_day",
            avoid_days=[] if exclude_time_slots else mentioned_days,
            exclude_time_slots=exclude_time_slots,
            is_persistent_constraint=True,
            confidence=0.94,
        )

    exclude_time_slots = _requested_time_slots_from_message(message, [])
    if exclude_time_slots and _contains_any(lowered, remove_day_terms):
        return _blank_intent(
            action="modify_schedule",
            operation="remove_day",
            exclude_time_slots=exclude_time_slots,
            is_persistent_constraint=True,
            confidence=0.9,
        )

    mentions_lab = _contains_any(lowered, ("實驗課", "實驗", "lab", "laboratory"))
    if mentions_lab and _contains_any(lowered, ("可以", "加回", "解除", "允許", "恢復", "不排除")) and not _contains_any(lowered, ("不要", "不想", "不修", "不上")):
        return _blank_intent(
            action="modify_schedule",
            operation="allow_category",
            category="lab",
            is_explicit_override=True,
            confidence=0.9,
        )

    if mentions_lab and _contains_any(lowered, ("不要", "不想", "不修", "不上", "去掉", "移除", "排除", "avoid")):
        return _blank_intent(
            action="modify_schedule",
            operation="remove_category",
            category="lab",
            is_persistent_constraint=True,
            confidence=0.93,
        )

    if _contains_any(lowered, ("老師", "評價", "心得", "ptt", "dcard", "review", "涼", "甜", "好不好")):
        course = _extract_after_patterns(
            message,
            [
                r"(?:想問|請問|查|看看|看一下)?\s*(.+?)(?:的)?(?:老師)?(?:評價|心得|review|好不好|涼不涼|甜不甜)",
                r"(.+?)(?:老師|教授).*(?:評價|心得|review|好不好)",
            ],
        )
        if not course:
            course = _resolve_pronoun_course(message, state_snapshot)
        intent = _blank_intent(
            action="review_course",
            operation="none",
            course_names=[course] if course else [],
            course_codes=course_codes,
            preferences={"use_review_search": True},
            confidence=0.88 if course or course_codes else 0.48,
        )
        if not course and not course_codes:
            intent["needs_clarification"] = True
            intent["clarification_question"] = "你想查哪一門課的老師評價？"
        return intent

    if _contains_any(lowered, ("輕鬆", "不要太硬", "太硬", "平衡", "涼一點", "好過")):
        return _blank_intent(
            action="modify_schedule",
            operation="replan",
            preferences={"avoid_difficult_courses": True, "balance_with_other_electives": True},
            confidence=0.9,
        )

    # Handle reduce-non-EE before generic removal.
    # This means replace/reduce non-EE courses.
    # Do not treat the phrase as a literal course name.
    if _is_reduce_non_ee_request(message):
        return _blank_intent(
            action="modify_schedule",
            operation="add_more_ee",
            course_count=2,
            preferences={"replace_non_ee_first": True, "reduce_non_ee_courses": True},
            confidence=0.94,
        )

    if any(value is not None for value in credit.values()):
        return _blank_intent(
            action="recommend_schedule",
            operation="none",
            confidence=0.92,
            **credit,
        )

    if _contains_any(lowered, ("排課", "課表", "選課", "推薦", "規劃", "schedule")):
        return _blank_intent(action="recommend_schedule", operation="none", confidence=0.82, **credit)

    if _contains_any(lowered, ("加回", "補回", "加入", "加", "想上", "想修", "我要上", "我要修", "add")) and not _contains_any(lowered, ("不要", "不想", "不修", "不上", "不加", "不用")):
        day_prefs = {
            "preferred_days": mentioned_days,
            "strict_preferred_days": bool(mentioned_days),
        } if mentioned_days else {}
        if _contains_any(lowered, ("通識", "ge", "general education")):
            return _blank_intent(action="modify_schedule", operation="add_general_education", preferences=day_prefs, confidence=0.88)
        if _contains_any(lowered, ("電機系", "電機課", "ee課", "ee 課", "eecs", "電資")):
            prefs = {
                "prefer_theory_ee_courses": _contains_any(lowered, ("理論", "非實驗", "non-lab", "non lab")),
                **day_prefs,
            }
            return _blank_intent(action="modify_schedule", operation="add_more_ee", preferences=prefs, confidence=0.86)
        course = _extract_after_patterns(
            message,
            [
                r"(?:加回|補回|加入|加|add)\s*(.+)",
                r"(?:我想上|我想修|想上|想修|我要上|我要修)\s*(.+)",
            ],
        ) or _resolve_pronoun_course(message, state_snapshot)
        if course or course_codes:
            return _blank_intent(
                action="modify_schedule",
                operation="add_course",
                course_names=[course] if course else [],
                course_codes=course_codes,
                preferences=day_prefs,
                is_explicit_override=True,
                confidence=0.9,
            )

    if _contains_any(lowered, ("不想上", "不要上", "不上", "不想修", "不要修", "不修", "不想要", "不要", "去掉", "移除", "刪掉", "拿掉", "退掉", "drop", "remove")):
        course = _extract_after_patterns(
            message,
            [
                r"(?:不想上|不要上|不上|不想修|不要修|不修|不想要|不要|去掉|移除|刪掉|拿掉|退掉|drop|remove)\s*(.+)",
            ],
        ) or _resolve_pronoun_course(message, state_snapshot)
        if course or course_codes:
            return _blank_intent(
                action="modify_schedule",
                operation="remove_course",
                course_names=[course] if course else [],
                course_codes=course_codes,
                is_persistent_constraint=True,
                confidence=0.88,
            )

    if _contains_any(lowered, ("換掉", "替換", "換一門", "replace")):
        course = _extract_after_patterns(message, [r"(?:換掉|替換|replace)\s*(.+)"]) or _resolve_pronoun_course(message, state_snapshot)
        return _blank_intent(
            action="modify_schedule",
            operation="replace_course",
            course_names=[course] if course else [],
            course_codes=course_codes,
            confidence=0.82 if course or course_codes else 0.5,
            needs_clarification=not bool(course or course_codes),
            clarification_question="你想換掉目前課表中的哪一門課？" if not (course or course_codes) else "",
        )

    return _blank_intent(confidence=0.25)


def _llm_prompt(user_message: str, state_snapshot: dict | None) -> str:
    return f"""
You are a strict intent parser for a course planning agent.
Return exactly one JSON object. Do not answer the user.

The LLM must only translate natural language into the schema. It must NOT recommend courses,
compute graduation progress, check timetable conflicts, invent reviews, or invent course availability.

Allowed actions: {sorted(ALLOWED_ACTIONS)}
Allowed operations: {sorted(ALLOWED_OPERATIONS)}
Day codes: M Monday, T Tuesday, W Wednesday, R Thursday, F Friday, S Saturday, U Sunday.

Schema:
{json.dumps(DEFAULT_INTENT, ensure_ascii=False, indent=2)}

Important examples:
- "我想知道畢業還缺什麼" -> action check_graduation, operation none
- "我要修20學分" -> action recommend_schedule, exact_credits 20
- "我不想上實驗課" -> action modify_schedule, operation remove_category, category lab, persistent true
- "但我想上固態電子實驗" -> action modify_schedule, operation add_course, explicit override true
- "我不要偏微分" -> remove_course with course_names ["偏微分"]
- "我想加回偏微分" -> add_course with explicit override true
- "我不要星期五的課" -> remove_day avoid_days ["F"]
- "星期五可以接受" -> allow_day allow_days ["F"]
- "早上十點到12點有什麼課可以選，PTT 哪個最甜" -> search_course_options query_time_slots and preferences.use_review_search true
- Teacher/course reviews -> review_course and preferences.use_review_search true
- Lighter schedule -> modify_schedule replan and avoid_difficult_courses/balance true

State snapshot:
{json.dumps(state_snapshot or {}, ensure_ascii=False, indent=2, default=str)}

User message:
{user_message}
""".strip()


@langsmith_trace("intent.ollama_parse", run_type="llm")
def _ollama_parse(prompt: str, model: str, timeout: int) -> dict:
    try:
        completed = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    return _json_object_from_text(completed.stdout)


@langsmith_trace("intent.gemini_parse", run_type="llm")
def _gemini_parse(prompt: str, model: str, timeout: int) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {}
    try:
        timeout = max(3, min(int(os.environ.get("GEMINI_TIMEOUT_SECONDS", timeout)), 20))
    except ValueError:
        timeout = min(timeout, 12)
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return {}
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    text = "\n".join(str(part.get("text") or "") for part in parts if isinstance(part, dict)).strip()
    return _json_object_from_text(text)


@langsmith_trace("intent.llm_parse")
def _llm_parse(
    user_message: str,
    state_snapshot: dict | None,
    model: str,
    provider: str = "ollama",
    timeout: int = 12,
) -> dict:
    prompt = _llm_prompt(user_message, state_snapshot)
    if provider == "gemini":
        return _gemini_parse(prompt, model, timeout)
    return _ollama_parse(prompt, model, timeout)


def validate_intent(raw_intent: dict | None, user_message: str = "") -> dict:
    raw_intent = raw_intent if isinstance(raw_intent, dict) else {}
    intent = deepcopy(DEFAULT_INTENT)

    action = str(raw_intent.get("action") or "unknown").strip()
    operation = str(raw_intent.get("operation") or "none").strip()
    if operation == "":
        operation = "none"
    intent["action"] = action if action in ALLOWED_ACTIONS else "unknown"
    intent["operation"] = operation if operation in ALLOWED_OPERATIONS else "none"

    intent["course_names"] = list(dict.fromkeys(_as_list(raw_intent.get("course_names"))))
    legacy_name = str(raw_intent.get("course_name") or "").strip()
    if legacy_name and legacy_name not in intent["course_names"]:
        intent["course_names"].insert(0, legacy_name)

    raw_codes = _as_list(raw_intent.get("course_codes")) + _extract_course_codes(user_message)
    intent["course_codes"] = list(dict.fromkeys([normalize_course_code(code) for code in raw_codes if normalize_course_code(code)]))

    category = str(raw_intent.get("category") or "").strip().lower()
    if category in {"實驗", "實驗課", "laboratory", "experiment"}:
        category = "lab"
    intent["category"] = category
    intent["teacher_names"] = list(dict.fromkeys(_as_list(raw_intent.get("teacher_names"))))
    intent["avoid_days"] = _normalize_days(_as_list(raw_intent.get("avoid_days")))
    intent["allow_days"] = _normalize_days(_as_list(raw_intent.get("allow_days")))
    intent["exclude_time_slots"] = _normalize_time_slot_values(
        _as_list(raw_intent.get("exclude_time_slots")),
        user_message,
        include_message_fallback=True,
    )
    intent["allow_time_slots"] = _normalize_time_slot_values(_as_list(raw_intent.get("allow_time_slots")), user_message)
    intent["query_time_slots"] = _normalize_time_slot_values(_as_list(raw_intent.get("query_time_slots")), user_message)

    intent["credit_min"] = _as_optional_int(raw_intent.get("credit_min"))
    intent["credit_max"] = _as_optional_int(raw_intent.get("credit_max"))
    intent["exact_credits"] = _as_optional_int(raw_intent.get("exact_credits"))
    course_count = _as_optional_int(raw_intent.get("course_count"))
    intent["course_count"] = max(1, min(course_count, 10)) if course_count is not None else None
    if intent["exact_credits"] is not None:
        intent["credit_min"] = None
        intent["credit_max"] = None
    elif intent["credit_min"] is not None and intent["credit_max"] is not None and intent["credit_min"] > intent["credit_max"]:
        intent["credit_min"], intent["credit_max"] = intent["credit_max"], intent["credit_min"]

    prefs = deepcopy(DEFAULT_PREFERENCES)
    raw_prefs = raw_intent.get("preferences") if isinstance(raw_intent.get("preferences"), dict) else {}
    # Backward compatibility with older flattened LLM parser outputs.
    for key in DEFAULT_PREFERENCES:
        if key in raw_intent:
            raw_prefs[key] = raw_intent[key]
    prefs["avoid_difficult_courses"] = _as_bool(raw_prefs.get("avoid_difficult_courses", False))
    prefs["balance_with_other_electives"] = _as_bool(raw_prefs.get("balance_with_other_electives", False))
    prefs["prefer_theory_ee_courses"] = _as_bool(raw_prefs.get("prefer_theory_ee_courses", False))
    prefs["preferred_days"] = _normalize_days(_as_list(raw_prefs.get("preferred_days")))
    prefs["strict_preferred_days"] = _as_bool(raw_prefs.get("strict_preferred_days", False))
    prefs["use_review_search"] = _as_bool(raw_prefs.get("use_review_search", False))
    prefs["allow_live_ptt_review_ranking"] = _as_bool(raw_prefs.get("allow_live_ptt_review_ranking", False))
    prefs["review_lookup_limit"] = _as_optional_int(raw_prefs.get("review_lookup_limit"))
    prefs["review_timeout"] = _as_optional_int(raw_prefs.get("review_timeout"))
    prefs["review_max_results"] = _as_optional_int(raw_prefs.get("review_max_results"))
    prefs["initial_lab_count"] = _as_optional_int(raw_prefs.get("initial_lab_count"))
    prefs["reduce_lab_courses"] = _as_bool(raw_prefs.get("reduce_lab_courses", False))
    prefs["prefer_general_education_courses"] = _as_bool(raw_prefs.get("prefer_general_education_courses", False))
    prefs["requested_general_education_count"] = _as_optional_int(raw_prefs.get("requested_general_education_count"))
    prefs["replace_non_ee_first"] = _as_bool(raw_prefs.get("replace_non_ee_first", False))
    prefs["reduce_non_ee_courses"] = _as_bool(raw_prefs.get("reduce_non_ee_courses", False))
    review_prefer = str(raw_prefs.get("review_prefer") or "").strip().lower()
    prefs["review_prefer"] = review_prefer if review_prefer in {"coolness", "sweetness"} else ""
    sources = []
    for source in _as_list(raw_prefs.get("review_sources")):
        value = source.lower()
        if value in {"local_cache", "ptt_rag", "ptt", "web"}:
            sources.append(value)
    prefs["review_sources"] = list(dict.fromkeys(sources))
    intent["preferences"] = prefs

    intent["is_persistent_constraint"] = _as_bool(raw_intent.get("is_persistent_constraint", False))
    if intent["exclude_time_slots"] and _contains_any(user_message, ("avoid", "no ", "not ", "don't", "do not", "不要", "避開", "排除")):
        intent["is_persistent_constraint"] = True
    intent["is_explicit_override"] = _as_bool(raw_intent.get("is_explicit_override", False))
    try:
        confidence = float(raw_intent.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    intent["confidence"] = max(0.0, min(1.0, confidence))
    intent["needs_clarification"] = _as_bool(raw_intent.get("needs_clarification", False))
    intent["clarification_question"] = str(raw_intent.get("clarification_question") or "").strip()

    if intent["confidence"] < 0.45 and intent["action"] not in {"help", "unknown"}:
        intent["needs_clarification"] = True
    if intent["action"] == "review_course" and not intent["course_names"] and not intent["course_codes"]:
        intent["needs_clarification"] = True
        intent["clarification_question"] = intent["clarification_question"] or "你想查哪一門課的老師評價？"
    if intent["operation"] in {"add_course", "remove_course", "replace_course"} and not intent["course_names"] and not intent["course_codes"]:
        intent["needs_clarification"] = True
        intent["clarification_question"] = intent["clarification_question"] or "你指的是哪一門課？"

    return intent


def _specificity_score(intent: dict) -> int:
    if intent.get("course_names") or intent.get("course_codes"):
        return 3
    if intent.get("teacher_names"):
        return 2
    if intent.get("category") or intent.get("avoid_days") or intent.get("allow_days"):
        return 1
    return 0


def _merge_rule_and_llm(rule_intent: dict, llm_intent: dict) -> dict:
    if not llm_intent or llm_intent.get("action") == "unknown":
        return rule_intent
    if rule_intent.get("action") == "unknown" or rule_intent.get("confidence", 0.0) < 0.45:
        return llm_intent

    # Safer interpretations: explicit course-level add/remove beats broad replan.
    rule_op = rule_intent.get("operation")
    llm_op = llm_intent.get("operation")
    specific_ops = {"add_course", "remove_course", "replace_course"}
    if rule_op in specific_ops and llm_op not in specific_ops:
        return rule_intent
    if llm_op in specific_ops and rule_op not in specific_ops:
        return llm_intent

    # Explicit override should override an older persistent exclusion.
    if llm_intent.get("is_explicit_override") and not rule_intent.get("is_explicit_override"):
        return llm_intent
    if rule_intent.get("is_explicit_override") and not llm_intent.get("is_explicit_override"):
        return rule_intent

    if _specificity_score(rule_intent) > _specificity_score(llm_intent):
        return rule_intent
    if _specificity_score(llm_intent) > _specificity_score(rule_intent):
        return llm_intent

    if rule_intent.get("action") != llm_intent.get("action") or rule_op != llm_op:
        cautious = deepcopy(rule_intent if rule_intent.get("confidence", 0.0) >= llm_intent.get("confidence", 0.0) else llm_intent)
        cautious["needs_clarification"] = True
        cautious["confidence"] = min(cautious.get("confidence", 0.5), 0.55)
        cautious["clarification_question"] = cautious.get("clarification_question") or "我不確定你是想改課表、查評價，還是檢查畢業進度，可以再指定一下嗎？"
        return cautious

    return rule_intent if rule_intent.get("confidence", 0.0) >= llm_intent.get("confidence", 0.0) else llm_intent


@langsmith_trace("intent.parse_user_intent")
def parse_user_intent(
    user_message: str,
    state_snapshot: dict | None = None,
    use_llm: bool = True,
    model: str = "phi4-mini:latest",
    provider: str = "ollama",
) -> dict:
    """Parse a natural Chinese/English course-planning request into a validated intent.

    The returned object is safe to pass to deterministic tools. It never contains course
    recommendations, graduation decisions, timetable judgments, or invented review facts.
    """
    rule_intent = validate_intent(_rule_based_parse(user_message, state_snapshot), user_message)
    rule_intent["parser_source"] = "rule_based"

    provider = str(provider or "ollama").lower()
    if not use_llm:
        return rule_intent
    if provider != "gemini" and rule_intent.get("confidence", 0.0) >= 0.78:
        return rule_intent
    if provider == "gemini" and rule_intent.get("confidence", 0.0) >= 0.94:
        return rule_intent

    llm_raw = _llm_parse(user_message, state_snapshot, model=model, provider=provider)
    llm_intent = validate_intent(llm_raw, user_message) if llm_raw else {}
    if llm_intent:
        llm_intent["parser_source"] = "gemini_json" if provider == "gemini" else "llm_json"

    merged = _merge_rule_and_llm(rule_intent, llm_intent) if llm_intent else rule_intent
    if llm_intent and merged not in (rule_intent, llm_intent):
        merged["parser_source"] = "hybrid_conflict_safe"
    elif merged is llm_intent:
        merged["parser_source"] = "gemini_json" if provider == "gemini" else "llm_json"
    else:
        merged["parser_source"] = "rule_based"
    return validate_intent(merged, user_message) | {"parser_source": merged.get("parser_source", "rule_based")}
