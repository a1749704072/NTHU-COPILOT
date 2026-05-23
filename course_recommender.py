from __future__ import annotations

import re
from typing import Any

import pandas as pd

try:
    from course_data_loader import normalize_course_code
    from course_review_searcher import search_course_reviews
    from schedule_checker import check_plan_conflicts, parse_time_slots
    from trace_utils import langsmith_trace
except ImportError:  # pragma: no cover - supports package-style imports
    from .course_data_loader import normalize_course_code
    from .course_review_searcher import search_course_reviews
    from .schedule_checker import check_plan_conflicts, parse_time_slots
    from .trace_utils import langsmith_trace


TARGET_SEMESTER = "11420"
NORMAL_SEMESTER_MIN_CREDITS = 16.0
NORMAL_SEMESTER_MAX_CREDITS = 25.0
GENERAL_EDUCATION_PREFIXES = ("GE", "GEC")
LANGUAGE_PREFIXES = ("LANG", "FL", "CL", "CLC")
SOFT_ELECTIVE_PREFIXES = GENERAL_EDUCATION_PREFIXES + LANGUAGE_PREFIXES
PE_PREFIXES = ("PE",)
TECHNICAL_ELECTIVE_PREFIXES = ("EE", "EECS", "CS")
HARD_ELECTIVE_PREFIXES = ("EE", "EECS", "MATH", "PHYS", "CHEM", "CS")
HARD_COURSE_KEYWORDS = (
    "微積分",
    "線性代數",
    "機率",
    "統計",
    "偏微分",
    "複變",
    "演算法",
    "電磁",
    "電子",
    "電路",
    "訊號",
    "資料結構",
    "機器學習",
    "力學",
    "calculus",
    "linear algebra",
    "probability",
    "statistics",
    "algorithm",
    "electromagnet",
    "electronics",
    "circuits",
    "signals",
    "data structures",
    "machine learning",
)
LAB_COURSE_CODES = {
    "EE2230",
    "EE4150",
    "EE3840",
    "EE2405",
    "EE4320",
    "EE4650",
    "EE3662",
    "EE4292",
}
EXCLUDED_SOFT_FILL_KEYWORDS = ("專題研究", "論文", "書報討論", "seminar", "thesis", "research")
DAY_LABELS = {
    "M": "Monday",
    "T": "Tuesday",
    "W": "Wednesday",
    "R": "Thursday",
    "F": "Friday",
    "S": "Saturday",
    "U": "Sunday",
}
REQUEST_DAY_KEYWORDS = {
    "M": ("星期一", "週一", "禮拜一", "monday"),
    "T": ("星期二", "週二", "禮拜二", "tuesday"),
    "W": ("星期三", "週三", "禮拜三", "wednesday"),
    "R": ("星期四", "週四", "禮拜四", "thursday"),
    "F": ("星期五", "週五", "禮拜五", "friday"),
    "S": ("星期六", "週六", "禮拜六", "saturday"),
    "U": ("星期日", "星期天", "週日", "週天", "禮拜日", "禮拜天", "sunday"),
}
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


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _target_only(target_df: pd.DataFrame) -> pd.DataFrame:
    df = target_df.copy()
    if "normalized_course_code" not in df.columns:
        source = "raw_course_code" if "raw_course_code" in df.columns else "科號"
        df["normalized_course_code"] = df[source].map(normalize_course_code)
    if "term" not in df.columns:
        source = "raw_course_code" if "raw_course_code" in df.columns else "科號"
        df["term"] = df[source].astype(str).str.extract(r"^(\d{5})", expand=False).fillna("")
    df["term"] = df["term"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.loc[df["term"] == TARGET_SEMESTER].copy()


def _course_to_dict(row: pd.Series, reason: str = "", requirement_code: str = "") -> dict:
    credits = row.get("credits")
    credits_value = 0.0 if pd.isna(credits) else float(credits)
    course = {
        "code": _clean(row.get("normalized_course_code")),
        "raw_course_code": _clean(row.get("raw_course_code")),
        "course_name_zh": _clean(row.get("course_name_zh")),
        "course_name_en": _clean(row.get("course_name_en")),
        "credits": credits_value,
        "time": _clean(row.get("time")),
        "time_slots": sorted(parse_time_slots(_clean(row.get("time")))),
        "teacher": _clean(row.get("teacher")),
        "classroom": _clean(row.get("classroom")),
        "term": _clean(row.get("term")),
        "course_level": _clean(row.get("course_level")),
        "listed_departments": _clean(row.get("listed_departments")),
        "prerequisite_text": _clean(row.get("prerequisite_text")),
        "recommendation_reason": reason,
    }
    review_summary = row.get("_review_summary") if "_review_summary" in row.index else None
    if not isinstance(review_summary, dict):
        review_summary = row.get("_ptt_review_summary") if "_ptt_review_summary" in row.index else None
    if isinstance(review_summary, dict):
        course["review_summary"] = review_summary
        course["ptt_review_summary"] = review_summary
    if requirement_code:
        course["requirement_code"] = requirement_code
    return {key: _json_safe(value) for key, value in course.items()}


def _target_credit_range(preferences: dict) -> tuple[float, float]:
    if not preferences:
        return NORMAL_SEMESTER_MIN_CREDITS, NORMAL_SEMESTER_MAX_CREDITS
    if "target_credit_range" in preferences:
        low, high = preferences["target_credit_range"]
        return float(low), float(high)
    if "target_credits" in preferences:
        value = preferences["target_credits"]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
        return float(value), float(value)
    return float(preferences.get("target_credit_min", NORMAL_SEMESTER_MIN_CREDITS)), float(
        preferences.get("target_credit_max", NORMAL_SEMESTER_MAX_CREDITS)
    )


def _semester_load_warnings(total_credits: float, target_min: float, target_max: float) -> list[str]:
    warnings: list[str] = []
    if target_max < NORMAL_SEMESTER_MIN_CREDITS:
        warnings.append(
            f"The requested target range {target_min:g}-{target_max:g} credits is below the normal semester minimum of "
            f"{NORMAL_SEMESTER_MIN_CREDITS:g} credits. A low-credit-load application may be required."
        )
    elif target_min < NORMAL_SEMESTER_MIN_CREDITS:
        warnings.append(
            f"The requested target range starts below the normal semester minimum of {NORMAL_SEMESTER_MIN_CREDITS:g} credits. "
            "If the final plan is below the minimum, a low-credit-load application may be required."
        )

    if target_min > NORMAL_SEMESTER_MAX_CREDITS:
        warnings.append(
            f"The requested target range {target_min:g}-{target_max:g} credits is above the normal semester maximum of "
            f"{NORMAL_SEMESTER_MAX_CREDITS:g} credits. An overload application may be required."
        )
    elif target_max > NORMAL_SEMESTER_MAX_CREDITS:
        warnings.append(
            f"The requested target range exceeds the normal semester maximum of {NORMAL_SEMESTER_MAX_CREDITS:g} credits. "
            "If the final plan is above the maximum, an overload application may be required."
        )

    if total_credits < NORMAL_SEMESTER_MIN_CREDITS:
        warnings.append(
            f"The recommended plan has {total_credits:g} credits, below the normal semester minimum of "
            f"{NORMAL_SEMESTER_MIN_CREDITS:g}. A low-credit-load application may be required."
        )
    if total_credits > NORMAL_SEMESTER_MAX_CREDITS:
        warnings.append(
            f"The recommended plan has {total_credits:g} credits, above the normal semester maximum of "
            f"{NORMAL_SEMESTER_MAX_CREDITS:g}. An overload application may be required."
        )
    return warnings


def _has_friday(course: dict | pd.Series) -> bool:
    if isinstance(course, pd.Series):
        time_value = _clean(course.get("time"))
    else:
        time_value = _clean(course.get("time"))
    return any(slot.startswith("F") for slot in parse_time_slots(time_value))


def _is_ee_course_code(code: str) -> bool:
    normalized = normalize_course_code(code)
    return normalized.startswith(("EE", "EECS"))


def _is_technical_course_code(code: str) -> bool:
    normalized = normalize_course_code(code)
    return normalized.startswith(TECHNICAL_ELECTIVE_PREFIXES)


def _is_general_education_code(code: str) -> bool:
    normalized = normalize_course_code(code)
    return normalized.startswith(GENERAL_EDUCATION_PREFIXES)


def _avoid_days(preferences: dict) -> set[str]:
    days = {str(day).upper() for day in preferences.get("avoid_days", []) if str(day).strip()}
    if bool(preferences.get("avoid_friday", False)):
        days.add("F")
    return {day for day in days if day in DAY_LABELS}


def _has_avoid_day(course: dict | pd.Series, avoid_days: set[str]) -> bool:
    if not avoid_days:
        return False
    time_value = _clean(course.get("time"))
    return any(slot[:1] in avoid_days for slot in parse_time_slots(time_value))


def _normalize_time_slot_token(slot: object) -> str:
    text = str(slot or "").strip()
    if len(text) < 2:
        return ""
    return text[:1].upper() + text[1:].lower()


def _excluded_time_slots(preferences: dict) -> set[str]:
    return {
        normalized
        for normalized in (_normalize_time_slot_token(slot) for slot in preferences.get("exclude_time_slots", []))
        if normalized
    }


def _course_time_slots(course: dict | pd.Series) -> set[str]:
    return {
        normalized
        for normalized in (_normalize_time_slot_token(slot) for slot in parse_time_slots(_clean(course.get("time"))))
        if normalized
    }


def _excluded_time_slot_overlap(course: dict | pd.Series, excluded_slots: set[str]) -> set[str]:
    if not excluded_slots:
        return set()
    return _course_time_slots(course) & excluded_slots


def _has_excluded_time_slot(course: dict | pd.Series, excluded_slots: set[str]) -> bool:
    return bool(_excluded_time_slot_overlap(course, excluded_slots))


def _required_section_time_slots_for_code(preferences: dict, code: str) -> set[str]:
    required_by_code = preferences.get("required_section_time_slots_by_code", {})
    if not isinstance(required_by_code, dict):
        return set()
    raw_slots = required_by_code.get(normalize_course_code(code), [])
    return {
        normalized
        for normalized in (_normalize_time_slot_token(slot) for slot in raw_slots)
        if normalized
    }


def _all_sections_have_excluded_time_slot(options: pd.DataFrame, excluded_slots: set[str]) -> bool:
    if options.empty or not excluded_slots:
        return False
    return all(_has_excluded_time_slot(row, excluded_slots) for _, row in options.iterrows())


def _avoid_day_names(avoid_days: set[str]) -> str:
    return ", ".join(DAY_LABELS[day] for day in sorted(avoid_days))


def _preferred_days(preferences: dict) -> set[str]:
    return {
        str(day).upper()
        for day in preferences.get("preferred_days", [])
        if str(day).upper() in DAY_LABELS
    }


def _preferred_day_names(preferred_days: set[str]) -> str:
    return ", ".join(DAY_LABELS[day] for day in sorted(preferred_days))


def _has_preferred_day(course: dict | pd.Series, preferred_days: set[str]) -> bool:
    if not preferred_days:
        return True
    time_value = _clean(course.get("time"))
    return any(slot[:1] in preferred_days for slot in parse_time_slots(time_value))


def _all_sections_miss_preferred_day(options: pd.DataFrame, preferred_days: set[str]) -> bool:
    if options.empty or not preferred_days:
        return False
    return all(not _has_preferred_day(row, preferred_days) for _, row in options.iterrows())


def _strict_avoid_days(preferences: dict) -> bool:
    return bool(preferences.get("strict_avoid_days", True))


def _strict_preferred_days(preferences: dict) -> bool:
    return bool(preferences.get("strict_preferred_days", False))


def _all_sections_have_avoid_day(options: pd.DataFrame, avoid_days: set[str]) -> bool:
    if options.empty or not avoid_days:
        return False
    return all(_has_avoid_day(row, avoid_days) for _, row in options.iterrows())


def _is_hard_course(row_or_course: dict | pd.Series) -> bool:
    code = _clean(row_or_course.get("normalized_course_code") or row_or_course.get("code"))
    name_zh = _clean(row_or_course.get("course_name_zh"))
    name_en = _clean(row_or_course.get("course_name_en")).lower()
    if code.startswith(HARD_ELECTIVE_PREFIXES):
        return True
    text = f"{name_zh} {name_en}".lower()
    return any(keyword.lower() in text for keyword in HARD_COURSE_KEYWORDS)


def _is_soft_elective(row_or_course: dict | pd.Series, preferences: dict) -> bool:
    code = _clean(row_or_course.get("normalized_course_code") or row_or_course.get("code"))
    name_zh = _clean(row_or_course.get("course_name_zh"))
    name_en = _clean(row_or_course.get("course_name_en")).lower()
    if any(keyword.lower() in f"{name_zh} {name_en}".lower() for keyword in EXCLUDED_SOFT_FILL_KEYWORDS):
        return False
    if code.startswith(PE_PREFIXES):
        return bool(preferences.get("include_pe", True))
    if code.startswith(SOFT_ELECTIVE_PREFIXES):
        return True
    if bool(preferences.get("include_outside_department", True)):
        if code.startswith(("EE", "EECS")):
            return False
        if bool(preferences.get("avoid_difficult_courses", True)) and _is_hard_course(row_or_course):
            return False
        course_level = _clean(row_or_course.get("course_level"))
        return (not course_level) or ("大學部" in course_level)
    return False


def _soft_elective_category_rank(row_or_course: dict | pd.Series) -> tuple[int, str]:
    code = _clean(row_or_course.get("normalized_course_code") or row_or_course.get("code"))
    if code.startswith(GENERAL_EDUCATION_PREFIXES):
        return 0, "general_education"
    if code.startswith(LANGUAGE_PREFIXES):
        return 1, "language"
    if code.startswith(PE_PREFIXES):
        return 3, "physical_education"
    return 2, "outside_department"


def _is_non_ge_non_technical_other_filler(course_or_candidate: dict | pd.Series) -> bool:
    """True for automatic other-elective fillers that are not EE/EECS/CS or GE/GEC."""
    code = _clean(
        course_or_candidate.get("normalized_course_code")
        or course_or_candidate.get("code")
    )
    if not code:
        return False
    if _is_technical_course_code(code) or _is_general_education_code(code):
        return False
    requirement_code = _clean(course_or_candidate.get("requirement_code"))
    reason = _clean(course_or_candidate.get("recommendation_reason") or course_or_candidate.get("reason")).lower()
    return requirement_code == "other_electives" or "other elective fill" in reason or "physical education" in reason


def _non_ge_non_technical_filler_count(selected: list[dict]) -> int:
    return sum(1 for course in selected if _is_non_ge_non_technical_other_filler(course))


def _max_non_ge_non_technical_fillers(preferences: dict) -> int:
    try:
        return max(0, int(preferences.get("max_non_ge_non_technical_fillers", 1)))
    except (TypeError, ValueError):
        return 1


def _is_protected_graduation_course(course: dict) -> bool:
    reason = _clean(course.get("recommendation_reason")).lower()
    requirement_code = _clean(course.get("requirement_code")).lower()
    return (
        "missing required" in reason
        or "probability" in reason
        or "required lab" in reason
        or requirement_code in {"required_lab_electives", "probability", "linear_algebra"}
        or bool(requirement_code and requirement_code == normalize_course_code(course.get("code", "")).lower())
    )


def _is_lab_course(course: dict | pd.Series) -> bool:
    code = _clean(course.get("normalized_course_code") or course.get("code"))
    name_zh = _clean(course.get("course_name_zh"))
    name_en = _clean(course.get("course_name_en")).lower()
    reason = _clean(course.get("recommendation_reason")).lower()
    requirement_code = _clean(course.get("requirement_code")).lower()
    return (
        "實驗" in name_zh
        or "lab" in name_en
        or "laboratory" in name_en
        or "required lab" in reason
        or requirement_code == "required_lab_electives"
        or code in LAB_COURSE_CODES
    )


def _is_ee_theory_course(course: dict | pd.Series) -> bool:
    code = _clean(course.get("normalized_course_code") or course.get("code"))
    if not _is_ee_course_code(code):
        return False
    return not _is_lab_course(course)


def _explicit_lab_override_codes(preferences: dict | None) -> set[str]:
    preferences = preferences or {}
    return {
        normalize_course_code(code)
        for code in preferences.get("explicitly_requested_lab_course_codes", [])
        if normalize_course_code(code)
    }


def _should_exclude_lab_course(course: dict | pd.Series, preferences: dict | None) -> bool:
    preferences = preferences or {}
    code = _clean(course.get("normalized_course_code") or course.get("code"))
    if not preferences.get("exclude_lab_courses", False):
        return False
    if code in _explicit_lab_override_codes(preferences):
        return False
    return _is_lab_course(course)


def _selected_codes(selected: list[dict]) -> set[str]:
    return {course.get("code", "") for course in selected if course.get("code")}


def _student_codes(student_df: pd.DataFrame, include_in_progress: bool = True) -> set[str]:
    codes: set[str] = set()
    for _, row in student_df.iterrows():
        status = _clean(row.get("status")).lower()
        code = _clean(row.get("normalized_course_code"))
        if not code:
            continue
        if status == "completed" or (include_in_progress and status == "in_progress"):
            codes.add(code)
    return codes


def _course_conflicts(candidate: dict, selected: list[dict]) -> list[dict]:
    conflicts = check_plan_conflicts(selected + [candidate]).get("conflicts", [])
    candidate_label = f"{candidate.get('code')} {candidate.get('course_name_zh')}".strip()
    return [
        conflict
        for conflict in conflicts
        if candidate_label in {conflict.get("course_a"), conflict.get("course_b")}
    ]


def search_target_courses(target_df: pd.DataFrame, course_codes: list[str]) -> pd.DataFrame:
    normalized = {normalize_course_code(code) for code in course_codes}
    df = _target_only(target_df)
    return df.loc[df["normalized_course_code"].isin(normalized)].copy()


def _requirement_candidates(graduation_result: dict, preferences: dict | None = None) -> list[dict]:
    candidates: list[dict] = []
    for item in graduation_result.get("missing_required", []):
        candidates.append(
            {
                "priority": 0,
                "code": item.get("code", ""),
                "reason": f"missing required course: {item.get('code', '')}",
                "requirement_code": item.get("code", ""),
            }
        )

    for group_key, status in graduation_result.get("alternative_requirements_status", {}).items():
        if status.get("satisfied"):
            continue
        label = status.get("label", group_key)
        for option in status.get("options", []):
            candidates.append(
                {
                    "priority": 1,
                    "code": option.get("code", ""),
                    "reason": f"missing alternative requirement: {label}",
                    "requirement_code": group_key,
                }
            )

    lab_status = graduation_result.get("lab_elective_status", {})
    if not lab_status.get("satisfied", False):
        counted_lab_codes = {course.get("code") for course in lab_status.get("completed_or_counted_courses", [])}
        for lab in lab_status.get("eligible_courses", []):
            lab_candidate = {
                "code": lab.get("code", ""),
                "course_name_zh": lab.get("name_zh", ""),
                "course_name_en": lab.get("name_en", ""),
                "requirement_code": "required_lab_electives",
            }
            if _should_exclude_lab_course(lab_candidate, preferences):
                continue
            if lab.get("code") not in counted_lab_codes:
                candidates.append(
                    {
                        "priority": 2,
                        "code": lab.get("code", ""),
                        "reason": "helps satisfy required lab elective requirement",
                        "requirement_code": "required_lab_electives",
                    }
                )
    return candidates


def _professional_candidates(
    target_df: pd.DataFrame,
    graduation_result: dict,
    excluded_codes: set[str],
    preferences: dict | None = None,
) -> list[dict]:
    preferences = preferences or {}
    professional_remaining = graduation_result.get("credit_summary", {}).get(
        "professional_elective_remaining_credits_mvp", 0
    )
    if professional_remaining <= 0:
        return []

    df = _target_only(target_df)
    required_codes = {item.get("code") for item in graduation_result.get("missing_required", [])}
    lab_codes = {
        item.get("code")
        for item in graduation_result.get("lab_elective_status", {}).get("eligible_courses", [])
    }
    rows = []
    for _, row in df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        if code in excluded_codes or code in required_codes or code in lab_codes:
            continue
        if not (code.startswith("EE") or code.startswith("EECS")):
            continue
        if _should_exclude_lab_course(row, preferences):
            continue
        if _clean(row.get("course_level")) and "大學部" not in _clean(row.get("course_level")):
            continue
        credits = pd.to_numeric(row.get("credits"), errors="coerce")
        if pd.isna(credits) or float(credits) <= 0:
            continue
        rows.append({"priority": 4, "code": code, "reason": "potential EE/EECS professional elective", "requirement_code": "professional_electives"})
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in rows:
        if item["code"] not in seen:
            deduped.append(item)
            seen.add(item["code"])
    return deduped


def _ee_theory_candidates(
    target_df: pd.DataFrame,
    graduation_result: dict,
    excluded_codes: set[str],
    preferences: dict | None = None,
) -> list[dict]:
    preferences = preferences or {}
    df = _target_only(target_df)
    missing_required_codes = {item.get("code") for item in graduation_result.get("missing_required", [])}
    rows: list[tuple[tuple[int, str], dict]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        if not code or code in excluded_codes or code in seen:
            continue
        if not _is_ee_theory_course(row):
            continue
        if _clean(row.get("course_level")) and "大學部" not in _clean(row.get("course_level")):
            continue
        credits = pd.to_numeric(row.get("credits"), errors="coerce")
        if pd.isna(credits) or float(credits) <= 0:
            continue
        if code in missing_required_codes:
            rank = 0
            reason = f"EE-first required/theory course: also missing required course {code}"
            requirement_code = code
        else:
            rank = 1
            reason = "EE-first theory/professional course"
            requirement_code = "ee_first_theory_professional"
        rows.append(
            (
                (rank, code),
                {
                    "priority": 2,
                    "code": code,
                    "reason": reason,
                    "requirement_code": requirement_code,
                },
            )
        )
        seen.add(code)
    return [candidate for _, candidate in sorted(rows, key=lambda item: item[0])]


def _lab_candidates(
    target_df: pd.DataFrame,
    graduation_result: dict,
    excluded_codes: set[str],
    preferences: dict | None = None,
) -> list[dict]:
    preferences = preferences or {}
    rows: list[tuple[tuple[int, str], dict]] = []
    seen: set[str] = set()
    lab_status = graduation_result.get("lab_elective_status", {})
    counted_lab_codes = {course.get("code") for course in lab_status.get("completed_or_counted_courses", [])}
    for lab in lab_status.get("eligible_courses", []):
        code = _clean(lab.get("code"))
        if not code or code in counted_lab_codes or code in excluded_codes or code in seen:
            continue
        lab_candidate = {
            "code": code,
            "course_name_zh": lab.get("name_zh", ""),
            "course_name_en": lab.get("name_en", ""),
            "requirement_code": "required_lab_electives",
        }
        if _should_exclude_lab_course(lab_candidate, preferences):
            continue
        rows.append(
            (
                (0, code),
                {
                    "priority": 3,
                    "code": code,
                    "reason": "EE-first default lab course",
                    "requirement_code": "required_lab_electives",
                },
            )
        )
        seen.add(code)

    df = _target_only(target_df)
    for _, row in df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        if not code or code in excluded_codes or code in seen:
            continue
        if not _is_ee_course_code(code) or not _is_lab_course(row):
            continue
        if _should_exclude_lab_course(row, preferences):
            continue
        if _clean(row.get("course_level")) and "大學部" not in _clean(row.get("course_level")):
            continue
        credits = pd.to_numeric(row.get("credits"), errors="coerce")
        if pd.isna(credits) or float(credits) <= 0:
            continue
        rows.append(
            (
                (1, code),
                {
                    "priority": 3,
                    "code": code,
                    "reason": "EE-first default lab course",
                    "requirement_code": "lab_filler",
                },
            )
        )
        seen.add(code)
    return [candidate for _, candidate in sorted(rows, key=lambda item: item[0])]


def _user_requested_ee_candidates(
    target_df: pd.DataFrame,
    graduation_result: dict,
    excluded_codes: set[str],
    preferences: dict | None = None,
) -> list[dict]:
    preferences = preferences or {}
    df = _target_only(target_df)
    missing_required_codes = {item.get("code") for item in graduation_result.get("missing_required", [])}
    lab_codes = {
        item.get("code")
        for item in graduation_result.get("lab_elective_status", {}).get("eligible_courses", [])
    }
    rows: list[tuple[tuple[int, str], dict]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        if not code or code in seen or code in excluded_codes:
            continue
        if not _is_ee_course_code(code):
            continue
        if preferences.get("prefer_theory_ee_courses") and _is_lab_course(row):
            continue
        if _should_exclude_lab_course(row, preferences):
            continue
        if _clean(row.get("course_level")) and "大學部" not in _clean(row.get("course_level")):
            continue
        credits = pd.to_numeric(row.get("credits"), errors="coerce")
        if pd.isna(credits) or float(credits) <= 0:
            continue
        if code in missing_required_codes:
            rank = 0
            reason = f"user requested more EE/EECS courses; also missing required course: {code}"
            requirement_code = code
        elif code in lab_codes:
            rank = 1
            reason = "user requested more EE/EECS courses; helps satisfy required lab elective requirement"
            requirement_code = "required_lab_electives"
        else:
            rank = 2
            reason = "user requested more EE/EECS course"
            requirement_code = "user_requested_ee_course"
        rows.append(
            (
                (rank, code),
                {
                    "priority": rank,
                    "code": code,
                    "reason": reason,
                    "requirement_code": requirement_code,
                },
            )
        )
        seen.add(code)
    return [candidate for _, candidate in sorted(rows, key=lambda item: item[0])]


def _user_requested_general_education_candidates(
    target_df: pd.DataFrame,
    excluded_codes: set[str],
) -> list[dict]:
    df = _target_only(target_df)
    rows: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        if not code or code in seen or code in excluded_codes:
            continue
        if not _is_general_education_code(code):
            continue
        credits = pd.to_numeric(row.get("credits"), errors="coerce")
        if pd.isna(credits) or float(credits) <= 0:
            continue
        rows.append(
            (
                code,
                {
                    "priority": 3,
                    "code": code,
                    "reason": "user requested more general education course",
                    "requirement_code": "user_requested_general_education",
                },
            )
        )
        seen.add(code)
    return [candidate for _, candidate in sorted(rows, key=lambda item: item[0])]


def _other_elective_candidates(
    target_df: pd.DataFrame,
    graduation_result: dict,
    excluded_codes: set[str],
    preferences: dict,
) -> list[dict]:
    other_remaining = graduation_result.get("credit_summary", {}).get("other_elective_remaining_credits_mvp", 0)
    should_balance = bool(preferences.get("balance_with_other_electives", True))
    if other_remaining <= 0 and not should_balance and not preferences.get("include_pe", False):
        return []

    df = _target_only(target_df)
    rows: list[tuple[tuple[int, int, int, str], dict]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        if not code or code in excluded_codes or code in seen:
            continue
        if not _is_soft_elective(row, preferences):
            continue
        credits = pd.to_numeric(row.get("credits"), errors="coerce")
        if pd.isna(credits):
            continue
        if float(credits) <= 0 and not code.startswith(PE_PREFIXES):
            continue
        if bool(preferences.get("avoid_difficult_courses", True)) and _is_hard_course(row) and not code.startswith(PE_PREFIXES):
            continue
        if code.startswith(PE_PREFIXES):
            if not bool(preferences.get("include_pe", True)):
                continue
            if not _clean(row.get("time")):
                continue
            reason = "optional physical education course for schedule balance; does not add graduation credits"
            priority = 5
        elif code.startswith(LANGUAGE_PREFIXES):
            reason = "other elective fill: language or college-language course"
            priority = 3
        elif code.startswith(GENERAL_EDUCATION_PREFIXES):
            reason = "other elective fill: general education course"
            priority = 3
        else:
            reason = "other elective fill: outside-department undergraduate course"
            priority = 3
        category_rank, _ = _soft_elective_category_rank(row)
        if bool(preferences.get("prefer_general_education_courses", False)):
            if code.startswith(GENERAL_EDUCATION_PREFIXES):
                category_rank = -1
            elif code.startswith(LANGUAGE_PREFIXES):
                category_rank = 1
        avoid_day_penalty = int(_has_avoid_day(row, _avoid_days(preferences)))
        no_time_penalty = int(not bool(_clean(row.get("time"))))
        sort_key = (category_rank, avoid_day_penalty, no_time_penalty, code)
        rows.append(
            (
                sort_key,
                {
                    "priority": priority,
                    "code": code,
                    "reason": reason,
                    "requirement_code": "other_electives",
                    "avoid_days_strict": code.startswith(PE_PREFIXES),
                    "avoid_friday_strict": code.startswith(PE_PREFIXES),
                },
            )
        )
        seen.add(code)
    return [candidate for _, candidate in sorted(rows, key=lambda item: item[0])]


def _ge_language_filler_candidates(
    target_df: pd.DataFrame,
    graduation_result: dict,
    excluded_codes: set[str],
    preferences: dict,
) -> list[dict]:
    allowed_prefixes = (
        GENERAL_EDUCATION_PREFIXES
        if bool(preferences.get("prefer_ge_gec_for_remaining_fillers", True))
        else SOFT_ELECTIVE_PREFIXES
    )
    return [
        candidate
        for candidate in _other_elective_candidates(target_df, graduation_result, excluded_codes, preferences)
        if normalize_course_code(candidate.get("code", "")).startswith(allowed_prefixes)
    ]


def _outside_department_filler_candidates(
    target_df: pd.DataFrame,
    graduation_result: dict,
    excluded_codes: set[str],
    preferences: dict,
) -> list[dict]:
    candidates: list[dict] = []
    for candidate in _other_elective_candidates(target_df, graduation_result, excluded_codes, preferences):
        code = normalize_course_code(candidate.get("code", ""))
        if code.startswith(SOFT_ELECTIVE_PREFIXES) or code.startswith(PE_PREFIXES):
            continue
        if _is_ee_course_code(code):
            continue
        candidates.append(candidate)
    return candidates


def _first_teacher_name(raw_teacher: str) -> str:
    text = _clean(raw_teacher)
    if not text:
        return ""
    return re.split(r"[,，、/]", text)[0].strip()


def _review_search_enabled(preferences: dict) -> bool:
    return bool(preferences.get("use_review_search", False) or preferences.get("use_ptt_reviews", False))


def _review_sources(preferences: dict) -> list[str]:
    if preferences.get("review_sources"):
        value = preferences["review_sources"]
        sources = [value] if isinstance(value, str) else list(value)
    elif preferences.get("use_ptt_reviews"):
        sources = ["local_cache", "ptt"]
    else:
        sources = ["local_cache"]
    normalized: list[str] = []
    for source in sources:
        source = str(source).strip().lower()
        if source in {"ptt", "ptt_rag", "local_cache", "web"} and source not in normalized:
            normalized.append(source)
    if "ptt" in normalized and "ptt_rag" not in normalized:
        normalized.insert(0, "ptt_rag")
    if "ptt" in normalized and not preferences.get("allow_live_ptt_review_ranking", False):
        normalized = [source for source in normalized if source != "ptt"]
    return normalized or ["local_cache"]


def _review_preference(preferences: dict) -> str:
    return str(preferences.get("review_prefer", preferences.get("ptt_prefer", "coolness"))).lower()


def _add_review_columns(options: pd.DataFrame, preferences: dict) -> pd.DataFrame:
    if options.empty or not _review_search_enabled(preferences):
        return options
    preference = _review_preference(preferences)
    score_key = "avg_sweetness" if preference == "sweetness" else "avg_coolness"
    max_results = min(int(preferences.get("review_max_results", preferences.get("ptt_max_pages", 2))), 2)
    timeout = max(1, min(int(preferences.get("review_timeout", preferences.get("ptt_timeout", 3))), 3))
    lookup_limit = max(0, int(preferences.get("review_lookup_limit", 8)))
    sources = _review_sources(preferences)
    ranked = options.copy()
    summaries: list[dict] = []
    scores: list[float] = []
    review_counts: list[int] = []
    for index, (_, row) in enumerate(ranked.iterrows()):
        teacher = _first_teacher_name(row.get("teacher", ""))
        course_name = _clean(row.get("course_name_zh"))
        if index >= lookup_limit:
            summary = {"review_count": 0, score_key: None, "warnings": ["Review lookup skipped to keep the demo responsive."]}
        elif not teacher or not course_name:
            summary = {"review_count": 0, score_key: None, "warnings": ["Missing teacher or course name for review lookup."]}
        else:
            summary = search_course_reviews(
                course_name=course_name,
                teacher_name=teacher,
                sources=sources,
                max_results=max_results,
                timeout=timeout,
            )
        score = summary.get(score_key)
        summaries.append(summary)
        scores.append(float(score) if score is not None else -1.0)
        review_counts.append(int(summary.get("review_count") or 0))
    ranked["_review_summary"] = summaries
    ranked["_review_score"] = scores
    ranked["_review_count"] = review_counts
    ranked["_ptt_review_summary"] = summaries
    ranked["_ptt_score"] = scores
    ranked["_ptt_review_count"] = review_counts
    return ranked


def _rank_options(options: pd.DataFrame, avoid_friday: bool, preferences: dict | None = None) -> pd.DataFrame:
    if options.empty:
        return options
    preferences = preferences or {}
    avoid_days = _avoid_days({**preferences, "avoid_friday": avoid_friday or preferences.get("avoid_friday", False)})
    preferred_days = _preferred_days(preferences)
    ranked = _add_review_columns(options.copy(), preferences)
    ranked["_avoid_day_penalty"] = ranked.apply(lambda row: int(_has_avoid_day(row, avoid_days)), axis=1)
    ranked["_preferred_day_penalty"] = ranked.apply(lambda row: int(not _has_preferred_day(row, preferred_days)), axis=1)
    ranked["_hard_penalty"] = ranked.apply(lambda row: int(_is_hard_course(row)), axis=1)
    ranked["_has_time"] = ranked["time"].map(lambda value: int(bool(_clean(value)))) if "time" in ranked else 0
    ranked["_raw_sort"] = ranked["raw_course_code"].map(_clean) if "raw_course_code" in ranked else ""
    ranked["_review_score"] = ranked["_review_score"] if "_review_score" in ranked else 0.0
    ranked["_review_count"] = ranked["_review_count"] if "_review_count" in ranked else 0
    return ranked.sort_values(
        by=["_avoid_day_penalty", "_preferred_day_penalty", "_hard_penalty", "_has_time", "_review_score", "_review_count", "_raw_sort"],
        ascending=[True, True, True, False, False, False, True],
    )


def _try_add_candidate(
    selected: list[dict],
    target_df: pd.DataFrame,
    candidate: dict,
    total_credits: float,
    max_credits: float,
    avoid_friday: bool,
    excluded_codes: set[str],
    preferences: dict,
) -> tuple[bool, float, list[dict], list[str]]:
    warnings: list[str] = []
    code = candidate["code"]
    avoid_days = _avoid_days({**preferences, "avoid_friday": avoid_friday or preferences.get("avoid_friday", False)})
    excluded_time_slots = _excluded_time_slots(preferences)
    preferred_days = _preferred_days(preferences)
    strict_avoid_days = _strict_avoid_days(preferences)
    strict_preferred_days = _strict_preferred_days(preferences)
    required_section_slots = _required_section_time_slots_for_code(preferences, code)
    if not code or code in _selected_codes(selected) or code in excluded_codes:
        return False, total_credits, [], warnings
    if (
        bool(preferences.get("limit_non_ge_non_technical_fillers", True))
        and _is_non_ge_non_technical_other_filler(candidate)
        and _non_ge_non_technical_filler_count(selected) >= _max_non_ge_non_technical_fillers(preferences)
    ):
        return False, total_credits, [], warnings

    options = _rank_options(search_target_courses(target_df, [code]), avoid_friday, preferences)
    conflict_rejections: list[dict] = []
    if options.empty:
        return False, total_credits, [], warnings

    for _, option in options.iterrows():
        course = _course_to_dict(option, candidate.get("reason", ""), candidate.get("requirement_code", ""))
        if required_section_slots and not required_section_slots.issubset(_course_time_slots(course)):
            continue
        if _should_exclude_lab_course(course, preferences):
            continue
        if _has_excluded_time_slot(course, excluded_time_slots):
            continue
        if (strict_avoid_days or candidate.get("avoid_days_strict")) and _has_avoid_day(course, avoid_days):
            continue
        if strict_preferred_days and preferred_days and not _has_preferred_day(course, preferred_days):
            continue
        if avoid_friday and candidate.get("avoid_friday_strict") and _has_friday(course):
            continue
        if total_credits + float(course["credits"] or 0) > max_credits:
            continue
        conflicts = _course_conflicts(course, selected)
        if conflicts:
            conflict_rejections.extend(conflicts)
            continue
        selected.append(course)
        total_credits += float(course["credits"] or 0)
        if _has_avoid_day(course, avoid_days):
            warnings.append(
                f"{course['code']} includes requested avoid-day time slots ({_avoid_day_names(avoid_days)}). "
                "It was selected because no alternative section fit the graduation, credit, and conflict constraints."
            )
        return True, total_credits, conflict_rejections, warnings

    if strict_avoid_days and candidate.get("priority", 99) <= 2 and _all_sections_have_avoid_day(options, avoid_days):
        warnings.append(
            f"{code} was not selected because its available 11420 sections include requested avoid-day time slots "
            f"({_avoid_day_names(avoid_days)})."
        )
    if excluded_time_slots and candidate.get("priority", 99) <= 2 and _all_sections_have_excluded_time_slot(options, excluded_time_slots):
        warnings.append(
            f"{code} was not selected because its available 11420 sections include excluded time slots "
            f"({', '.join(sorted(excluded_time_slots))})."
        )
    if strict_preferred_days and preferred_days and _all_sections_miss_preferred_day(options, preferred_days):
        warnings.append(
            f"{code} was not selected because no available 11420 section falls on the preferred day(s) "
            f"({_preferred_day_names(preferred_days)})."
        )

    return False, total_credits, conflict_rejections, warnings


def _format_conflict_rejection_detail(code: str, conflicts: list[dict]) -> str:
    parts: list[str] = []
    for conflict in conflicts[:4]:
        course_a = conflict.get("course_a", "")
        course_b = conflict.get("course_b", "")
        slots = ", ".join(conflict.get("overlap_slots", []))
        other = course_b if str(course_a).startswith(code) else course_a
        parts.append(f"與 {other} 衝堂（{slots}）")
    return "；".join(parts)


def _specific_add_failure_details(
    selected: list[dict],
    target_df: pd.DataFrame,
    code: str,
    total_credits: float,
    max_credits: float,
    avoid_friday: bool,
    preferences: dict,
) -> list[str]:
    """Explain why a user-requested course could not be added.

    This is used only for explanation. It does not choose or execute changes.
    """
    availability = _rank_options(search_target_courses(target_df, [code]), avoid_friday, preferences)
    if availability.empty:
        return [f"{code} 在目標學期 11420 沒有找到可用開課資料。"]

    avoid_days = _avoid_days({**preferences, "avoid_friday": avoid_friday or preferences.get("avoid_friday", False)})
    excluded_time_slots = _excluded_time_slots(preferences)
    preferred_days = _preferred_days(preferences)
    strict_avoid_days = _strict_avoid_days(preferences)
    strict_preferred_days = _strict_preferred_days(preferences)
    required_section_slots = _required_section_time_slots_for_code(preferences, code)
    details: list[str] = []
    for _, option in availability.iterrows():
        course = _course_to_dict(option, f"user requested specific course: {code}", "user_requested_specific_course")
        section_label = f"{course.get('code')} {course.get('course_name_zh')} {course.get('time') or 'TBA'}".strip()
        section_reasons: list[str] = []
        if required_section_slots and not required_section_slots.issubset(_course_time_slots(course)):
            section_reasons.append(f"不是你剛才候選清單中選到的 section 時段（需要 {', '.join(sorted(required_section_slots))}）")
        if _should_exclude_lab_course(course, preferences):
            section_reasons.append("目前仍被『不要實驗課』限制排除")
        excluded_overlap = sorted(_excluded_time_slot_overlap(course, excluded_time_slots))
        if excluded_overlap:
            section_reasons.append(f"落在你要求排除的時段（{', '.join(excluded_overlap)}）")
        if (strict_avoid_days or preferences.get("strict_avoid_days")) and _has_avoid_day(course, avoid_days):
            section_reasons.append(f"落在你要求避開的星期（{_avoid_day_names(avoid_days)}）")
        if strict_preferred_days and preferred_days and not _has_preferred_day(course, preferred_days):
            section_reasons.append(f"沒有落在你指定的星期（{_preferred_day_names(preferred_days)}）")
        if total_credits + float(course.get("credits") or 0) > max_credits:
            section_reasons.append(
                f"加入後會超過目前學分上限 {max_credits:g} 學分"
            )
        conflicts = _course_conflicts(course, selected)
        if conflicts:
            conflict_text = _format_conflict_rejection_detail(code, conflicts)
            if conflict_text:
                section_reasons.append(conflict_text)
        if not section_reasons:
            section_reasons.append("沒有通過排課器的可加入條件，可能是重複、已修/修課中或資料不足")
        details.append(f"{section_label}：" + "；".join(section_reasons))
    return details[:6]


def _candidate_add_failure_summary(
    selected: list[dict],
    target_df: pd.DataFrame,
    candidate: dict,
    total_credits: float,
    max_credits: float,
    avoid_friday: bool,
    preferences: dict,
) -> str:
    code = candidate.get("code", "")
    if not code:
        return "候選課程沒有課號，無法查詢 11420 開課。"
    availability = _rank_options(search_target_courses(target_df, [code]), avoid_friday, preferences)
    if availability.empty:
        return f"{code}：11420 沒有找到開課資料。"

    avoid_days = _avoid_days({**preferences, "avoid_friday": avoid_friday or preferences.get("avoid_friday", False)})
    excluded_time_slots = _excluded_time_slots(preferences)
    preferred_days = _preferred_days(preferences)
    strict_avoid_days = _strict_avoid_days(preferences)
    strict_preferred_days = _strict_preferred_days(preferences)
    required_section_slots = _required_section_time_slots_for_code(preferences, code)
    section_reasons: list[str] = []
    seen_sections: set[str] = set()
    for _, option in availability.iterrows():
        course = _course_to_dict(option, candidate.get("reason", ""), candidate.get("requirement_code", ""))
        reasons: list[str] = []
        if required_section_slots and not required_section_slots.issubset(_course_time_slots(course)):
            reasons.append(f"不是候選清單指定 section 時段（需要 {', '.join(sorted(required_section_slots))}）")
        if course.get("code") in _selected_codes(selected):
            reasons.append("已在目前課表中")
        if _should_exclude_lab_course(course, preferences):
            reasons.append("被目前『不要實驗課』限制排除")
        excluded_overlap = sorted(_excluded_time_slot_overlap(course, excluded_time_slots))
        if excluded_overlap:
            reasons.append(f"落在排除時段（{', '.join(excluded_overlap)}）")
        if (strict_avoid_days or candidate.get("avoid_days_strict")) and _has_avoid_day(course, avoid_days):
            reasons.append(f"落在避開星期（{_avoid_day_names(avoid_days)}）")
        if strict_preferred_days and preferred_days and not _has_preferred_day(course, preferred_days):
            reasons.append(f"不在指定星期（{_preferred_day_names(preferred_days)}）")
        if total_credits + float(course.get("credits") or 0) > max_credits:
            reasons.append(f"加入後超過學分上限 {max_credits:g}")
        conflicts = _course_conflicts(course, selected)
        conflict_text = _format_conflict_rejection_detail(code, conflicts)
        if conflict_text:
            reasons.append(conflict_text)
        if not reasons:
            reasons.append("沒有通過排課器條件，可能是資料不完整或重複候選")
        section_label = f"{course.get('code')} {course.get('course_name_zh')} {course.get('time') or 'TBA'}".strip()
        detail = f"{section_label}: " + "、".join(reasons)
        if detail in seen_sections:
            continue
        seen_sections.add(detail)
        section_reasons.append(detail)
    return "；".join(section_reasons[:3])


def _target_credit_center(preferences: dict, min_credits: float, max_credits: float) -> float:
    target = preferences.get("target_credits")
    if target is not None:
        try:
            return float(target)
        except (TypeError, ValueError):
            pass
    return (float(min_credits) + float(max_credits)) / 2


def _desired_ee_theory_count(preferences: dict, min_credits: float, max_credits: float) -> int:
    center = _target_credit_center(preferences, min_credits, max_credits)
    if center <= 20:
        return int(preferences.get("initial_ee_theory_count_under_or_equal_20", 4))
    return int(preferences.get("initial_ee_theory_count_above_20", 5))


def _default_composition_enabled(preferences: dict) -> bool:
    if preferences.get("composition_policy") != "ee_first":
        return False
    if preferences.get("locked_courses"):
        return False
    return not any(
        bool(preferences.get(key))
        for key in (
            "prefer_general_education_courses",
            "user_requested_lighter_schedule",
            "avoid_ee_courses",
            "reduce_professional_courses",
        )
    )


def _count_ee_theory_courses(courses: list[dict]) -> int:
    return sum(1 for course in courses if _is_ee_theory_course(course))


def _count_lab_courses(courses: list[dict]) -> int:
    return sum(1 for course in courses if _is_lab_course(course))


def _unresolved_reason_for_candidate(
    target_df: pd.DataFrame,
    candidate: dict,
    preferences: dict,
    avoid_days: set[str],
) -> str:
    availability = search_target_courses(target_df, [candidate["code"]])
    if availability.empty:
        return "not offered in target semester 11420"
    if _strict_avoid_days(preferences) and _all_sections_have_avoid_day(availability, avoid_days):
        return "not selected because available sections use requested avoid-day time slots"
    return "not selected due to credit limit or time conflict"


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.get("code", ""), candidate.get("requirement_code", ""))
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return deduped


@langsmith_trace("recommender.recommend_courses")
def recommend_courses(student_df, target_df, graduation_result, preferences: dict) -> dict:
    preferences = dict(preferences or {})
    preferences.setdefault("composition_policy", "ee_first")
    preferences.setdefault("initial_ee_theory_count_under_or_equal_20", 4)
    preferences.setdefault("initial_ee_theory_count_above_20", 5)
    preferences.setdefault("initial_lab_count", 1)
    preferences.setdefault("outside_department_fill_last", True)
    preferences.setdefault("prefer_ge_gec_for_remaining_fillers", True)
    preferences.setdefault("limit_non_ge_non_technical_fillers", True)
    preferences.setdefault("max_non_ge_non_technical_fillers", 1)
    min_credits, max_credits = _target_credit_range(preferences)
    avoid_friday = bool(preferences.get("avoid_friday", False))
    avoid_days = _avoid_days(preferences)
    preferred_days = _preferred_days(preferences)

    selected: list[dict] = [dict(course) for course in preferences.get("locked_courses", [])]
    total_credits = sum(float(course.get("credits") or 0) for course in selected)
    excluded_codes = {normalize_course_code(code) for code in preferences.get("exclude_course_codes", [])}
    excluded_codes.difference_update(_explicit_lab_override_codes(preferences))
    already_taken_codes = _student_codes(student_df, include_in_progress=True)
    excluded_codes |= already_taken_codes

    warnings: list[str] = []
    unresolved: list[dict] = []
    conflict_rejections: list[dict] = []

    if graduation_result.get("in_progress_courses_counted_in_planning"):
        warnings.append(
            "Courses currently in progress are treated as expected-to-pass for this planning recommendation."
        )
    if _review_search_enabled(preferences):
        warnings.append(
            "Online course reviews are used only as subjective soft references for teacher/section ranking. "
            "They may be biased, incomplete, outdated, or based on small samples."
        )
    if preferences.get("credit_target_relaxed_for_followup"):
        center = preferences.get("target_credit_soft_center")
        if center is not None:
            warnings.append(
                f"Previous exact credit target ({float(center):g} credits) was relaxed for this follow-up edit. "
                "The planner will stay close when possible, but the new user request is prioritized."
            )
    if avoid_days and _strict_avoid_days(preferences):
        warnings.append(
            f"Requested avoid days are treated as hard scheduling rules: {_avoid_day_names(avoid_days)}."
        )
    if preferred_days and _strict_preferred_days(preferences):
        warnings.append(
            f"Requested preferred days are treated as hard add-course rules: {_preferred_day_names(preferred_days)}."
        )
    if preferences.get("exclude_lab_courses"):
        warnings.append(
            "因為你前面已經說過不要實驗課，所以這次排課會排除實驗課，除非你明確指定某一門實驗課。"
        )
    if preferences.get("prefer_theory_ee_courses"):
        warnings.append(
            "你要求的是電機系理論/非實驗課，所以 EE/EECS 候選會排除實驗課。"
        )

    if _default_composition_enabled(preferences):
        desired_ee_count = _desired_ee_theory_count(preferences, min_credits, max_credits)
        target_center = _target_credit_center(preferences, min_credits, max_credits)
        if target_center <= 20:
            warnings.append(
                "這份初始課表採用 EE-first 排課策略：20 學分以下先嘗試安排約 4 門 EE/EECS 理論或專業課，"
                "再加入 1 門實驗課；其餘學分優先用 GE/GEC 補足，非 EE/EECS/CS 且非 GE/GEC 的課最多只自動放 1 門。"
            )
        else:
            warnings.append(
                "因為你目標超過 20 學分，所以系統先嘗試安排約 5 門 EE/EECS 課，再加入 1 門實驗課，"
                "其餘學分優先用 GE/GEC 補足，非 EE/EECS/CS 且非 GE/GEC 的課最多只自動放 1 門。"
            )
        if preferences.get("exclude_lab_courses"):
            warnings.append("由於你指定不要實驗課，本次排課不會強制加入實驗課。")

        requirement_candidates = _dedupe_candidates(_requirement_candidates(graduation_result, preferences))
        non_lab_requirements = [candidate for candidate in requirement_candidates if candidate.get("requirement_code") != "required_lab_electives"]
        for candidate in non_lab_requirements:
            added, total_credits, rejected, add_warnings = _try_add_candidate(
                selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
            )
            conflict_rejections.extend(rejected)
            warnings.extend(add_warnings)
            if not added and candidate["priority"] <= 1:
                unresolved.append(
                    {
                        "code": candidate["code"],
                        "requirement_code": candidate.get("requirement_code", ""),
                        "reason": _unresolved_reason_for_candidate(target_df, candidate, preferences, avoid_days),
                    }
                )

        for candidate in _ee_theory_candidates(target_df, graduation_result, excluded_codes | _selected_codes(selected), preferences):
            if _count_ee_theory_courses(selected) >= desired_ee_count:
                break
            added, total_credits, rejected, add_warnings = _try_add_candidate(
                selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
            )
            conflict_rejections.extend(rejected)
            warnings.extend(add_warnings)

        desired_labs = 0 if preferences.get("exclude_lab_courses") else int(preferences.get("initial_lab_count", 1))
        for candidate in _lab_candidates(target_df, graduation_result, excluded_codes | _selected_codes(selected), preferences):
            if _count_lab_courses(selected) >= desired_labs:
                break
            added, total_credits, rejected, add_warnings = _try_add_candidate(
                selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
            )
            conflict_rejections.extend(rejected)
            warnings.extend(add_warnings)
            if not added:
                continue

        if _count_ee_theory_courses(selected) < desired_ee_count:
            warnings.append(
                f"EE-first policy could only fit {_count_ee_theory_courses(selected)} of the desired {desired_ee_count} "
                "EE/EECS theory/professional courses because of offering, credits, avoid-day rules, or conflicts."
            )
        if desired_labs and _count_lab_courses(selected) < desired_labs:
            warnings.append(
                "EE-first policy could not fit the default lab course because of offering, credits, avoid-day rules, or conflicts."
            )

        if total_credits < min_credits:
            for candidate in _ge_language_filler_candidates(
                target_df,
                graduation_result,
                excluded_codes | _selected_codes(selected),
                preferences,
            ):
                added, total_credits, rejected, add_warnings = _try_add_candidate(
                    selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
                )
                conflict_rejections.extend(rejected)
                warnings.extend(add_warnings)
                if total_credits >= min_credits:
                    break

        if total_credits < min_credits and preferences.get("outside_department_fill_last", True):
            for candidate in _outside_department_filler_candidates(
                target_df,
                graduation_result,
                excluded_codes | _selected_codes(selected),
                preferences,
            ):
                added, total_credits, rejected, add_warnings = _try_add_candidate(
                    selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
                )
                conflict_rejections.extend(rejected)
                warnings.extend(add_warnings)
                if total_credits >= min_credits:
                    break
    else:
        if preferences.get("user_requested_lighter_schedule") or preferences.get("prefer_general_education_courses"):
            warnings.append("由於你指定不要太硬 / 想要輕鬆一點 / 想多一點通識，本次排課沒有強制加入大量 EE/EECS 課。")
        candidates = _requirement_candidates(graduation_result, preferences)
        if bool(preferences.get("prefer_more_ee_courses", False)):
            candidates.extend(_user_requested_ee_candidates(target_df, graduation_result, excluded_codes, preferences))
            candidates.extend(_other_elective_candidates(target_df, graduation_result, excluded_codes, preferences))
        else:
            candidates.extend(_other_elective_candidates(target_df, graduation_result, excluded_codes, preferences))
            candidates.extend(_professional_candidates(target_df, graduation_result, excluded_codes, preferences))

        seen_candidates: set[tuple[int, str, str]] = set()
        deduped_candidates: list[dict] = []
        for candidate in candidates:
            key = (candidate["priority"], candidate["code"], candidate["requirement_code"])
            if key not in seen_candidates:
                deduped_candidates.append(candidate)
                seen_candidates.add(key)

        for candidate in deduped_candidates:
            if total_credits >= min_credits and candidate["priority"] >= 3:
                break
            added, total_credits, rejected, add_warnings = _try_add_candidate(
                selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
            )
            conflict_rejections.extend(rejected)
            warnings.extend(add_warnings)
            if not added and candidate["priority"] <= 2:
                unresolved.append(
                    {
                        "code": candidate["code"],
                        "requirement_code": candidate.get("requirement_code", ""),
                        "reason": _unresolved_reason_for_candidate(target_df, candidate, preferences, avoid_days),
                    }
                )

        # If still below target, continue with professional electives that fit.
        if total_credits < min_credits:
            for candidate in _professional_candidates(target_df, graduation_result, excluded_codes | _selected_codes(selected), preferences):
                added, total_credits, rejected, add_warnings = _try_add_candidate(
                    selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
                )
                conflict_rejections.extend(rejected)
                warnings.extend(add_warnings)
                if total_credits >= min_credits:
                    break

    if bool(preferences.get("limit_non_ge_non_technical_fillers", True)):
        warnings.append(
            "其餘選修補學分時，系統會優先選 GE/GEC；非 EE/EECS/CS 且非 GE/GEC 的自動 filler 課最多只放 1 門。"
        )

    if bool(preferences.get("include_pe", False)) and not any(
        course.get("code", "").startswith(PE_PREFIXES) for course in selected
    ):
        pe_candidates = [
            candidate
            for candidate in _other_elective_candidates(
                target_df,
                graduation_result,
                excluded_codes | _selected_codes(selected),
                {**preferences, "include_pe": True},
            )
            if candidate.get("code", "").startswith(PE_PREFIXES)
        ]
        for candidate in pe_candidates:
            added, total_credits, rejected, add_warnings = _try_add_candidate(
                selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
            )
            conflict_rejections.extend(rejected)
            warnings.extend(add_warnings)
            if added:
                warnings.append(
                    "A physical education course was added for schedule balance. It is listed in the plan but contributes 0 graduation credits."
                )
                break

    if total_credits < min_credits:
        warnings.append(
            f"The recommendation only reaches {total_credits:g} credits, below the target minimum of {min_credits:g}."
        )
    if total_credits > max_credits:
        warnings.append(
            f"The recommendation reaches {total_credits:g} credits, above the target maximum of {max_credits:g}."
        )
    warnings.extend(_semester_load_warnings(total_credits, min_credits, max_credits))

    target_has_related_electronics = not search_target_courses(target_df, ["EE2250", "EE2260"]).empty
    unresolved_codes = {item.get("code") for item in unresolved}
    missing_codes = {item.get("code") for item in graduation_result.get("missing_required", [])}
    if "EE2255" in missing_codes and target_has_related_electronics:
        warnings.append(
            "Target semester 11420 has EE2250/EE2260-related electronics courses, but they are not treated as EE2255 for EE112 unless the department confirms the substitution."
        )

    plan_conflicts = check_plan_conflicts(selected)
    warnings.extend(graduation_result.get("warnings", []))
    warnings = list(dict.fromkeys(warnings))

    return {
        "target_semester": TARGET_SEMESTER,
        "recommended_courses": selected,
        "total_credits": total_credits,
        "target_credit_range": {"min": min_credits, "max": max_credits},
        "course_mix_policy": {
            "composition_policy": preferences.get("composition_policy", "ee_first"),
            "initial_ee_theory_target": _desired_ee_theory_count(preferences, min_credits, max_credits),
            "initial_lab_count": int(preferences.get("initial_lab_count", 1)),
            "outside_department_fill_last": bool(preferences.get("outside_department_fill_last", True)),
            "prefer_ge_gec_for_remaining_fillers": bool(preferences.get("prefer_ge_gec_for_remaining_fillers", True)),
            "max_non_ge_non_technical_fillers": _max_non_ge_non_technical_fillers(preferences),
            "balance_with_other_electives": bool(preferences.get("balance_with_other_electives", True)),
            "include_general_education": True,
            "include_language": True,
            "include_outside_department": bool(preferences.get("include_outside_department", True)),
            "include_pe": bool(preferences.get("include_pe", False)),
            "avoid_difficult_courses_when_filling_electives": bool(preferences.get("avoid_difficult_courses", True)),
            "avoid_days": sorted(avoid_days),
            "strict_avoid_days": _strict_avoid_days(preferences),
            "exclude_time_slots": sorted(_excluded_time_slots(preferences)),
            "preferred_days": sorted(preferred_days),
            "strict_preferred_days": _strict_preferred_days(preferences),
        },
        "review_search_policy": {
            "enabled": _review_search_enabled(preferences),
            "sources": _review_sources(preferences),
            "preference": _review_preference(preferences),
            "max_results": int(preferences.get("review_max_results", preferences.get("ptt_max_pages", 3))),
            "used_as_soft_signal_only": True,
        },
        "ptt_review_policy": {
            "enabled": _review_search_enabled(preferences),
            "source": "PTT is one optional source within multi-source review search.",
            "preference": _review_preference(preferences),
            "max_pages": int(preferences.get("ptt_max_pages", preferences.get("review_max_results", 3))),
            "used_as_soft_signal_only": True,
        },
        "semester_credit_policy": {
            "normal_min_credits": NORMAL_SEMESTER_MIN_CREDITS,
            "normal_max_credits": NORMAL_SEMESTER_MAX_CREDITS,
            "low_credit_load_application_required": total_credits < NORMAL_SEMESTER_MIN_CREDITS,
            "overload_application_required": total_credits > NORMAL_SEMESTER_MAX_CREDITS,
        },
        "conflicts": plan_conflicts.get("conflicts", []),
        "has_conflict": plan_conflicts.get("has_conflict", False),
        "unresolved_requirements": unresolved,
        "conflict_rejections_considered": conflict_rejections,
        "warnings": warnings,
        "updated_preferences": _json_safe(preferences),
    }


def _extract_requested_codes(user_request: str, current_courses: list[dict]) -> set[str]:
    request = user_request or ""
    codes = {normalize_course_code(match.group(0)) for match in re.finditer(r"[A-Za-z]{2,6}\s*[0-9]{4}", request)}
    lowered = request.lower()
    for course in current_courses:
        code = course.get("code", "")
        names = [course.get("course_name_zh", ""), course.get("course_name_en", ""), course.get("raw_course_code", "")]
        if code and code.lower() in lowered:
            codes.add(code)
        for name in names:
            if name and name.lower() in lowered:
                codes.add(code)
    return {code for code in codes if code}


def _clean_requested_course_query(user_request: str) -> str:
    query = str(user_request or "").strip().lower()
    query = re.split(r"[，。！？!?;；\n]", query, maxsplit=1)[0]
    query = re.sub(r"^(但|可是|不過|我|幫我|請|請你|把|將|想要|想|要|可以|能不能|可不可以)\s*", "", query)
    query = re.sub(r"^(加回|補回|加入|加|修|上|想上|想修|我要上|我要修|add)\s*", "", query)
    query = re.sub(r"(這門|這堂|這個|那門|那堂|那個)$", "", query).strip()
    return query.strip(" ：:。,.，")


def _extract_requested_target_codes(user_request: str, target_df) -> set[str]:
    request = str(user_request or "")
    lowered = request.lower()
    clean_query = _clean_requested_course_query(request)
    codes = {normalize_course_code(match.group(0)) for match in re.finditer(r"[A-Za-z]{2,6}\s*[0-9]{4}", request)}
    if codes:
        return {code for code in codes if code}
    if target_df is None:
        return {code for code in codes if code}
    df = _target_only(target_df)
    for _, row in df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        raw_code = _clean(row.get("raw_course_code"))
        names = [_clean(row.get("course_name_zh")), _clean(row.get("course_name_en"))]
        if code and code.lower() in lowered:
            codes.add(code)
        if raw_code and raw_code.lower() in lowered:
            codes.add(code)
        for name in names:
            name_lower = name.lower()
            if not name_lower:
                continue
            if name_lower in lowered or (clean_query and len(clean_query) >= 2 and clean_query in name_lower):
                codes.add(code)
    return {code for code in codes if code}


def _extract_requested_days(user_request: str) -> set[str]:
    lowered = str(user_request or "").lower()
    wants_avoid = any(
        keyword in lowered
        for keyword in (
            "不要",
            "不想",
            "不上",
            "不要上",
            "不要有",
            "避開",
            "去掉",
            "移除",
            "刪掉",
            "拿掉",
            "退掉",
            "avoid",
            "remove",
            "drop",
        )
    )
    if not wants_avoid:
        return set()
    days: set[str] = set()
    for day_code, keywords in REQUEST_DAY_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            days.add(day_code)
    return days


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


def _global_time_context(text: str) -> str:
    for prefix in ("晚上", "晚間", "夜間", "下午", "中午", "上午", "早上"):
        if prefix in text:
            return prefix
    return ""


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
    return {
        period
        for period, (period_start, _period_end) in PERIOD_INTERVALS.items()
        if start_minute <= period_start < end_minute
    }


def _sort_periods(periods: set[str]) -> list[str]:
    return sorted(periods, key=lambda item: PERIOD_ORDER.index(item) if item in PERIOD_ORDER else 99)


def _extract_general_time_periods(user_request: str) -> tuple[set[str], bool]:
    compact = re.sub(r"\s+", "", str(user_request or "").lower())
    periods: set[str] = set()
    matched_range = False
    hour_token = r"\d{1,2}|十二|十一|十|九|八|七|六|五|四|三|兩|二|一"
    prefix_token = r"早上|上午|中午|下午|晚上|晚間|夜間"
    global_context = _global_time_context(compact)
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
        if "早上" in compact or "上午" in compact:
            periods.update({"1", "2", "3", "4"})
        elif "中午" in compact:
            periods.add("n")
        elif "下午" in compact:
            periods.update({"5", "6", "7", "8", "9"})
        elif "晚上" in compact or "晚間" in compact or "夜間" in compact:
            periods.update({"a", "b", "c", "d"})
    return periods, matched_range


def _extract_requested_periods(user_request: str) -> set[str]:
    text = str(user_request or "").lower()
    compact = re.sub(r"\s+", "", text)
    periods, matched_general_range = _extract_general_time_periods(user_request)

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
        return set(_sort_periods(periods))

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

    for match in re.finditer(r"第\s*([1-9a-dA-D])\s*節", text):
        period = match.group(1)
        periods.add(period.lower() if period.isalpha() else period)
    return set(_sort_periods(periods))


def _extract_requested_time_slots(user_request: str) -> set[str]:
    periods = _extract_requested_periods(user_request)
    if not periods:
        return set()
    days = _extract_requested_days(user_request) or set(DAY_LABELS)
    return {f"{day}{period}" for day in days for period in periods}


def _has_requested_time_slot(course: dict | pd.Series, requested_slots: set[str]) -> bool:
    if not requested_slots:
        return False
    time_value = course.get("time") if isinstance(course, dict) else course.get("time")
    return bool(parse_time_slots(time_value) & requested_slots)


def _extract_requested_categories(user_request: str) -> set[str]:
    lowered = str(user_request or "").lower()
    wants_remove = any(
        keyword in lowered
        for keyword in ("不要", "不想", "不上", "不修", "去掉", "移除", "刪掉", "拿掉", "退掉", "drop", "remove")
    )
    if not wants_remove:
        return set()
    categories: set[str] = set()
    if "實驗" in lowered or "lab" in lowered or "experiment" in lowered:
        categories.add("lab")
    return categories


def _non_ee_replacement_rank(course: dict) -> tuple[int, float, str]:
    code = normalize_course_code(course.get("code", ""))
    credits = float(course.get("credits") or 0)
    if code.startswith(("CL", "CLC", "LANG", "FL", "GE", "GEC")):
        category = 0
    elif code.startswith(PE_PREFIXES):
        category = 2
    else:
        category = 1
    return category, -credits, code


def _general_education_replacement_rank(course: dict) -> tuple[int, float, str]:
    code = normalize_course_code(course.get("code", ""))
    credits = float(course.get("credits") or 0)
    if code.startswith(("CL", "CLC", "LANG", "FL")):
        category = 0
    elif not _is_ee_course_code(code) and not code.startswith(PE_PREFIXES):
        category = 1
    elif code.startswith(PE_PREFIXES):
        category = 2
    else:
        category = 3
    return category, -credits, code


def _build_update_plan_result(
    selected: list[dict],
    current_plan: dict,
    preferences: dict,
    graduation_result: dict,
    warnings: list[str],
    conflict_rejections: list[dict] | None = None,
) -> dict:
    min_credits, max_credits = _target_credit_range(preferences)
    total_credits = sum(float(course.get("credits") or 0) for course in selected)
    if total_credits < min_credits:
        warnings.append(
            f"After the update, the plan has {total_credits:g} credits, below the current target minimum of {min_credits:g}."
        )
    if total_credits > max_credits:
        warnings.append(
            f"After the update, the plan has {total_credits:g} credits, above the current target maximum of {max_credits:g}."
        )
    if preferences.get("exclude_lab_courses"):
        if _explicit_lab_override_codes(preferences):
            warnings.append(
                "目前仍保留不要實驗課的限制；但你明確指定的實驗課會作為例外候選，仍需通過衝堂與學分檢查。"
            )
        else:
            warnings.append(
                "因為你前面已經說過不要實驗課，所以這次修改會排除實驗課，除非你明確指定某一門實驗課。"
            )
    if preferences.get("prefer_theory_ee_courses"):
        warnings.append(
            "因為你要求電機系理論/非實驗課，所以這次新增電機系課程時，我排除了實驗課，只搜尋 EE/EECS 的非實驗課。"
        )
    warnings.extend(_semester_load_warnings(total_credits, min_credits, max_credits))
    warnings.extend(graduation_result.get("warnings", []))
    plan_conflicts = check_plan_conflicts(selected)
    avoid_days = _avoid_days(preferences)
    preferred_days = _preferred_days(preferences)
    return {
        "target_semester": TARGET_SEMESTER,
        "recommended_courses": selected,
        "total_credits": total_credits,
        "target_credit_range": {"min": min_credits, "max": max_credits},
        "course_mix_policy": {
            "balance_with_other_electives": bool(preferences.get("balance_with_other_electives", True)),
            "include_general_education": True,
            "include_language": True,
            "include_outside_department": bool(preferences.get("include_outside_department", True)),
            "include_pe": bool(preferences.get("include_pe", False)),
            "avoid_difficult_courses_when_filling_electives": bool(preferences.get("avoid_difficult_courses", True)),
            "avoid_days": sorted(avoid_days),
            "strict_avoid_days": _strict_avoid_days(preferences),
            "exclude_time_slots": sorted(_excluded_time_slots(preferences)),
            "preferred_days": sorted(preferred_days),
            "strict_preferred_days": _strict_preferred_days(preferences),
        },
        "review_search_policy": {
            "enabled": _review_search_enabled(preferences),
            "sources": _review_sources(preferences),
            "preference": _review_preference(preferences),
            "max_results": int(preferences.get("review_max_results", preferences.get("ptt_max_pages", 3))),
            "used_as_soft_signal_only": True,
        },
        "ptt_review_policy": {
            "enabled": _review_search_enabled(preferences),
            "source": "PTT is one optional source within multi-source review search.",
            "preference": _review_preference(preferences),
            "max_pages": int(preferences.get("ptt_max_pages", preferences.get("review_max_results", 3))),
            "used_as_soft_signal_only": True,
        },
        "semester_credit_policy": {
            "normal_min_credits": NORMAL_SEMESTER_MIN_CREDITS,
            "normal_max_credits": NORMAL_SEMESTER_MAX_CREDITS,
            "low_credit_load_application_required": total_credits < NORMAL_SEMESTER_MIN_CREDITS,
            "overload_application_required": total_credits > NORMAL_SEMESTER_MAX_CREDITS,
        },
        "conflicts": plan_conflicts.get("conflicts", []),
        "has_conflict": plan_conflicts.get("has_conflict", False),
        "unresolved_requirements": current_plan.get("unresolved_requirements", []),
        "conflict_rejections_considered": conflict_rejections or [],
        "warnings": list(dict.fromkeys(warnings)),
        "updated_preferences": _json_safe(preferences),
    }


def _update_more_ee_courses(
    current_plan: dict,
    user_request: str,
    target_df,
    graduation_result: dict,
    preferences: dict,
) -> dict:
    selected = [dict(course) for course in current_plan.get("recommended_courses", [])]
    requested_count = max(1, int(preferences.get("requested_ee_course_count", 1) or 1))
    _, max_credits = _target_credit_range(preferences)
    total_credits = sum(float(course.get("credits") or 0) for course in selected)
    avoid_friday = bool(preferences.get("avoid_friday", False))
    replace_non_ee_first = bool(preferences.get("replace_non_ee_first") or preferences.get("reduce_non_ee_courses"))
    excluded_codes = {normalize_course_code(code) for code in preferences.get("exclude_course_codes", [])}
    excluded_codes.difference_update(_explicit_lab_override_codes(preferences))
    excluded_codes |= _selected_codes(selected)
    already_counted_codes = {
        course.get("code", "")
        for course in graduation_result.get("counted_courses", [])
        if course.get("code")
    }
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("completed_courses_official", [])
        if course.get("code")
    )
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("in_progress_courses_counted_in_planning", [])
        if course.get("code")
    )
    excluded_codes |= already_counted_codes

    if replace_non_ee_first:
        warnings: list[str] = [
            (
                f"User requested reducing non-EE courses and replacing them with {requested_count} EE/EECS course(s). "
                "The planner will try replace-first instead of simply adding another EE/EECS course."
            )
        ]
    else:
        warnings = [
            (
                f"User requested {requested_count} additional EE/EECS course(s). "
                "The planner first tries to add EE/EECS courses, then replaces non-EE courses only if adding does not fit."
            )
        ]
    if graduation_result.get("in_progress_courses_counted_in_planning"):
        warnings.append("Courses currently in progress are treated as expected-to-pass for this planning recommendation.")
    if preferences.get("credit_target_relaxed_for_followup"):
        center = preferences.get("target_credit_soft_center")
        if center is not None:
            warnings.append(
                f"Previous exact credit target ({float(center):g} credits) was relaxed for this follow-up edit. "
                "The planner will stay close when possible, but the new user request is prioritized."
            )

    candidates = _user_requested_ee_candidates(target_df, graduation_result, excluded_codes, preferences)
    added_courses: list[dict] = []
    removed_courses: list[dict] = []
    replacement_pairs: list[dict] = []
    conflict_rejections: list[dict] = []

    def try_replace_one(candidate: dict, direct_add_was_attempted: bool) -> bool:
        nonlocal selected, total_credits, conflict_rejections
        non_ee_courses = sorted(
            [course for course in selected if not _is_ee_course_code(course.get("code", ""))],
            key=_non_ee_replacement_rank,
        )
        for removed in non_ee_courses:
            trial_selected = [course for course in selected if course.get("code") != removed.get("code")]
            trial_total = total_credits - float(removed.get("credits") or 0)
            added, new_total, rejected, add_warnings = _try_add_candidate(
                trial_selected,
                target_df,
                candidate,
                trial_total,
                max_credits,
                avoid_friday,
                excluded_codes | {removed.get("code", "")},
                preferences,
            )
            conflict_rejections.extend(rejected)
            if not added:
                continue
            warnings.extend(add_warnings)
            selected = trial_selected
            total_credits = new_total
            new_course = selected[-1]
            added_courses.append(new_course)
            removed_courses.append(removed)
            excluded_codes.add(new_course.get("code", ""))
            excluded_codes.add(removed.get("code", ""))
            if replace_non_ee_first:
                why_removed = (
                    f"{removed.get('code')} is not an EE/EECS course. The user asked to reduce non-EE courses, "
                    "so the planner used replace-first instead of simply adding credits."
                )
            elif direct_add_was_attempted:
                why_removed = (
                    f"{removed.get('code')} is not an EE/EECS course, and adding another EE/EECS course directly did not fit. "
                    "It was replaced only after the add-first step failed."
                )
            else:
                why_removed = f"{removed.get('code')} is not an EE/EECS course and was selected as a replaceable non-protected course."
            replacement_pairs.append(
                {
                    "removed_code": removed.get("code", ""),
                    "removed_name_zh": removed.get("course_name_zh", ""),
                    "added_code": new_course.get("code", ""),
                    "added_name_zh": new_course.get("course_name_zh", ""),
                    "why_removed": why_removed,
                    "why_added": (
                        f"{new_course.get('code')} was selected from target semester 11420 EE/EECS courses. "
                        "The replacement was checked against time conflicts, avoid-day rules, completed courses, and the current credit ceiling."
                    ),
                }
            )
            return True
        return False

    if replace_non_ee_first:
        for candidate in candidates:
            if len(added_courses) >= requested_count:
                break
            if candidate.get("code") in _selected_codes(selected):
                continue
            try_replace_one(candidate, direct_add_was_attempted=False)
    else:
        for candidate in candidates:
            if len(added_courses) >= requested_count:
                break
            added, total_credits, rejected, add_warnings = _try_add_candidate(
                selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
            )
            conflict_rejections.extend(rejected)
            warnings.extend(add_warnings)
            if added:
                new_course = selected[-1]
                added_courses.append(new_course)
                excluded_codes.add(new_course.get("code", ""))
                replacement_pairs.append(
                    {
                        "removed_code": "",
                        "removed_name_zh": "",
                        "added_code": new_course.get("code", ""),
                        "added_name_zh": new_course.get("course_name_zh", ""),
                        "why_removed": "No non-EE course was removed because the EE/EECS course could be added directly.",
                        "why_added": (
                            f"{new_course.get('code')} was added because the user requested more EE/EECS courses. "
                            "The add-first policy succeeded without creating a time conflict or exceeding the current credit ceiling."
                        ),
                    }
                )

    if replace_non_ee_first and not removed_courses:
        warnings.append(
            "No non-EE course could be safely replaced by an EE/EECS course under the current credit, conflict, avoid-day, and completed-course constraints."
        )
    if len(added_courses) < requested_count:
        action_word = "added or swapped in" if replace_non_ee_first else "added"
        warnings.append(
            f"Only {len(added_courses)} EE/EECS course(s) could be {action_word} out of the requested {requested_count}."
        )
        if not replace_non_ee_first:
            warnings.append(
                "The user asked to add EE/EECS course(s), not to replace a specific course, so the planner did not remove existing courses automatically."
            )
        failure_details = []
        seen_failure_codes: set[str] = set()
        for candidate in candidates:
            candidate_code = candidate.get("code")
            if candidate_code in seen_failure_codes or candidate_code in _selected_codes(selected):
                continue
            seen_failure_codes.add(candidate_code)
            failure_details.append(
                _candidate_add_failure_summary(
                    selected,
                    target_df,
                    candidate,
                    total_credits,
                    max_credits,
                    avoid_friday,
                    preferences,
                )
            )
            if len(failure_details) >= 4:
                break
        if failure_details:
            warnings.append("加課失敗原因：" + "；".join(failure_details))

    updated = _build_update_plan_result(
        selected=selected,
        current_plan=current_plan,
        preferences=preferences,
        graduation_result=graduation_result,
        warnings=warnings,
        conflict_rejections=conflict_rejections,
    )
    updated["previous_courses_removed"] = removed_courses
    updated["update_request"] = user_request
    if replace_non_ee_first:
        selection_policy = (
            "For a request such as '不要那麼多不是電機系的課', the agent uses replace-first: it tries to remove replaceable non-EE courses "
            "and swap in EE/EECS courses. It does not simply add an EE course while leaving all non-EE courses unchanged. "
            "All candidates still come only from target semester 11420 and are checked deterministically for conflicts."
        )
    else:
        selection_policy = (
            "For a request to take more EE/EECS courses, the agent first tries to add the requested number of EE/EECS courses. "
            "If adding would exceed the credit ceiling, miss the requested weekday, or create a time conflict, it reports the reason instead of replacing existing courses automatically. "
            "All candidates still come only from target semester 11420 and are checked deterministically for conflicts."
        )
    updated["replacement_summary"] = {
        "user_request": user_request,
        "trigger": "more_ee_courses",
        "triggered_by_ptt_review": False,
        "triggered_by_review_search": False,
        "removed_courses": removed_courses,
        "added_courses": added_courses,
        "replacement_pairs": replacement_pairs,
        "selection_policy": selection_policy,
    }
    return updated


def _update_more_general_education_courses(
    current_plan: dict,
    user_request: str,
    target_df,
    graduation_result: dict,
    preferences: dict,
) -> dict:
    selected = [dict(course) for course in current_plan.get("recommended_courses", [])]
    requested_count = max(1, int(preferences.get("requested_general_education_count", 1) or 1))
    _, max_credits = _target_credit_range(preferences)
    total_credits = sum(float(course.get("credits") or 0) for course in selected)
    avoid_friday = bool(preferences.get("avoid_friday", False))
    excluded_codes = {normalize_course_code(code) for code in preferences.get("exclude_course_codes", [])}
    excluded_codes.difference_update(_explicit_lab_override_codes(preferences))
    excluded_codes |= _selected_codes(selected)

    already_counted_codes = {
        course.get("code", "")
        for course in graduation_result.get("counted_courses", [])
        if course.get("code")
    }
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("completed_courses_official", [])
        if course.get("code")
    )
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("in_progress_courses_counted_in_planning", [])
        if course.get("code")
    )
    excluded_codes |= already_counted_codes

    warnings: list[str] = [
        (
            f"User requested {requested_count} additional general education course(s). "
            "The planner first tries to add GE/GEC courses, then replaces non-protected non-GE courses only if adding does not fit."
        )
    ]
    if graduation_result.get("in_progress_courses_counted_in_planning"):
        warnings.append("Courses currently in progress are treated as expected-to-pass for this planning recommendation.")
    if preferences.get("credit_target_relaxed_for_followup"):
        center = preferences.get("target_credit_soft_center")
        if center is not None:
            warnings.append(
                f"Previous exact credit target ({float(center):g} credits) was relaxed for this follow-up edit. "
                "The planner will stay close when possible, but the new user request is prioritized."
            )

    candidates = _user_requested_general_education_candidates(target_df, excluded_codes)
    added_courses: list[dict] = []
    removed_courses: list[dict] = []
    replacement_pairs: list[dict] = []
    conflict_rejections: list[dict] = []

    for candidate in candidates:
        if len(added_courses) >= requested_count:
            break
        added, total_credits, rejected, add_warnings = _try_add_candidate(
            selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
        )
        conflict_rejections.extend(rejected)
        warnings.extend(add_warnings)
        if added:
            new_course = selected[-1]
            added_courses.append(new_course)
            excluded_codes.add(new_course.get("code", ""))
            replacement_pairs.append(
                {
                    "removed_code": "",
                    "removed_name_zh": "",
                    "added_code": new_course.get("code", ""),
                    "added_name_zh": new_course.get("course_name_zh", ""),
                    "why_removed": "No course was removed because the general education course could be added directly.",
                    "why_added": (
                        f"{new_course.get('code')} was added because the user requested more general education courses. "
                        "The add-first policy succeeded without creating a time conflict or exceeding the current credit ceiling."
                    ),
                }
            )

    if len(added_courses) < requested_count:
        for candidate in candidates:
            if len(added_courses) >= requested_count:
                break
            if candidate.get("code") in _selected_codes(selected):
                continue
            removable_courses = sorted(
                [
                    course
                    for course in selected
                    if not _is_general_education_code(course.get("code", ""))
                    and not _is_protected_graduation_course(course)
                ],
                key=_general_education_replacement_rank,
            )
            for removed in removable_courses:
                trial_selected = [course for course in selected if course.get("code") != removed.get("code")]
                trial_total = total_credits - float(removed.get("credits") or 0)
                added, new_total, rejected, add_warnings = _try_add_candidate(
                    trial_selected,
                    target_df,
                    candidate,
                    trial_total,
                    max_credits,
                    avoid_friday,
                    excluded_codes | {removed.get("code", "")},
                    preferences,
                )
                conflict_rejections.extend(rejected)
                if not added:
                    continue
                warnings.extend(add_warnings)
                selected = trial_selected
                total_credits = new_total
                new_course = selected[-1]
                added_courses.append(new_course)
                removed_courses.append(removed)
                excluded_codes.add(new_course.get("code", ""))
                excluded_codes.add(removed.get("code", ""))
                replacement_pairs.append(
                    {
                        "removed_code": removed.get("code", ""),
                        "removed_name_zh": removed.get("course_name_zh", ""),
                        "added_code": new_course.get("code", ""),
                        "added_name_zh": new_course.get("course_name_zh", ""),
                        "why_removed": (
                            f"{removed.get('code')} is not a GE/GEC course and is not a protected graduation-gap course. "
                            "It was replaced only after adding a general education course directly did not fit."
                        ),
                        "why_added": (
                            f"{new_course.get('code')} was selected from 11420 GE/GEC courses because the user requested a general education course. "
                            "The replacement was checked against time conflicts and the current credit ceiling."
                        ),
                    }
                )
                break

    if len(added_courses) < requested_count:
        warnings.append(
            f"Only {len(added_courses)} general education course(s) could be added or swapped in out of the requested {requested_count}."
        )

    updated = _build_update_plan_result(
        selected=selected,
        current_plan=current_plan,
        preferences=preferences,
        graduation_result=graduation_result,
        warnings=warnings,
        conflict_rejections=conflict_rejections,
    )
    updated["previous_courses_removed"] = removed_courses
    updated["update_request"] = user_request
    updated["replacement_summary"] = {
        "user_request": user_request,
        "trigger": "more_general_education_courses",
        "triggered_by_ptt_review": False,
        "triggered_by_review_search": False,
        "removed_courses": removed_courses,
        "added_courses": added_courses,
        "replacement_pairs": replacement_pairs,
        "selection_policy": (
            "For a request to take general education courses, the agent first tries to add the requested number of GE/GEC courses. "
            "If adding would exceed the credit ceiling or create a time conflict, it then tries to replace non-protected non-GE courses already in the plan. "
            "Required, alternative-required, and required-lab courses are protected from this replacement."
        ),
    }
    return updated


def _update_add_specific_courses(
    current_plan: dict,
    user_request: str,
    target_df,
    graduation_result: dict,
    preferences: dict,
) -> dict:
    selected = [dict(course) for course in current_plan.get("recommended_courses", [])]
    requested_codes = _extract_requested_target_codes(user_request, target_df)
    min_credits, max_credits = _target_credit_range(preferences)
    total_credits = sum(float(course.get("credits") or 0) for course in selected)
    avoid_friday = bool(preferences.get("avoid_friday", False))
    selected_codes = _selected_codes(selected)
    excluded_codes = {normalize_course_code(code) for code in preferences.get("exclude_course_codes", [])}
    excluded_codes.difference_update(_explicit_lab_override_codes(preferences))
    excluded_codes |= selected_codes

    already_counted_codes = {
        course.get("code", "")
        for course in graduation_result.get("counted_courses", [])
        if course.get("code")
    }
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("completed_courses_official", [])
        if course.get("code")
    )
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("in_progress_courses_counted_in_planning", [])
        if course.get("code")
    )
    excluded_codes |= already_counted_codes

    warnings: list[str] = []
    explicit_lab_codes = _explicit_lab_override_codes(preferences)
    requested_lab_overrides = sorted(code for code in requested_codes if code in explicit_lab_codes)
    if preferences.get("exclude_lab_courses") and requested_lab_overrides:
        warnings.append(
            "雖然你之前說不要實驗課，但這次你明確指定這門實驗課，所以我會嘗試加入並重新檢查衝堂與學分。"
        )
    if graduation_result.get("in_progress_courses_counted_in_planning"):
        warnings.append("Courses currently in progress are treated as expected-to-pass for this planning recommendation.")
    if preferences.get("credit_target_relaxed_for_followup"):
        center = preferences.get("target_credit_soft_center")
        if center is not None:
            warnings.append(
                f"Previous exact credit target ({float(center):g} credits) was relaxed for this follow-up edit. "
                "The planner will stay close when possible, but the new user request is prioritized."
            )

    added_courses: list[dict] = []
    conflict_rejections: list[dict] = []
    replacement_pairs: list[dict] = []
    if not requested_codes:
        warnings.append("No exact 11420 course was identified in the add-course request, so no course was added.")

    for code in sorted(requested_codes):
        if code in selected_codes:
            warnings.append(f"{code} is already in the current plan, so it was not added again.")
            continue
        candidate = {
            "priority": 0,
            "code": code,
            "reason": f"user requested specific course: {code}",
            "requirement_code": "user_requested_specific_course",
        }
        added, total_credits, rejected, add_warnings = _try_add_candidate(
            selected, target_df, candidate, total_credits, max_credits, avoid_friday, excluded_codes, preferences
        )
        conflict_rejections.extend(rejected)
        warnings.extend(add_warnings)
        if not added:
            availability = search_target_courses(target_df, [code])
            if availability.empty:
                warnings.append(f"{code} was not found in target semester 11420, so it was not added.")
            else:
                details = _specific_add_failure_details(
                    selected,
                    target_df,
                    code,
                    total_credits,
                    max_credits,
                    avoid_friday,
                    preferences,
                )
                if details:
                    warnings.append("指定課程加入失敗：" + "；".join(details))
                else:
                    warnings.append(
                        f"{code} could not be added because of the current credit limit, requested avoid-day rules, or time conflicts."
                    )
            continue
        new_course = selected[-1]
        added_courses.append(new_course)
        excluded_codes.add(new_course.get("code", ""))
        selected_codes.add(new_course.get("code", ""))
        replacement_pairs.append(
            {
                "removed_code": "",
                "removed_name_zh": "",
                "added_code": new_course.get("code", ""),
                "added_name_zh": new_course.get("course_name_zh", ""),
                "why_removed": "No course was removed because the user asked to add a specific course.",
                "why_added": (
                    f"{new_course.get('code')} was added because it exactly matched the user's course-name/course-code request. "
                    "The addition was checked against target semester 11420, time conflicts, and the current credit ceiling."
                ),
            }
        )

    updated = _build_update_plan_result(
        selected=selected,
        current_plan=current_plan,
        preferences=preferences,
        graduation_result=graduation_result,
        warnings=warnings,
        conflict_rejections=conflict_rejections,
    )
    updated["previous_courses_removed"] = []
    updated["update_request"] = user_request
    updated["replacement_summary"] = {
        "user_request": user_request,
        "trigger": "add_specific_courses",
        "triggered_by_ptt_review": False,
        "triggered_by_review_search": False,
        "removed_courses": [],
        "added_courses": added_courses,
        "replacement_pairs": replacement_pairs,
        "selection_policy": (
            "For an explicit add-course request, the agent only tries to add the course names or course codes mentioned by the user. "
            "It does not regenerate the whole plan or add related courses from the same requirement category."
        ),
    }
    return updated


@langsmith_trace("recommender.update_plan")
def update_plan(current_plan: dict, user_request: str, target_df, graduation_result, preferences: dict) -> dict:
    preferences = dict(preferences or {})
    current_courses = [dict(course) for course in current_plan.get("recommended_courses", [])]
    if preferences.get("update_mode") == "add_or_replace_ee":
        return _update_more_ee_courses(current_plan, user_request, target_df, graduation_result, preferences)
    if preferences.get("update_mode") == "add_or_replace_general_education":
        return _update_more_general_education_courses(current_plan, user_request, target_df, graduation_result, preferences)
    if preferences.get("update_mode") == "add_specific_course":
        return _update_add_specific_courses(current_plan, user_request, target_df, graduation_result, preferences)
    remove_codes = _extract_requested_codes(user_request, current_courses)
    remove_only = preferences.get("update_mode") == "remove_only"
    requested_remove_days = _extract_requested_days(user_request) if remove_only else set()
    requested_remove_time_slots = _extract_requested_time_slots(user_request) if remove_only else set()
    requested_remove_categories = (
        _extract_requested_categories(user_request)
        if remove_only and not remove_codes and not requested_remove_time_slots
        else set()
    )
    if requested_remove_time_slots:
        remove_codes.update(
            course.get("code", "")
            for course in current_courses
            if course.get("code") and _has_requested_time_slot(course, requested_remove_time_slots)
        )
    elif requested_remove_days:
        remove_codes.update(
            course.get("code", "")
            for course in current_courses
            if course.get("code") and _has_avoid_day(course, requested_remove_days)
        )
    if "lab" in requested_remove_categories:
        remove_codes.update(
            course.get("code", "")
            for course in current_courses
            if course.get("code") and _is_lab_course(course)
        )
    matched_user_request = bool(remove_codes)

    if not remove_codes and current_courses and not remove_only:
        friday_courses = [course for course in current_courses if _has_friday(course)]
        fallback_course = friday_courses[0] if friday_courses else current_courses[-1]
        remove_codes = {fallback_course.get("code", "")}

    removed_courses = [course for course in current_courses if course.get("code") in remove_codes]
    kept_courses = [course for course in current_courses if course.get("code") not in remove_codes]

    preferences["locked_courses"] = kept_courses
    already_counted_codes = {
        course.get("code", "")
        for course in graduation_result.get("counted_courses", [])
        if course.get("code")
    }
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("completed_courses_official", [])
        if course.get("code")
    )
    already_counted_codes.update(
        course.get("code", "")
        for course in graduation_result.get("in_progress_courses_counted_in_planning", [])
        if course.get("code")
    )
    preferences["exclude_course_codes"] = list(
        set(preferences.get("exclude_course_codes", [])) | remove_codes | already_counted_codes
    )

    if remove_only:
        min_credits, max_credits = _target_credit_range(preferences)
        total_credits = sum(float(course.get("credits") or 0) for course in kept_courses)
        warnings: list[str] = []
        if graduation_result.get("in_progress_courses_counted_in_planning"):
            warnings.append(
                "Courses currently in progress are treated as expected-to-pass for this planning recommendation."
            )
        if preferences.get("credit_target_relaxed_for_followup"):
            center = preferences.get("target_credit_soft_center")
            if center is not None:
                warnings.append(
                    f"Previous exact credit target ({float(center):g} credits) was relaxed for this follow-up edit. "
                    "The planner will stay close when possible, but the new user request is prioritized."
                )
        if total_credits < min_credits:
            warnings.append(
                f"After removal, the plan has {total_credits:g} credits, below the current target minimum of {min_credits:g}."
            )
        if total_credits > max_credits:
            warnings.append(
                f"After removal, the plan has {total_credits:g} credits, above the current target maximum of {max_credits:g}."
            )
        warnings.extend(_semester_load_warnings(total_credits, min_credits, max_credits))
        warnings.extend(graduation_result.get("warnings", []))
        plan_conflicts = check_plan_conflicts(kept_courses)
        updated = {
            "target_semester": TARGET_SEMESTER,
            "recommended_courses": kept_courses,
            "total_credits": total_credits,
            "target_credit_range": {"min": min_credits, "max": max_credits},
            "course_mix_policy": {
                "balance_with_other_electives": bool(preferences.get("balance_with_other_electives", True)),
                "include_general_education": True,
                "include_language": True,
                "include_outside_department": bool(preferences.get("include_outside_department", True)),
                "include_pe": bool(preferences.get("include_pe", False)),
                "avoid_difficult_courses_when_filling_electives": bool(preferences.get("avoid_difficult_courses", True)),
                "avoid_days": sorted(_avoid_days(preferences)),
                "strict_avoid_days": _strict_avoid_days(preferences),
                "exclude_time_slots": sorted(_excluded_time_slots(preferences)),
                "preferred_days": sorted(_preferred_days(preferences)),
                "strict_preferred_days": _strict_preferred_days(preferences),
            },
            "review_search_policy": {
                "enabled": _review_search_enabled(preferences),
                "sources": _review_sources(preferences),
                "preference": _review_preference(preferences),
                "max_results": int(preferences.get("review_max_results", preferences.get("ptt_max_pages", 3))),
                "used_as_soft_signal_only": True,
            },
            "ptt_review_policy": {
                "enabled": _review_search_enabled(preferences),
                "source": "PTT is one optional source within multi-source review search.",
                "preference": _review_preference(preferences),
                "max_pages": int(preferences.get("ptt_max_pages", preferences.get("review_max_results", 3))),
                "used_as_soft_signal_only": True,
            },
            "semester_credit_policy": {
                "normal_min_credits": NORMAL_SEMESTER_MIN_CREDITS,
                "normal_max_credits": NORMAL_SEMESTER_MAX_CREDITS,
                "low_credit_load_application_required": total_credits < NORMAL_SEMESTER_MIN_CREDITS,
                "overload_application_required": total_credits > NORMAL_SEMESTER_MAX_CREDITS,
            },
            "conflicts": plan_conflicts.get("conflicts", []),
            "has_conflict": plan_conflicts.get("has_conflict", False),
            "unresolved_requirements": current_plan.get("unresolved_requirements", []),
            "conflict_rejections_considered": [],
            "warnings": list(dict.fromkeys(warnings)),
            "previous_courses_removed": removed_courses,
            "update_request": user_request,
        }
        updated["replacement_summary"] = {
            "user_request": user_request,
            "trigger": "remove_only" if matched_user_request else "remove_only_no_match",
            "triggered_by_ptt_review": False,
            "triggered_by_review_search": False,
            "remove_only_days": sorted(requested_remove_days),
            "remove_only_time_slots": sorted(requested_remove_time_slots),
            "remove_only_categories": sorted(requested_remove_categories),
            "removed_courses": removed_courses,
            "added_courses": [],
            "replacement_pairs": [
                {
                    "removed_code": course.get("code", ""),
                    "removed_name_zh": course.get("course_name_zh", ""),
                    "added_code": "",
                    "added_name_zh": "",
                    "why_removed": "Matched the user's remove-only request. Online reviews did not trigger this removal.",
                    "why_added": "No replacement was added because the user asked to remove the course only.",
                }
                for course in removed_courses
            ],
            "selection_policy": (
                "The user asked to remove courses only, so the agent locked the remaining current courses, "
                "removed the matched course or weekday time slots, recalculated credits and time conflicts, and did not add a replacement course. "
                "Online reviews are only a soft reference and never automatically remove a course from the plan."
            ),
        }
        if not removed_courses:
            if requested_remove_time_slots:
                no_match_message = "No current courses matched the requested weekday/time-slot removal, so no course was removed."
            elif requested_remove_days:
                no_match_message = "No current courses matched the requested weekday removal, so no course was removed."
            elif requested_remove_categories:
                no_match_message = "No current courses matched the requested category removal, so no course was removed."
            else:
                no_match_message = "No exact course was identified in the remove-only request, so no course was removed."
            updated["warnings"] = list(
                dict.fromkeys(
                    updated.get("warnings", [])
                    + [no_match_message]
                )
            )
        return updated

    # The real graduation progress stays in graduation_result. Courses already
    # completed or in progress are excluded through preferences above.
    empty_student = pd.DataFrame(columns=["normalized_course_code", "status"])
    updated = recommend_courses(empty_student, target_df, graduation_result, preferences)
    updated["previous_courses_removed"] = removed_courses
    updated["update_request"] = user_request

    kept_codes = {course.get("code", "") for course in kept_courses if course.get("code")}
    added_courses = [
        course
        for course in updated.get("recommended_courses", [])
        if course.get("code") and course.get("code") not in kept_codes
    ]
    why_removed = (
        "Matched the user's replacement request. Online reviews did not trigger this removal."
        if matched_user_request
        else "No exact course was identified, so the system chose a fallback course from the current plan. Online reviews did not trigger this removal."
    )
    replacement_pairs: list[dict] = []
    pair_count = max(len(removed_courses), len(added_courses))
    for index in range(pair_count):
        removed = removed_courses[index] if index < len(removed_courses) else {}
        added = added_courses[index] if index < len(added_courses) else {}
        added_reason = added.get("recommendation_reason", "Selected by the recommender") if added else ""
        if added_reason and not str(added_reason).rstrip().endswith((".", "!", "?")):
            added_reason = f"{added_reason}."
        replacement_pairs.append(
            {
                "removed_code": removed.get("code", ""),
                "removed_name_zh": removed.get("course_name_zh", ""),
                "added_code": added.get("code", ""),
                "added_name_zh": added.get("course_name_zh", ""),
                "why_removed": why_removed if removed else "No additional course was removed.",
                "why_added": (
                    f"{added_reason} "
                    "It keeps the updated plan within the target credit range while avoiding time conflicts and excluding courses already completed or in progress."
                    if added
                    else "No replacement course was added because the remaining plan already satisfied the planner constraints or no suitable course fit."
                ),
            }
        )

    updated["replacement_summary"] = {
        "user_request": user_request,
        "trigger": "user_request" if matched_user_request else "fallback",
        "triggered_by_ptt_review": False,
        "triggered_by_review_search": False,
        "removed_courses": removed_courses,
        "added_courses": added_courses,
        "replacement_pairs": replacement_pairs,
        "selection_policy": (
            "After removing the requested course, the recommender locked the remaining courses, "
            "excluded courses already completed or in progress, searched only target semester 11420, "
            "then selected courses that stay near the current target credit range when possible, prioritize remaining graduation requirements, and avoid time conflicts. "
            "Online reviews are only a soft teacher/section ranking signal and never automatically remove a course from the plan."
        ),
    }

    if not removed_courses:
        updated["warnings"] = list(
            dict.fromkeys(
                updated.get("warnings", [])
                + ["No exact course was identified in the request, so the plan was regenerated from current preferences."]
            )
        )
    elif not added_courses:
        updated["warnings"] = list(
            dict.fromkeys(
                updated.get("warnings", [])
                + [
                    "替換沒有補上新課：可能是 11420 沒有符合候選、候選課已修或已在課表中、加入後超過學分上限，或與目前課表衝堂。"
                ]
            )
        )
    return updated
