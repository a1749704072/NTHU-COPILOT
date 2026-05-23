from __future__ import annotations

import math
import re
from typing import Any


TIME_TOKEN_RE = re.compile(r"([MTWRFSU])([0-9A-Za-z])", re.IGNORECASE)


def parse_time_slots(time_str: str) -> set[str]:
    """Parse NTHU time strings like M3M4W1W2 into comparable slot tokens."""
    if time_str is None:
        return set()
    if isinstance(time_str, float) and math.isnan(time_str):
        return set()

    text = str(time_str).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()

    slots: set[str] = set()
    for day, period in TIME_TOKEN_RE.findall(text):
        normalized_period = period.lower() if period.isalpha() else period
        slots.add(f"{day.upper()}{normalized_period}")
    return slots


def _course_time_slots(course: dict) -> set[str]:
    if "time_slots" in course and course["time_slots"] is not None:
        return set(course["time_slots"])
    for key in ("time", "上課時間", "class_time", "schedule"):
        if key in course:
            return parse_time_slots(course.get(key))
    return set()


def _course_label(course: dict) -> str:
    code = course.get("normalized_course_code") or course.get("code") or course.get("raw_course_code") or "UNKNOWN"
    name = course.get("course_name_zh") or course.get("name_zh") or course.get("中文課名") or ""
    return f"{code} {name}".strip()


def has_time_conflict(course_a: dict, course_b: dict) -> bool:
    return bool(_course_time_slots(course_a) & _course_time_slots(course_b))


def check_plan_conflicts(selected_courses: list[dict]) -> dict:
    conflicts: list[dict] = []
    for i, course_a in enumerate(selected_courses):
        for course_b in selected_courses[i + 1 :]:
            overlap = sorted(_course_time_slots(course_a) & _course_time_slots(course_b))
            if overlap:
                conflicts.append(
                    {
                        "course_a": _course_label(course_a),
                        "course_b": _course_label(course_b),
                        "overlap_slots": overlap,
                    }
                )

    return {
        "has_conflict": bool(conflicts),
        "conflicts": conflicts,
        "checked_course_count": len(selected_courses),
    }
