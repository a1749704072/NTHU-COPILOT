from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from schedule_checker import parse_time_slots
except ImportError:  # pragma: no cover - package style import
    from .schedule_checker import parse_time_slots


DAY_TO_WEEKDAY = {
    "M": ("MO", 0, "Monday"),
    "T": ("TU", 1, "Tuesday"),
    "W": ("WE", 2, "Wednesday"),
    "R": ("TH", 3, "Thursday"),
    "F": ("FR", 4, "Friday"),
    "S": ("SA", 5, "Saturday"),
    "U": ("SU", 6, "Sunday"),
}

WEEKDAY_ZH = {
    "MO": "星期一",
    "TU": "星期二",
    "WE": "星期三",
    "TH": "星期四",
    "FR": "星期五",
    "SA": "星期六",
    "SU": "星期日",
}

PERIOD_TIMES = {
    "1": ("08:00", "08:50"),
    "2": ("09:00", "09:50"),
    "3": ("10:10", "11:00"),
    "4": ("11:10", "12:00"),
    "n": ("12:10", "13:00"),
    "5": ("13:20", "14:10"),
    "6": ("14:20", "15:10"),
    "7": ("15:30", "16:20"),
    "8": ("16:30", "17:20"),
    "9": ("17:30", "18:20"),
    "a": ("18:30", "19:20"),
    "b": ("19:30", "20:20"),
    "c": ("20:30", "21:20"),
    "d": ("21:30", "22:20"),
}

PERIOD_ORDER = {period: index for index, period in enumerate(PERIOD_TIMES)}
DEFAULT_SEMESTER_MONDAY = date(2026, 2, 16)


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def _course_value(course: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = course.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _ics_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace(",", "\\,").replace(";", "\\;")


def slot_to_weekday_and_time(slot: str) -> tuple[str, str, str]:
    """Convert a slot like T3 into (weekday_code, start_time, end_time)."""
    if not slot or len(slot) < 2:
        raise ValueError(f"Invalid time slot: {slot!r}")
    day = slot[0].upper()
    period = slot[1:].lower()
    if day not in DAY_TO_WEEKDAY or period not in PERIOD_TIMES:
        raise ValueError(f"Invalid NTHU time slot: {slot!r}")
    start, end = PERIOD_TIMES[period]
    return DAY_TO_WEEKDAY[day][0], start, end


def _group_slots(slots: set[str]) -> list[tuple[str, list[str]]]:
    by_day: dict[str, list[str]] = defaultdict(list)
    for slot in slots:
        if len(slot) >= 2:
            day = slot[0].upper()
            period = slot[1:].lower()
            if day in DAY_TO_WEEKDAY and period in PERIOD_TIMES:
                by_day[day].append(period)

    groups: list[tuple[str, list[str]]] = []
    for day, periods in by_day.items():
        ordered = sorted(set(periods), key=lambda item: PERIOD_ORDER[item])
        current: list[str] = []
        previous_index: int | None = None
        for period in ordered:
            index = PERIOD_ORDER[period]
            if previous_index is None or index == previous_index + 1:
                current.append(period)
            else:
                groups.append((day, current))
                current = [period]
            previous_index = index
        if current:
            groups.append((day, current))
    return sorted(groups, key=lambda item: (DAY_TO_WEEKDAY[item[0]][1], PERIOD_ORDER[item[1][0]]))


def course_to_calendar_events(course: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one recommended course row into weekly calendar event dictionaries."""
    slots = set(course.get("time_slots") or [])
    if not slots:
        slots = parse_time_slots(_course_value(course, "time", "class_time", "schedule"))
    if not slots:
        return []

    code = _course_value(course, "code", "normalized_course_code", "raw_course_code")
    name = _course_value(course, "course_name_zh", "name_zh", "course_name_en")
    teacher = _course_value(course, "teacher", "instructor")
    classroom = _course_value(course, "classroom", "room", "location")
    original_time = _course_value(course, "time", "class_time", "schedule")

    events: list[dict[str, Any]] = []
    for day, periods in _group_slots(slots):
        weekday_code, weekday_offset, _weekday_name = DAY_TO_WEEKDAY[day]
        event_date = DEFAULT_SEMESTER_MONDAY + timedelta(days=weekday_offset)
        first_start, _first_end = PERIOD_TIMES[periods[0]]
        _last_start, last_end = PERIOD_TIMES[periods[-1]]
        events.append(
            {
                "course_code": code,
                "course_name": name,
                "teacher": teacher,
                "classroom": classroom,
                "time": original_time,
                "weekday": weekday_code,
                "periods": periods,
                "start": datetime.combine(event_date, _parse_hhmm(first_start)),
                "end": datetime.combine(event_date, _parse_hhmm(last_end)),
                "summary": f"{code} {name}".strip(),
            }
        )
    return events


def _event_to_ics(event: dict[str, Any]) -> list[str]:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = f"{uuid4()}@nthu-copilot"
    description_parts = [
        f"Course: {event.get('course_code', '')} {event.get('course_name', '')}".strip(),
        f"Teacher: {event.get('teacher', '')}",
        f"Time code: {event.get('time', '')}",
    ]
    description = "\n".join(description_parts)
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{now}",
        f"DTSTART;TZID=Asia/Taipei:{event['start'].strftime('%Y%m%dT%H%M%S')}",
        f"DTEND;TZID=Asia/Taipei:{event['end'].strftime('%Y%m%dT%H%M%S')}",
        "RRULE:FREQ=WEEKLY;COUNT=18",
        f"SUMMARY:{_ics_escape(event.get('summary', ''))}",
        f"DESCRIPTION:{_ics_escape(description)}",
        f"LOCATION:{_ics_escape(event.get('classroom', ''))}",
        "END:VEVENT",
    ]
    return lines


def _write_readable_preview(
    exported_courses: list[dict[str, Any]],
    skipped_courses: list[dict[str, Any]],
    preview_path: Path,
) -> str:
    rows = sorted(
        exported_courses,
        key=lambda row: (
            row.get("start", ""),
            row.get("code", ""),
            row.get("course_name", ""),
        ),
    )
    lines = [
        "# NTHU COPILOT Final Schedule Preview",
        "",
        "這份檔案是給人閱讀的課表預覽；真正匯入 Google Calendar / Apple Calendar 請使用 `.ics` 檔。",
        "",
        "## 每週課表事件",
        "",
        "| 星期 | 時間 | 課號 | 課名 | 授課老師 | 教室 | 節次 |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        weekday = WEEKDAY_ZH.get(str(row.get("weekday", "")), str(row.get("weekday", "")))
        start_time = str(row.get("start", ""))[11:16]
        end_time = str(row.get("end", ""))[11:16]
        lines.append(
            "| {weekday} | {time_range} | {code} | {name} | {teacher} | {classroom} | {periods} |".format(
                weekday=weekday,
                time_range=f"{start_time}-{end_time}",
                code=row.get("code", ""),
                name=row.get("course_name", ""),
                teacher=row.get("teacher", ""),
                classroom=row.get("classroom", ""),
                periods=row.get("periods", ""),
            )
        )

    if skipped_courses:
        lines.extend(["", "## 未匯入行事曆的課程", "", "| 課號 | 課名 | 原因 |", "|---|---|---|"])
        for row in skipped_courses:
            lines.append(
                f"| {row.get('code', '')} | {row.get('course_name', '')} | {row.get('reason', '')} |"
            )

    preview_text = "\n".join(lines) + "\n"
    preview_path.write_text(preview_text, encoding="utf-8")
    return preview_text


def export_schedule_to_ics(
    courses: list[dict[str, Any]],
    output_path: str = "final_schedule.ics",
    include_preview: bool = False,
) -> dict[str, Any]:
    """Export a confirmed schedule to an ICS calendar file.

    By default, this function only creates the .ics file.
    Set include_preview=True only if a Markdown preview is explicitly needed.
    """
    exported_courses: list[dict[str, Any]] = []
    skipped_courses: list[dict[str, Any]] = []
    event_lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//NTHU COPILOT//Course Planning Agent//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    for course in courses or []:
        events = course_to_calendar_events(course)
        code = _course_value(course, "code", "normalized_course_code", "raw_course_code")
        name = _course_value(course, "course_name_zh", "name_zh", "course_name_en")
        if not events:
            skipped_courses.append({"code": code, "course_name": name, "reason": "TBA or no parseable time code"})
            continue
        for event in events:
            event_lines.extend(_event_to_ics(event))
            exported_courses.append(
                {
                    "code": event["course_code"],
                    "course_name": event["course_name"],
                    "teacher": event["teacher"],
                    "classroom": event["classroom"],
                    "weekday": event["weekday"],
                    "periods": "".join(event["periods"]),
                    "start": event["start"].strftime("%Y-%m-%d %H:%M"),
                    "end": event["end"].strftime("%Y-%m-%d %H:%M"),
                }
            )

    event_lines.append("END:VCALENDAR")
    path = Path(output_path)
    path.write_text("\r\n".join(event_lines) + "\r\n", encoding="utf-8")

    # Keep backward-compatible keys so old notebook cells will not crash,
    # but do not generate or display any Markdown preview by default.
    result: dict[str, Any] = {
        "ics_path": str(path.resolve()),
        "exported_courses": exported_courses,
        "skipped_courses": skipped_courses,
        "preview_markdown_path": "",
        "preview_markdown": "",
    }

    preview_path = path.with_name(f"{path.stem}_preview.md")
    if include_preview:
        preview_markdown = _write_readable_preview(exported_courses, skipped_courses, preview_path)
        result["preview_markdown_path"] = str(preview_path.resolve())
        result["preview_markdown"] = preview_markdown
    elif preview_path.exists():
        # Keep the default demo clean: remove stale Markdown preview files.
        preview_path.unlink()

    return result
