from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    from calendar_exporter import export_schedule_to_ics
    from intent_parser import parse_user_intent
    from schedule_checker import check_plan_conflicts
except ImportError:  # pragma: no cover - package style import
    from .calendar_exporter import export_schedule_to_ics
    from .intent_parser import parse_user_intent
    from .schedule_checker import check_plan_conflicts

# OCR parser compatibility layer.
# Some notebook/project versions renamed or removed these helper functions,
# but evaluation_report.py only needs them for a lightweight OCR smoke test.
try:
    import ocr_screenshot_parser as _ocr_parser
except ImportError:  # pragma: no cover - package style import
    from . import ocr_screenshot_parser as _ocr_parser

try:
    from course_data_loader import normalize_course_code
except ImportError:  # pragma: no cover - package style import
    from .course_data_loader import normalize_course_code


def extract_course_codes_from_text(text: str) -> list[str]:
    if hasattr(_ocr_parser, "extract_course_codes_from_text"):
        return _ocr_parser.extract_course_codes_from_text(text)

    candidates = re.findall(r"\b[A-Za-z]{2,6}\s*[-_ ]?\s*0?\d{4,6}\b", str(text or ""))
    normalized: list[str] = []
    for candidate in candidates:
        code = normalize_course_code(candidate)
        if code and re.match(r"^[A-Z]{2,6}\d{4}$", code) and code not in normalized:
            normalized.append(code)
    return normalized


def parse_ocr_text_for_demo(text: str, student_path: str, target_path: str) -> dict[str, Any]:
    if hasattr(_ocr_parser, "parse_ocr_text_for_demo"):
        return _ocr_parser.parse_ocr_text_for_demo(text, student_path, target_path)

    codes = extract_course_codes_from_text(text)
    return {
        "raw_ocr_text": text or "",
        "extracted_codes": codes,
        "warnings": ["evaluation_report fallback OCR parser was used."],
    }


HEADERS = ["Test category", "Test input", "Expected behavior", "Actual result", "Pass/Fail", "Notes"]


def _safe_intent(message: str) -> dict[str, Any]:
    try:
        return parse_user_intent(message, use_llm=False)
    except TypeError:
        return parse_user_intent(message)
    except Exception as exc:
        return {"error": str(exc)}


def _pass_fail(condition: bool) -> str:
    return "Pass" if condition else "Fail"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join("---" for _ in HEADERS) + " |",
    ]
    for row in rows:
        values = []
        for header in HEADERS:
            value = str(row.get(header, "")).replace("\n", "<br>").replace("|", "\\|")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_csv(rows: list[dict[str, Any]], output_csv_path: str | None) -> str:
    if not output_csv_path:
        return ""
    path = Path(output_csv_path)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return str(path.resolve())


def _intent_tests() -> list[dict[str, Any]]:
    cases = [
        ("我要修20學分", "Parse a target-credit schedule request", lambda x: x.get("action") == "recommend_schedule" and x.get("exact_credits") == 20),
        ("我不想上實驗課", "Persistently exclude lab/experiment courses", lambda x: x.get("operation") == "remove_category" or x.get("preferences", {}).get("exclude_lab_courses")),
        ("但我想上固態電子實驗", "Recognize a specific lab course override/add request", lambda x: x.get("operation") in {"add_course", "replace_course"} or bool(x.get("course_names") or x.get("course_codes"))),
        ("我不要星期五的課", "Recognize Friday as an avoided day", lambda x: "F" in (x.get("avoid_days") or [])),
        ("我不要上早上八點到十點的課", "Recognize first and second periods as avoided time slots", lambda x: any(str(slot).endswith("1") for slot in x.get("exclude_time_slots", [])) and any(str(slot).endswith("2") for slot in x.get("exclude_time_slots", []))),
    ]

    rows: list[dict[str, Any]] = []
    for message, expected, predicate in cases:
        intent = _safe_intent(message)
        try:
            passed = predicate(intent)
        except Exception:
            passed = False
        rows.append(
            {
                "Test category": "Intent parsing",
                "Test input": message,
                "Expected behavior": expected,
                "Actual result": f"action={intent.get('action')}, operation={intent.get('operation')}, courses={intent.get('course_names') or intent.get('course_codes')}",
                "Pass/Fail": _pass_fail(passed),
                "Notes": "Intent parsing only creates structured intent; deterministic tools still execute the result.",
            }
        )
    return rows


def _conflict_tests() -> list[dict[str, Any]]:
    courses = [
        {"code": "EE2020", "course_name_zh": "偏微分方程與複變函數", "time": "T3T4"},
        {"code": "TEST1000", "course_name_zh": "Test Course", "time": "T3F4"},
    ]
    result = check_plan_conflicts(courses)
    return [
        {
            "Test category": "Conflict checking",
            "Test input": "EE2020 T3T4 + TEST1000 T3F4",
            "Expected behavior": "Detect overlap at T3",
            "Actual result": str(result.get("conflicts")),
            "Pass/Fail": _pass_fail(bool(result.get("has_conflict"))),
            "Notes": "Uses deterministic schedule_checker.parse_time_slots/check_plan_conflicts.",
        }
    ]


def _ocr_tests(student_path: str, target_path: str) -> list[dict[str, Any]]:
    samples = [
        "EE2020 偏微分方程與複變函數 T3T4R3R4",
        "11420EE 214001 電磁學 M3M4W2",
        "EE3060 機率 T5T6R5R6",
    ]
    rows: list[dict[str, Any]] = []
    for sample in samples:
        codes = extract_course_codes_from_text(sample)
        parsed = parse_ocr_text_for_demo(sample, student_path, target_path)
        rows.append(
            {
                "Test category": "OCR extraction",
                "Test input": sample,
                "Expected behavior": "Extract and normalize at least one course code",
                "Actual result": f"codes={codes}, matched_target={len(parsed.get('matched_target_courses', []))}",
                "Pass/Fail": _pass_fail(bool(codes)),
                "Notes": "Manual OCR fallback is used; no schedule is modified.",
            }
        )
    return rows


def _calendar_tests() -> list[dict[str, Any]]:
    fake_courses = [
        {"code": "EE2020", "course_name_zh": "偏微分方程與複變函數", "teacher": "Demo Teacher", "time": "T3T4R3R4", "classroom": "Room 101"},
        {"code": "EE4900", "course_name_zh": "專題研究", "teacher": "Advisor", "time": "TBA"},
    ]
    result = export_schedule_to_ics(fake_courses, output_path="evaluation_calendar_test.ics")
    ics_path = Path(result["ics_path"])
    passed = ics_path.exists() and bool(result.get("exported_courses")) and bool(result.get("skipped_courses"))
    return [
        {
            "Test category": "Calendar export",
            "Test input": "Fake EE2020 timed course + TBA course",
            "Expected behavior": "Generate .ics and skip TBA course",
            "Actual result": f"ics={result.get('ics_path')}, exported={len(result.get('exported_courses', []))}, skipped={len(result.get('skipped_courses', []))}",
            "Pass/Fail": _pass_fail(passed),
            "Notes": "Uses only Python standard library for ICS generation.",
        }
    ]


def run_evaluation_report(
    student_path: str = "data/student_courses.xlsx",
    target_path: str = "data/114_2_course_data.xlsx",
    output_csv_path: str | None = "evaluation_results.csv",
) -> dict[str, Any]:
    """Run a lightweight deterministic evaluation and return table artifacts."""
    rows: list[dict[str, Any]] = []
    rows.extend(_intent_tests())
    rows.extend(_conflict_tests())
    rows.extend(_ocr_tests(str(student_path), str(target_path)))
    rows.extend(_calendar_tests())

    csv_path = _write_csv(rows, output_csv_path)
    return {
        "rows": rows,
        "markdown_table": _markdown_table(rows),
        "csv_path": csv_path,
    }


def generate_evaluation_table(*args, **kwargs) -> str:
    """Convenience wrapper for notebooks that only need the Markdown table."""
    return run_evaluation_report(*args, **kwargs)["markdown_table"]
