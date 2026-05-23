from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PASSING_GRADES = {
    "pass",
    "passed",
    "p",
    "a+",
    "a",
    "a-",
    "b+",
    "b",
    "b-",
    "c+",
    "c",
    "c-",
    "d",
    "及格",
    "通過",
}


def load_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    text = _clean(value)
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _course_record(row: pd.Series | dict) -> dict:
    credits_value = _safe_float(row.get("credits"), 0.0)
    return {
        "code": _clean(row.get("normalized_course_code")),
        "raw_course_code": _clean(row.get("raw_course_code")),
        "course_name_zh": _clean(row.get("course_name_zh")),
        "course_name_en": _clean(row.get("course_name_en")),
        "credits": credits_value,
        "term": _clean(row.get("term")),
        "status": _clean(row.get("status")),
        "grade": _clean(row.get("grade")),
    }


def _grade_is_pass(grade: Any) -> bool:
    text = _clean(grade).lower()
    return text in PASSING_GRADES


def _required_lookup(rules: dict) -> dict[str, dict]:
    return {course["code"]: course for course in rules.get("required_courses", [])}


def _all_alternative_codes(rules: dict) -> set[str]:
    codes: set[str] = set()
    for group in rules.get("alternative_requirements", {}).values():
        for option in group.get("options", []):
            codes.add(option["code"])
    return codes


def _lab_codes(rules: dict) -> set[str]:
    return {course["code"] for course in rules.get("required_lab_electives", {}).get("courses", [])}


def _counted_course_rows(student_df: pd.DataFrame, planning_mode: bool) -> tuple[list[dict], list[dict], list[dict]]:
    completed: list[dict] = []
    in_progress: list[dict] = []
    not_counted: list[dict] = []

    for _, row in student_df.iterrows():
        code = _clean(row.get("normalized_course_code"))
        if not code:
            continue
        record = _course_record(row)
        status = _clean(row.get("status")).lower()
        grade = _clean(row.get("grade")).lower()
        if status == "completed" and _grade_is_pass(grade):
            completed.append(record)
        elif status == "in_progress":
            if planning_mode:
                in_progress.append(record)
            else:
                not_counted.append(record)
        else:
            not_counted.append(record)

    # Prefer completed records over in-progress records for duplicate course codes.
    by_code: dict[str, dict] = {}
    for record in in_progress:
        by_code.setdefault(record["code"], record)
    for record in completed:
        by_code[record["code"]] = record

    counted = list(by_code.values())
    completed_codes = {record["code"] for record in completed}
    effective_in_progress = [
        record for record in in_progress if record["code"] not in completed_codes
    ]
    return counted, completed, effective_in_progress + not_counted


def _sum_credits(records: list[dict]) -> float:
    return float(sum(float(record.get("credits") or 0) for record in records))


def _status_for_code(code: str, completed_codes: set[str], in_progress_codes: set[str], planning_mode: bool) -> str:
    if code in completed_codes:
        return "completed"
    if planning_mode and code in in_progress_codes:
        return "in_progress_counted"
    return "missing"


def _apply_ee2255_electronics_substitution(
    completed_records: list[dict],
    in_progress_records: list[dict],
    counted_by_code: dict[str, dict],
    counted_codes: set[str],
    completed_codes: set[str],
    in_progress_codes: set[str],
    planning_mode: bool,
) -> tuple[dict | None, str]:
    """Treat EE2250 or EE2260 as satisfying EE2255 for the demo audit.

    This creates a virtual required-course record for EE2255 without adding
    extra credits to the total-credit summary, so the graduation checker will
    not double-count Electronics I/II credits.
    """
    if "EE2255" in counted_codes:
        return None, ""

    related_codes = {"EE2250", "EE2260"}
    completed_related = [record for record in completed_records if record.get("code") in related_codes]
    in_progress_related = [record for record in in_progress_records if record.get("code") in related_codes]

    if completed_related:
        source = completed_related[0]
        virtual = {
            "code": "EE2255",
            "raw_course_code": source.get("raw_course_code", ""),
            "course_name_zh": "電子學",
            "course_name_en": "Electronics",
            "credits": 3.0,
            "term": source.get("term", ""),
            "status": "completed",
            "grade": source.get("grade", "pass"),
            "substituted_by": source.get("code", ""),
        }
        counted_by_code["EE2255"] = virtual
        counted_codes.add("EE2255")
        completed_codes.add("EE2255")
        return virtual, "EE2255 電子學已由已修過的 EE2250/EE2260 相關課程暫時視為滿足。"

    if planning_mode and in_progress_related:
        source = in_progress_related[0]
        virtual = {
            "code": "EE2255",
            "raw_course_code": source.get("raw_course_code", ""),
            "course_name_zh": "電子學",
            "course_name_en": "Electronics",
            "credits": 3.0,
            "term": source.get("term", ""),
            "status": "in_progress",
            "grade": "",
            "substituted_by": source.get("code", ""),
        }
        counted_by_code["EE2255"] = virtual
        counted_codes.add("EE2255")
        in_progress_codes.add("EE2255")
        return virtual, "EE2255 電子學已由修課中的 EE2250/EE2260 相關課程暫時視為規劃中滿足。"

    return None, ""


def check_graduation_progress(student_df: pd.DataFrame, rules: dict, planning_mode: bool = True) -> dict:
    counted, completed_records, not_counted_records = _counted_course_rows(student_df, planning_mode)
    completed_codes = {record["code"] for record in completed_records}
    in_progress_records = [
        _course_record(row)
        for _, row in student_df.iterrows()
        if _clean(row.get("status")).lower() == "in_progress"
        and _clean(row.get("normalized_course_code"))
        and _clean(row.get("normalized_course_code")) not in completed_codes
    ]
    in_progress_codes = {record["code"] for record in in_progress_records}
    counted_by_code = {record["code"]: record for record in counted}
    counted_codes = set(counted_by_code)

    electronics_virtual_record, electronics_substitution_message = _apply_ee2255_electronics_substitution(
        completed_records=completed_records,
        in_progress_records=in_progress_records,
        counted_by_code=counted_by_code,
        counted_codes=counted_codes,
        completed_codes=completed_codes,
        in_progress_codes=in_progress_codes,
        planning_mode=planning_mode,
    )

    required_lookup = _required_lookup(rules)
    completed_required: list[dict] = []
    in_progress_required: list[dict] = []
    missing_required: list[dict] = []

    for code, info in required_lookup.items():
        status = _status_for_code(code, completed_codes, in_progress_codes, planning_mode)
        if status == "completed":
            completed_required.append(counted_by_code.get(code, {"code": code, **info}))
        elif status == "in_progress_counted":
            in_progress_required.append(counted_by_code.get(code, {"code": code, **info}))
        else:
            missing_required.append({"code": code, "name_zh": info.get("name_zh", ""), "credits": info.get("credits", 0)})

    alternative_status: dict[str, dict] = {}
    for key, group in rules.get("alternative_requirements", {}).items():
        options = group.get("options", [])
        satisfied_options = [option for option in options if option["code"] in counted_codes]
        selected = satisfied_options[0] if satisfied_options else None
        selected_code = selected["code"] if selected else ""
        status = _status_for_code(selected_code, completed_codes, in_progress_codes, planning_mode) if selected else "missing"
        alternative_status[key] = {
            "label": group.get("label", key),
            "name_zh": group.get("name_zh", ""),
            "satisfied": selected is not None,
            "status": status,
            "selected_code": selected_code,
            "options": options,
            "missing_options": [] if selected else options,
        }

    lab_rule = rules.get("required_lab_electives", {})
    lab_codes = _lab_codes(rules)
    lab_records = [record for record in counted if record["code"] in lab_codes]
    lab_credits = _sum_credits(lab_records)
    lab_course_count = len({record["code"] for record in lab_records})
    lab_status = {
        "satisfied": lab_credits >= lab_rule.get("min_credits", 0)
        and lab_course_count >= lab_rule.get("min_courses", 0),
        "completed_or_counted_courses": lab_records,
        "credits_counted": lab_credits,
        "course_count": lab_course_count,
        "min_credits": lab_rule.get("min_credits", 0),
        "min_courses": lab_rule.get("min_courses", 0),
        "remaining_credits": max(0.0, float(lab_rule.get("min_credits", 0)) - lab_credits),
        "remaining_courses": max(0, int(lab_rule.get("min_courses", 0)) - lab_course_count),
        "eligible_courses": lab_rule.get("courses", []),
    }

    math_science_rule = rules.get("math_science_elective", {})
    math_science_codes = set(math_science_rule.get("course_codes", []))
    math_science_records = [record for record in counted if record["code"] in math_science_codes]
    math_science_credits = _sum_credits(math_science_records)
    math_science_status = {
        "satisfied": math_science_credits >= float(math_science_rule.get("min_credits", 0)),
        "credits_counted": math_science_credits,
        "min_credits": math_science_rule.get("min_credits", 0),
        "remaining_credits": max(0.0, float(math_science_rule.get("min_credits", 0)) - math_science_credits),
        "counted_courses": math_science_records,
        "eligible_course_codes": sorted(math_science_codes),
    }

    required_codes = set(required_lookup)
    alternative_codes = _all_alternative_codes(rules)
    professional_prefixes = tuple(rules.get("professional_electives", {}).get("eligible_prefixes", []))
    professional_records = [
        record
        for record in counted
        if record["code"].startswith(professional_prefixes)
        and record["code"] not in required_codes
        and record["code"] not in alternative_codes
        and record["code"] not in lab_codes
    ]
    professional_credits = _sum_credits(professional_records)
    professional_min = float(rules.get("professional_electives", {}).get("min_credits", 0))

    used_for_known_categories = required_codes | alternative_codes | lab_codes | math_science_codes
    other_records = [record for record in counted if record["code"] not in used_for_known_categories]
    other_credits = _sum_credits(other_records)
    other_min = float(rules.get("other_electives", {}).get("min_credits", 0))

    total_completed_credits = _sum_credits(completed_records)
    total_planning_credits = _sum_credits(counted)
    minimum_total = float(rules.get("minimum_total_graduation_credits", 0))
    credit_summary = {
        "completed_credits_official": total_completed_credits,
        "planning_credits_with_in_progress": total_planning_credits,
        "in_progress_credits_counted": max(0.0, total_planning_credits - total_completed_credits),
        "minimum_total_graduation_credits": minimum_total,
        "remaining_total_credits_in_planning_mode": max(0.0, minimum_total - total_planning_credits),
        "professional_elective_credits_counted_mvp": professional_credits,
        "professional_elective_min_credits": professional_min,
        "professional_elective_remaining_credits_mvp": max(0.0, professional_min - professional_credits),
        "other_elective_credits_counted_mvp": other_credits,
        "other_elective_min_credits": other_min,
        "other_elective_remaining_credits_mvp": max(0.0, other_min - other_credits),
    }

    all_record_codes = {
        _clean(code)
        for code in student_df.get("normalized_course_code", pd.Series(dtype=str)).tolist()
        if _clean(code)
    }
    substitution_issues: list[dict] = []
    warnings: list[str] = list(student_df.attrs.get("warnings", []))
    if electronics_substitution_message:
        warnings.append(electronics_substitution_message)
    for policy in rules.get("substitution_policies", []):
        required_code = policy.get("required_code", "")
        related_codes = set(policy.get("related_codes", []))
        related_present = sorted(all_record_codes & related_codes)
        if related_present and required_code not in counted_codes:
            issue = {
                "status": policy.get("status", "possible_substitution_needs_department_confirmation"),
                "required_code": required_code,
                "related_codes_present": related_present,
                "message": policy.get("message", ""),
            }
            substitution_issues.append(issue)
            warnings.append(policy.get("message", "Possible substitution requires department confirmation."))
        if required_code not in counted_codes:
            warnings.append(
                "EE112 required-course checking uses EE2255 電子學 exactly. EE2250 + EE2260 are not automatically accepted as EE2255 without department confirmation."
            )

    if planning_mode and in_progress_records:
        warnings.append(
            "Courses currently in progress are only counted under the assumption that the student will pass them. "
            "If any in-progress course is not passed, the graduation progress and recommendation need to be recalculated."
        )

    unknown_not_counted = [
        record for record in not_counted_records if record.get("status") not in {"in_progress", "completed"}
    ]
    if unknown_not_counted:
        warnings.append(
            "Courses with missing/unknown status are not counted by the graduation checker: "
            + ", ".join(record["code"] for record in unknown_not_counted if record.get("code"))
        )

    warnings.append("This MVP is not an official graduation audit. The final graduation decision must be confirmed by the department office.")
    warnings = list(dict.fromkeys(warnings))

    return {
        "metadata": {
            "department": rules.get("department", "EE"),
            "admission_year": rules.get("admission_year", "112"),
            "rule_id": rules.get("rule_id", "EE_112"),
            "official_rule_source": rules.get("source", {}).get("official_file", "EE112.pdf"),
            "planning_mode": "assume_in_progress_pass" if planning_mode else "completed_only",
        },
        "completed_required": completed_required,
        "in_progress_counted_with_warning": in_progress_required,
        "missing_required": missing_required,
        "alternative_requirements_status": alternative_status,
        "lab_elective_status": lab_status,
        "math_science_elective_status": math_science_status,
        "credit_summary": credit_summary,
        "counted_courses": counted,
        "completed_courses_official": completed_records,
        "in_progress_courses_counted_in_planning": in_progress_records if planning_mode else [],
        "not_counted_courses": not_counted_records,
        "substitution_issues": substitution_issues,
        "warnings": warnings,
    }
