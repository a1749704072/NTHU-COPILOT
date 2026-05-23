from __future__ import annotations

import json
import re
import subprocess
from html import escape
from pathlib import Path
from typing import Any

try:
    from course_data_loader import load_student_courses, load_target_courses, normalize_course_code
    from course_recommender import recommend_courses, update_plan
    from course_review_searcher import compare_teachers_for_course, search_course_reviews
    from graduation_rules import check_graduation_progress, load_rules
    from schedule_checker import parse_time_slots
except ImportError:  # pragma: no cover - supports package-style imports
    from .course_data_loader import load_student_courses, load_target_courses, normalize_course_code
    from .course_recommender import recommend_courses, update_plan
    from .course_review_searcher import compare_teachers_for_course, search_course_reviews
    from .graduation_rules import check_graduation_progress, load_rules
    from .schedule_checker import parse_time_slots

try:
    from intent_parser import parse_user_intent
except ImportError:  # pragma: no cover - supports package-style imports
    from .intent_parser import parse_user_intent

try:
    from trace_utils import langsmith_trace
except ImportError:  # pragma: no cover - supports package-style imports
    from .trace_utils import langsmith_trace


SYSTEM_MESSAGE = """
Only use provided tool outputs and course database results.
Do NOT make up course availability, graduation requirements, prerequisites, or timetable results.
If the tool output is uncertain or incomplete, clearly say so.
The final graduation decision must be confirmed by the department office.
""".strip()

AGENT_CODE_VERSION = "course-agent-intent-v3"


def create_autogen_agents(llm_config: dict | bool | None = None):
    """Create HW2-style AutoGen/AG2 agents when the package is available."""
    try:
        from autogen import AssistantAgent, UserProxyAgent
    except ImportError:
        try:
            from ag2 import AssistantAgent, UserProxyAgent
        except ImportError as exc:
            raise ImportError("AutoGen/AG2 is not installed; use CoursePlanningAgent fallback.") from exc

    assistant = AssistantAgent(
        name="course_planning_assistant",
        system_message=SYSTEM_MESSAGE,
        llm_config=llm_config,
    )
    user_proxy = UserProxyAgent(
        name="tool_output_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )
    return assistant, user_proxy


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def explain_with_ollama(question: str, tool_outputs: dict, model: str = "phi4-mini:latest", timeout: int = 90) -> str:
    prompt = f"""
{SYSTEM_MESSAGE}

User question:
{question}

Structured tool outputs:
{_compact_json(tool_outputs)}

Explain the result in concise natural language. Do not add facts that are not present in the tool outputs.
"""
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
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


ALLOWED_INTENT_ACTIONS = {
    "recommend_schedule",
    "modify_schedule",
    "search_course_options",
    "review_course",
    "review_rerank_schedule",
    "check_graduation",
    "help",
    "unknown",
}

ALLOWED_INTENT_OPERATIONS = {
    "",
    "replan",
    "remove_course",
    "remove_category",
    "remove_day",
    "add_course",
    "add_more_ee",
    "add_general_education",
    "replace_course",
    "allow_category",
}

ALLOWED_REVIEW_SOURCES = {"local_cache", "ptt_rag", "ptt", "web"}
ALLOWED_REVIEW_PREFERENCES = {"coolness", "sweetness"}
VALID_DAY_CODES = {"M", "T", "W", "R", "F", "S", "U"}


def _json_object_from_text(text: str) -> dict:
    """Extract one JSON object from an LLM response without trusting extra prose."""
    if not text:
        return {}
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _as_optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_llm_intent(raw_intent: dict, user_message: str) -> dict:
    """Validate the LLM JSON intent. Invalid fields are dropped, not executed."""
    if not isinstance(raw_intent, dict):
        raw_intent = {}
    action = str(raw_intent.get("action") or "unknown").strip()
    operation = str(raw_intent.get("operation") or "").strip()
    review_prefer = str(raw_intent.get("review_prefer") or "").strip().lower()

    days = [
        day.upper()
        for day in _as_string_list(raw_intent.get("avoid_days"))
        if day.upper() in VALID_DAY_CODES
    ]
    sources = [
        source.lower()
        for source in _as_string_list(raw_intent.get("review_sources"))
        if source.lower() in ALLOWED_REVIEW_SOURCES
    ]

    exact_credits = _as_optional_int(raw_intent.get("exact_credits"))
    credit_min = _as_optional_int(raw_intent.get("credit_min"))
    credit_max = _as_optional_int(raw_intent.get("credit_max"))
    if exact_credits is not None:
        credit_min = exact_credits
        credit_max = exact_credits
    if credit_min is not None and credit_max is not None and credit_min > credit_max:
        credit_min, credit_max = credit_max, credit_min

    course_count = _as_optional_int(raw_intent.get("course_count"))
    if course_count is not None:
        course_count = max(1, min(course_count, 10))

    normalized = {
        "action": action if action in ALLOWED_INTENT_ACTIONS else "unknown",
        "operation": operation if operation in ALLOWED_INTENT_OPERATIONS else "",
        "course_name": str(raw_intent.get("course_name") or "").strip(),
        "course_names": _as_string_list(raw_intent.get("course_names")),
        "category": str(raw_intent.get("category") or "").strip().lower(),
        "avoid_days": list(dict.fromkeys(days)),
        "query_time_slots": _as_string_list(raw_intent.get("query_time_slots")),
        "credit_min": credit_min,
        "credit_max": credit_max,
        "exact_credits": exact_credits,
        "course_count": course_count,
        "use_review_search": bool(raw_intent.get("use_review_search", False)),
        "review_sources": list(dict.fromkeys(sources)),
        "review_prefer": review_prefer if review_prefer in ALLOWED_REVIEW_PREFERENCES else "",
        "prefer_theory_ee_courses": bool(raw_intent.get("prefer_theory_ee_courses", False)),
        "wants_remove_only": bool(raw_intent.get("wants_remove_only", False)),
        "wants_replacement": bool(raw_intent.get("wants_replacement", False)),
        "confidence": raw_intent.get("confidence", None),
        "original_message": user_message,
    }

    if normalized["course_name"] and normalized["course_name"] not in normalized["course_names"]:
        normalized["course_names"].insert(0, normalized["course_name"])
    return normalized


def _intent_state_snapshot(last_recommendation: dict | None, last_teacher_review: dict | None) -> dict:
    current_courses = []
    for course in (last_recommendation or {}).get("recommended_courses", []):
        current_courses.append(
            {
                "code": course.get("code", ""),
                "course_name_zh": course.get("course_name_zh", ""),
                "time": course.get("time", ""),
                "credits": course.get("credits", ""),
            }
        )
    return {
        "has_current_schedule": bool(current_courses),
        "current_schedule_courses": current_courses,
        "last_reviewed_course_name": _last_reviewed_course_name(last_teacher_review),
    }


def parse_intent_with_ollama(
    user_message: str,
    state_snapshot: dict | None = None,
    model: str = "phi4-mini:latest",
    timeout: int = 45,
) -> dict:
    """Ask the local model to translate natural language into a constrained JSON intent.

    The returned intent is only used to choose deterministic Python tools. It never
    contains course availability, graduation decisions, timetable results, or review facts.
    """
    state_snapshot = state_snapshot or {}
    prompt = f"""
You are an intent parser for a course planning agent.
Return exactly one JSON object and no prose.

Rules:
- Do not answer the user.
- Do not recommend courses.
- Do not invent course availability, graduation results, teacher reviews, scores, or timetable facts.
- Only translate the user's natural language into the JSON schema.
- Copy course names from the user when present. If the user says "this course" and state has last_reviewed_course_name, use that.
- For "我不要電動機械實驗", use remove_course with course_name "電動機械實驗", not remove_category.
- For "我不想上實驗課", use remove_category with category "lab".
- For "加電動機械實驗", use add_course with course_name "電動機械實驗", not add all labs.
- For "實驗課可以加回來" or "解除不要實驗課", use allow_category with category "lab".
- For "我要多一點電機系課" or "多一門 EE 課", use add_more_ee.
- For "我要加一門電機系理論課" or "加一門 EE 非實驗課", use add_more_ee and set prefer_theory_ee_courses true.
- For "我要通識課" or "加一門通識", use add_general_education.
- For "早上十點到12點有什麼課可以選" or "10-12 有哪些課，PTT 哪個最甜", use search_course_options and set use_review_search true, review_prefer sweetness when the user asks for 甜.
- Reviews are only soft reference; never set remove/replace just because the user asks for reviews.

Allowed action values:
recommend_schedule, modify_schedule, search_course_options, review_course, review_rerank_schedule, check_graduation, help, unknown

Allowed operation values:
"", replan, remove_course, remove_category, remove_day, add_course, add_more_ee, add_general_education, replace_course, allow_category

JSON schema:
{{
  "action": "modify_schedule",
  "operation": "remove_course",
  "course_name": "",
  "course_names": [],
  "category": "",
  "avoid_days": [],
  "query_time_slots": [],
  "credit_min": null,
  "credit_max": null,
  "exact_credits": null,
  "course_count": null,
  "use_review_search": false,
  "review_sources": [],
  "review_prefer": "",
  "prefer_theory_ee_courses": false,
  "wants_remove_only": false,
  "wants_replacement": false,
  "confidence": 0.0
}}

Day codes: M=Monday, T=Tuesday, W=Wednesday, R=Thursday, F=Friday, S=Saturday, U=Sunday.
Review sources: ptt_rag, local_cache, ptt, web.
Review preference: coolness or sweetness.

State:
{_compact_json(state_snapshot)}

User message:
{user_message}
"""
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
    return _normalize_llm_intent(_json_object_from_text(completed.stdout), user_message)


def explain_graduation_result(result: dict) -> str:
    def course_names(items: list[dict]) -> str:
        labels = []
        for item in items:
            code = str(item.get("code") or "").strip()
            name = str(item.get("name_zh") or item.get("course_name_zh") or "").strip()
            if code and name:
                labels.append(f"`{code}` {name}")
            elif code:
                labels.append(f"`{code}`")
            elif name:
                labels.append(name)
        return "<br>".join(labels) if labels else "無"

    def alternative_missing_names(groups: dict) -> str:
        labels = []
        for key, status in groups.items():
            if status.get("satisfied"):
                continue
            options = status.get("missing_options") or status.get("options") or []
            option_labels = []
            for option in options:
                code = str(option.get("code") or "").strip()
                name = str(option.get("name_zh") or option.get("course_name_zh") or status.get("name_zh") or "").strip()
                if code and name:
                    option_labels.append(f"`{code}` {name}")
                elif code:
                    option_labels.append(f"`{code}`")
                elif name:
                    option_labels.append(name)
            if option_labels:
                labels.append(" / ".join(option_labels))
            else:
                fallback_name = status.get("name_zh") or status.get("label") or key
                labels.append(str(fallback_name))
        return "<br>".join(labels) if labels else "無"

    completed = course_names(result.get("completed_required", []))
    in_progress = course_names(result.get("in_progress_counted_with_warning", []))
    missing = course_names(result.get("missing_required", []))
    alt_missing = alternative_missing_names(result.get("alternative_requirements_status", {}))
    lab = result.get("lab_elective_status", {})
    credits = result.get("credit_summary", {})

    lines = [
        "### 畢業進度檢查",
        "",
        "已依照 **EE112 入學年度規則** 進行檢查。",
        "",
        "| 項目 | 結果 |",
        "|---|---|",
        f"| 已完成並正式採計的必修 | {completed} |",
        f"| 規劃模式下暫時計入的修課中必修 | {in_progress} |",
        f"| 尚缺必修 | {missing} |",
        f"| 尚缺替代必修群組 | {alt_missing} |",
        (
            "| 必選實驗進度 | "
            f"{lab.get('credits_counted', 0):g}/{lab.get('min_credits', 0):g} 學分，"
            f"{lab.get('course_count', 0)}/{lab.get('min_courses', 0)} 門 |"
        ),
        (
            "| 學分摘要 | "
            f"正式完成 {credits.get('completed_credits_official', 0):g} 學分；"
            f"若修課中皆通過，規劃採計 {credits.get('planning_credits_with_in_progress', 0):g} 學分 |"
        ),
    ]
    if result.get("warnings"):
        lines.extend(["", "#### 注意事項"])
        translated_warnings = list(dict.fromkeys(_warning_text_in_zh(warning) for warning in result["warnings"]))
        lines.extend(f"- {warning}" for warning in translated_warnings)
    return "\n".join(lines)


def _warning_text_in_zh(warning: object) -> str:
    text = str(warning or "")
    if "EE2255" in text and "EE2250" in text and "EE2260" in text:
        return "EE112 規則要求 `EE2255 電子學`；`EE2250 電子學一` 與 `EE2260 電子學二` 可能和課程調整有關，但 NTHU COPILOT 不能自動認定可抵免，請向系辦確認。"
    if "in progress" in text or "修課中" in text:
        return "修課中的課只是在規劃模式下假設會通過；如果未通過，畢業進度與推薦課表需要重新計算。"
    if "official graduation audit" in text:
        return "NTHU COPILOT 不是正式畢業審查，最後仍須以系辦確認為準。"
    if "Grade column was not present" in text:
        return "原始資料沒有成績欄位；本版本會把標示為 completed 的課暫視為通過。"
    if "missing/unknown status" in text or "unknown status" in text:
        return text.replace("Courses with missing/unknown status are not counted by the graduation checker:", "狀態不明的課不會被畢業檢查採計：")
    return _escape_markdown_cell(text)


DAY_LABELS = [
    ("M", "星期一<br>Monday"),
    ("T", "星期二<br>Tuesday"),
    ("W", "星期三<br>Wednesday"),
    ("R", "星期四<br>Thursday"),
    ("F", "星期五<br>Friday"),
    ("S", "星期六<br>Saturday"),
    ("U", "星期日<br>Sunday"),
]

PERIOD_TIMES = {
    "1": "08:00-08:50",
    "2": "09:00-09:50",
    "3": "10:10-11:00",
    "4": "11:10-12:00",
    "n": "12:10-13:00",
    "5": "13:20-14:10",
    "6": "14:20-15:10",
    "7": "15:30-16:20",
    "8": "16:30-17:20",
    "9": "17:30-18:20",
    "a": "18:30-19:20",
    "b": "19:30-20:20",
    "c": "20:30-21:20",
    "d": "21:30-22:20",
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


def _escape_markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _course_label(course: dict) -> str:
    code = course.get("code", "")
    name = course.get("course_name_zh", "")
    return _escape_markdown_cell(f"{code}<br>{name}".strip())


def _escape_html_cell(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def _course_html_label(course: dict) -> str:
    code = _escape_html_cell(course.get("code", ""))
    name = _escape_html_cell(course.get("course_name_zh", ""))
    time = _escape_html_cell(course.get("time") or "TBA")
    return (
        '<div style="line-height:1.25;color:#f7f9fc;">'
        f'<div style="font-weight:700;color:#ffffff;">{code}</div>'
        f"<div>{name}</div>"
        '<div style="display:inline-block;margin-top:2px;padding:1px 4px;'
        f'background:#202633;color:#b8c2d2;border:1px solid #334055;border-radius:3px;font-size:11px;">{time}</div>'
        "</div>"
    )


def _reason_in_zh(course: dict) -> str:
    reason = str(course.get("recommendation_reason", ""))
    if "user requested specific course" in reason:
        return "依照你明確指定的課名/課號加入，並已檢查 114 第二學期開課、衝堂與學分限制。"
    if "user requested more EE/EECS" in reason:
        if "missing required" in reason:
            return "依照你想多修電機/電資課的要求選入，而且它也能補 EE112 必修缺口。"
        if "required lab elective" in reason:
            return "依照你想多修電機/電資課的要求選入，而且它也能補必選實驗需求。"
        return "依照你想多修電機/電資課的要求，從 114 第二學期的 EE/EECS 課號中選入。"
    if "user requested more general education" in reason:
        return "依照你想修通識課的要求，從 114 第二學期的 GE/GEC 通識課中選入。"
    if "EE-first required/theory course" in reason:
        return "依照初始 EE-first 排課策略選入，並且它也能補 EE112 必修或專業課需求。"
    if "EE-first theory/professional course" in reason:
        return "依照初始 EE-first 排課策略，先補 EE/EECS 理論或專業課。"
    if "EE-first default lab course" in reason:
        return "依照初始 EE-first 排課策略，在未排除實驗課時加入 1 門電機實驗課。"
    if reason.startswith("missing required course"):
        return "補 EE112 必修缺口。"
    if "Probability" in reason:
        return "補機率替代必修需求。"
    if "required lab elective" in reason:
        return "補必選實驗需求，EE112 至少需要 6 學分且至少 3 門。"
    if "language" in reason or "college-language" in reason:
        return "作為本次課表中最多 1 門的非 EE/EECS/CS、非 GE/GEC 其餘選修 filler。"
    if "general education" in reason:
        return "用 GE/GEC 通識課補其餘選修與學分。"
    if "outside-department" in reason:
        return "作為本次課表中最多 1 門的非 EE/EECS/CS、非 GE/GEC 其餘選修 filler。"
    if "physical education" in reason:
        return "體育課可放進課表平衡生活節奏，但這門是 0 學分，不計入畢業總學分。"
    if "professional elective" in reason:
        return "電機/電資專業選修候選，用來補專業選修或學分。"
    return _escape_markdown_cell(reason or "由推薦器依照學分、衝堂與畢業需求選入。")


def _review_note(course: dict) -> str:
    summary = course.get("review_summary") or course.get("ptt_review_summary")
    if not isinstance(summary, dict) or not summary.get("review_count"):
        return ""
    sources = ", ".join(summary.get("sources_used", []))
    parts = [f"樣本 {summary.get('review_count')} 篇"]
    if sources:
        parts.append(f"來源：{sources}")
    if summary.get("avg_coolness") is not None:
        parts.append(f"涼度 {summary.get('avg_coolness')}/5")
    if summary.get("avg_sweetness") is not None:
        parts.append(f"甜度 {summary.get('avg_sweetness')}/5")
    evidence = summary.get("evidence") or []
    if evidence and evidence[0].get("url"):
        parts.append(f"[來源]({evidence[0]['url']})")
    return "<br>".join(parts)


def _build_reason_table(courses: list[dict]) -> str:
    if not courses:
        return "目前沒有可推薦的課程。"
    rows = [
        "| 課號 | 課名 | 授課老師 | 學分 | 時間 | 為什麼建議修 | 評價參考 |",
        "|---|---|---|---:|---|---|---|",
    ]
    for course in courses:
        rows.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown_cell(course.get("code", "")),
                    _escape_markdown_cell(course.get("course_name_zh", "")),
                    _escape_markdown_cell(course.get("teacher", "") or "未提供"),
                    f"{float(course.get('credits') or 0):g}",
                    f"`{_escape_markdown_cell(course.get('time') or 'TBA')}`",
                    _reason_in_zh(course),
                    _review_note(course),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _build_timetable(courses: list[dict]) -> str:
    grid: dict[tuple[str, str], list[str]] = {}
    for course in courses:
        slots = parse_time_slots(course.get("time", ""))
        if not slots:
            continue
        label = _course_html_label(course)
        for slot in slots:
            day = slot[0]
            period = slot[1:]
            grid.setdefault((period, day), [])
            if label not in grid[(period, day)]:
                grid[(period, day)].append(label)

    table_style = (
        "width:100%;table-layout:fixed;border-collapse:collapse;"
        "font-size:13px;line-height:1.25;background:#0f141c;color:#f7f9fc;"
    )
    th_style = (
        "border:1px solid #334055;padding:6px 4px;text-align:center;"
        "vertical-align:middle;background:#202838;color:#f7f9fc;font-weight:700;"
    )
    td_style = (
        "border:1px solid #334055;padding:6px 4px;vertical-align:top;"
        "text-align:left;overflow-wrap:anywhere;word-break:break-word;"
        "background:#111720;color:#f7f9fc;"
    )
    time_style = (
        "border:1px solid #334055;padding:6px 4px;vertical-align:top;"
        "text-align:center;white-space:nowrap;background:#111720;color:#f7f9fc;"
    )
    slot_style = (
        "margin:0 0 4px 0;padding:3px 4px;border-left:3px solid #8ec5ff;"
        "background:#202633;color:#f7f9fc;border-radius:3px;"
    )
    rows = [
        f'<table style="{table_style}">',
        "<colgroup>",
        '<col style="width:6%;">',
        '<col style="width:13%;">',
        *['<col style="width:11.57%;">' for _ in DAY_LABELS],
        "</colgroup>",
        "<thead>",
        "<tr>",
        f'<th style="{th_style}">節次</th>',
        f'<th style="{th_style}">時間</th>',
    ]
    for day, label in DAY_LABELS:
        zh_label = label.split("<br>")[0].replace("星期", "")
        rows.append(f'<th style="{th_style}">{_escape_html_cell(day)}<br>{_escape_html_cell(zh_label)}</th>')
    rows.extend(["</tr>", "</thead>", "<tbody>"])
    for period in PERIOD_ORDER:
        background = "#151b26" if PERIOD_ORDER.index(period) % 2 else "#111720"
        rows.append(f'<tr style="background:{background};">')
        rows.append(f'<td style="{time_style}font-weight:600;">{_escape_html_cell(period)}</td>')
        rows.append(f'<td style="{time_style}">{_escape_html_cell(PERIOD_TIMES[period])}</td>')
        for day, _ in DAY_LABELS:
            entries = grid.get((period, day), [])
            cell_html = "".join(f'<div style="{slot_style}">{entry}</div>' for entry in entries)
            rows.append(f'<td style="{td_style}">{cell_html}</td>')
        rows.append("</tr>")
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _build_time_code_notes() -> str:
    day_rows = ["| 代碼 | 星期 | English |", "|---|---|---|"]
    for code, label in DAY_LABELS:
        zh, en = label.split("<br>")
        day_rows.append(f"| `{code}` | {zh} | {en} |")

    period_rows = ["| 節次 | 時間 |", "|---|---|"]
    for period, time in PERIOD_TIMES.items():
        period_rows.append(f"| `{period}` | {time} |")

    return "\n".join(
        [
            "##### 星期代碼",
            *day_rows,
            "",
            "##### 節次時間",
            *period_rows,
            "",
            "例：`T3T4R3R4` 代表星期二第 3、4 節，和星期四第 3、4 節。",
        ]
    )


def _important_warning_lines(result: dict) -> list[str]:
    warnings = result.get("warnings", [])
    lines: list[str] = []
    if any("in progress" in warning for warning in warnings):
        lines.append("修課中的課只是在規劃模式假設會通過；如果沒有通過，畢業進度和推薦要重算。")
    if any("low-credit-load" in warning for warning in warnings):
        lines.append("目前規劃低於正常最低學分，可能需要低修申請。")
    if any("EE/EECS" in warning and ("could be added or swapped" in warning or "could be added" in warning) for warning in warnings):
        lines.append("因為學分、衝堂或避開星期限制，系統沒有辦法加入你要求數量的 EE/EECS 課。")
    for warning in warnings:
        if str(warning).startswith("指定課程加入失敗："):
            lines.append(str(warning).replace("指定課程加入失敗：", "").replace("11420", "114 第二學期"))
        if str(warning).startswith("加課失敗原因："):
            lines.append(str(warning).replace("加課失敗原因：", "").replace("11420", "114 第二學期"))
        if str(warning).startswith("替換沒有補上新課："):
            lines.append(str(warning).replace("11420", "114 第二學期"))
    if any("general education" in warning.lower() and "could be added or swapped" in warning for warning in warnings):
        lines.append("因為學分、衝堂或避開星期限制，系統沒有辦法加入你要求數量的通識課。")
    if any("不要實驗課" in warning or "非實驗課" in warning for warning in warnings):
        lines.append("因為你前面已經說過不要實驗課，所以後續排課會排除實驗課；只有在你解除限制或明確指定某一門實驗課時才會重新考慮。")
    if any("明確指定這門實驗課" in warning for warning in warnings):
        lines.append("雖然你之前說不要實驗課，但這次你明確指定這門實驗課，所以系統會嘗試加入並重新檢查衝堂與學分。")
    if any("解除不要實驗課" in warning or "重新考慮實驗課" in warning for warning in warnings):
        lines.append("你已經解除不要實驗課的限制，所以後續可以重新考慮實驗課。")
    if any("preferred day" in warning.lower() or "指定的星期" in warning for warning in warnings):
        lines.append("你指定加課星期時，系統會只嘗試加入那天有課的課程；如果沒有符合、衝堂或超過學分上限，會列出原因。")
    if any("official graduation audit" in warning for warning in warnings):
        lines.append("這不是正式畢業審查，最後仍要以系辦確認為準。")
    return list(dict.fromkeys(lines))


def _replacement_reason_in_zh(reason: object, is_removed: bool) -> str:
    text = "" if reason is None else str(reason)
    lowered = text.lower()
    if is_removed:
        if "not a ge/gec course" in lowered:
            return "因為你要求通識課，而這門不是 GE/GEC，也不是受保護的畢業缺口課；直接加課放不進去時，才用它替換。"
        if "no course was removed" in lowered:
            return "沒有移除其他課，因為可以直接新增通識課。"
        if "specific course" in lowered:
            return "沒有移除其他課，因為你是要求新增特定課程。"
        if "not an ee/eecs course" in lowered:
            return "因為你要求多一點電機/電資課，而這門不是 EE/EECS 課；直接加課放不進去時，才用它替換。"
        if "no non-ee course was removed" in lowered:
            return "沒有移除其他課，因為可以直接新增電機/電資課。"
        if "matched" in lowered:
            return "因為你的文字要求替換這門課。"
        if "fallback" in lowered or "no exact" in lowered:
            return "因為系統沒有精準辨識到課名，所以用目前課表中的候選課做替換示範。"
        return text

    base = "補上這門是因為它能讓更新後的課表維持在目標學分範圍內、不衝堂，且不重複已完成或修課中的課。"
    if "user requested more ee" in lowered or "ee/eecs" in lowered:
        return f"它是 114 第二學期的 EE/EECS 課號課程，符合你想多修電機系課的要求。{base}"
    if "general education" in lowered or "ge/gec" in lowered:
        return f"它是 114 第二學期的 GE/GEC 通識課，符合你想修通識課的要求。{base}"
    if "specific course" in lowered or "exactly matched" in lowered:
        return f"它正是你指定要加入的課程。{base}"
    if "language" in lowered or "college-language" in lowered:
        return f"它是本次課表中最多 1 門的非 EE/EECS/CS、非 GE/GEC 其餘選修 filler。{base}"
    if "general education" in lowered:
        return f"它屬於通識課，可作其餘選修候選並平衡課表。{base}"
    if "outside-department" in lowered:
        return f"它是本次課表中最多 1 門的非 EE/EECS/CS、非 GE/GEC 其餘選修 filler。{base}"
    if "physical education" in lowered:
        return f"它是體育課，可平衡課表節奏；但若為 0 學分，不計入畢業總學分。{base}"
    if "professional elective" in lowered:
        return f"它是電機/電資專業選修候選。{base}"
    if "missing required" in lowered:
        return f"它能補必修缺口。{base}"
    if "probability" in lowered:
        return f"它能補機率替代必修需求。{base}"
    if "required lab" in lowered:
        return f"它能補必選實驗需求。{base}"
    return text or base


def _clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _split_teacher_names(raw_teacher: object) -> list[str]:
    text = _clean_text(raw_teacher)
    if not text:
        return []
    names = re.split(r"[,，、/／;；\s]+", text)
    return [name.strip() for name in names if name.strip()]


def _find_target_course_sections(target_df, course_query: str) -> list[dict]:
    query = _clean_text(course_query)
    if target_df is None or not query:
        return []

    query_code = normalize_course_code(query)
    query_lower = query.lower()
    exact_sections: list[dict] = []
    fuzzy_sections: list[dict] = []
    for _, row in target_df.iterrows():
        term = _clean_text(row.get("term"))
        code = _clean_text(row.get("normalized_course_code"))
        raw_code = _clean_text(row.get("raw_course_code"))
        name_zh = _clean_text(row.get("course_name_zh"))
        name_en = _clean_text(row.get("course_name_en"))
        if term and term != "11420":
            continue

        match_code = bool(query_code and query_code in {code, normalize_course_code(raw_code)})
        exact_name = bool(query_lower and query_lower in {name_zh.lower(), name_en.lower()})
        fuzzy_name = bool(
            query_lower
            and (
                query_lower in name_zh.lower()
                or query_lower in name_en.lower()
                or name_zh.lower() in query_lower
            )
        )
        if not match_code and not exact_name and not fuzzy_name:
            continue

        section = {
            "code": code,
            "raw_course_code": raw_code,
            "course_name_zh": name_zh,
            "course_name_en": name_en,
            "teacher": _clean_text(row.get("teacher")),
            "time": _clean_text(row.get("time")),
            "credits": row.get("credits"),
        }
        if match_code or exact_name:
            exact_sections.append(section)
        else:
            fuzzy_sections.append(section)
    return exact_sections or fuzzy_sections


def _target_course_row_to_option(row, reason: str = "符合指定時段的 114 第二學期候選課程") -> dict:
    credits = row.get("credits")
    try:
        credits_value = 0.0 if credits in (None, "") else float(credits)
    except (TypeError, ValueError):
        credits_value = 0.0
    time_text = _clean_text(row.get("time"))
    return {
        "code": _clean_text(row.get("normalized_course_code")),
        "raw_course_code": _clean_text(row.get("raw_course_code")),
        "course_name_zh": _clean_text(row.get("course_name_zh")),
        "course_name_en": _clean_text(row.get("course_name_en")),
        "teacher": _clean_text(row.get("teacher")) or "未提供",
        "credits": credits_value,
        "time": time_text,
        "time_slots": sorted(parse_time_slots(time_text)),
        "classroom": _clean_text(row.get("classroom")),
        "course_level": _clean_text(row.get("course_level")),
        "recommendation_reason": reason,
    }


def _counted_student_course_codes(student_df) -> set[str]:
    if student_df is None:
        return set()
    codes: set[str] = set()
    for _, row in student_df.iterrows():
        status = _clean_text(row.get("status")).lower()
        code = normalize_course_code(row.get("normalized_course_code") or row.get("raw_course_code") or "")
        if code and status in {"completed", "in_progress"}:
            codes.add(code)
    return codes


def _course_option_conflicts(candidate: dict, current_courses: list[dict]) -> list[str]:
    candidate_slots = set(candidate.get("time_slots") or parse_time_slots(candidate.get("time", "")))
    if not candidate_slots:
        return []
    conflicts: list[str] = []
    for course in current_courses:
        other_slots = set(course.get("time_slots") or parse_time_slots(course.get("time", "")))
        overlap = sorted(candidate_slots & other_slots)
        if overlap:
            label = f"{course.get('code', '')} {course.get('course_name_zh', '')}".strip()
            conflicts.append(f"{label}（{', '.join(overlap)}）")
    return conflicts


def _normalize_time_slot_token(slot: object) -> str:
    text = str(slot or "").strip()
    if len(text) < 2:
        return ""
    return text[:1].upper() + text[1:].lower()


def _excluded_time_slots_from_preferences(preferences: dict) -> set[str]:
    return {
        normalized
        for normalized in (_normalize_time_slot_token(slot) for slot in preferences.get("exclude_time_slots", []))
        if normalized
    }


def _normalized_option_time_slots(option: dict) -> set[str]:
    return {
        normalized
        for normalized in (
            _normalize_time_slot_token(slot)
            for slot in (option.get("time_slots") or parse_time_slots(option.get("time", "")))
        )
        if normalized
    }


def _review_score_for_preference(summary: dict, preference: str) -> float | None:
    if not isinstance(summary, dict):
        return None
    key = "avg_sweetness" if preference == "sweetness" else "avg_coolness"
    score = summary.get(key)
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def _short_review_evidence(summary: dict) -> str:
    if not isinstance(summary, dict):
        return ""
    evidence = summary.get("evidence") or []
    if not evidence:
        return ""
    first = evidence[0]
    title = _escape_markdown_cell(first.get("title", ""))
    snippet = _escape_markdown_cell(first.get("snippet") or first.get("short_comment") or "")
    url = first.get("url", "")
    source = _escape_markdown_cell(first.get("source", ""))
    parts = []
    if source:
        parts.append(source)
    if title:
        parts.append(title[:36])
    if snippet:
        parts.append(snippet[:60])
    text = "<br>".join(parts)
    if url:
        text += f"<br>[來源]({url})"
    return text


def _review_option_note(option: dict, preference: str) -> str:
    if option.get("review_lookup_skipped"):
        return "未查詢評價"
    summary = option.get("review_summary")
    if not isinstance(summary, dict) or not summary.get("review_count"):
        return "未找到可靠心得"
    score = _review_score_for_preference(summary, preference)
    score_label = "甜度" if preference == "sweetness" else "涼度"
    pieces = [f"樣本 {summary.get('review_count')} 篇"]
    if score is not None:
        pieces.append(f"{score_label} {score:g}/5")
    evidence_sources = [
        str(item.get("source", "")).strip()
        for item in summary.get("evidence", [])
        if str(item.get("source", "")).strip()
    ]
    sources = ", ".join(dict.fromkeys(evidence_sources or summary.get("sources_used", [])))
    if sources:
        pieces.append(f"來源：{sources}")
    return "<br>".join(pieces)


def explain_course_options_result(result: dict) -> str:
    slots = ", ".join(f"`{slot}`" for slot in result.get("query_time_slots", [])) or "未指定"
    candidates = result.get("candidate_courses", [])
    preference = result.get("review_preference", "sweetness")
    score_label = "甜度" if preference == "sweetness" else "涼度"
    review_search_enabled = bool(result.get("review_search_enabled", True))
    lines = [
        "### 指定時段可選課程",
        "",
        f"查詢時段：{slots}",
        "",
    ]
    if result.get("category_filter") == "general_education":
        lines.extend(["篩選類型：通識 GE/GEC", ""])
    if candidates:
        if not review_search_enabled:
            lines.append("這次沒有要求 PTT/評價排序，所以先用穩定順序列出候選課。")
        elif any(_review_score_for_preference(course.get("review_summary", {}), preference) is not None for course in candidates):
            lines.append(f"排序方式：有 PTT/評價 `{score_label}` 分數的課排前面，樣本數較多者再往前；沒有心得的課放後面，不會亂猜分數。")
        else:
            lines.append(f"目前候選課沒有找到足夠可靠的 PTT/評價{score_label}樣本，所以先用穩定順序列出；系統不會猜哪門課比較{score_label[0]}。")
        lines.extend(
            [
                "",
                "| 排名 | 課號 | 課名 | 授課老師 | 學分 | 時間 | 評價參考 | 參考片段 |",
                "|---:|---|---|---|---:|---|---|---|",
            ]
        )
        for index, course in enumerate(candidates, start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        _escape_markdown_cell(course.get("code", "")),
                        _escape_markdown_cell(course.get("course_name_zh", "")),
                        _escape_markdown_cell(course.get("teacher", "") or "未提供"),
                        f"{float(course.get('credits') or 0):g}",
                        f"`{_escape_markdown_cell(course.get('time') or 'TBA')}`",
                        _review_option_note(course, preference),
                        _short_review_evidence(course.get("review_summary", {})),
                    ]
                )
                + " |"
            )
    else:
        lines.append("目前沒有找到符合這個時段、且可作為候選的課程。")

    rejected = result.get("rejected_courses", [])
    if rejected:
        lines.extend(["", "#### 沒列入候選的原因摘要"])
        for item in rejected[:8]:
            lines.append(
                f"- `{_escape_markdown_cell(item.get('code', ''))}` {_escape_markdown_cell(item.get('course_name_zh', ''))}：{_escape_markdown_cell(item.get('reason', ''))}"
            )

    return "\n".join(lines)


def _build_replacement_summary_lines(replacement_summary: dict) -> list[str]:
    if not replacement_summary:
        return []

    trigger = replacement_summary.get("trigger", "")
    if trigger == "more_ee_courses":
        lines = ["#### 電機系課程調整說明"]
    elif trigger == "more_general_education_courses":
        lines = ["#### 通識課程調整說明"]
    elif trigger == "add_specific_courses":
        lines = ["#### 加課說明"]
    elif str(trigger).startswith("remove_only"):
        lines = ["#### 刪減說明"]
    else:
        lines = ["#### 替換說明"]
    user_request = str(replacement_summary.get("user_request", "")).strip()
    if user_request:
        if trigger == "more_ee_courses":
            label = "你的電機系課程調整要求"
        elif trigger == "more_general_education_courses":
            label = "你的通識課程調整要求"
        elif trigger == "add_specific_courses":
            label = "你的加課要求"
        elif str(trigger).startswith("remove_only"):
            label = "你的刪減要求"
        else:
            label = "你的替換要求"
        lines.append(f"- {label}：`{user_request}`")
    if str(trigger).startswith("remove_only") and replacement_summary.get("remove_only_time_slots"):
        slots = ", ".join(replacement_summary.get("remove_only_time_slots", []))
        lines.append(f"- 這次只刪減符合指定時段的課：`{slots}`。例如 `W1/W2` 代表星期三第 1、2 節。")
    if str(trigger).startswith("remove_only"):
        lines.append("- 這次是純刪減：系統只移除符合你要求的課，重新檢查學分與衝堂，不會自動補課或替換。")
    elif replacement_summary.get("triggered_by_review_search") is False or replacement_summary.get("triggered_by_ptt_review") is False:
        lines.append("- 這次調整不是因為網路評價自動決定；多來源評價只作老師/班別的主觀參考，不會自動把課從課表移除。")
    if trigger == "fallback":
        lines.append("- 因為系統沒有從文字中精準辨識到要換哪一門課，所以才用目前課表中的候選課做替換示範。")

    for pair in replacement_summary.get("replacement_pairs", []):
        removed = f"{pair.get('removed_code')} {pair.get('removed_name_zh')}".strip() or "none"
        added = f"{pair.get('added_code')} {pair.get('added_name_zh')}".strip() or "none"
        if removed == "none" and added != "none":
            lines.append(f"- 我新增 **{added}**，沒有移除原本課表中的課。")
            lines.append(f"  加入原因：{_replacement_reason_in_zh(pair.get('why_added'), is_removed=False)}")
            continue
        elif str(trigger).startswith("remove_only") and added == "none":
            lines.append(f"- 刪減成功：已移除 **{removed}**。")
            continue
        elif added == "none":
            lines.append(f"- 我已經從目前課表移除 **{removed}**，沒有自動補其他課。")
        else:
            lines.append(f"- 我把 **{removed}** 換成 **{added}**。")
        lines.append(f"  換掉原因：{_replacement_reason_in_zh(pair.get('why_removed'), is_removed=True)}")
        lines.append(f"  補上原因：{_replacement_reason_in_zh(pair.get('why_added'), is_removed=False)}")

    policy = str(replacement_summary.get("selection_policy", "")).strip()
    if policy:
        if str(trigger).startswith("remove_only"):
            lines.append("- 刪減後已重新檢查：學分、低修/超修提醒與衝堂狀態。")
        else:
            lines.append("- 已重新檢查：學分、低修/超修提醒與衝堂狀態。")
    lines.append("")
    return lines


def _build_review_plan_review_lines(result: dict) -> list[str]:
    if not (result.get("show_review_block") or result.get("review_block_requested")):
        return []

    policy = result.get("review_search_policy") or result.get("ptt_review_policy", {})
    if not policy.get("enabled"):
        return []

    courses = result.get("recommended_courses", [])
    reviewed_courses = [
        course
        for course in courses
        if isinstance(course.get("review_summary") or course.get("ptt_review_summary"), dict)
        and int((course.get("review_summary") or course.get("ptt_review_summary")).get("review_count") or 0) > 0
    ]
    sources = ", ".join(policy.get("sources", [])) or "available sources"
    lines = ["#### 多來源評價輔助檢查"]
    if reviewed_courses:
        lines.append(
            f"- 我有用多來源評價搜尋當主觀參考；來源包含：**{sources}**。目前這份課表中有 **{len(reviewed_courses)}** 門課找到心得樣本。"
        )
        for course in reviewed_courses[:4]:
            summary = course.get("review_summary") or course.get("ptt_review_summary", {})
            note = _review_note(course).replace("<br>", "；")
            lines.append(f"- `{course.get('code')}` {course.get('course_name_zh')}：{note}")
    else:
        lines.append(f"- 我有嘗試查多來源評價資料（{sources}），但目前這份課表沒有找到足夠心得樣本。")
    lines.extend(
        [
            "- 網路評價只用來輔助老師/班別排序，不會自動換掉必修、實驗或畢業缺口課。",
            "- 如果你看完某門課評價後不想修，可以再用文字要求替換，系統才會重新排一次並說明換掉原因。",
            "",
        ]
    )
    return lines


def explain_recommendation_result(result: dict, include_process_sections: bool = True) -> str:
    courses = result.get("recommended_courses", [])
    semester_policy = result.get("semester_credit_policy", {})
    target_semester = result.get("target_semester", "目標學期")
    total_credits = float(result.get("total_credits", 0) or 0)
    normal_min = semester_policy.get("normal_min_credits", 16)
    normal_max = semester_policy.get("normal_max_credits", 25)
    semester_title = "114 第二學期" if str(target_semester) == "11420" else str(target_semester)
    lines = [
        f"### {semester_title}課表建議",
        "",
    ]
    if include_process_sections:
        lines.extend(_build_review_plan_review_lines(result))
        lines.extend(_build_replacement_summary_lines(result.get("replacement_summary", {})))
    lines.extend(
        [
            "#### 推薦課程與理由",
            _build_reason_table(courses),
            "",
            "#### 每週課表",
            _build_timetable(courses),
            "",
        ]
    )
    if semester_policy:
        low_required = bool(semester_policy.get("low_credit_load_application_required", False))
        overload_required = bool(semester_policy.get("overload_application_required", False))
        lines.extend(
            [
                "#### 學分狀態",
                (
                    f"- 目前總學分：**{total_credits:g}**\n"
                    f"- 正常學分範圍：**{normal_min:g}-{normal_max:g}**\n"
                    f"- 低修：**{low_required}**\n"
                    f"- 超修：**{overload_required}**"
                ),
                "",
            ]
        )
        if low_required or overload_required:
            form_name = "低修申請表" if low_required else "超修申請表"
            lines.extend(
                [
                    "#### 超修/低修流程",
                    "1. 第 1 次選課開始起 ～ 至加退選截止日。",
                    f"2. 下載 **{form_name}**。",
                    "3. 經導師與系主任簽核後，提交至系辦公室設定。",
                    "4. 設定完成後，務必回選課系統確認狀態。",
                    "",
                ]
            )
    important_warnings = _important_warning_lines(result)
    if important_warnings:
        lines.extend(["#### 重要提醒"])
        lines.extend(f"- {warning}" for warning in important_warnings)
        lines.append("")
    lines.extend(["#### 時間代碼說明", _build_time_code_notes()])
    return "\n".join(lines)


def explain_teacher_review_result(result: dict) -> str:
    course_name = result.get("course_name", "")
    preference = "甜度" if result.get("preference") == "sweetness" else "涼度"
    best_teacher = result.get("best_teacher")
    discovered_teachers = result.get("discovered_teacher_names", [])
    lines = [
        f"### {course_name} 多來源評價參考",
        "",
    ]
    if result.get("teacher_discovery_source"):
        if discovered_teachers:
            lines.append(f"我先從 114 第二學期課表資料找到這門課的老師：**{', '.join(discovered_teachers)}**。")
        else:
            lines.append("我先查了 114 第二學期課表資料，但沒有找到可比較的老師名單，所以不會亂猜。")
        lines.append("")
    if best_teacher:
        lines.append(f"以目前搜尋到的主觀心得樣本來看，**{best_teacher}** 的{preference}資料比較有利；這不是客觀排名，也不保證涼、甜或高分。")
    else:
        lines.append("我目前沒有在搜尋範圍內找到足夠可靠的網路心得，所以不能判斷哪位老師比較涼或甜，也不會亂猜。")
    sections = result.get("course_sections", [])
    if sections:
        lines.extend(
            [
                "",
                "#### 114 第二學期開課資料",
                "| 課號 | 課名 | 老師 | 時間 | 學分 |",
                "|---|---|---|---|---:|",
            ]
        )
        for section in sections:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown_cell(section.get("code", "")),
                        _escape_markdown_cell(section.get("course_name_zh", "")),
                        _escape_markdown_cell(section.get("teacher", "")),
                        f"`{_escape_markdown_cell(section.get('time', ''))}`",
                        str(section.get("credits", "")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "| 老師 | 心得篇數 | 平均涼度 | 平均甜度 | 來源 | 參考片段 |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for summary in result.get("teacher_summaries", []):
        evidence = summary.get("evidence") or []
        if evidence and evidence[0].get("url"):
            source_label = evidence[0].get("source", "review source")
            source = f"{_escape_markdown_cell(source_label)}: [{_escape_markdown_cell(evidence[0].get('title', '心得'))}]({evidence[0]['url']})"
        elif evidence:
            source = _escape_markdown_cell(evidence[0].get("source", "review source"))
        else:
            source = "未找到"
        snippet = _escape_markdown_cell(evidence[0].get("snippet", evidence[0].get("short_comment", ""))) if evidence else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown_cell(summary.get("teacher_name", "")),
                    str(summary.get("review_count", 0)),
                    str(summary.get("avg_coolness") if summary.get("avg_coolness") is not None else ""),
                    str(summary.get("avg_sweetness") if summary.get("avg_sweetness") is not None else ""),
                    source,
                    snippet,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("看完評價後，如果你想換掉某門課，請明確說「我不想修某某課，幫我換一門課」；系統不會只因為評價自動換課。")
    lines.extend(
        [
            "",
            "#### 使用限制",
            "- 網路課程心得是學生主觀評價，不是官方資料。",
            "- PTT、Dcard、部落格、論壇或網站資料都可能偏誤、過時、不完整或樣本很小。",
            "- 不能保證老師一定涼、甜或高分，也不能宣稱誰客觀比較好。",
            "- 排課時仍應先看畢業需求、衝堂、學分上下限與官方規則，再把評價當輔助排序。",
        ]
    )
    return "\n".join(lines)


def explain_current_plan_review_result(result: dict) -> str:
    reviewed = result.get("reviewed_courses", [])
    missing = result.get("courses_without_reviews", [])
    source_label = ", ".join(result.get("review_sources", [])) or "PTT"
    lines = [
        "### 目前課表 PTT 心得掃描",
        "",
        f"我依照目前課表逐門查 `{source_label}` 心得；有命中的課會列在下面，沒有命中的不會亂猜。",
        "",
    ]
    if reviewed:
        lines.extend(
            [
                "| 課號 | 課名 | 老師 | 時間 | 心得篇數 | 涼度 | 甜度 | 來源 |",
                "|---|---|---|---|---:|---:|---:|---|",
            ]
        )
        for item in reviewed:
            evidence = item.get("evidence") or []
            first = evidence[0] if evidence else {}
            source = "有找到"
            if first.get("url"):
                source = f"[{_escape_markdown_cell(first.get('title', 'PTT 心得'))}]({first.get('url')})"
            elif first.get("title"):
                source = _escape_markdown_cell(first.get("title", "PTT 心得"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown_cell(item.get("code", "")),
                        _escape_markdown_cell(item.get("course_name_zh", "")),
                        _escape_markdown_cell(item.get("teacher", "") or "未提供"),
                        f"`{_escape_markdown_cell(item.get('time', '') or 'TBA')}`",
                        str(item.get("review_count", 0)),
                        str(item.get("avg_coolness") if item.get("avg_coolness") is not None else ""),
                        str(item.get("avg_sweetness") if item.get("avg_sweetness") is not None else ""),
                        source,
                    ]
                )
                + " |"
            )
    else:
        lines.append("目前課表中的課沒有找到可靠 PTT 心得樣本。")

    if missing:
        missing_text = "、".join(
            f"`{_escape_markdown_cell(item.get('code', ''))}` {_escape_markdown_cell(item.get('course_name_zh', ''))}"
            for item in missing[:10]
        )
        lines.extend(["", "#### 沒找到心得的課", missing_text])
        if len(missing) > 10:
            lines.append(f"另外還有 {len(missing) - 10} 門未列出。")

    lines.extend(
        [
            "",
            "#### 使用限制",
            "- PTT 心得是主觀資料，只能當輔助參考。",
            "- 沒找到心得不代表課不好，也不代表一定很硬或很涼。",
            "- 是否要換課仍要看畢業需求、衝堂、學分上下限與官方規則。",
        ]
    )
    return "\n".join(lines)


DEFAULT_CHAT_PREFERENCES = {
    "target_credit_range": (16, 25),
    "avoid_friday": False,
    "avoid_days": [],
    "strict_avoid_days": True,
    "preferred_days": [],
    "strict_preferred_days": False,
    "prioritize_graduation_requirements": True,
    "balance_with_other_electives": True,
    "include_outside_department": True,
    "include_pe": True,
    "avoid_difficult_courses": True,
    "use_review_search": False,
    "review_sources": ["ptt"],
    "review_prefer": "coolness",
    "review_max_results": 3,
    "exclude_lab_courses": False,
    "prefer_theory_ee_courses": False,
    "explicitly_requested_lab_course_codes": [],
    "composition_policy": "ee_first",
    "initial_ee_theory_count_under_or_equal_20": 4,
    "initial_ee_theory_count_above_20": 5,
    "initial_lab_count": 1,
    "outside_department_fill_last": True,
    "prefer_ge_gec_for_remaining_fillers": True,
    "limit_non_ge_non_technical_fillers": True,
    "max_non_ge_non_technical_fillers": 1,
}

EPHEMERAL_REVIEW_KEYS = {
    "use_review_search",
    "review_sources",
    "review_prefer",
    "ptt_prefer",
    "review_max_results",
    "ptt_max_pages",
    "allow_live_ptt_review_ranking",
    "review_lookup_limit",
    "review_timeout",
    "show_review_block",
    "review_block_requested",
}


def _strip_ephemeral_review_preferences(preferences: dict | None) -> dict:
    clean = dict(preferences or {})
    for key in EPHEMERAL_REVIEW_KEYS:
        clean.pop(key, None)
    clean.setdefault("use_review_search", False)
    clean.setdefault("review_sources", ["ptt"])
    clean.setdefault("review_prefer", "coolness")
    clean.setdefault("review_max_results", 3)
    return clean

CHAT_DAY_KEYWORDS = {
    "M": ("星期一", "週一", "禮拜一", "monday"),
    "T": ("星期二", "週二", "禮拜二", "tuesday"),
    "W": ("星期三", "週三", "禮拜三", "wednesday"),
    "R": ("星期四", "週四", "禮拜四", "thursday"),
    "F": ("星期五", "週五", "禮拜五", "friday"),
    "S": ("星期六", "週六", "禮拜六", "saturday"),
    "U": ("星期日", "星期天", "週日", "週天", "禮拜日", "禮拜天", "sunday"),
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
        if "早上" in compact or "上午" in compact:
            periods.update({"1", "2", "3", "4"})
        elif "中午" in compact:
            periods.add("n")
        elif "下午" in compact:
            periods.update({"5", "6", "7", "8", "9"})
        elif "晚上" in compact or "晚間" in compact or "夜間" in compact:
            periods.update({"a", "b", "c", "d"})
    return periods


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


def _message_mentions_time_slot_constraint(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    return bool(
        _extract_general_time_periods(compact)
        or
        "早八" in compact
        or "八點" in compact
        or "九點" in compact
        or "第1節" in compact
        or "第一節" in compact
        or "第2節" in compact
        or "第二節" in compact
        or re.search(r"(?<!\d)0?[89][:：]00", compact)
        or re.search(r"(?<!\d)8[~-]10(?!\d)", compact)
    )


def _extract_query_periods(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    periods: set[str] = _extract_general_time_periods(text)
    matched_general_range = _has_general_time_range(text)
    matched_range = False
    if any(
        re.search(pattern, compact)
        for pattern in (
            r"早八",
            r"8點到10點?",
            r"8點[~-]10點?",
            r"八點到十點?",
            r"八點[~-]十點?",
            r"八到十",
            r"(?<!\d)0?8[:：]00?[~-](?:10|十)[:：]?00?",
            r"(?<!\d)8[~-]10(?!\d)",
        )
    ):
        periods.update({"1", "2"})
        matched_range = True
    if any(
        re.search(pattern, compact)
        for pattern in (
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
    ):
        periods.update({"3", "4"})
        matched_range = True
    if any(
        re.search(pattern, compact)
        for pattern in (
            r"下午(?:1|一)點到(?:3|三)點?",
            r"下午(?:1|一)點[~-](?:3|三)點?",
            r"下午(?:1|一)到(?:3|三)",
            r"13點到15點?",
            r"13點[~-]15點?",
            r"(?<!\d)13[:：]00?[~-]15[:：]?00?",
            r"(?<!\d)13[~-]15(?!\d)",
        )
    ):
        periods.update({"5", "6"})
        matched_range = True
    if matched_range or matched_general_range:
        for match in re.finditer(r"第\s*([1-9a-dA-D])\s*節", compact):
            period = match.group(1)
            periods.add(period.lower() if period.isalpha() else period)
        return _sort_periods(periods)
    direct_patterns = {
        "1": (r"(?<!\d)0?8[:：]00", r"(?<!\d)8點", r"八點", r"第1節", r"第一節"),
        "2": (r"(?<!\d)0?9[:：]00", r"(?<!\d)9點", r"九點", r"第2節", r"第二節"),
        "3": (r"(?<!\d)10[:：]00", r"(?<!\d)10點", r"十點", r"第3節", r"第三節"),
        "4": (r"(?<!\d)11[:：]00", r"(?<!\d)11點", r"十一點", r"第4節", r"第四節"),
    }
    for period, patterns in direct_patterns.items():
        if any(re.search(pattern, compact) for pattern in patterns):
            periods.add(period)
    for match in re.finditer(r"第\s*([1-9a-dA-D])\s*節", compact):
        period = match.group(1)
        periods.add(period.lower() if period.isalpha() else period)
    return _sort_periods(periods)


def _query_time_slots_from_message(text: str, explicit_slots: list[str] | None = None) -> list[str]:
    if explicit_slots:
        return list(dict.fromkeys(str(slot).upper() if str(slot)[:1].isalpha() else str(slot) for slot in explicit_slots))
    periods = _extract_query_periods(text)
    if not periods:
        return []
    mentioned_days: list[str] = []
    lowered = str(text or "").lower()
    for day_code, keywords in CHAT_DAY_KEYWORDS.items():
        if _contains_any(lowered, keywords):
            mentioned_days.append(day_code)
    days = mentioned_days or [day for day, _ in DAY_LABELS]
    return [f"{day}{period}" for day in days for period in periods]


def _is_course_option_query(message: str, parsed_intent: dict | None = None) -> bool:
    if (parsed_intent or {}).get("action") == "search_course_options":
        return True
    lowered = str(message or "").lower()
    course_words = ("課", "通識", "選修", "必修", "外文", "體育", "ge", "gec", "course")
    return bool(
        _query_time_slots_from_message(message, (parsed_intent or {}).get("query_time_slots"))
        and (
            _contains_any(
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
            or re.search(r"(?:有什麼|有哪些|有沒有|哪幾門|哪一些|什麼).*(?:課|通識|選修|必修|外文|體育|ge|gec)", lowered)
            or re.search(r"(?:課|通識|選修|必修|外文|體育|ge|gec).*(?:可以選|可選|推薦|候選)", lowered)
        )
        and _contains_any(lowered, course_words)
    )


def _is_current_plan_review_query(message: str) -> bool:
    lowered = str(message or "").lower()
    return bool(
        _contains_any(lowered, ("目前課表", "現在課表", "課表裡", "課表中", "current plan"))
        and _contains_any(lowered, ("ptt", "心得", "評價", "評論", "review"))
        and _contains_any(lowered, ("哪幾門", "哪些", "哪一些", "有", "找到", "列出", "掃", "查"))
    )


def _option_selection_index(message: str) -> int | None:
    text = str(message or "").strip().lower()
    chinese_numbers = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    match = re.search(r"第\s*(\d{1,2}|[一二兩三四五六七八九十])\s*(?:個|門|堂|項)?", text)
    if not match:
        match = re.search(r"(?:我要|我想要|選|加|修|上)\s*(\d{1,2}|[一二兩三四五六七八九十])\s*(?:個|門|堂|項)", text)
    if not match:
        return None
    raw = match.group(1)
    value = int(raw) if raw.isdigit() else chinese_numbers.get(raw)
    return value if value and value > 0 else None


def _resolve_course_option_selection(message: str, last_course_options: dict | None) -> dict | None:
    if not isinstance(last_course_options, dict):
        return None
    candidates = last_course_options.get("candidate_courses") or []
    if not candidates:
        return None
    index = _option_selection_index(message)
    if index is not None and 1 <= index <= len(candidates):
        return candidates[index - 1]

    lowered = str(message or "").strip().lower()
    if not lowered:
        return None
    if _contains_any(lowered, ("不要", "不想", "去掉", "移除", "刪掉", "退掉", "remove", "drop")):
        return None
    if not _contains_any(lowered, ("我要", "我想要", "想修", "想上", "要修", "要上", "加", "加入", "選", "修", "上", "add", "take", "choose")):
        return None

    requested_codes = {
        normalize_course_code(match.group(0))
        for match in re.finditer(r"[A-Za-z]{2,6}\s*[0-9]{4}", str(message or ""))
        if normalize_course_code(match.group(0))
    }
    if requested_codes:
        for candidate in candidates:
            code = normalize_course_code(candidate.get("code", ""))
            raw_code = normalize_course_code(candidate.get("raw_course_code", ""))
            if code in requested_codes or raw_code in requested_codes:
                return candidate

    for candidate in candidates:
        for name in (candidate.get("course_name_zh", ""), candidate.get("course_name_en", "")):
            name_text = str(name or "").strip().lower()
            if name_text and name_text in lowered:
                return candidate
    return None


COURSE_COUNT_WORDS = {
    "一": 1,
    "二": 2,
    "兩": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _extract_requested_course_count(message: str, default: int = 1) -> int:
    text = str(message or "")
    digit_match = re.search(r"(\d{1,2})\s*(?:堂|門|个|個)", text)
    if digit_match:
        return max(1, int(digit_match.group(1)))
    word_match = re.search(r"([一二兩俩三四五六七八九十])\s*(?:堂|門|个|個)", text)
    if word_match:
        return COURSE_COUNT_WORDS.get(word_match.group(1), default)
    return default


def _is_reduce_non_ee_request(message: str) -> bool:
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


def _is_more_ee_request(message: str) -> bool:
    if _is_reduce_non_ee_request(message):
        return True
    ee_terms = (
        "電機系",
        "電機係",
        "電機課",
        "電機的課",
        "電機系的課",
        "電機係的課",
        "ee課",
        "ee 課",
        "ee理論",
        "ee 理論",
        "ee非實驗",
        "ee 非實驗",
        "ee課號",
        "ee 課號",
        "eecs",
        "電資",
    )
    more_terms = ("多一點", "多修", "多上", "多加", "加", "增加", "再加", "多", "別的", "其他")
    lowered = str(message or "").lower()
    if any(term in lowered for term in ("不要加", "不想加", "不用加", "不要多")):
        return False
    return any(term.lower() in lowered for term in ee_terms) and any(term.lower() in lowered for term in more_terms)


def _is_general_education_request(message: str) -> bool:
    lowered = str(message or "").lower()
    if any(term in lowered for term in ("不要通識", "不想要通識", "不用通識")):
        return False
    wants = ("我要", "想要", "想修", "想上", "加", "增加", "再加", "多", "多修", "多上", "一堂", "一門", "兩堂", "兩門")
    return "通識" in lowered and any(term in lowered for term in wants)


def _is_add_specific_course_request(message: str, target_df, current_plan: dict | None) -> bool:
    if not current_plan:
        return False
    lowered = str(message or "").lower()
    add_words = ("加", "加入", "加回", "補回", "想加", "我要上", "我要修", "想上", "想修", "再上", "再修")
    remove_words = ("不要", "不想", "不修", "不上", "去掉", "移除", "刪掉", "拿掉", "退掉")
    if not any(word in lowered for word in add_words) or any(word in lowered for word in remove_words):
        return False
    course_name = _infer_course_name_from_message(target_df, message)
    if not _has_target_course_match(target_df, course_name):
        return False
    return not _contains_any(message, ("通識", "電機系", "電機係", "電機課", "ee課", "ee 課", "eecs", "電資"))


def _message_mentions_credit_target(message: str) -> bool:
    return bool(
        re.search(r"\d{1,2}\s*(?:-|~|到|至)?\s*\d{0,2}\s*學?分", message)
        or _contains_any(message, ("學分", "少一點", "多一點", "少修", "多修", "至少", "最多", "不超過", "不要超過"))
    )


def _relax_credit_target_for_followup(message: str, preferences: dict, last_recommendation: dict | None) -> dict:
    """Keep previous exact credit requests as a soft center during follow-up edits."""
    if not last_recommendation or _message_mentions_credit_target(message):
        return preferences

    low, high = preferences.get("target_credit_range", (16, 25))
    is_exact = bool(preferences.get("credit_target_is_exact")) or float(low) == float(high)
    if not is_exact:
        return preferences

    relaxed = dict(preferences)
    center = float(
        relaxed.get("target_credits")
        or relaxed.get("target_credit_soft_center")
        or last_recommendation.get("total_credits", low)
        or low
    )
    tolerance = float(relaxed.get("credit_flex_tolerance", 2))
    relaxed["target_credit_soft_center"] = center
    relaxed["target_credit_range"] = (max(0.0, center - tolerance), center + tolerance)
    relaxed.pop("target_credits", None)
    relaxed["credit_target_is_exact"] = False
    relaxed["credit_target_relaxed_for_followup"] = True
    return relaxed


def _is_remove_only_request(message: str) -> bool:
    remove_words = ("移除", "去掉", "刪掉", "拿掉", "退掉", "不想修", "不要修", "不修", "drop", "remove")
    replace_words = ("換", "替換", "replace", "另一門", "補一門", "重新排")
    return _contains_any(message, remove_words) and not _contains_any(message, replace_words)


def _message_requests_avoid_day(message: str) -> bool:
    wants_avoid_day = _contains_any(message, ("不要", "不想", "不上", "不要上", "不要有", "避開", "avoid"))
    return wants_avoid_day and any(_contains_any(message, keywords) for keywords in CHAT_DAY_KEYWORDS.values())


def _message_requests_remove_category(message: str) -> bool:
    lowered = str(message or "").lower()
    wants_remove = _contains_any(message, ("不要", "不想", "不上", "不修", "去掉", "移除", "刪掉", "拿掉", "退掉", "drop", "remove"))
    return wants_remove and ("實驗" in lowered or "lab" in lowered or "experiment" in lowered)


def _message_allows_lab_courses(message: str, parsed_intent: dict | None = None) -> bool:
    lowered = str(message or "").lower()
    if (parsed_intent or {}).get("operation") == "allow_category" and (parsed_intent or {}).get("category") == "lab":
        return True
    mentions_lab = "實驗" in lowered or "lab" in lowered or "experiment" in lowered
    if not mentions_lab:
        return False
    allow_terms = ("可以", "加回", "考慮", "解除", "允許", "恢復", "不排除", "可以修", "可以上")
    return any(term in lowered for term in allow_terms)


def _message_requests_theory_ee(message: str, parsed_intent: dict | None = None) -> bool:
    lowered = str(message or "").lower()
    if (parsed_intent or {}).get("prefer_theory_ee_courses"):
        return True
    wants_ee = _is_more_ee_request(message)
    theory_terms = ("理論課", "理論", "非實驗", "不要實驗", "不用實驗", "不含實驗", "non-lab", "non lab", "not lab")
    return wants_ee and any(term in lowered for term in theory_terms)


def _course_like_lab(course: dict) -> bool:
    code = normalize_course_code(course.get("code") or course.get("normalized_course_code") or "")
    name_zh = _clean_text(course.get("course_name_zh"))
    name_en = _clean_text(course.get("course_name_en")).lower()
    requirement_code = _clean_text(course.get("requirement_code")).lower()
    return (
        code in LAB_COURSE_CODES
        or "實驗" in name_zh
        or "lab" in name_en
        or "laboratory" in name_en
        or requirement_code == "required_lab_electives"
    )


def _course_like_general_education(course: dict) -> bool:
    code = normalize_course_code(course.get("code") or course.get("normalized_course_code") or "")
    return code.startswith("GE") or code.startswith("GEC")


def _course_option_category_filter(message: str, preferences: dict | None = None) -> str:
    lowered = str(message or "").lower()
    preferences = preferences or {}
    if preferences.get("prefer_general_education_courses") or _contains_any(lowered, ("通識", "ge", "gec", "general education")):
        return "general_education"
    return ""


def _target_course_codes_from_message(target_df, message: str) -> set[str]:
    codes = {normalize_course_code(match.group(0)) for match in re.finditer(r"[A-Za-z]{2,6}\s*[0-9]{4}", message or "")}
    if target_df is None:
        return {code for code in codes if code}
    query = _infer_course_name_from_message(target_df, message)
    for section in _find_target_course_sections(target_df, query):
        code = normalize_course_code(section.get("code", ""))
        if code:
            codes.add(code)
    return {code for code in codes if code}


def _target_lab_course_codes_from_message(target_df, message: str) -> set[str]:
    codes: set[str] = set()
    if target_df is None:
        return codes
    query = _infer_course_name_from_message(target_df, message)
    for section in _find_target_course_sections(target_df, query):
        code = normalize_course_code(section.get("code", ""))
        if code and _course_like_lab(section):
            codes.add(code)
    for code in _target_course_codes_from_message(target_df, message):
        if code in LAB_COURSE_CODES:
            codes.add(code)
    return codes


def _message_excludes_lab_category(
    message: str,
    parsed_intent: dict | None,
    target_df,
    current_plan: dict | None,
) -> bool:
    if (parsed_intent or {}).get("operation") == "remove_category" and (parsed_intent or {}).get("category") == "lab":
        return True
    if not _message_requests_remove_category(message):
        return False
    if _message_mentions_current_course(message, current_plan):
        return False
    query = _infer_course_name_from_message(target_df, message)
    return not _has_target_course_match(target_df, query)


def _is_avoid_day_remove_only_followup(message: str) -> bool:
    if not _message_requests_avoid_day(message):
        return False
    explicit_replan = _contains_any(
        message,
        ("重新排", "重排", "幫我排", "排課", "生成", "推薦", "規劃", "選課", "11420", "學分", "換"),
    )
    return not explicit_replan


def _message_mentions_current_course(message: str, current_plan: dict | None) -> bool:
    lowered = str(message or "").lower()
    for course in (current_plan or {}).get("recommended_courses", []):
        values = [
            course.get("code", ""),
            course.get("raw_course_code", ""),
            course.get("course_name_zh", ""),
            course.get("course_name_en", ""),
        ]
        if any(value and str(value).lower() in lowered for value in values):
            return True
    return False


def _last_reviewed_course_name(last_teacher_review: dict | None) -> str:
    if not isinstance(last_teacher_review, dict):
        return ""
    return str(last_teacher_review.get("course_name") or "").strip()


def _is_course_remove_only_followup(
    message: str,
    current_plan: dict | None,
    last_teacher_review: dict | None,
) -> bool:
    if not current_plan:
        return False
    negative = _contains_any(message, ("不要", "不想要", "不用", "不修", "算了", "去掉", "移除", "刪掉", "拿掉", "退掉"))
    explicit_replan = _contains_any(message, ("重新排", "重排", "幫我排", "推薦", "補一門", "換一門", "替換", "換課"))
    if not negative or explicit_replan:
        return False
    if _message_mentions_current_course(message, current_plan):
        return True
    reviewed_name = _last_reviewed_course_name(last_teacher_review)
    references_reviewed_course = _contains_any(message, ("這堂", "這門", "這個", "它", "他", "那堂", "那門"))
    return bool(reviewed_name and references_reviewed_course)


def _chat_preferences_from_message(message: str, base_preferences: dict | None = None) -> dict:
    preferences = dict(DEFAULT_CHAT_PREFERENCES)
    preferences.update(base_preferences or {})
    lowered = message.lower()

    range_match = re.search(r"(\d{1,2})\s*(?:-|~|到|至)\s*(\d{1,2})\s*學?分?", message)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        if low <= high:
            preferences["target_credit_range"] = (low, high)
            preferences.pop("target_credits", None)
            preferences["credit_target_is_exact"] = False
            preferences.pop("credit_target_relaxed_for_followup", None)
    else:
        at_least_match = re.search(r"(?:至少|最少|不少於|>=)\s*(\d{1,2})\s*學分", message)
        at_most_match = re.search(r"(?:最多|至多|不要超過|不超過|<=)\s*(\d{1,2})\s*學分", message)
        exact_match = re.search(r"(\d{1,2})\s*學分", message)
        current_low, current_high = preferences.get("target_credit_range", (16, 25))
        if at_least_match:
            low = int(at_least_match.group(1))
            preferences["target_credit_range"] = (low, max(float(current_high), float(low)))
            preferences.pop("target_credits", None)
            preferences["credit_target_is_exact"] = False
            preferences.pop("credit_target_relaxed_for_followup", None)
        elif at_most_match:
            high = int(at_most_match.group(1))
            preferences["target_credit_range"] = (min(float(current_low), float(high)), high)
            preferences.pop("target_credits", None)
            preferences["credit_target_is_exact"] = False
            preferences.pop("credit_target_relaxed_for_followup", None)
        elif exact_match:
            credits = int(exact_match.group(1))
            if _contains_any(lowered, ("大概", "約", "左右", "上下", "附近")):
                preferences["target_credit_range"] = (max(0, credits - 1), credits + 1)
                preferences.pop("target_credits", None)
                preferences["credit_target_is_exact"] = False
            else:
                preferences["target_credits"] = credits
                preferences["target_credit_range"] = (credits, credits)
                preferences["target_credit_soft_center"] = credits
                preferences["credit_target_is_exact"] = True
            preferences.pop("credit_target_relaxed_for_followup", None)
    if _contains_any(lowered, ("少一點學分", "學分少一點", "少修一點", "降低學分")):
        low, high = preferences.get("target_credit_range", (16, 25))
        preferences["target_credit_range"] = (float(low), max(float(low), min(float(high), 18.0)))
        preferences.pop("target_credits", None)
        preferences["credit_target_is_exact"] = False
        preferences.pop("credit_target_relaxed_for_followup", None)

    avoid_days = set(preferences.get("avoid_days", []))
    wants_avoid_day = _contains_any(
        lowered,
        ("不要", "不想", "不上", "不要上", "不要有", "避開", "去掉", "移除", "刪掉", "拿掉", "退掉", "avoid", "remove", "drop"),
    )
    time_specific_avoid_day = wants_avoid_day and _message_mentions_time_slot_constraint(lowered)
    if wants_avoid_day and not time_specific_avoid_day:
        for day_code, keywords in CHAT_DAY_KEYWORDS.items():
            if _contains_any(lowered, keywords):
                avoid_days.add(day_code)
                preferences["strict_avoid_days"] = True
    if not time_specific_avoid_day and _contains_any(lowered, ("不要星期五", "不要週五", "避開星期五", "avoid friday")):
        avoid_days.add("F")
        preferences["avoid_friday"] = True
        preferences["strict_avoid_days"] = True
    if avoid_days:
        preferences["avoid_days"] = sorted(avoid_days)
        preferences["avoid_friday"] = "F" in avoid_days or bool(preferences.get("avoid_friday", False))

    preferred_days = set()
    wants_preferred_day = (
        not wants_avoid_day
        and _contains_any(lowered, ("加", "加入", "排", "安排", "排到", "排在", "修", "想修", "想上", "我要修", "我要上"))
        and not _is_course_option_query(message)
        and not _contains_any(lowered, ("可以接受", "解除", "恢復", "不排除", "沒關係"))
    )
    if wants_preferred_day:
        for day_code, keywords in CHAT_DAY_KEYWORDS.items():
            if _contains_any(lowered, keywords):
                preferred_days.add(day_code)
    if preferred_days:
        preferences["preferred_days"] = sorted(preferred_days)
        preferences["strict_preferred_days"] = True
        remaining_avoid_days = set(preferences.get("avoid_days", [])) - preferred_days
        preferences["avoid_days"] = sorted(remaining_avoid_days)
        preferences["avoid_friday"] = "F" in remaining_avoid_days

    review_disabled = _contains_any(lowered, ("不用ptt", "不要ptt", "不看ptt", "不用評價", "不要評價", "不看評價"))
    if review_disabled:
        preferences["use_review_search"] = False
        preferences["use_ptt_reviews"] = False
    if not review_disabled and _contains_any(lowered, ("ptt", "dcard", "評價", "老師", "涼", "甜", "心得", "網路", "網站")):
        preferences["use_review_search"] = True
        sources = ["ptt"]
        if _contains_any(lowered, ("dcard", "網路", "網站", "web", "blog", "論壇")):
            sources.append("web")
        sources.append("local_cache")
        preferences["review_sources"] = list(dict.fromkeys(sources))
        if "ptt" in lowered:
            preferences["allow_live_ptt_review_ranking"] = True
            preferences.setdefault("review_lookup_limit", 3)
            preferences.setdefault("review_timeout", 3)
            preferences.setdefault("review_max_results", 2)
    if _contains_any(lowered, ("甜", "甜度", "高分")):
        preferences["review_prefer"] = "sweetness"
        preferences["ptt_prefer"] = "sweetness"
    if _contains_any(lowered, ("涼", "涼度", "輕鬆")):
        preferences["review_prefer"] = "coolness"
        preferences["ptt_prefer"] = "coolness"
    if _contains_any(lowered, ("不要太硬", "太硬", "輕鬆", "涼一點", "平衡")):
        preferences["avoid_difficult_courses"] = True
        preferences["balance_with_other_electives"] = True
        preferences["include_outside_department"] = True
        preferences["include_pe"] = True
        preferences["user_requested_lighter_schedule"] = True
    if _contains_any(lowered, ("不要電機系課", "不要電機課", "不要ee", "不要 ee", "不想要ee", "少一點專業課", "少一點電機", "少一點ee")):
        preferences["avoid_ee_courses"] = True
        preferences["reduce_professional_courses"] = True
    if _contains_any(lowered, ("通識", "外文", "中文", "外系", "體育")):
        preferences["balance_with_other_electives"] = True
        preferences["include_outside_department"] = True
        preferences["include_pe"] = True
    if not _is_more_ee_request(message):
        preferences.pop("prefer_more_ee_courses", None)
        preferences.pop("requested_ee_course_count", None)
    if _is_more_ee_request(message):
        preferences["prefer_more_ee_courses"] = True
        preferences["requested_ee_course_count"] = _extract_requested_course_count(message, default=2 if _is_reduce_non_ee_request(message) else 1)
        preferences["avoid_difficult_courses"] = False
        if _is_reduce_non_ee_request(message):
            preferences["replace_non_ee_first"] = True
            preferences["reduce_non_ee_courses"] = True
    if _is_general_education_request(message):
        preferences["prefer_general_education_courses"] = True
        preferences["requested_general_education_count"] = _extract_requested_course_count(message)
    else:
        preferences.pop("prefer_general_education_courses", None)
        preferences.pop("requested_general_education_count", None)
    preferences.setdefault("review_max_results", preferences.get("ptt_max_pages", 3))
    preferences.setdefault("ptt_max_pages", preferences.get("review_max_results", 3))
    return preferences


def _clean_course_query_from_message(message: str) -> str:
    query = str(message or "").strip().lower()
    query = re.split(r"[，。！？!?;；\n]", query, maxsplit=1)[0]
    query = re.sub(r"^(但|可是|不過|我|幫我|請|請你|把|將|想要|想|要|可以|能不能|可不可以)\s*", "", query)
    query = re.sub(r"^(加回|補回|加入|加|修|上|想上|想修|我要上|我要修|add)\s*", "", query)
    query = re.sub(r"(這門|這堂|這個|那門|那堂|那個)$", "", query).strip()
    return query.strip(" ：:。,.，")


def _infer_course_name_from_message(target_df, message: str) -> str:
    query_code = normalize_course_code(message)
    if target_df is None:
        return message.strip()

    lowered_message = str(message or "").lower()
    clean_query = _clean_course_query_from_message(message)
    candidates: list[dict] = []
    for _, row in target_df.iterrows():
        term = _clean_text(row.get("term"))
        if term and term != "11420":
            continue
        code = _clean_text(row.get("normalized_course_code"))
        raw_code = _clean_text(row.get("raw_course_code"))
        name_zh = _clean_text(row.get("course_name_zh"))
        name_en = _clean_text(row.get("course_name_en"))
        if query_code and query_code in {code, normalize_course_code(raw_code)} and name_zh:
            return name_zh
        for name in (name_zh, name_en):
            name_lower = name.lower()
            if not name_lower:
                continue
            if name_lower in lowered_message or (clean_query and len(clean_query) >= 2 and clean_query in name_lower):
                candidates.append({"name": name_zh or name, "length": len(name)})
    if candidates:
        return sorted(candidates, key=lambda item: item["length"], reverse=True)[0]["name"]
    return message.strip()


def _has_target_course_match(target_df, course_query: str) -> bool:
    return bool(_find_target_course_sections(target_df, course_query))


def _days_text_from_codes(day_codes: list[str]) -> str:
    labels = {
        "M": "星期一",
        "T": "星期二",
        "W": "星期三",
        "R": "星期四",
        "F": "星期五",
        "S": "星期六",
        "U": "星期日",
    }
    return " ".join(labels.get(day, day) for day in day_codes)


def _message_from_llm_intent(original_message: str, parsed_intent: dict | None) -> str:
    """Turn validated JSON intent into a deterministic parser-friendly message."""
    if not parsed_intent:
        return original_message

    operation = parsed_intent.get("operation", "")
    action = parsed_intent.get("action", "")
    names = parsed_intent.get("course_names") or []
    name_text = " ".join(names).strip()
    count = parsed_intent.get("course_count") or 1
    intent_prefs = parsed_intent.get("preferences") if isinstance(parsed_intent.get("preferences"), dict) else {}

    if operation == "remove_course" and name_text:
        return f"去掉 {name_text}"
    if operation == "add_course":
        codes = parsed_intent.get("course_codes") or []
        if codes and name_text:
            return f"加 {codes[0]} {name_text}"
        if codes:
            return f"加 {codes[0]}"
        if name_text:
            return f"加 {name_text}"
    if operation == "replace_course" and name_text:
        return f"換掉 {name_text}"
    if operation == "remove_category" and parsed_intent.get("category") == "lab":
        return "不要實驗課"
    if operation == "allow_category" and parsed_intent.get("category") == "lab":
        return "實驗課可以加回來"
    if operation == "remove_day" and parsed_intent.get("exclude_time_slots"):
        return original_message
    if operation == "remove_day" and parsed_intent.get("avoid_days"):
        return f"不要 {_days_text_from_codes(parsed_intent['avoid_days'])} 的課"
    if operation == "add_more_ee":
        if intent_prefs.get("replace_non_ee_first") or intent_prefs.get("reduce_non_ee_courses"):
            return f"不要那麼多不是電機系的課，換成 {count} 門電機系課"
        if parsed_intent.get("prefer_theory_ee_courses") or intent_prefs.get("prefer_theory_ee_courses"):
            return f"加 {count} 門電機系理論課，不要實驗"
        return f"加 {count} 門電機系課"
    if operation == "add_general_education":
        return f"加 {count} 門通識課"
    if action == "review_course" and name_text:
        return f"{name_text} 評價"
    if action == "review_rerank_schedule":
        return f"幫我根據網路評價重新排課 {original_message}"
    if action == "check_graduation":
        return "檢查畢業還缺什麼"
    if action == "recommend_schedule":
        return f"幫我排 11420 課表，{original_message}"
    return original_message


def _apply_llm_intent_to_preferences(preferences: dict, parsed_intent: dict | None) -> dict:
    if not parsed_intent:
        return preferences

    updated = dict(preferences)
    intent_prefs = parsed_intent.get("preferences") if isinstance(parsed_intent.get("preferences"), dict) else {}
    exact_credits = parsed_intent.get("exact_credits")
    credit_min = parsed_intent.get("credit_min")
    credit_max = parsed_intent.get("credit_max")
    if exact_credits is not None:
        updated["target_credits"] = int(exact_credits)
        updated["target_credit_range"] = (int(exact_credits), int(exact_credits))
        updated["target_credit_soft_center"] = int(exact_credits)
        updated["credit_target_is_exact"] = True
        updated.pop("credit_target_relaxed_for_followup", None)
    elif credit_min is not None or credit_max is not None:
        current_low, current_high = updated.get("target_credit_range", (16, 25))
        low = float(credit_min if credit_min is not None else current_low)
        high = float(credit_max if credit_max is not None else current_high)
        if low <= high:
            updated["target_credit_range"] = (low, high)
            updated.pop("target_credits", None)
            updated["credit_target_is_exact"] = False
            updated.pop("credit_target_relaxed_for_followup", None)

    if parsed_intent.get("avoid_days") and not parsed_intent.get("exclude_time_slots"):
        avoid_days = set(updated.get("avoid_days", []))
        avoid_days.update(parsed_intent["avoid_days"])
        updated["avoid_days"] = sorted(avoid_days)
        updated["avoid_friday"] = "F" in avoid_days
        updated["strict_avoid_days"] = True
    if parsed_intent.get("exclude_time_slots"):
        existing_slots = list(updated.get("exclude_time_slots", []))
        updated["exclude_time_slots"] = list(dict.fromkeys(existing_slots + list(parsed_intent["exclude_time_slots"])))
    intent_preferred_days = intent_prefs.get("preferred_days") or parsed_intent.get("preferred_days")
    if intent_preferred_days:
        preferred_days = set(updated.get("preferred_days", []))
        preferred_days.update(intent_preferred_days)
        updated["preferred_days"] = sorted(preferred_days)
        updated["strict_preferred_days"] = True
        remaining_avoid_days = set(updated.get("avoid_days", [])) - preferred_days
        updated["avoid_days"] = sorted(remaining_avoid_days)
        updated["avoid_friday"] = "F" in remaining_avoid_days

    if parsed_intent.get("use_review_search") or intent_prefs.get("use_review_search"):
        updated["use_review_search"] = True
    review_sources = parsed_intent.get("review_sources") or intent_prefs.get("review_sources")
    if review_sources:
        updated["review_sources"] = review_sources
    review_prefer = parsed_intent.get("review_prefer") or intent_prefs.get("review_prefer")
    if review_prefer:
        updated["review_prefer"] = review_prefer
        updated["ptt_prefer"] = review_prefer
    if intent_prefs.get("allow_live_ptt_review_ranking"):
        updated["allow_live_ptt_review_ranking"] = True
    for numeric_key in ("review_lookup_limit", "review_timeout", "review_max_results"):
        if intent_prefs.get(numeric_key) is not None:
            updated[numeric_key] = int(intent_prefs[numeric_key])
    if parsed_intent.get("query_time_slots"):
        updated["query_time_slots"] = list(dict.fromkeys(parsed_intent.get("query_time_slots", [])))
    if isinstance(intent_prefs.get("required_section_time_slots_by_code"), dict):
        required_sections = dict(updated.get("required_section_time_slots_by_code", {}))
        for code, slots in intent_prefs["required_section_time_slots_by_code"].items():
            normalized_code = normalize_course_code(code)
            if normalized_code:
                required_sections[normalized_code] = [
                    slot
                    for slot in (_normalize_time_slot_token(item) for item in slots)
                    if slot
                ]
        updated["required_section_time_slots_by_code"] = required_sections

    if intent_prefs.get("avoid_difficult_courses"):
        updated["avoid_difficult_courses"] = True
    if intent_prefs.get("balance_with_other_electives"):
        updated["balance_with_other_electives"] = True
    if intent_prefs.get("initial_lab_count") is not None:
        updated["initial_lab_count"] = max(0, int(intent_prefs["initial_lab_count"]))
    if intent_prefs.get("reduce_lab_courses"):
        updated["reduce_lab_courses"] = True
        updated["initial_lab_count"] = min(int(updated.get("initial_lab_count", 1) or 0), 0)
    if intent_prefs.get("prefer_theory_ee_courses"):
        updated["prefer_theory_ee_courses"] = True
    if intent_prefs.get("prefer_general_education_courses"):
        updated["prefer_general_education_courses"] = True
        updated["requested_general_education_count"] = int(intent_prefs.get("requested_general_education_count") or 1)
    if intent_prefs.get("replace_non_ee_first") or intent_prefs.get("reduce_non_ee_courses"):
        updated["replace_non_ee_first"] = True
        updated["reduce_non_ee_courses"] = True

    if parsed_intent.get("operation") == "add_more_ee":
        updated["prefer_more_ee_courses"] = True
        updated["requested_ee_course_count"] = parsed_intent.get("course_count") or (2 if updated.get("replace_non_ee_first") else 1)
        updated["avoid_difficult_courses"] = False
        updated.pop("prefer_general_education_courses", None)
        updated.pop("requested_general_education_count", None)
    elif parsed_intent.get("operation") == "add_general_education":
        updated["prefer_general_education_courses"] = True
        updated["requested_general_education_count"] = parsed_intent.get("course_count") or 1
        updated.pop("prefer_more_ee_courses", None)
        updated.pop("requested_ee_course_count", None)

    return updated


def _intent_requests_schedule_update(parsed_intent: dict | None) -> bool:
    if not parsed_intent:
        return False
    return parsed_intent.get("action") == "modify_schedule" or parsed_intent.get("operation") in {
        "remove_course",
        "remove_category",
        "remove_day",
        "add_course",
        "add_more_ee",
        "add_general_education",
        "replace_course",
    }


def _update_mode_from_intents(
    parsed_intent: dict | None,
    avoid_day_remove_only_intent: bool,
    category_remove_only_intent: bool,
    course_remove_only_intent: bool,
    add_specific_course_intent: bool,
    more_ee_update_intent: bool,
    general_education_update_intent: bool,
    message: str,
) -> str:
    operation = (parsed_intent or {}).get("operation", "")
    if operation == "add_course" or add_specific_course_intent:
        return "add_specific_course"
    if operation == "add_more_ee" or more_ee_update_intent:
        return "add_or_replace_ee"
    if operation == "add_general_education" or general_education_update_intent:
        return "add_or_replace_general_education"
    if (
        operation in {"remove_course", "remove_category", "remove_day"}
        or avoid_day_remove_only_intent
        or category_remove_only_intent
        or course_remove_only_intent
        or (_is_remove_only_request(message) and not add_specific_course_intent)
    ):
        return "remove_only"
    return "replace"


def _prepend_chat_header(result: dict, action: str, tool_names: list[str]) -> dict:
    header = [
        "### Course Planning",
        "",
        f"**{action.replace('11420', '114 第二學期')}**",
        "",
    ]
    result["agent_explanation"] = "\n".join(header) + result.get("agent_explanation", "")
    return result


def _chat_fallback_response() -> dict:
    explanation = "\n".join(
        [
            "### Agent 聊天入口",
            "",
            "我目前可以用聊天方式幫你控制這幾種流程：",
            "",
            "- `我想知道畢業還缺什麼`",
            "- `幫我排 114 第二學期課表，16-25 學分，不要太硬，可以參考網路評價`",
            "- `偏微分方程與複變函數老師評價怎麼樣`",
            "- `我看完評價後不想修偏微分方程與複變函數，幫我換一門課`",
            "",
            "每一步都會先查工具輸出，不會直接編造畢業規則、開課資料或網路評價結果。",
        ]
    )
    return {"intent": "help", "agent_explanation": explanation}


class CoursePlanningAgent:
    """Small HW2-style agent wrapper that explains deterministic tool outputs."""

    def __init__(
        self,
        student_path: str | Path = "data/student_courses.xlsx",
        target_path: str | Path = "data/114_2_course_data.xlsx",
        rules_path: str | Path = "data/rules/EE_112_rules.json",
        use_llm: bool = False,
        use_llm_intent: bool = False,
        model: str = "phi4-mini:latest",
        intent_provider: str = "ollama",
    ):
        self.student_path = Path(student_path)
        self.target_path = Path(target_path)
        self.rules_path = Path(rules_path)
        self.use_llm = use_llm
        self.use_llm_intent = use_llm_intent
        self.model = model
        self.intent_provider = intent_provider
        self.student_df = None
        self.target_df = None
        self.rules = None
        self.last_graduation_result = None
        self.last_recommendation = None
        self.last_teacher_review = None
        self.last_course_options = None
        self.last_parsed_intent = None
        self.user_constraints = {
            "exclude_course_codes": set(),
            "avoid_categories": set(),
            "exclude_time_slots": set(),
            "exclude_lab_courses": False,
            "prefer_theory_ee_courses": False,
            "explicitly_requested_lab_course_codes": set(),
        }
        self.last_preferences = dict(DEFAULT_CHAT_PREFERENCES)
        self.request_history: list[dict] = []

    @langsmith_trace("agent.load")
    def load(self):
        self.student_df = load_student_courses(str(self.student_path))
        self.target_df = load_target_courses(str(self.target_path))
        self.rules = load_rules(str(self.rules_path))
        return {
            "student_rows": len(self.student_df),
            "target_rows": len(self.target_df),
            "rule_id": self.rules.get("rule_id"),
            "agent_version": AGENT_CODE_VERSION,
        }

    def _record_user_request(self, message: str, parsed_intent: dict | None) -> None:
        text = str(message or "").strip()
        if not text:
            return
        parsed_intent = parsed_intent or {}
        if parsed_intent.get("needs_clarification"):
            return
        if parsed_intent.get("action") in {"help", "confirm_final", "unknown"}:
            return
        if self.request_history and self.request_history[-1].get("message") == text:
            return
        self.request_history.append(
            {
                "message": text,
                "action": parsed_intent.get("action", "unknown"),
                "operation": parsed_intent.get("operation", "none"),
            }
        )
        self.request_history = self.request_history[-12:]

    def _request_history_markdown(self) -> str:
        if not self.request_history:
            return ""
        lines = ["### 聊天記錄", ""]
        for index, item in enumerate(self.request_history, start=1):
            message = str(item.get("message") or "").strip()
            if message:
                lines.append(f"{index}. `{message}`")
        return "\n".join(lines)

    def _constraints_as_preferences(self) -> dict:
        constraints = self.user_constraints
        exclude_codes = set(constraints.get("exclude_course_codes", set()))
        explicit_lab_codes = set(constraints.get("explicitly_requested_lab_course_codes", set()))
        if constraints.get("exclude_lab_courses", False):
            exclude_codes.update(LAB_COURSE_CODES)
        exclude_codes.difference_update(explicit_lab_codes)
        return {
            "exclude_course_codes": sorted(exclude_codes),
            "avoid_categories": sorted(constraints.get("avoid_categories", set())),
            "exclude_time_slots": sorted(constraints.get("exclude_time_slots", set())),
            "exclude_lab_courses": bool(constraints.get("exclude_lab_courses", False)),
            "prefer_theory_ee_courses": bool(constraints.get("prefer_theory_ee_courses", False)),
            "explicitly_requested_lab_course_codes": sorted(
                explicit_lab_codes
            ),
        }

    def _merge_user_constraints_into_preferences(self, preferences: dict) -> dict:
        merged = dict(preferences or {})
        constraint_preferences = self._constraints_as_preferences()
        explicit_lab_codes = {
            normalize_course_code(code)
            for code in list(merged.get("explicitly_requested_lab_course_codes", []))
            + list(constraint_preferences.get("explicitly_requested_lab_course_codes", []))
            if normalize_course_code(code)
        }
        exclude_course_codes = {
            normalize_course_code(code)
            for code in list(merged.get("exclude_course_codes", []))
            + list(constraint_preferences.get("exclude_course_codes", []))
            if normalize_course_code(code)
        }
        exclude_course_codes.difference_update(explicit_lab_codes)
        merged["exclude_course_codes"] = sorted(exclude_course_codes)
        merged["avoid_categories"] = sorted(
            set(merged.get("avoid_categories", [])) | set(constraint_preferences.get("avoid_categories", []))
        )
        exclude_time_slots = {
            _normalize_time_slot_token(slot)
            for slot in list(merged.get("exclude_time_slots", []))
            + list(constraint_preferences.get("exclude_time_slots", []))
            if _normalize_time_slot_token(slot)
        }
        merged["exclude_time_slots"] = sorted(exclude_time_slots)
        merged["exclude_lab_courses"] = bool(
            merged.get("exclude_lab_courses", False) or constraint_preferences["exclude_lab_courses"]
        )
        merged["prefer_theory_ee_courses"] = bool(
            merged.get("prefer_theory_ee_courses", False) or constraint_preferences["prefer_theory_ee_courses"]
        )
        merged["explicitly_requested_lab_course_codes"] = sorted(explicit_lab_codes)
        return merged

    def _sync_constraints_from_preferences(self, preferences: dict | None) -> None:
        if not preferences:
            return
        if preferences.get("exclude_lab_courses"):
            self.user_constraints["exclude_lab_courses"] = True
            self.user_constraints["avoid_categories"].add("lab")
            self.user_constraints["exclude_course_codes"].update(LAB_COURSE_CODES)
        if preferences.get("prefer_theory_ee_courses"):
            self.user_constraints["prefer_theory_ee_courses"] = True
        for slot in preferences.get("exclude_time_slots", []):
            normalized_slot = _normalize_time_slot_token(slot)
            if normalized_slot:
                self.user_constraints["exclude_time_slots"].add(normalized_slot)
        for code in preferences.get("explicitly_requested_lab_course_codes", []):
            normalized = normalize_course_code(code)
            if normalized:
                self.user_constraints["explicitly_requested_lab_course_codes"].add(normalized)

    def _apply_user_constraint_updates(
        self,
        original_message: str,
        effective_message: str,
        parsed_intent: dict | None,
    ) -> list[str]:
        messages: list[str] = []
        parsed_slots = {
            _normalize_time_slot_token(slot)
            for slot in (parsed_intent or {}).get("exclude_time_slots", [])
            if _normalize_time_slot_token(slot)
        }
        if parsed_slots:
            before = set(self.user_constraints.get("exclude_time_slots", set()))
            self.user_constraints["exclude_time_slots"].update(parsed_slots)
            merged_slots = set(self.last_preferences.get("exclude_time_slots", [])) | parsed_slots
            self.last_preferences["exclude_time_slots"] = sorted(merged_slots)
            new_slots = sorted(parsed_slots - before)
            if new_slots:
                messages.append(
                    "我已記住你要避開這些時段："
                    + ", ".join(new_slots)
                    + "；後續新增或推薦課程都會避開它們。"
                )
        allow_lab = _message_allows_lab_courses(original_message, parsed_intent) or _message_allows_lab_courses(effective_message, parsed_intent)
        if allow_lab:
            self.user_constraints["exclude_lab_courses"] = False
            self.user_constraints["prefer_theory_ee_courses"] = False
            self.user_constraints["avoid_categories"].discard("lab")
            self.user_constraints["exclude_course_codes"].difference_update(LAB_COURSE_CODES)
            self.last_preferences["exclude_lab_courses"] = False
            self.last_preferences["prefer_theory_ee_courses"] = False
            self.last_preferences["avoid_categories"] = [
                category for category in self.last_preferences.get("avoid_categories", []) if category != "lab"
            ]
            self.last_preferences["exclude_course_codes"] = [
                code for code in self.last_preferences.get("exclude_course_codes", []) if normalize_course_code(code) not in LAB_COURSE_CODES
            ]
            messages.append("你已經解除不要實驗課的限制，所以後續可以重新考慮實驗課。")

        for text in (effective_message, original_message):
            if _message_excludes_lab_category(text, parsed_intent, self.target_df, self.last_recommendation) and not allow_lab:
                self.user_constraints["exclude_lab_courses"] = True
                self.user_constraints["avoid_categories"].add("lab")
                self.user_constraints["exclude_course_codes"].update(LAB_COURSE_CODES)
                messages.append("我已記住你不要實驗課；後續排課會排除實驗課，除非你明確解除或指定某一門實驗課。")
                break

        if _message_requests_theory_ee(effective_message, parsed_intent) or _message_requests_theory_ee(original_message, parsed_intent):
            self.user_constraints["exclude_lab_courses"] = True
            self.user_constraints["prefer_theory_ee_courses"] = True
            self.user_constraints["avoid_categories"].add("lab")
            self.user_constraints["exclude_course_codes"].update(LAB_COURSE_CODES)
            messages.append("你要求電機系理論/非實驗課；後續 EE/EECS 候選會排除實驗課。")

        add_lab_intent = (
            (parsed_intent or {}).get("operation") == "add_course"
            or _is_add_specific_course_request(effective_message, self.target_df, self.last_recommendation)
        )
        if add_lab_intent:
            lab_codes = _target_lab_course_codes_from_message(self.target_df, effective_message)
            if lab_codes:
                self.user_constraints["explicitly_requested_lab_course_codes"].update(lab_codes)
                if self.user_constraints.get("exclude_lab_courses", False):
                    messages.append("雖然你之前說不要實驗課，但這次你明確指定這門實驗課，所以我會嘗試加入並重新檢查衝堂與學分。")
        return list(dict.fromkeys(messages))

    @langsmith_trace("agent.check_graduation")
    def check_graduation(self, planning_mode: bool = True) -> dict:
        if self.student_df is None or self.rules is None:
            self.load()
        result = check_graduation_progress(self.student_df, self.rules, planning_mode=planning_mode)
        explanation = ""
        if self.use_llm:
            explanation = explain_with_ollama("Check graduation progress.", {"graduation_result": result}, self.model)
        result["agent_explanation"] = explanation or explain_graduation_result(result)
        self.last_graduation_result = result
        return result

    @langsmith_trace("agent.recommend")
    def recommend(self, preferences: dict | None = None, planning_mode: bool = True) -> dict:
        if self.student_df is None or self.target_df is None or self.rules is None:
            self.load()
        merged_preferences = self._merge_user_constraints_into_preferences(preferences or {})
        graduation_result = check_graduation_progress(self.student_df, self.rules, planning_mode=planning_mode)
        result = recommend_courses(self.student_df, self.target_df, graduation_result, merged_preferences)
        result["show_review_block"] = bool(
            merged_preferences.get("show_review_block") or merged_preferences.get("review_block_requested")
        )
        explanation = ""
        if self.use_llm:
            explanation = explain_with_ollama("Recommend target-semester courses.", {"recommendation_result": result}, self.model)
        result["agent_explanation"] = explanation or explain_recommendation_result(result)
        self.last_recommendation = result
        self.last_preferences.update(
            _strip_ephemeral_review_preferences(result.get("updated_preferences") or merged_preferences)
        )
        self._sync_constraints_from_preferences(result.get("updated_preferences") or merged_preferences)
        return result

    @langsmith_trace("agent.replace_course")
    def replace_course(self, current_plan: dict, user_request: str, preferences: dict | None = None, planning_mode: bool = True) -> dict:
        if self.student_df is None or self.target_df is None or self.rules is None:
            self.load()
        merged_preferences = self._merge_user_constraints_into_preferences(preferences or {})
        graduation_result = check_graduation_progress(self.student_df, self.rules, planning_mode=planning_mode)
        result = update_plan(current_plan, user_request, self.target_df, graduation_result, merged_preferences)
        result["show_review_block"] = bool(
            merged_preferences.get("show_review_block") or merged_preferences.get("review_block_requested")
        )
        explanation = ""
        if self.use_llm:
            explanation = explain_with_ollama("Update a recommended course plan.", {"updated_plan": result}, self.model)
        result["agent_explanation"] = explanation or explain_recommendation_result(result)
        self.last_recommendation = result
        self.last_preferences.update(
            _strip_ephemeral_review_preferences(result.get("updated_preferences") or merged_preferences)
        )
        self._sync_constraints_from_preferences(result.get("updated_preferences") or merged_preferences)
        return result

    @langsmith_trace("agent.compare_teacher_reviews")
    def compare_teacher_reviews(
        self,
        course_name: str,
        teacher_names: list[str],
        preference: str = "coolness",
        sources: list[str] | None = None,
        max_pages: int = 5,
    ) -> dict:
        result = compare_teachers_for_course(
            course_name=course_name,
            teacher_names=teacher_names,
            preference=preference,
            sources=sources,
            max_pages=max_pages,
        )
        result["agent_explanation"] = explain_teacher_review_result(result)
        self.last_teacher_review = result
        return result

    @langsmith_trace("agent.compare_course_teachers")
    def compare_course_teachers(
        self,
        course_name: str,
        preference: str = "coolness",
        sources: list[str] | None = None,
        max_pages: int = 5,
    ) -> dict:
        if self.target_df is None:
            self.load()

        sections = _find_target_course_sections(self.target_df, course_name)
        teacher_names: list[str] = []
        for section in sections:
            for teacher in _split_teacher_names(section.get("teacher")):
                if teacher not in teacher_names:
                    teacher_names.append(teacher)

        if not teacher_names:
            result = {
                "course_name": course_name,
                "preference": preference,
                "best_teacher": None,
                "teacher_summaries": [],
                "course_sections": sections,
                "discovered_teacher_names": [],
                "teacher_discovery_source": "114_2_course_data.xlsx",
                "warnings": [
                    "No teacher names were found for the requested course in target semester 11420.",
                    "The system must not guess online reviews without course database matches.",
                ],
            }
        else:
            result = compare_teachers_for_course(
                course_name=course_name,
                teacher_names=teacher_names,
                preference=preference,
                sources=sources,
                max_pages=max_pages,
            )
            result["course_sections"] = sections
            result["discovered_teacher_names"] = teacher_names
            result["teacher_discovery_source"] = "114_2_course_data.xlsx"
            result["warnings"] = list(
                dict.fromkeys(
                    result.get("warnings", [])
                    + ["Teacher names were discovered from 11420 course data before querying multi-source reviews."]
                )
            )

        result["agent_explanation"] = explain_teacher_review_result(result)
        self.last_teacher_review = result
        return result

    @langsmith_trace("agent.scan_current_plan_reviews")
    def scan_current_plan_reviews(self, preferences: dict | None = None) -> dict:
        if self.student_df is None or self.target_df is None or self.rules is None:
            self.load()

        if not self.last_recommendation:
            result = {
                "intent": "current_plan_reviews",
                "reviewed_courses": [],
                "courses_without_reviews": [],
                "review_sources": ["ptt"],
                "warnings": ["No current recommendation exists yet. Generate a schedule before scanning current-plan reviews."],
                "agent_explanation": "### 目前課表 PTT 心得掃描\n\n目前還沒有課表可以掃描。請先排一份課表，再問「目前課表裡哪幾門有 PTT 心得」。",
            }
            return result

        preferences = preferences or {}
        sources = list(preferences.get("review_sources") or ["ptt"])
        timeout = max(1, int(preferences.get("review_timeout") or 3))
        max_results = max(1, int(preferences.get("review_max_results") or 2))
        courses = list((self.last_recommendation or {}).get("recommended_courses", []))
        reviewed_courses: list[dict] = []
        courses_without_reviews: list[dict] = []

        for course in courses:
            course_name = str(course.get("course_name_zh") or course.get("course_name_en") or "").strip()
            if not course_name:
                continue
            teacher_names = _split_teacher_names(course.get("teacher"))
            teacher_name = ""
            if len(teacher_names) == 1 and teacher_names[0] not in {"指導教授", "未提供", "TBA"}:
                teacher_name = teacher_names[0]

            summary = search_course_reviews(
                course_name=course_name,
                teacher_name=teacher_name,
                sources=sources,
                max_results=max_results,
                timeout=timeout,
            )
            if teacher_name and int(summary.get("review_count") or 0) == 0:
                summary = search_course_reviews(
                    course_name=course_name,
                    teacher_name="",
                    sources=sources,
                    max_results=max_results,
                    timeout=timeout,
                )

            item = {
                "code": course.get("code", ""),
                "raw_course_code": course.get("raw_course_code", ""),
                "course_name_zh": course.get("course_name_zh", ""),
                "teacher": course.get("teacher", ""),
                "time": course.get("time", ""),
                "credits": course.get("credits", 0),
                "review_count": int(summary.get("review_count") or 0),
                "avg_coolness": summary.get("avg_coolness"),
                "avg_sweetness": summary.get("avg_sweetness"),
                "evidence": summary.get("evidence", []),
                "review_summary": summary,
            }
            if item["review_count"] > 0:
                reviewed_courses.append(item)
            else:
                courses_without_reviews.append(item)

        reviewed_courses.sort(
            key=lambda item: (
                -int(item.get("review_count") or 0),
                str(item.get("code", "")),
            )
        )
        result = {
            "intent": "current_plan_reviews",
            "target_semester": self.last_recommendation.get("target_semester", "11420"),
            "review_sources": sources,
            "reviewed_courses": reviewed_courses,
            "courses_without_reviews": courses_without_reviews,
            "reviewed_count": len(reviewed_courses),
            "total_courses_checked": len(reviewed_courses) + len(courses_without_reviews),
            "warnings": [
                "Current-plan review scan uses the latest recommendation stored in session state.",
                "PTT reviews are subjective soft references and must not override timetable conflicts, official rules, or course availability.",
            ],
        }
        result["agent_explanation"] = explain_current_plan_review_result(result)
        return result

    @langsmith_trace("agent.search_course_options")
    def search_course_options(
        self,
        user_request: str,
        preferences: dict | None = None,
    ) -> dict:
        if self.student_df is None or self.target_df is None or self.rules is None:
            self.load()

        preferences = self._merge_user_constraints_into_preferences(preferences or {})
        self.last_preferences.update(_strip_ephemeral_review_preferences(preferences))
        query_slots = {
            _normalize_time_slot_token(slot)
            for slot in _query_time_slots_from_message(user_request, preferences.get("query_time_slots"))
            if _normalize_time_slot_token(slot)
        }
        warnings = [
            "Course option search uses only target semester 11420 course data.",
            "This result is only a candidate list. No course is added until the user explicitly asks to add one.",
            "Online course reviews are subjective soft references and must not override timetable conflicts, official rules, or course availability.",
        ]
        category_filter = _course_option_category_filter(user_request, preferences)
        if category_filter == "general_education":
            warnings.append("This option search is filtered to GE/GEC general education courses because the request mentions 通識.")
        if not query_slots:
            result = {
                "intent": "search_course_options",
                "target_semester": "11420",
                "query_time_slots": [],
                "category_filter": category_filter,
                "candidate_courses": [],
                "rejected_courses": [],
                "review_preference": "sweetness",
                "warnings": warnings + ["No recognizable time range was found in the request."],
            }
            result["agent_explanation"] = explain_course_options_result(result)
            self.last_course_options = result
            return result

        counted_codes = _counted_student_course_codes(self.student_df)
        current_courses = list((self.last_recommendation or {}).get("recommended_courses", []))
        current_codes = {normalize_course_code(course.get("code", "")) for course in current_courses if course.get("code")}
        excluded_codes = {normalize_course_code(code) for code in preferences.get("exclude_course_codes", []) if normalize_course_code(code)}
        excluded_time_slots = _excluded_time_slots_from_preferences(preferences)
        explicit_lab_codes = {
            normalize_course_code(code)
            for code in preferences.get("explicitly_requested_lab_course_codes", [])
            if normalize_course_code(code)
        }
        candidates: list[dict] = []
        rejected: list[dict] = []
        seen_sections: set[tuple[str, str, str]] = set()

        for _, row in self.target_df.iterrows():
            term = _clean_text(row.get("term"))
            raw_code = _clean_text(row.get("raw_course_code"))
            if term and term != "11420":
                continue
            if not term and raw_code and not raw_code.startswith("11420"):
                continue
            option = _target_course_row_to_option(row)
            code = normalize_course_code(option.get("code", ""))
            if not code:
                continue
            option["code"] = code
            section_key = (code, option.get("teacher", ""), option.get("time", ""))
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            option_slots = _normalized_option_time_slots(option)
            if not (option_slots & query_slots):
                continue
            if category_filter == "general_education" and not _course_like_general_education(option):
                continue

            reason = ""
            excluded_overlap = sorted(option_slots & excluded_time_slots)
            if code in counted_codes:
                reason = "已完成或修課中，不列入可新增候選。"
            elif code in current_codes:
                reason = "已經在目前課表中。"
            elif code in excluded_codes and code not in explicit_lab_codes:
                reason = "符合目前持續性排除限制。"
            elif excluded_overlap:
                reason = "落在你要求排除的時段：" + ", ".join(excluded_overlap)
            elif preferences.get("exclude_lab_courses") and code not in explicit_lab_codes and (
                code in LAB_COURSE_CODES or "實驗" in option.get("course_name_zh", "")
            ):
                reason = "你前面設定不要實驗課，所以先排除。"
            else:
                conflicts = _course_option_conflicts(option, current_courses)
                if conflicts:
                    reason = "與目前課表衝堂：" + "；".join(conflicts[:3])

            if reason:
                rejected.append({"code": code, "course_name_zh": option.get("course_name_zh", ""), "reason": reason})
                continue
            candidates.append(option)

        candidates.sort(key=lambda item: (item.get("code", ""), item.get("course_name_zh", ""), item.get("teacher", ""), item.get("time", "")))

        review_preference = "sweetness"
        lower_request = str(user_request or "").lower()
        if "涼" in lower_request or preferences.get("review_prefer") == "coolness":
            review_preference = "coolness"
        elif preferences.get("review_prefer") in {"sweetness", "coolness"}:
            review_preference = preferences["review_prefer"]

        sources = list(preferences.get("review_sources") or [])
        if not sources:
            sources = ["ptt_rag", "ptt"]
        if "ptt" in lower_request and "ptt" not in sources:
            sources.append("ptt")
        if "ptt" in sources and "ptt_rag" not in sources:
            sources.insert(0, "ptt_rag")
        max_display = max(1, int(preferences.get("course_option_limit", 12)))
        max_review_lookup = max(0, int(preferences.get("course_option_review_limit", 0)))
        review_timeout = max(1, int(preferences.get("review_timeout", 4)))
        review_max_results = max(1, int(preferences.get("review_max_results", 2)))
        review_search_enabled = bool(preferences.get("use_review_search")) or _contains_any(
            lower_request,
            ("ptt", "dcard", "網路", "網站", "評語", "評價", "心得", "甜", "甜度", "高分", "涼", "涼度", "輕鬆", "好過"),
        )

        if review_search_enabled:
            retrieval_sources = [source for source in sources if source in {"ptt_rag", "local_cache"}] or ["ptt_rag"]
            for option in candidates:
                summary = search_course_reviews(
                    course_name=option.get("course_name_zh", ""),
                    teacher_name=option.get("teacher", ""),
                    sources=retrieval_sources,
                    max_results=review_max_results,
                    timeout=review_timeout,
                )
                option["review_summary"] = summary
                option["ptt_review_summary"] = summary
                option["review_score"] = _review_score_for_preference(summary, review_preference)
                option["review_count"] = int(summary.get("review_count") or 0)
        else:
            for option in candidates:
                summary = {
                    "course_name": option.get("course_name_zh", ""),
                    "teacher_name": option.get("teacher", ""),
                    "sources_used": [],
                    "review_count": 0,
                    "avg_coolness": None,
                    "avg_sweetness": None,
                    "evidence": [],
                    "warnings": ["Review lookup was skipped because the request did not ask for reviews."],
                }
                option["review_summary"] = summary
                option["ptt_review_summary"] = summary
                option["review_lookup_skipped"] = True
                option["review_score"] = None
                option["review_count"] = 0

        live_sources = [source for source in sources if source not in {"local_cache", "ptt_rag"}]
        live_lookup_count = 0
        if review_search_enabled and live_sources and max_review_lookup > 0:
            candidates.sort(
                key=lambda item: (
                    item.get("review_score") is None,
                    -(item.get("review_score") if item.get("review_score") is not None else -1),
                    -int(item.get("review_count") or 0),
                    item.get("code", ""),
                    item.get("course_name_zh", ""),
                )
            )
            for option in candidates:
                if live_lookup_count >= max_review_lookup:
                    break
                if option.get("review_count"):
                    continue
                summary = search_course_reviews(
                    course_name=option.get("course_name_zh", ""),
                    teacher_name=option.get("teacher", ""),
                    sources=live_sources,
                    max_results=review_max_results,
                    timeout=review_timeout,
                )
                live_lookup_count += 1
                if int(summary.get("review_count") or 0) > 0:
                    option["review_summary"] = summary
                    option["ptt_review_summary"] = summary
                    option["review_score"] = _review_score_for_preference(summary, review_preference)
                    option["review_count"] = int(summary.get("review_count") or 0)

        candidates.sort(
            key=lambda item: (
                item.get("review_score") is None,
                -(item.get("review_score") if item.get("review_score") is not None else -1),
                -int(item.get("review_count") or 0),
                item.get("code", ""),
                item.get("course_name_zh", ""),
            )
        )
        if not review_search_enabled:
            warnings.append("本次只是查詢指定時段候選課，未要求 PTT/心得/評價，因此跳過評價搜尋以避免 demo 等太久。")
        elif live_sources and max_review_lookup <= 0:
            warnings.append(
                "為了避免 demo 卡住，指定時段候選排序預設只用本地/seed RAG 評價資料；即時 PTT/web 補查預設關閉，不會猜測沒有命中的課。"
            )
        elif live_sources and len([course for course in candidates if not course.get("review_count")]) > live_lookup_count:
            warnings.append(
                f"為了避免 demo 等太久，PTT RAG 會掃 seed corpus 中所有候選課；即時 PTT/web 查詢只補查最多 {max_review_lookup} 門沒有 RAG 命中的候選課，其餘不會被猜測甜度。"
            )
        if any((course.get("review_summary") or {}).get("review_count") for course in candidates):
            warnings.append("候選排序只把 PTT/多來源心得當主觀參考；樣本少時不可靠。")
        else:
            warnings.append("目前沒有找到足夠可靠的 PTT/多來源心得樣本，因此不會宣稱哪門課最甜。")

        result = {
            "intent": "search_course_options",
            "target_semester": "11420",
            "query_time_slots": sorted(query_slots),
            "category_filter": category_filter,
            "exclude_time_slots": sorted(excluded_time_slots),
            "candidate_courses": candidates[:max_display],
            "candidate_count_before_display_limit": len(candidates),
            "rejected_courses": rejected,
            "review_preference": review_preference,
            "review_sources": sources,
            "review_search_enabled": review_search_enabled,
            "warnings": list(dict.fromkeys(warnings)),
        }
        result["agent_explanation"] = explain_course_options_result(result)
        self.last_course_options = result
        return result

    @langsmith_trace("agent.chat")
    def chat(self, user_message: str, preferences: dict | None = None, planning_mode: bool = True) -> dict:
        message = str(user_message or "").strip()
        if not message:
            return _chat_fallback_response()

        if self.student_df is None or self.target_df is None or self.rules is None:
            self.load()

        parse_message = message
        previous_clarification = self.last_parsed_intent if isinstance(self.last_parsed_intent, dict) else {}
        previous_question = str(previous_clarification.get("clarification_question") or "")
        if previous_clarification.get("needs_clarification") and _contains_any(
            previous_question,
            ("評價", "老師", "涼", "甜", "心得", "ptt", "PTT"),
        ):
            inferred_followup_course = _infer_course_name_from_message(self.target_df, message)
            if _has_target_course_match(self.target_df, inferred_followup_course):
                parse_message = f"{message} 老師評價 PTT"

        state_snapshot = _intent_state_snapshot(self.last_recommendation, self.last_teacher_review)
        parsed_intent = parse_user_intent(
            parse_message,
            state_snapshot=state_snapshot,
            use_llm=self.use_llm_intent,
            model=self.model,
            provider=self.intent_provider,
        )
        inferred_review_course = _infer_course_name_from_message(self.target_df, message)
        if (
            parsed_intent.get("action") == "search_course_options"
            and not (parsed_intent.get("query_time_slots") or [])
            and _has_target_course_match(self.target_df, inferred_review_course)
            and _contains_any(
                message,
                (
                    "老師",
                    "評價",
                    "評論",
                    "心得",
                    "涼",
                    "甜",
                    "好不好",
                    "ptt",
                    "PTT",
                    "dcard",
                    "Dcard",
                ),
            )
        ):
            intent_preferences = dict(parsed_intent.get("preferences") if isinstance(parsed_intent.get("preferences"), dict) else {})
            intent_preferences["use_review_search"] = True
            if _contains_any(message, ("甜", "甜度", "高分")):
                intent_preferences["review_prefer"] = "sweetness"
            elif _contains_any(message, ("涼", "涼度", "輕鬆")):
                intent_preferences["review_prefer"] = "coolness"
            else:
                intent_preferences.setdefault("review_prefer", "coolness")
            intent_preferences.setdefault("review_sources", ["ptt"])
            parsed_intent = dict(parsed_intent)
            parsed_intent.update(
                {
                    "action": "review_course",
                    "operation": "none",
                    "course_name": inferred_review_course,
                    "course_names": [inferred_review_course],
                    "query_time_slots": [],
                    "preferences": intent_preferences,
                    "needs_clarification": False,
                    "clarification_question": "",
                    "confidence": min(0.98, max(float(parsed_intent.get("confidence") or 0), 0.9)),
                }
            )
        if parsed_intent.get("needs_clarification") and _contains_any(
            message,
            ("評價", "老師", "涼", "甜", "心得", "ptt", "PTT"),
        ) and _has_target_course_match(self.target_df, inferred_review_course):
            parsed_intent = dict(parsed_intent)
            parsed_intent.update(
                {
                    "action": "review_course",
                    "operation": "",
                    "course_name": inferred_review_course,
                    "course_names": [inferred_review_course],
                    "use_review_search": True,
                    "review_sources": ["ptt"],
                    "needs_clarification": False,
                    "clarification_question": "",
                    "confidence": 0.9,
                }
            )
        parser_source = parsed_intent.get("parser_source", "rule_based")
        selected_option = _resolve_course_option_selection(message, self.last_course_options)
        if selected_option:
            selected_code = normalize_course_code(selected_option.get("code", ""))
            selected_slots = sorted(_normalized_option_time_slots(selected_option))
            selected_days = sorted({slot[:1] for slot in selected_slots if slot[:1] in VALID_DAY_CODES})
            intent_preferences = dict(parsed_intent.get("preferences") if isinstance(parsed_intent.get("preferences"), dict) else {})
            if selected_days:
                intent_preferences["preferred_days"] = sorted(
                    set(intent_preferences.get("preferred_days", [])) | set(selected_days)
                )
            if selected_code and selected_slots:
                required_sections = dict(intent_preferences.get("required_section_time_slots_by_code") or {})
                required_sections[selected_code] = selected_slots
                intent_preferences["required_section_time_slots_by_code"] = required_sections
            parsed_intent = dict(parsed_intent)
            parsed_intent.update(
                {
                    "action": "modify_schedule",
                    "operation": "add_course",
                    "course_names": [selected_option.get("course_name_zh", "")],
                    "course_codes": [selected_code],
                    "preferences": intent_preferences,
                    "is_explicit_override": True,
                    "needs_clarification": False,
                    "clarification_question": "",
                    "confidence": 0.96,
                }
            )
        self.last_parsed_intent = parsed_intent
        self._record_user_request(message, parsed_intent)

        effective_message = _message_from_llm_intent(message, parsed_intent)
        constraint_messages = self._apply_user_constraint_updates(message, effective_message, parsed_intent)

        def finish(result: dict, action: str, tool_names: list[str]) -> dict:
            result["intent_parser"] = parser_source
            result["agent_version"] = AGENT_CODE_VERSION
            if parsed_intent:
                result["parsed_intent"] = parsed_intent
            result["user_constraints"] = self._constraints_as_preferences()
            return _prepend_chat_header(result, action, tool_names)

        if (
            parsed_intent.get("needs_clarification")
            and parsed_intent.get("action") == "search_course_options"
            and (parsed_intent.get("query_time_slots") or [])
        ):
            intent_preferences = dict(parsed_intent.get("preferences") if isinstance(parsed_intent.get("preferences"), dict) else {})
            if _contains_any(effective_message, ("甜", "甜度", "高分")):
                intent_preferences["review_prefer"] = "sweetness"
            elif _contains_any(effective_message, ("涼", "涼度", "輕鬆", "好過", "不要太硬", "不太硬")):
                intent_preferences["review_prefer"] = "coolness"
                intent_preferences["avoid_difficult_courses"] = True
                intent_preferences["use_review_search"] = True
            parsed_intent = dict(parsed_intent)
            parsed_intent.update(
                {
                    "needs_clarification": False,
                    "clarification_question": "",
                    "preferences": intent_preferences,
                    "confidence": max(float(parsed_intent.get("confidence") or 0), 0.9),
                }
            )

        if parsed_intent.get("needs_clarification"):
            result = {
                "intent": "needs_clarification",
                "agent_explanation": "### 需要再確認\n\n" + (parsed_intent.get("clarification_question") or "我不確定你想做哪一個動作，可以再說清楚一點嗎？"),
            }
            return finish(result, "確認使用者意圖", ["intent_parser"])

        if parsed_intent.get("action") == "confirm_final":
            if self.last_recommendation:
                result = dict(self.last_recommendation)
                result["intent"] = "confirm_final"
                result["agent_explanation"] = "### 已確認課表\n\n" + explain_recommendation_result(
                    result,
                    include_process_sections=False,
                ) + "\n\n#### 結語\n謝謝你使用 NTHU COPILOT。祝你選課順利，這學期也加油！"
            else:
                result = {
                    "intent": "confirm_final",
                    "agent_explanation": "### 還不能確認\n\n目前還沒有產生課表，所以不能確認最終課表。請先輸入例如：`我要修23學分`。",
                }
            result["intent_parser"] = parser_source
            result["agent_version"] = AGENT_CODE_VERSION
            result["user_constraints"] = self._constraints_as_preferences()
            if parsed_intent:
                result["parsed_intent"] = parsed_intent
            return result

        base_preferences = preferences if preferences is not None else _strip_ephemeral_review_preferences(self.last_preferences)
        chat_preferences = _chat_preferences_from_message(effective_message, base_preferences)
        chat_preferences = _apply_llm_intent_to_preferences(chat_preferences, parsed_intent)
        more_ee_intent = _is_more_ee_request(effective_message) or parsed_intent.get("operation") == "add_more_ee"
        general_education_intent = (
            _is_general_education_request(effective_message) or parsed_intent.get("operation") == "add_general_education"
        )
        if more_ee_intent:
            intent_prefs = parsed_intent.get("preferences") if isinstance(parsed_intent.get("preferences"), dict) else {}
            reduce_non_ee = _is_reduce_non_ee_request(effective_message) or intent_prefs.get("replace_non_ee_first") or intent_prefs.get("reduce_non_ee_courses")
            chat_preferences["prefer_more_ee_courses"] = True
            chat_preferences["requested_ee_course_count"] = parsed_intent.get("course_count") or _extract_requested_course_count(effective_message, default=2 if reduce_non_ee else 1)
            chat_preferences["avoid_difficult_courses"] = False
            if reduce_non_ee:
                chat_preferences["replace_non_ee_first"] = True
                chat_preferences["reduce_non_ee_courses"] = True
            chat_preferences.pop("prefer_general_education_courses", None)
            chat_preferences.pop("requested_general_education_count", None)
        elif general_education_intent:
            chat_preferences.pop("prefer_more_ee_courses", None)
            chat_preferences.pop("requested_ee_course_count", None)
            chat_preferences["prefer_general_education_courses"] = True
            chat_preferences["requested_general_education_count"] = parsed_intent.get("course_count") or _extract_requested_course_count(effective_message)
        else:
            chat_preferences.pop("prefer_more_ee_courses", None)
            chat_preferences.pop("requested_ee_course_count", None)
            chat_preferences.pop("prefer_general_education_courses", None)
            chat_preferences.pop("requested_general_education_count", None)
        chat_preferences = self._merge_user_constraints_into_preferences(chat_preferences)

        replace_intent = _contains_any(
            effective_message,
            ("不想修", "不要修", "不修", "換掉", "替換", "換一門", "換課", "replace", "移除", "去掉", "刪掉", "拿掉", "退掉", "drop", "remove"),
        )
        avoid_day_remove_only_intent = bool(self.last_recommendation) and (
            _is_avoid_day_remove_only_followup(effective_message) or parsed_intent.get("operation") == "remove_day"
        )
        category_remove_only_intent = (
            bool(self.last_recommendation)
            and (
                _message_requests_remove_category(effective_message)
                or parsed_intent.get("operation") == "remove_category"
            )
            and not _message_mentions_current_course(effective_message, self.last_recommendation)
        )
        course_remove_only_intent = _is_course_remove_only_followup(
            effective_message,
            self.last_recommendation,
            self.last_teacher_review,
        ) or parsed_intent.get("operation") == "remove_course"
        fresh_schedule_request = (
            parsed_intent.get("action") == "recommend_schedule"
            and parsed_intent.get("operation") in {"", "none"}
            and (
                parsed_intent.get("exact_credits") is not None
                or parsed_intent.get("credit_min") is not None
                or parsed_intent.get("credit_max") is not None
                or _contains_any(effective_message, ("幫我排", "重新排", "重排", "排一份", "排課"))
            )
        )
        more_ee_update_intent = bool(self.last_recommendation) and more_ee_intent and not fresh_schedule_request
        general_education_update_intent = bool(self.last_recommendation) and general_education_intent and not fresh_schedule_request
        add_specific_course_intent = (
            _is_add_specific_course_request(effective_message, self.target_df, self.last_recommendation)
            or parsed_intent.get("operation") == "add_course"
        )
        replace_intent = (
            replace_intent
            or avoid_day_remove_only_intent
            or category_remove_only_intent
            or course_remove_only_intent
            or more_ee_update_intent
            or general_education_update_intent
            or add_specific_course_intent
            or _intent_requests_schedule_update(parsed_intent)
        )
        plan_preference_intent = _contains_any(
            effective_message,
            (
                "不要上",
                "不想上",
                "不要有",
                "避開",
                "星期五",
                "週五",
                "禮拜五",
                "學分",
                "太硬",
                "輕鬆",
                "平衡",
                "通識",
                "外文",
                "中文",
                "外系",
                "體育",
                "星期一",
                "星期二",
                "星期三",
                "星期四",
                "週一",
                "週二",
                "週三",
                "週四",
                "禮拜一",
                "禮拜二",
                "禮拜三",
                "禮拜四",
                "早八",
                "早上",
                "下午",
                "晚上",
                "甜",
                "涼",
                "甜的課",
                "涼的課",
                "好過",
                "少一點",
                "多一點",
            ),
        )
        plan_intent = _contains_any(
            effective_message,
            ("排課", "課表", "選課", "推薦", "規劃", "11420", "下學期"),
        ) or plan_preference_intent or parsed_intent.get("action") == "recommend_schedule"
        review_intent = _contains_any(
            effective_message,
            ("老師", "評價", "ptt", "dcard", "網路", "網站", "涼", "甜", "好不好", "心得"),
        ) or parsed_intent.get("action") == "review_course"
        review_rerank_intent = review_intent and _contains_any(
            effective_message,
            ("根據網路評價", "參考網路評價", "評價比較好的老師", "比較好的老師", "甜的課", "涼的課", "甜一點", "涼一點", "很甜", "很涼", "好過的課"),
        ) or parsed_intent.get("action") == "review_rerank_schedule"
        if review_intent:
            chat_preferences["show_review_block"] = True
        else:
            chat_preferences.pop("show_review_block", None)
        graduation_intent = _contains_any(
            effective_message,
            ("畢業", "缺什麼", "還缺", "缺哪些", "畢業門檻", "graduation"),
        ) or parsed_intent.get("action") == "check_graduation"
        if not self.last_recommendation and not replace_intent and not graduation_intent:
            plan_intent = True
        if self.last_recommendation and (replace_intent or plan_intent):
            chat_preferences = _relax_credit_target_for_followup(effective_message, chat_preferences, self.last_recommendation)
            chat_preferences = self._merge_user_constraints_into_preferences(chat_preferences)
        self.last_preferences = _strip_ephemeral_review_preferences(chat_preferences)
        review_course_name = ""
        review_has_course_match = False
        if review_intent:
            review_course_name = _infer_course_name_from_message(self.target_df, effective_message)
            review_has_course_match = _has_target_course_match(self.target_df, review_course_name)
        current_plan_review_query = _is_current_plan_review_query(effective_message)

        if constraint_messages and not (replace_intent or plan_intent or review_intent or graduation_intent):
            self.last_preferences.update(_strip_ephemeral_review_preferences(chat_preferences))
            result = {
                "intent": "update_constraints",
                "agent_explanation": "### 已更新排課偏好\n\n" + "\n".join(f"- {line}" for line in constraint_messages),
            }
            return finish(result, "更新持續性排課限制", ["update_user_constraints"])

        if current_plan_review_query:
            review_preferences = dict(chat_preferences)
            review_preferences["use_review_search"] = True
            review_preferences["review_sources"] = ["ptt"]
            review_preferences.setdefault("review_timeout", 3)
            review_preferences.setdefault("review_max_results", 2)
            parsed_intent = dict(parsed_intent)
            parsed_intent.update(
                {
                    "action": "review_course",
                    "operation": "none",
                    "course_names": [course.get("course_name_zh", "") for course in (self.last_recommendation or {}).get("recommended_courses", [])],
                    "query_time_slots": [],
                    "preferences": review_preferences,
                    "needs_clarification": False,
                    "clarification_question": "",
                }
            )
            result = self.scan_current_plan_reviews(preferences=review_preferences)
            result["intent"] = "current_plan_reviews"
            return finish(
                result,
                "掃描目前課表中的 PTT 心得",
                ["search_course_reviews"],
            )

        course_option_query_intent = _is_course_option_query(effective_message, parsed_intent)
        if course_option_query_intent:
            result = self.search_course_options(effective_message, preferences=chat_preferences)
            result["intent"] = "search_course_options"
            return finish(
                result,
                "查詢指定時段可選課程並用多來源評價排序",
                ["search_11420_course_sections", "search_course_reviews", "check_plan_conflicts"],
            )

        if replace_intent:
            if self.last_recommendation is None:
                seed_plan = self.recommend(preferences=chat_preferences, planning_mode=planning_mode)
                self.last_recommendation = seed_plan
            update_request = effective_message
            reviewed_course_name = _last_reviewed_course_name(self.last_teacher_review)
            if (
                course_remove_only_intent
                and reviewed_course_name
                and not _message_mentions_current_course(effective_message, self.last_recommendation)
            ):
                update_request = f"{effective_message} {reviewed_course_name}"
            result = self.replace_course(
                self.last_recommendation,
                update_request,
                preferences={
                    **chat_preferences,
                    "update_mode": _update_mode_from_intents(
                        parsed_intent,
                        avoid_day_remove_only_intent,
                        category_remove_only_intent,
                        course_remove_only_intent,
                        add_specific_course_intent,
                        more_ee_update_intent,
                        general_education_update_intent,
                        effective_message,
                    ),
                },
                planning_mode=planning_mode,
            )
            result["intent"] = "replace_course"
            if more_ee_update_intent:
                action = "依照你的要求新增或替換成更多電機系課程"
            elif general_education_update_intent:
                action = "依照你的要求新增或替換成通識課程"
            elif add_specific_course_intent:
                action = "依照你的明確課名新增指定課程"
            elif avoid_day_remove_only_intent:
                action = "依照你的星期限制從目前課表移除課程"
            elif category_remove_only_intent:
                action = "依照你的類別限制從目前課表移除課程"
            elif course_remove_only_intent:
                action = "依照你的文字要求從目前課表移除課程"
            else:
                action = "依照你的文字要求替換目前課表中的課"
            return finish(result, action, ["update_plan", "check_plan_conflicts"])

        if review_intent and review_has_course_match:
            result = self.compare_course_teachers(
                course_name=review_course_name,
                preference=chat_preferences.get("review_prefer", chat_preferences.get("ptt_prefer", "coolness")),
                sources=chat_preferences.get("review_sources"),
                max_pages=int(chat_preferences.get("review_max_results", chat_preferences.get("ptt_max_pages", 3))),
            )
            result["intent"] = "compare_course_teachers"
            return finish(
                result,
                "查詢某門課的老師/多來源評價參考",
                ["search_11420_course_sections", "search_course_reviews"],
            )

        if review_rerank_intent and not review_has_course_match:
            plan_intent = True

        if plan_intent:
            result = self.recommend(preferences=chat_preferences, planning_mode=planning_mode)
            result["intent"] = "recommend_courses"
            return finish(
                result,
                "產生 114 第二學期課表建議",
                ["check_graduation_progress", "recommend_courses", "check_plan_conflicts"],
            )

        if review_intent:
            course_name = review_course_name
            has_course_match = review_has_course_match
            if not has_course_match and self.last_teacher_review and _contains_any(
                effective_message, ("ptt", "dcard", "網路", "網站", "心得", "評價")
            ):
                course_name = self.last_teacher_review.get("course_name", course_name)
                has_course_match = _has_target_course_match(self.target_df, course_name)
            if not has_course_match and review_rerank_intent:
                result = self.recommend(preferences=chat_preferences, planning_mode=planning_mode)
                result["intent"] = "recommend_courses_with_reviews"
                return finish(
                    result,
                    "依照你的評價偏好重新產生 114 第二學期課表建議",
                    ["check_graduation_progress", "recommend_courses", "search_course_reviews", "check_plan_conflicts"],
                )
            if not has_course_match:
                result = {
                    "intent": "need_course_name_for_review",
                    "agent_explanation": "\n".join(
                        [
                            "### 需要指定課名",
                            "",
                            "我判斷你想查網路評價，但訊息裡沒有對應到 114 第二學期課表中的明確課名，所以我不會亂猜。",
                            "",
                            "你可以這樣問：",
                            "",
                            "- `偏微分方程與複變函數老師評價怎麼樣`",
                            "- `機率 Dcard 或 PTT 有沒有心得`",
                            "- `幫我根據網路評價重新排課`",
                            "",
                            "如果你是想用評價輔助整份課表，請直接說「幫我根據網路評價重新排課」。",
                        ]
                    ),
                }
                return finish(result, "查詢評價前先確認課名", ["search_11420_course_sections"])
            result = self.compare_course_teachers(
                course_name=course_name,
                preference=chat_preferences.get("review_prefer", chat_preferences.get("ptt_prefer", "coolness")),
                sources=chat_preferences.get("review_sources"),
                max_pages=int(chat_preferences.get("review_max_results", chat_preferences.get("ptt_max_pages", 3))),
            )
            result["intent"] = "compare_course_teachers"
            return finish(
                result,
                "查詢某門課的老師/多來源評價參考",
                ["search_11420_course_sections", "search_course_reviews"],
            )

        if graduation_intent:
            result = self.check_graduation(planning_mode=planning_mode)
            result["intent"] = "check_graduation"
            return finish(result, "檢查 EE112 畢業進度", ["check_graduation_progress"])

        return _chat_fallback_response()
