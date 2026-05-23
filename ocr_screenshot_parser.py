from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from course_data_loader import load_student_courses, load_target_courses, normalize_course_code
except ImportError:  # pragma: no cover
    from .course_data_loader import load_student_courses, load_target_courses, normalize_course_code


TARGET_SEMESTER = "11420"
CURRENT_IN_PROGRESS_TERM = "11410"
FUZZY_MATCH_THRESHOLD = 0.86
CONFIRMATION_MESSAGE = "系統已根據截圖與歷史課程資料辨識出以下修課紀錄，請確認。確認後才會送進 agent。"
HISTORICAL_CACHE_FILENAME = "ocr_historical_course_cache.csv"
HISTORICAL_CACHE_META_FILENAME = "ocr_historical_course_cache.meta"

DISPLAY_COLUMNS = [
    "term",
    "academic_year",
    "semester",
    "normalized_course_code",
    "raw_course_code",
    "course_name_zh",
    "course_name_en",
    "credits",
    "status",
    "grade",
    "teacher",
    "time",
    "classroom",
    "needs_user_confirmation",
    "match_confidence",
    "match_source",
]

OCR_CONFIRM_DISPLAY_COLUMNS = [
    "term",
    "academic_year",
    "semester",
    "normalized_course_code",
    "raw_course_code",
    "course_name_zh",
    "course_name_en",
    "credits",
    "status",
    "grade",
]

SAVE_COLUMNS = [
    "term",
    "academic_year",
    "semester",
    "raw_course_code",
    "normalized_course_code",
    "course_name_zh",
    "course_name_en",
    "credits",
    "status",
    "grade",
    "teacher",
    "time",
    "classroom",
]

EMPTY_COURSE_COLUMNS = SAVE_COLUMNS.copy()

# Common course-code shapes in NTHU data.
# Raw: EE231001 / EE 231001 / EECS101001. Normalized: EE2310 / EECS1010.
CODE_RE = re.compile(r"(?P<prefix>[A-Za-z]{2,6})\s*(?P<digits>[0-9OIL|S$]{4,6})", re.IGNORECASE)
COURSE_LINE_RE = re.compile(
    r"(?P<year>1\d{2})\s*(?P<semester>10|20)\s*(?P<code>[A-Za-z]{2,6}\s*[0-9OIL|S$]{4,6})",
    re.IGNORECASE,
)
SEM_CODE_RE = re.compile(r"^(?P<semester>10|20)\s*(?P<code>[A-Za-z]{2,6}\s*[0-9OIL|S$]{4,6})$", re.IGNORECASE)
YEAR_ONLY_RE = re.compile(r"^(?P<year>1\d{2})$|^(?P<year2>1\d{2})\s*$")

VALID_PREFIXES = {
    "EE", "EECS", "CS", "ISA", "COM", "MATH", "PHYS", "CHEM", "CHE", "BMES", "BME",
    "GE", "GEC", "LANG", "CL", "PE", "AIA", "AI", "STAT", "ECON", "ANTH", "CSR", "ESS",
    "NEMS", "PME", "LS", "LST", "LIFE", "HSS", "TH", "TIGP",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", "", _clean(value).lower())


def _display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except Exception:  # pragma: no cover
        print(text)


def _display_dataframe(df: pd.DataFrame) -> None:
    try:
        from IPython.display import display
        display(df)
    except Exception:  # pragma: no cover
        print(df.to_string(index=False))


def _split_term(term: Any) -> tuple[str, str]:
    digits = re.sub(r"\D", "", _clean(term))
    if len(digits) >= 5:
        return digits[:3], digits[3:5]
    return "", ""


def _make_term(year: Any, semester: Any) -> str:
    year_digits = re.sub(r"\D", "", _clean(year))
    semester_digits = re.sub(r"\D", "", _clean(semester))
    if len(year_digits) == 3 and semester_digits:
        sem = semester_digits[:2]
        if sem in {"1", "01"}:
            sem = "10"
        elif sem in {"2", "02"}:
            sem = "20"
        return year_digits + sem
    return ""


def _semester_display_to_code(text: str) -> str:
    value = _clean(text)
    digits = re.sub(r"\D", "", value)
    # Accept compact semester formats.
    if re.fullmatch(r"1\d{2}[12]", digits):
        return digits[:3] + ("10" if digits[3] == "1" else "20")
    if re.fullmatch(r"1\d{4}", digits):
        year, sem = digits[:3], digits[3:5]
        if sem in {"10", "20"}:
            return year + sem
    year_match = re.search(r"(1\d{2})", value)
    if year_match:
        year = year_match.group(1)
        if "第一" in value or "上" in value or "1" in value[-3:]:
            return year + "10"
        if "第二" in value or "下" in value or "2" in value[-3:]:
            return year + "20"
    return ""


def cache_path_for_image(image_path: str | Path) -> Path:
    path = Path(image_path)
    return path.with_name(f"{path.stem}_ocr.txt")


def load_ocr_cache_if_available(image_path: str | Path) -> str:
    path = cache_path_for_image(image_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _fix_ocr_code_text(raw: Any) -> str:
    text = _clean(raw).upper()
    text = (
        text.replace("Ｏ", "O")
        .replace("Ｉ", "I")
        .replace("Ｌ", "L")
        .replace("｜", "|")
        .replace("＿", "_")
        .replace("－", "-")
    )
    text = re.sub(r"[^A-Z0-9|$\s_-]", "", text)
    text = re.sub(r"\s+", "", text)
    prefix_fixes = [
        ("LAIG", "LANG"), ("LA1G", "LANG"), ("LAG", "LANG"), ("LHIG", "LANG"),
        ("EBCS", "EECS"), ("ERCS", "EECS"), ("EEC5", "EECS"),
        ("MTH", "MATH"), ("M4TH", "MATH"),
        ("PHY5", "PHYS"), ("FHY5", "PHYS"), ("FHYS", "PHYS"),
        ("GELL", "GE"),
    ]
    for bad, good in prefix_fixes:
        if text.startswith(bad):
            text = good + text[len(bad):]
            break
    return text


def _split_code_prefix_digits(raw: Any) -> tuple[str, str]:
    fixed = _fix_ocr_code_text(raw)
    match = CODE_RE.search(fixed)
    if not match:
        return "", ""
    prefix = match.group("prefix").upper()
    digit_map = str.maketrans({"O": "0", "I": "1", "L": "1", "|": "1", "S": "5", "$": "5"})
    digits = match.group("digits").upper().translate(digit_map)
    digits = re.sub(r"\D", "", digits)
    return prefix, digits


def normalize_ocr_code(raw: Any) -> str:
    prefix, digits = _split_code_prefix_digits(raw)
    if not prefix or len(digits) < 4:
        return ""
    return normalize_course_code(prefix + digits)


def clean_raw_course_code(raw: Any) -> str:
    prefix, digits = _split_code_prefix_digits(raw)
    if not prefix or len(digits) < 4:
        return _clean(raw)
    return prefix + digits


def _looks_like_valid_course_code(code: str) -> bool:
    code = normalize_ocr_code(code)
    if not re.fullmatch(r"[A-Z]{2,6}\d{4}", code or ""):
        return False
    prefix = re.match(r"[A-Z]+", code).group(0)  # type: ignore[union-attr]
    return prefix in VALID_PREFIXES


# ---------------------------------------------------------------------------
# Loading structured data
# ---------------------------------------------------------------------------

def _safe_load_student(path: str | Path | None, warnings: list[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)
    try:
        return load_student_courses(str(path))
    except FileNotFoundError:
        warnings.append(f"找不到學生修課檔：{path}。")
    except Exception as exc:
        warnings.append(f"學生修課檔讀取失敗：{exc}")
    return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)


def _safe_load_courses(path: str | Path | None, warnings: list[str], label: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)
    try:
        return load_target_courses(str(path))
    except FileNotFoundError:
        warnings.append(f"找不到{label}：{path}。")
    except Exception as exc:
        warnings.append(f"{label}讀取失敗：{exc}")
    return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)


def _safe_load_courses_for_term(path: str | Path | None, term: str, warnings: list[str]) -> pd.DataFrame:
    """Load only one historical term sheet when possible.

    Manual OCR correction often needs one term, such as 11320. Loading the whole
    historical workbook can take minutes in HW2, so this avoids reading unrelated
    sheets.
    """
    if not path:
        return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)
    path = Path(path)
    try:
        sheet_name: str | int = str(term)
        filter_term_after_load = False
        try:
            sheet_names = pd.ExcelFile(path).sheet_names
        except Exception:
            sheet_names = []
        if sheet_names and str(term) not in sheet_names:
            year = str(term)[:3]
            combined_candidates = [
                name for name in sheet_names
                if year in name or (year in {"111", "112"} and "111" in name and "112" in name)
            ]
            if not combined_candidates:
                return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)
            sheet_name = combined_candidates[0]
            filter_term_after_load = True

        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        from course_data_loader import _detect_header_row, _standardize_columns  # type: ignore

        header_row = _detect_header_row(df)
        if header_row is not None:
            headers = []
            for i, value in enumerate(df.iloc[header_row].tolist()):
                header = _clean(value)
                headers.append(header if header else f"unnamed_{i}")
            df = df.iloc[header_row + 1 :].copy()
            df.columns = headers
        standardized = _standardize_columns(df.reset_index(drop=True))
        if "raw_course_code" in standardized.columns:
            raw_terms = standardized["raw_course_code"].astype(str).str.extract(r"^(\d{5})", expand=False).fillna("")
            standardized.loc[raw_terms.ne(""), "term"] = raw_terms[raw_terms.ne("")]
        blank_term = standardized["term"].map(_clean) == ""
        standardized.loc[blank_term, "term"] = str(term)
        if filter_term_after_load:
            standardized = standardized[standardized["term"].astype(str).str.replace(r"\.0$", "", regex=True).eq(str(term))]
        return standardized
    except ValueError:
        return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)
    except Exception as exc:
        warnings.append(f"讀取 {path.name} 的 {term} sheet 失敗：{exc}")
        return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)


def _historical_cache_paths(paths: list[str] | None) -> tuple[Path, Path]:
    base = Path(paths[0]).parent if paths else Path("data")
    return base / HISTORICAL_CACHE_FILENAME, base / HISTORICAL_CACHE_META_FILENAME


def _historical_cache_signature(paths: list[str] | None) -> str:
    parts: list[str] = []
    for path_value in paths or []:
        path = Path(path_value)
        if path.exists():
            stat = path.stat()
            parts.append(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}")
        else:
            parts.append(f"{path.resolve()}|missing|0")
    return "\n".join(parts)


def _read_historical_cache(paths: list[str] | None) -> pd.DataFrame | None:
    cache_path, meta_path = _historical_cache_paths(paths)
    if not cache_path.exists() or not meta_path.exists():
        return None
    if meta_path.read_text(encoding="utf-8", errors="replace") != _historical_cache_signature(paths):
        return None
    try:
        return pd.read_csv(cache_path, dtype=str, keep_default_na=False)
    except Exception:
        return None


def _write_historical_cache(paths: list[str] | None, df: pd.DataFrame) -> None:
    cache_path, meta_path = _historical_cache_paths(paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keep_columns = [
        "term",
        "academic_year",
        "semester",
        "raw_course_code",
        "normalized_course_code",
        "course_name_zh",
        "course_name_en",
        "credits",
        "status",
        "grade",
        "teacher",
        "time",
        "classroom",
    ]
    slim = df[[column for column in keep_columns if column in df.columns]].copy()
    slim.to_csv(cache_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(_historical_cache_signature(paths), encoding="utf-8")


def _load_historical(paths: list[str] | None, warnings: list[str]) -> pd.DataFrame:
    cached = _read_historical_cache(paths)
    if cached is not None:
        return cached

    frames: list[pd.DataFrame] = []
    for path in paths or []:
        df = _safe_load_courses(path, warnings, "歷史課程資料")
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=EMPTY_COURSE_COLUMNS)
    historical_df = pd.concat(frames, ignore_index=True)
    try:
        _write_historical_cache(paths, historical_df)
    except Exception as exc:
        warnings.append(f"歷史課程快取寫入失敗：{exc}")
    return historical_df


def rebuild_historical_course_cache(historical_course_paths: list[str]) -> dict[str, Any]:
    """Rebuild the OCR historical course cache explicitly for the notebook demo."""
    warnings: list[str] = []
    cache_path, meta_path = _historical_cache_paths(historical_course_paths)
    if cache_path.exists():
        cache_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    df = _load_historical(historical_course_paths, warnings)
    return {
        "cache_path": str(cache_path),
        "meta_path": str(meta_path),
        "rows": len(df),
        "warnings": warnings,
    }


def _load_existing_confirmed_record(image_path: str | Path) -> pd.DataFrame:
    data_dir = Path(image_path).parent
    confirmed_path = data_dir / "ocr_confirmed_student_courses.csv"
    if not confirmed_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(confirmed_path, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


def _canonical_code_set(*dfs: pd.DataFrame) -> set[str]:
    codes: set[str] = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        for column in ["normalized_course_code", "raw_course_code"]:
            if column in df.columns:
                for value in df[column].dropna().astype(str):
                    code = normalize_ocr_code(value) or normalize_course_code(value)
                    if code:
                        codes.add(code)
    return codes


# ---------------------------------------------------------------------------
# OCR text parsing
# ---------------------------------------------------------------------------

def _is_header_line(line: str) -> bool:
    lowered = line.lower().replace(" ", "_")
    return any(
        token in lowered
        for token in [
            "academic", "semester", "raw_course", "normalized", "course_cod", "course_name",
            "credit", "status", "grade", "teacher", "classroom", "學年", "學期", "課號", "學分", "狀態", "成績",
        ]
    )


def _status_grade_from_text(text: str) -> tuple[str, str]:
    lowered = _clean(text).lower().replace("-", "_")
    compact = re.sub(r"[^a-z_]", "", lowered)
    if any(token in lowered for token in ["in_progress", "in progress", "inprogress", "修課中", "正在修"]):
        return "in_progress", ""
    if "progress" in compact:
        return "in_progress", ""
    if any(token in lowered for token in ["completed", "complete", "passed", "pass", "已修", "通過", "及格"]):
        return "completed", "pass"
    if any(token in lowered for token in ["fail", "failed"]) or any(token in text for token in ["不及格", "未通過", "沒過"]):
        return "completed", "fail"
    return "", ""


def _credit_status_entries(lines: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in lines:
        if _is_header_line(line):
            continue
        if CODE_RE.search(line):
            continue
        status, grade = _status_grade_from_text(line)
        # Google/Paddle usually produce "3 completed" or "2 in progress" in one line.
        m = re.search(r"\b([0-6](?:\.0)?)\b", line)
        if m and status:
            entries.append({"credits": float(m.group(1)), "status": status, "grade": grade})
            continue
        if entries and line.strip().lower() in {"pass", "passed", "p"}:
            if entries[-1].get("status") == "completed":
                entries[-1]["grade"] = "pass"
        elif entries and line.strip().lower() in {"fail", "failed", "f"}:
            entries[-1]["status"] = "completed"
            entries[-1]["grade"] = "fail"
    return entries


def _course_entries(lines: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    pending_year = ""
    for line in lines:
        text = line.strip()
        if not text or _is_header_line(text):
            continue
        # Full row: 112 20 EE 223002
        for m in COURSE_LINE_RE.finditer(text):
            raw = clean_raw_course_code(m.group("code"))
            code = normalize_ocr_code(raw)
            if _looks_like_valid_course_code(code):
                entries.append({"academic_year": m.group("year"), "semester": m.group("semester"), "raw_course_code": raw, "normalized_course_code": code})
            pending_year = ""
        if COURSE_LINE_RE.search(text):
            continue
        y = YEAR_ONLY_RE.match(text)
        if y:
            pending_year = y.group("year") or y.group("year2") or ""
            continue
        sm = SEM_CODE_RE.match(text)
        if sm and pending_year:
            raw = clean_raw_course_code(sm.group("code"))
            code = normalize_ocr_code(raw)
            if _looks_like_valid_course_code(code):
                entries.append({"academic_year": pending_year, "semester": sm.group("semester"), "raw_course_code": raw, "normalized_course_code": code})
            pending_year = ""
            continue
    # Deduplicate while keeping OCR order and term/raw distinctions.
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        key = (entry.get("academic_year", ""), entry.get("semester", ""), entry.get("normalized_course_code", ""))
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def parse_ocr_rows_from_text(text: str, valid_codes: set[str] | None = None) -> list[dict[str, Any]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    courses = _course_entries(lines)
    statuses = _credit_status_entries(lines)
    valid_codes = valid_codes or set()
    rows: list[dict[str, Any]] = []
    for index, course in enumerate(courses):
        code = course["normalized_course_code"]
        # Strongly filter OCR hallucinations such as TOSOLI/TO5011. Keep a course only
        # if it is in the known structured data, or has a valid NTHU-like prefix.
        if valid_codes and code not in valid_codes and not _looks_like_valid_course_code(code):
            continue
        status_info = statuses[index] if index < len(statuses) else {}
        status = status_info.get("status", "") or "completed"
        grade = status_info.get("grade", "")
        if status == "completed" and not grade:
            grade = "pass"
        if status == "in_progress":
            grade = ""
        term = _make_term(course.get("academic_year"), course.get("semester"))
        row = {
            "term": term,
            "academic_year": course.get("academic_year", ""),
            "semester": course.get("semester", ""),
            "raw_course_code": course.get("raw_course_code", ""),
            "normalized_course_code": code,
            "credits": status_info.get("credits", ""),
            "status": status,
            "grade": grade,
            "needs_user_confirmation": False,
            "match_confidence": 1.0,
            "match_source": "ocr_text",
        }
        rows.append(row)
    return rows


def parse_ocr_rows_from_cache(ocr_cache_path: str | Path = "data/course_screenshot_ocr.txt") -> dict[str, dict[str, Any]]:
    text = Path(ocr_cache_path).read_text(encoding="utf-8", errors="replace")
    rows = parse_ocr_rows_from_text(text)
    return {row["normalized_course_code"]: row for row in rows if row.get("normalized_course_code")}


# ---------------------------------------------------------------------------
# Matching/enrichment
# ---------------------------------------------------------------------------

def _matching_rows(df: pd.DataFrame, code: str, term: str = "", raw: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    if "normalized_course_code" in result.columns:
        result["_norm"] = result["normalized_course_code"].astype(str).map(normalize_ocr_code)
    else:
        result["_norm"] = ""
    if "raw_course_code" in result.columns:
        result["_raw_clean"] = result["raw_course_code"].astype(str).map(clean_raw_course_code)
    else:
        result["_raw_clean"] = ""
    code = normalize_ocr_code(code) or normalize_course_code(code)
    raw_clean = clean_raw_course_code(raw)
    mask = result["_norm"].eq(code)
    if raw_clean:
        mask = mask | result["_raw_clean"].eq(raw_clean)
    result = result[mask].copy()
    if term and "term" in result.columns:
        exact = result[result["term"].astype(str).str.replace(r"\.0$", "", regex=True).eq(str(term))]
        if not exact.empty:
            result = exact
    return result.drop(columns=[c for c in ["_norm", "_raw_clean"] if c in result.columns], errors="ignore")


def _best_info_for_row(ocr_row: dict[str, Any], student_df: pd.DataFrame, historical_df: pd.DataFrame) -> dict[str, Any]:
    code = _clean(ocr_row.get("normalized_course_code"))
    term = _clean(ocr_row.get("term"))
    raw = _clean(ocr_row.get("raw_course_code"))
    student_matches = _matching_rows(student_df, code, term, raw)
    hist_matches = _matching_rows(historical_df, code, term, raw)
    info: dict[str, Any] = {}
    if not hist_matches.empty:
        info.update(hist_matches.iloc[0].fillna("").to_dict())
    if not student_matches.empty:
        # Student reference is stronger than historical DB for term/credits/status/grade.
        info.update(student_matches.iloc[0].fillna("").to_dict())
    return info


def _as_credit(value: Any, fallback: Any = "") -> Any:
    if _clean(value) == "":
        return fallback
    try:
        return float(value)
    except Exception:
        return value


def _recognized_courses_from_ocr_rows(ocr_rows: list[dict[str, Any]], student_df: pd.DataFrame, historical_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ocr_row in ocr_rows:
        info = _best_info_for_row(ocr_row, student_df, historical_df)
        term = _clean(ocr_row.get("term")) or _clean(info.get("term"))
        academic_year = _clean(ocr_row.get("academic_year")) or _clean(info.get("academic_year"))
        semester = _clean(ocr_row.get("semester")) or _clean(info.get("semester"))
        if (not academic_year or not semester) and term:
            academic_year, semester = _split_term(term)
        status = _clean(ocr_row.get("status")) or _clean(info.get("status")) or "completed"
        grade = _clean(ocr_row.get("grade"))
        if not grade and status == "completed":
            grade = _clean(info.get("grade")) or "pass"
        if status == "in_progress":
            grade = ""
        record = {
            "term": term,
            "academic_year": academic_year,
            "semester": semester,
            "normalized_course_code": _clean(ocr_row.get("normalized_course_code")) or _clean(info.get("normalized_course_code")),
            "raw_course_code": _clean(ocr_row.get("raw_course_code")) or _clean(info.get("raw_course_code")),
            "course_name_zh": _clean(info.get("course_name_zh")),
            "course_name_en": _clean(info.get("course_name_en")),
            "credits": _as_credit(ocr_row.get("credits"), _as_credit(info.get("credits"))),
            "status": status,
            "grade": grade,
            "teacher": _clean(info.get("teacher")),
            "time": _clean(info.get("time")),
            "classroom": _clean(info.get("classroom")),
            "needs_user_confirmation": bool(ocr_row.get("needs_user_confirmation", False)),
            "match_confidence": ocr_row.get("match_confidence", 1.0),
            "match_source": "ocr_exact" if info else "ocr_provisional",
        }
        # Drop hallucinated provisional codes. If it was not in the Excel lookup and not a
        # common NTHU prefix, it is almost certainly an English-word OCR artifact.
        if record["match_source"] == "ocr_provisional" and not _looks_like_valid_course_code(record["normalized_course_code"]):
            continue
        records.append(record)
    return records


def extract_course_code_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in CODE_RE.finditer(str(text or "")):
        candidate = match.group(0)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_course_codes_from_text(text: str) -> list[str]:
    normalized: list[str] = []
    for candidate in extract_course_code_candidates(text):
        code = normalize_ocr_code(candidate)
        if code and _looks_like_valid_course_code(code) and code not in normalized:
            normalized.append(code)
    return normalized


def match_codes_to_database(codes, student_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, Any]:
    normalized_codes = [normalize_ocr_code(code) or normalize_course_code(code) for code in (codes or [])]
    normalized_codes = [code for code in dict.fromkeys(normalized_codes) if code]
    student_matches = student_df[student_df.get("normalized_course_code", pd.Series(dtype=str)).astype(str).map(normalize_ocr_code).isin(normalized_codes)] if normalized_codes and not student_df.empty else pd.DataFrame()
    target_matches = target_df[target_df.get("normalized_course_code", pd.Series(dtype=str)).astype(str).map(normalize_ocr_code).isin(normalized_codes)] if normalized_codes and not target_df.empty else pd.DataFrame()
    return {
        "extracted_codes": normalized_codes,
        "matched_student_courses": student_matches.fillna("").to_dict("records"),
        "matched_target_courses": target_matches.fillna("").to_dict("records"),
        "unmatched_codes": [],
    }


def _result_from_text(
    text: str,
    student_path: str | None = None,
    target_path: str | None = None,
    warnings: list[str] | None = None,
    historical_course_paths: list[str] | None = None,
) -> dict[str, Any]:
    result_warnings = list(warnings or [])
    student_df = _safe_load_student(student_path, result_warnings)
    historical_df = _load_historical(historical_course_paths, result_warnings)
    valid_codes = _canonical_code_set(student_df, historical_df)
    ocr_rows = parse_ocr_rows_from_text(text, valid_codes=valid_codes)
    recognized = _recognized_courses_from_ocr_rows(ocr_rows, student_df, historical_df)
    if not ocr_rows:
        result_warnings.append("OCR 文字中沒有解析出可用的學生修課列。請確認 OCR cache 是否更新，或改用更清楚的截圖。")
    if len(ocr_rows) != len(recognized):
        result_warnings.append("部分 OCR 候選看起來不像有效課號，已自動過濾。")
    return {
        "raw_ocr_text": text or "",
        "ocr_rows": ocr_rows,
        "recognized_courses": recognized,
        "ocr_confirmed_student_record": recognized,
        "matched_student_courses": recognized,
        "warnings": result_warnings,
        "confirmation_message": CONFIRMATION_MESSAGE,
        "student_path": str(student_path) if student_path else "",
        "historical_course_paths": list(historical_course_paths or []),
        "target_path": str(target_path) if target_path else "",
    }


def parse_course_screenshot(image_path: str, student_path: str, target_path: str) -> dict[str, Any]:
    # Kept for compatibility; OCR should normally be generated by ocr_preprocess_demo.py.
    warnings = ["建議先用 Python (OCR) kernel 產生 data/course_screenshot_ocr.txt，再用 parse_course_screenshot_from_cache。"]
    return _result_from_text("", student_path=student_path, target_path=target_path, warnings=warnings)


def parse_course_screenshot_from_cache(
    image_path: str,
    student_path: str | None = None,
    target_path: str | None = None,
    historical_course_paths: list[str] | None = None,
    force_reparse: bool = False,
) -> dict[str, Any]:
    cache_path = cache_path_for_image(image_path)
    existing_confirmed = pd.DataFrame() if force_reparse else _load_existing_confirmed_record(image_path)
    if not existing_confirmed.empty:
        records = _course_record_to_df(existing_confirmed).fillna("").to_dict("records")
        return {
            "raw_ocr_text": load_ocr_cache_if_available(image_path),
            "ocr_rows": [],
            "recognized_courses": records,
            "ocr_confirmed_student_record": records,
            "matched_student_courses": records,
            "warnings": ["已讀取既有 OCR confirmed CSV；若你剛更換截圖或 OCR cache，請重新跑 Demo 0 並覆蓋確認檔。"],
            "confirmation_message": CONFIRMATION_MESSAGE,
            "student_path": str(student_path) if student_path else "",
            "historical_course_paths": list(historical_course_paths or []),
            "target_path": str(target_path) if target_path else "",
            "image_path": str(image_path),
            "ocr_cache_path": str(cache_path),
            "used_cache": bool(cache_path.exists()),
            "used_existing_confirmed_csv": True,
        }

    text = load_ocr_cache_if_available(image_path)
    warnings: list[str] = []
    if not text:
        warnings.append("找不到 OCR cache，請先在 Python (OCR) kernel 執行 ocr_preprocess_demo.py 或 Google Vision OCR。")
    result = _result_from_text(
        text,
        student_path=student_path,
        target_path=target_path,
        warnings=warnings,
        historical_course_paths=historical_course_paths,
    )
    result.update({"image_path": str(image_path), "ocr_cache_path": str(cache_path), "used_cache": bool(text)})
    return result


def parse_ocr_text_for_demo(text: str, student_path: str, target_path: str) -> dict[str, Any]:
    return _result_from_text(text, student_path=student_path, target_path=target_path, warnings=["本次使用手貼 OCR 文字 fallback。"])


# ---------------------------------------------------------------------------
# Manual correction helpers
# ---------------------------------------------------------------------------

def _course_record_to_df(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    for col in DISPLAY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def display_ocr_confirmed_record(courses: list[dict[str, Any]] | pd.DataFrame) -> None:
    df = _course_record_to_df(courses)
    # Demo 0 confirms student history. Teacher/time/classroom are not required for
    # graduation checking and are often unavailable when the screenshot only contains
    # course code, credits, status, and grade. Hide them here to avoid making normal
    # missing schedule metadata look like an OCR failure.
    available = [col for col in OCR_CONFIRM_DISPLAY_COLUMNS if col in df.columns]
    _display_dataframe(df[available].fillna(""))


def _search_sources_for_term(term: str, query: str, student_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df in [student_df, historical_df]:
        if df is None or df.empty:
            continue
        temp = df.copy()
        if "term" in temp.columns:
            temp = temp[temp["term"].astype(str).str.replace(r"\.0$", "", regex=True).eq(term)]
        frames.append(temp)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).fillna("")
    q = _norm_text(query)
    q_code = normalize_ocr_code(query) or normalize_course_code(query)
    rows = []
    for _, row in df.iterrows():
        norm = normalize_ocr_code(row.get("normalized_course_code", "")) or normalize_course_code(row.get("normalized_course_code", ""))
        raw = clean_raw_course_code(row.get("raw_course_code", ""))
        fields = [row.get("course_name_zh", ""), row.get("course_name_en", ""), row.get("teacher", ""), raw, norm]
        if (q_code and q_code == norm) or (q and any(q in _norm_text(field) for field in fields)):
            rows.append(row.to_dict())
    result = pd.DataFrame(rows).drop_duplicates(subset=[c for c in ["raw_course_code", "normalized_course_code", "teacher", "time"] if c in pd.DataFrame(rows).columns])
    if result.empty:
        return result
    result = result.reset_index(drop=True)
    result.insert(0, "編號", range(1, len(result) + 1))
    return result


def _search_historical_paths_for_term(term: str, query: str, paths: list[str] | None, warnings: list[str]) -> pd.DataFrame:
    cached = _read_historical_cache(paths)
    if cached is not None and not cached.empty:
        cached_result = _search_sources_for_term(term, query, pd.DataFrame(), cached)
        if not cached_result.empty:
            return cached_result

    frames = []
    for path in paths or []:
        df = _safe_load_courses_for_term(path, term, warnings)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _search_sources_for_term(term, query, pd.DataFrame(), pd.concat(frames, ignore_index=True))


def _record_from_selected_historical(row: dict[str, Any], term: str, status: str, grade: str) -> dict[str, Any]:
    academic_year, semester = _split_term(term)
    code = normalize_ocr_code(row.get("normalized_course_code", "")) or normalize_ocr_code(row.get("raw_course_code", ""))
    raw = clean_raw_course_code(row.get("raw_course_code", "")) or code
    return {
        "term": term,
        "academic_year": academic_year,
        "semester": semester,
        "raw_course_code": raw,
        "normalized_course_code": code,
        "course_name_zh": _clean(row.get("course_name_zh")),
        "course_name_en": _clean(row.get("course_name_en")),
        "credits": _as_credit(row.get("credits"), ""),
        "status": status,
        "grade": "" if status == "in_progress" else grade,
        "teacher": _clean(row.get("teacher")),
        "time": _clean(row.get("time")),
        "classroom": _clean(row.get("classroom")),
        "needs_user_confirmation": False,
        "match_confidence": 1.0,
        "match_source": "manual_historical_selection",
    }


def _extract_query_after_keywords(message: str, keywords: list[str]) -> str:
    """Extract the course name/code part from a status/grade correction sentence.

    Examples:
    - 我要修正電路學的成績 -> 電路學
    - 修正大學中文 grade -> 大學中文
    - EE2260 的狀態 -> EE2260
    """
    text = _clean(message)
    for kw in keywords:
        text = re.sub(re.escape(kw), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"我要|我想|請|幫我|麻煩|修正|更正|更改|修改|改|調整", " ", text)
    text = re.sub(r"這門課|這堂課|該課|課程|科目|狀態|成績|status|grade", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[的之了呢啊喔啦]|[：:，,。；;？?！!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_manual_course_query(message: str) -> str:
    """Extract the actual course query from natural manual correction text.

    Examples:
    - 我要加機率 -> 機率
    - 漏了 113第二學期 機率 -> 機率
    - 幫我補上電子學二 -> 電子學二
    """
    text = _clean(message)
    text = re.sub(r"1\d{2}\s*(第?[一二12]學期|[12])?", " ", text)
    text = re.sub(r"我要|我想|請|幫我|麻煩|可以|一下", " ", text)
    text = re.sub(r"漏了|缺了|少了|新增|加入|加上|補上|補入|補|加|修|選|課程|一門|這門課", " ", text)
    text = re.sub(r"哪一年|哪一學期|第?[一二]學期|上學期|下學期", " ", text)
    text = re.sub(r"[：:，,。；;]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_status_update(message: str) -> bool:
    return any(token in message.lower() for token in ["status", "狀態"])


def _looks_like_grade_update(message: str) -> bool:
    return any(token in message.lower() for token in ["grade", "成績"])


def _select_working_row(query: str, working_df: pd.DataFrame) -> int | None:
    current_query = _clean(query)
    while True:
        # This search is intentionally limited to the current OCR confirmation table.
        # It should not search 114_2 or historical databases when the user says
        # Handle status or grade fixes.
        cleaned_query = _extract_query_after_keywords(current_query, ["status", "狀態", "grade", "成績"])
        if not cleaned_query:
            cleaned_query = _clean(current_query)
        q = _norm_text(cleaned_query)
        q_code = normalize_ocr_code(cleaned_query) or normalize_course_code(cleaned_query)
        candidates = []
        for idx, row in working_df.fillna("").iterrows():
            fields = [
                row.get("normalized_course_code", ""), row.get("raw_course_code", ""), row.get("course_name_zh", ""),
                row.get("course_name_en", ""), row.get("teacher", ""),
            ]
            norm = normalize_ocr_code(row.get("normalized_course_code", "")) or normalize_course_code(row.get("normalized_course_code", ""))
            raw_norm = normalize_ocr_code(row.get("raw_course_code", ""))
            field_norms = [_norm_text(field) for field in fields]
            if (q_code and q_code in {norm, raw_norm}) or (q and any(q in field for field in field_norms)):
                candidates.append(idx)
        if not candidates:
            current_query = input("在目前確認表找不到這門課，請重新輸入課名/課號（直接 Enter 取消）：").strip()
            if not current_query:
                return None
            continue
        if len(candidates) == 1:
            return int(candidates[0])
        display = working_df.loc[candidates].copy().reset_index().rename(columns={"index": "原始列"})
        display.insert(0, "編號", range(1, len(display) + 1))
        _display_dataframe(display[[c for c in ["編號", "normalized_course_code", "raw_course_code", "course_name_zh", "status", "grade"] if c in display.columns]].fillna(""))
        sel = input("找到多筆，請輸入要修正的編號：").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(candidates):
            return int(candidates[int(sel) - 1])


def _ask_status() -> str:
    while True:
        ans = input("這門課是進行中還是已修完畢？請輸入：進行中 / 已修完畢：").strip()
        lower = ans.lower().replace(" ", "_")
        if lower in {"in_progress", "inprogress", "ongoing"} or any(t in ans for t in ["進行中", "修課中", "還在進行", "正在修"]):
            return "in_progress"
        if lower in {"completed", "complete", "done", "finished"} or any(t in ans for t in ["已修完畢", "已修完", "修完了", "修完", "完成", "已修"]):
            return "completed"


def _ask_grade() -> str:
    while True:
        ans = input("這門課成績是及格還是不及格？請輸入：及格 / 不及格：").strip()
        lower = ans.lower()
        if lower in {"pass", "passed", "p"} or ans in {"及格", "通過", "過了"}:
            return "pass"
        if lower in {"fail", "failed", "f"} or ans in {"不及格", "沒過", "未通過"}:
            return "fail"


def _handle_status_update(message: str, working_df: pd.DataFrame, corrections: list[dict[str, Any]]) -> pd.DataFrame:
    query = _extract_query_after_keywords(message, ["status", "狀態"])
    idx = _select_working_row(query, working_df)
    if idx is None:
        return working_df
    term = _clean(working_df.at[idx, "term"])
    label = f"{working_df.at[idx, 'normalized_course_code']} {working_df.at[idx, 'course_name_zh']}"
    old_status = _clean(working_df.at[idx, "status"])
    old_grade = _clean(working_df.at[idx, "grade"])
    if term == CURRENT_IN_PROGRESS_TERM:
        working_df.at[idx, "status"] = "in_progress"
        working_df.at[idx, "grade"] = ""
        corrections.append({"operation": "auto_lock_current_term_status", "course": label, "old_status": old_status, "old_grade": old_grade, "new_status": "in_progress", "new_grade": ""})
        _display_markdown(f"`{label}` 是 114 第一學期課程，系統已固定為 `in_progress`，不需要另外修改 status。")
        return working_df
    new_status = _ask_status()
    working_df.at[idx, "status"] = new_status
    if new_status == "in_progress":
        working_df.at[idx, "grade"] = ""
    elif not _clean(working_df.at[idx, "grade"]):
        working_df.at[idx, "grade"] = "pass"
    corrections.append({"operation": "update_status", "course": label, "old_status": old_status, "old_grade": old_grade, "new_status": new_status, "new_grade": _clean(working_df.at[idx, "grade"])})
    _display_markdown(f"已更新 `{label}` 的 status。")
    return working_df


def _handle_grade_update(message: str, working_df: pd.DataFrame, corrections: list[dict[str, Any]]) -> pd.DataFrame:
    query = _extract_query_after_keywords(message, ["grade", "成績"])
    idx = _select_working_row(query, working_df)
    if idx is None:
        return working_df
    term = _clean(working_df.at[idx, "term"])
    label = f"{working_df.at[idx, 'normalized_course_code']} {working_df.at[idx, 'course_name_zh']}"
    old_status = _clean(working_df.at[idx, "status"])
    old_grade = _clean(working_df.at[idx, "grade"])
    if term == CURRENT_IN_PROGRESS_TERM:
        working_df.at[idx, "status"] = "in_progress"
        working_df.at[idx, "grade"] = ""
        corrections.append({"operation": "reject_current_term_grade", "course": label, "old_status": old_status, "old_grade": old_grade, "new_status": "in_progress", "new_grade": ""})
        _display_markdown(f"`{label}` 是 114 第一學期進行中課程，尚未有成績；系統已保留為 `in_progress`。")
        return working_df
    new_grade = _ask_grade()
    working_df.at[idx, "status"] = "completed"
    working_df.at[idx, "grade"] = new_grade
    corrections.append({"operation": "update_grade", "course": label, "old_status": old_status, "old_grade": old_grade, "new_status": "completed", "new_grade": new_grade})
    _display_markdown(f"已更新 `{label}` 的 grade。")
    return working_df


def _handle_missing_course_addition(query: str, working_df: pd.DataFrame, ocr_result: dict[str, Any], corrections: list[dict[str, Any]]) -> pd.DataFrame:
    term = _semester_display_to_code(query)
    course_query = _clean_manual_course_query(query)
    while not term:
        term_text = input("這門課是哪一年哪一學期？例如：113第二學期 / 1132 / 114第一學期：").strip()
        term = _semester_display_to_code(term_text)
        if not term:
            _display_markdown("無法辨識學期，請再輸入一次。")
    if term == TARGET_SEMESTER:
        _display_markdown("114 第二學期是目標排課學期，不能加入 OCR 修課紀錄。請在 Demo 1 用『我要修某某課』新增。")
        return working_df
    if not course_query:
        course_query = input("請輸入要補上的課名、課號或老師姓名：").strip()
        course_query = _clean_manual_course_query(course_query)
    warnings: list[str] = []
    student_df = _safe_load_student(ocr_result.get("student_path"), warnings)
    candidates = _search_historical_paths_for_term(term, course_query, ocr_result.get("historical_course_paths", []), warnings)
    if candidates.empty and not student_df.empty:
        candidates = _search_sources_for_term(term, course_query, student_df, pd.DataFrame())
    while candidates.empty:
        _display_markdown(f"在 {term} 找不到 `{course_query}`，請重新輸入課名/課號/老師姓名。")
        next_input = input("請輸入要補上的課名、課號或老師姓名（也可以輸入新的學期；直接 Enter 取消）：").strip()
        if not next_input:
            return working_df
        next_term = _semester_display_to_code(next_input)
        if next_term:
            term = next_term
            _display_markdown(f"已切換搜尋學期為 `{term}`，會繼續搜尋 `{course_query}`。")
        else:
            course_query = _clean_manual_course_query(next_input)
        candidates = _search_historical_paths_for_term(term, course_query, ocr_result.get("historical_course_paths", []), warnings)
        if candidates.empty and not student_df.empty:
            candidates = _search_sources_for_term(term, course_query, student_df, pd.DataFrame())
    cols = ["編號", "raw_course_code", "normalized_course_code", "course_name_zh", "teacher", "credits", "time", "classroom"]
    _display_dataframe(candidates[[c for c in cols if c in candidates.columns]].fillna(""))
    selection = input("請輸入要加入的編號或 raw_course_code：").strip()
    selected = None
    if selection.isdigit() and int(selection) in set(candidates["編號"]):
        selected = candidates[candidates["編號"] == int(selection)].iloc[0].to_dict()
    else:
        raw_selection = clean_raw_course_code(selection)
        raw_series = candidates.get("raw_course_code", pd.Series(dtype=str)).astype(str).map(clean_raw_course_code)
        matches = candidates[raw_series == raw_selection]
        if not matches.empty:
            selected = matches.iloc[0].to_dict()
    if selected is None:
        _display_markdown("沒有選到有效課程，取消這次新增。")
        return working_df
    if term == CURRENT_IN_PROGRESS_TERM:
        status, grade = "in_progress", ""
    else:
        grade = _ask_grade()
        status = "completed"
    record = _record_from_selected_historical(selected, term, status, grade)
    code = record.get("normalized_course_code", "")
    if code:
        working_df = working_df[working_df["normalized_course_code"].astype(str) != str(code)]
    working_df = pd.concat([working_df, pd.DataFrame([record])], ignore_index=True)
    corrections.append({"operation": "manual_add_missing_course", "course": code, "term": term, "raw_course_code": record.get("raw_course_code")})
    _display_markdown(f"已加入/更新 `{record.get('raw_course_code')}` `{code}` `{record.get('course_name_zh')}`。")
    return working_df


def _enforce_term_status_rules(working_df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic status rules for the OCR confirmation table.

    In this demo, 11410 is the current/in-progress semester. Therefore, any
    student record from 11410 should be treated as in_progress and should not
    contain a grade yet. 11420 is the target planning semester and should not be
    added as an OCR historical record.
    """
    if working_df.empty or "term" not in working_df.columns:
        return working_df
    mask = working_df["term"].astype(str).str.replace(r"\.0$", "", regex=True).eq(CURRENT_IN_PROGRESS_TERM)
    if mask.any():
        working_df.loc[mask, "status"] = "in_progress"
        working_df.loc[mask, "grade"] = ""
    return working_df


def interactive_confirm_ocr_record(
    ocr_result: dict[str, Any],
    target_path: str,
    output_path: str = "data/ocr_confirmed_student_courses.csv",
) -> dict[str, Any]:
    records = list(
        ocr_result.get("ocr_confirmed_student_record")
        or ocr_result.get("recognized_courses")
        or ocr_result.get("matched_student_courses")
        or []
    )
    working_df = _enforce_term_status_rules(_course_record_to_df(records))
    corrections: list[dict[str, Any]] = []
    _display_markdown("### OCR 修課紀錄確認")
    _display_markdown(ocr_result.get("confirmation_message") or CONFIRMATION_MESSAGE)
    if ocr_result.get("warnings"):
        _display_markdown("\n".join(f"- {w}" for w in ocr_result["warnings"]))

    while True:
        working_df = _enforce_term_status_rules(working_df)
        if "normalized_course_code" in working_df.columns:
            working_df = working_df.drop_duplicates(subset=["normalized_course_code"], keep="last").reset_index(drop=True)
        display_ocr_confirmed_record(working_df)
        _display_markdown(
            "如果都正確，請直接按 **Enter** 繼續。  \n"
            "如果有漏課，請輸入例如：`漏了機率`。  \n"
            "如果要修正狀態/成績，請輸入例如：`我要修正電子學二的狀態` 或 `我要修正電子學二的成績`。"
        )
        query = input("請輸入修正內容（直接 Enter 確認）：").strip()
        if not query:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            save_df = _course_record_to_df(working_df)
            save_cols = [c for c in SAVE_COLUMNS if c in save_df.columns]
            save_df[save_cols].to_csv(output, index=False, encoding="utf-8-sig")
            _display_markdown(f"已確認 OCR 修課紀錄，並儲存為 `{output}`。")
            display_ocr_confirmed_record(save_df)
            return {
                "confirmed": True,
                "output_path": str(output),
                "confirmed_courses": save_df[save_cols].fillna("").to_dict("records"),
                "corrections": corrections,
            }
        if _looks_like_grade_update(query):
            working_df = _handle_grade_update(query, working_df, corrections)
            continue
        if _looks_like_status_update(query):
            working_df = _handle_status_update(query, working_df, corrections)
            continue
        working_df = _handle_missing_course_addition(query, working_df, ocr_result, corrections)
