from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd


STANDARD_COLUMNS = [
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
]


COLUMN_ALIASES = {
    "semester": "semester",
    "term": "term",
    "學年期": "term",
    "學期": "semester",
    "academic_year": "academic_year",
    "學年": "academic_year",
    "course_code": "raw_course_code",
    "course code": "raw_course_code",
    "科號": "raw_course_code",
    "課號": "raw_course_code",
    "raw_course_code": "raw_course_code",
    "course_name": "course_name_zh",
    "中文課名": "course_name_zh",
    "課程名稱": "course_name_zh",
    "科目名稱": "course_name_zh",
    "英文課名": "course_name_en",
    "english_name": "course_name_en",
    "credits": "credits",
    "學分": "credits",
    "status": "status",
    "修課狀態": "status",
    "grade": "grade",
    "成績": "grade",
    "上課時間": "time",
    "time": "time",
    "教師": "teacher",
    "teacher": "teacher",
    "教室": "classroom",
    "限制條件": "restrictions",
    "擋修": "prerequisite_text",
    "此課程已列入之系所班別": "listed_departments",
    "課程屬性": "course_level",
    "代碼": "department_code",
}


def normalize_course_code(raw_code: str) -> str:
    """Normalize NTHU course identifiers to catalog codes like EE2255."""
    if raw_code is None or pd.isna(raw_code):
        return ""

    text = str(raw_code).strip().upper().replace("\u3000", " ")
    if not text or text in {"NAN", "NONE", "NULL"}:
        return ""

    text = re.sub(r"\.0$", "", text)
    text = text.replace("-", " ").replace("_", " ")

    # Full NTHU course number with term and section, e.g. 11420EE  214001.
    match = re.match(r"^\s*\d{5}\s*([A-Z]+)\s*0*([0-9]{4})(?:[0-9]{0,2})?\s*$", text)
    if match:
        return f"{match.group(1)}{match.group(2)}"

    # Catalog code with optional section suffix, e.g. EE 225500 or EE2255.
    match = re.search(r"([A-Z]{2,6})\s*0*([0-9]{4})(?:[0-9]{0,2})?", text)
    if match:
        return f"{match.group(1)}{match.group(2)}"

    compact = re.sub(r"[^A-Z0-9]", "", text)
    match = re.search(r"([A-Z]{2,6})0*([0-9]{4})(?:[0-9]{0,2})?", compact)
    if match:
        return f"{match.group(1)}{match.group(2)}"

    return compact


def _clean_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_term(value: Any) -> str:
    text = _clean_string(value)
    if not text:
        return ""
    text = re.sub(r"\.0$", "", text)
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 5:
        return digits[:5]
    return text


def _split_term(term: Any) -> tuple[str, str]:
    text = _normalize_term(term)
    if re.fullmatch(r"\d{5}", text):
        return text[:3], text[3:]
    return "", ""


def _normalize_status(value: Any) -> str:
    text = _clean_string(value).lower()
    if not text:
        return "unknown"
    completed = {"completed", "complete", "done", "passed", "pass", "已修", "已通過", "修畢"}
    in_progress = {"in_progress", "in progress", "taking", "current", "修課中", "正在修"}
    failed = {"failed", "fail", "not_passed", "not passed", "未通過", "不及格"}
    if text in completed:
        return "completed"
    if text in in_progress:
        return "in_progress"
    if text in failed:
        return "failed"
    return text


def _normalize_grade(value: Any) -> str:
    text = _clean_string(value)
    return text.lower()


def _detect_header_row(raw: pd.DataFrame) -> int | None:
    markers = {
        "course_code",
        "raw_course_code",
        "normalized_course_code",
        "科號",
        "課號",
        "course name",
        "course_name",
        "course_name_zh",
        "中文課名",
    }
    for idx, row in raw.iterrows():
        values = {str(v).strip().lower() for v in row.tolist() if not pd.isna(v)}
        if values & markers:
            return int(idx)
    return None


def _column_letters_to_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return max(0, index - 1)


def _read_xlsx_without_openpyxl(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Small stdlib-only xlsx reader for the MVP when HW2 lacks openpyxl."""
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                parts = [node.text or "" for node in si.findall(".//main:t", ns)]
                shared_strings.append("".join(parts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", ns)
        }
        sheets = []
        for sheet in workbook.findall("main:sheets/main:sheet", ns):
            rid = sheet.attrib.get(f"{{{ns['rel']}}}id", "")
            target = rel_targets.get(rid, "")
            if not target.startswith("xl/"):
                target = f"xl/{target.lstrip('/')}"
            sheets.append((sheet.attrib.get("name", ""), target))

        if isinstance(sheet_name, int):
            selected_name, selected_target = sheets[sheet_name]
        else:
            matches = [item for item in sheets if item[0] == str(sheet_name)]
            if not matches:
                raise ValueError(f"Sheet {sheet_name!r} not found in {path}")
            selected_name, selected_target = matches[0]

        worksheet = ET.fromstring(zf.read(selected_target))
        rows: list[list[Any]] = []
        max_col = 0
        for row in worksheet.findall("main:sheetData/main:row", ns):
            values: list[Any] = []
            for cell in row.findall("main:c", ns):
                cell_ref = cell.attrib.get("r", "")
                col_idx = _column_letters_to_index(cell_ref)
                while len(values) <= col_idx:
                    values.append(None)

                cell_type = cell.attrib.get("t", "")
                value_node = cell.find("main:v", ns)
                inline_node = cell.find("main:is/main:t", ns)
                if cell_type == "s" and value_node is not None:
                    raw_value = value_node.text or ""
                    value = shared_strings[int(raw_value)] if raw_value.isdigit() else raw_value
                elif cell_type == "inlineStr":
                    value = inline_node.text if inline_node is not None else ""
                elif value_node is None:
                    value = None
                else:
                    raw_value = value_node.text or ""
                    try:
                        number = float(raw_value)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw_value
                values[col_idx] = value
            max_col = max(max_col, len(values))
            rows.append(values)

        normalized_rows = [row + [None] * (max_col - len(row)) for row in rows]
        df = pd.DataFrame(normalized_rows)
        df.attrs["xlsx_reader"] = f"stdlib fallback ({selected_name})"
        return df


def _excel_sheet_names_without_openpyxl(path: str | Path) -> list[str]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    return [sheet.attrib.get("name", "") for sheet in workbook.findall("main:sheets/main:sheet", ns)]


def _read_excel_with_flexible_header(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        raw = _read_xlsx_without_openpyxl(path, sheet_name=sheet_name)
    header_row = _detect_header_row(raw)
    if header_row is None:
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except ImportError as exc:
            if "openpyxl" not in str(exc):
                raise
            return raw

    headers = []
    for i, value in enumerate(raw.iloc[header_row].tolist()):
        header = _clean_string(value)
        headers.append(header if header else f"unnamed_{i}")

    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    return df.reset_index(drop=True)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip()
        alias = COLUMN_ALIASES.get(key, COLUMN_ALIASES.get(key.lower()))
        if alias:
            rename_map[col] = alias
    df = df.rename(columns=rename_map)

    if "raw_course_code" not in df.columns:
        df["raw_course_code"] = ""

    df["raw_course_code"] = df["raw_course_code"].map(_clean_string)
    df["normalized_course_code"] = df["raw_course_code"].map(normalize_course_code)

    if "term" not in df.columns:
        if "semester" in df.columns:
            possible_terms = df["semester"].map(_normalize_term)
            if possible_terms.str.fullmatch(r"\d{5}").any():
                df["term"] = possible_terms
                df["semester"] = ""
            else:
                df["term"] = df["raw_course_code"].str.extract(r"^(\d{5})", expand=False).fillna("")
        else:
            df["term"] = df["raw_course_code"].str.extract(r"^(\d{5})", expand=False).fillna("")
    df["term"] = df["term"].map(_normalize_term)

    if "academic_year" not in df.columns:
        df["academic_year"] = ""
    if "semester" not in df.columns:
        df["semester"] = ""
    split_terms = df["term"].map(_split_term)
    df["academic_year"] = [
        existing or split[0] for existing, split in zip(df["academic_year"].map(_clean_string), split_terms)
    ]
    df["semester"] = [
        existing or split[1] for existing, split in zip(df["semester"].map(_clean_string), split_terms)
    ]

    if "course_name_zh" not in df.columns:
        df["course_name_zh"] = ""
    if "course_name_en" not in df.columns:
        df["course_name_en"] = ""
    df["course_name_zh"] = df["course_name_zh"].map(_clean_string)
    df["course_name_en"] = df["course_name_en"].map(_clean_string)

    if "credits" not in df.columns:
        df["credits"] = pd.NA
    df["credits"] = pd.to_numeric(df["credits"], errors="coerce")

    if "status" not in df.columns:
        df["status"] = "unknown"
    df["status"] = df["status"].map(_normalize_status)

    grade_column_missing = "grade" not in df.columns
    if grade_column_missing:
        df["grade"] = ""
    df["grade"] = df["grade"].map(_normalize_grade)
    df["grade_inferred"] = False
    completed_without_grade = (df["status"] == "completed") & (df["grade"] == "")
    df.loc[completed_without_grade, "grade"] = "pass"
    df.loc[completed_without_grade, "grade_inferred"] = True

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["is_completed_passed"] = (df["status"] == "completed") & df["grade"].isin(
        ["pass", "passed", "p", "a+", "a", "a-", "b+", "b", "b-", "c+", "c", "c-", "d", "及格", "通過"]
    )
    df["is_in_progress"] = df["status"] == "in_progress"

    blank_rows = (df["normalized_course_code"] == "") & (df["course_name_zh"] == "") & df["credits"].isna()
    df = df.loc[~blank_rows].reset_index(drop=True)

    warnings: list[str] = []
    if grade_column_missing:
        warnings.append("Grade column was not present. Rows marked completed are treated as pass for this MVP.")
    missing_status_codes = df.loc[df["status"] == "unknown", "normalized_course_code"].dropna().tolist()
    if missing_status_codes:
        warnings.append(
            "Some course rows have unknown status and are not counted unless corrected: "
            + ", ".join(code for code in missing_status_codes if code)
        )
    df.attrs["warnings"] = warnings
    return df


def _read_student_table(path: str | Path) -> pd.DataFrame:
    """Read a student record from CSV or Excel.

    Demo 0 writes OCR-confirmed student records as CSV, while the original
    baseline student record is often XLSX.  CoursePlanningAgent passes either
    file into load_student_courses(), so this loader must support both.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return _read_excel_with_flexible_header(path)

    # Fallback for paths without a clear suffix. Try CSV first because the OCR
    # confirmed file is usually a CSV; then try Excel.
    try:
        return pd.read_csv(path)
    except Exception:
        return _read_excel_with_flexible_header(path)


def load_student_courses(path: str) -> pd.DataFrame:
    """Load the student's completed and in-progress courses from CSV or Excel."""
    df = _read_student_table(path)
    df = _standardize_columns(df)
    return df


def _looks_like_course_sheet(df: pd.DataFrame) -> bool:
    columns = {str(col).strip() for col in df.columns}
    return bool(columns & {"科號", "課號", "raw_course_code", "course_code"})


def _ensure_target_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["time", "teacher", "classroom", "course_level", "listed_departments", "prerequisite_text"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_clean_string)
    return df


def _candidate_sheet_names(sheet_names: list[str]) -> list[str]:
    candidates: list[str] = []
    skip_tokens = ["日期", "data日期", "教師data", "遠距data", "停修後", "加退選截止選課資料庫"]
    for name in sheet_names:
        name_text = str(name)
        if any(token in name_text for token in skip_tokens):
            continue
        candidates.append(name)
    return candidates or [sheet_names[0]]


def _fill_missing_from_same_code(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill descriptive fields from other offerings of the same catalog code.

    Some historical workbook sheets contain only Chinese names and enrollment
    metadata. This fills missing English names/credits for display only. It does
    not fill or override term/status/grade.
    """
    if df.empty or "normalized_course_code" not in df.columns:
        return df
    df = df.copy()
    for col in ["course_name_zh", "course_name_en"]:
        if col not in df.columns:
            df[col] = ""
    if "credits" not in df.columns:
        df["credits"] = pd.NA

    for code, group in df.groupby("normalized_course_code", dropna=False):
        if not _clean_string(code):
            continue
        for col in ["course_name_zh", "course_name_en"]:
            values = group.loc[group[col].map(_clean_string) != "", col]
            if values.empty:
                continue
            value = values.iloc[0]
            missing = df.index.isin(group.index) & (df[col].map(_clean_string) == "")
            df.loc[missing, col] = value
        values = group["credits"].dropna()
        if not values.empty:
            value = values.iloc[0]
            missing = df.index.isin(group.index) & df["credits"].isna()
            df.loc[missing, "credits"] = value
    return df


def load_target_courses(path: str) -> pd.DataFrame:
    """Load course data from Excel and normalize key columns.

    This version reads all likely course sheets and concatenates them. It is
    necessary for historical databases such as 111-113 _course_data.xlsx where
    different semesters are stored in different sheets.
    """
    try:
        excel = pd.ExcelFile(path)
        sheet_names = excel.sheet_names
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        sheet_names = _excel_sheet_names_without_openpyxl(path)

    frames: list[pd.DataFrame] = []
    for sheet_name in _candidate_sheet_names(sheet_names):
        try:
            raw_df = _read_excel_with_flexible_header(path, sheet_name=sheet_name)
        except Exception:
            continue
        if raw_df.empty or not _looks_like_course_sheet(raw_df):
            continue
        df = _standardize_columns(raw_df)
        if re.fullmatch(r"\d{5}", str(sheet_name)):
            blank_term = df["term"].map(_clean_string) == ""
            df.loc[blank_term, "term"] = str(sheet_name)
            split_terms = df["term"].map(_split_term)
            df["academic_year"] = [
                existing or split[0]
                for existing, split in zip(df["academic_year"].map(_clean_string), split_terms)
            ]
            df["semester"] = [
                existing or split[1]
                for existing, split in zip(df["semester"].map(_clean_string), split_terms)
            ]
        df = _ensure_target_extra_columns(df)
        df["source_sheet"] = str(sheet_name)
        frames.append(df)

    if not frames:
        sheet_name = sheet_names[0]
        df = _read_excel_with_flexible_header(path, sheet_name=sheet_name)
        df = _ensure_target_extra_columns(_standardize_columns(df))
        df["source_sheet"] = str(sheet_name)
        return df.loc[df["normalized_course_code"] != ""].reset_index(drop=True)

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = _fill_missing_from_same_code(df)
    df = df.loc[df["normalized_course_code"] != ""].reset_index(drop=True)
    return df
