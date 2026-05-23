from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from calendar_exporter import export_schedule_to_ics
from course_agent import CoursePlanningAgent
from course_data_loader import load_target_courses
from ocr_preprocess_demo import DEFAULT_GOOGLE_VISION_KEY_PATH, run_ocr
from ocr_screenshot_parser import (
    CURRENT_IN_PROGRESS_TERM,
    SAVE_COLUMNS,
    TARGET_SEMESTER,
    _clean_manual_course_query,
    _extract_query_after_keywords,
    _record_from_selected_historical,
    _safe_load_student,
    _search_historical_paths_for_term,
    _search_sources_for_term,
    _semester_display_to_code,
    cache_path_for_image,
    clean_raw_course_code,
    normalize_ocr_code,
    parse_course_screenshot_from_cache,
)
from schedule_checker import parse_time_slots


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_STUDENT_PATH = DATA_DIR / "ocr_confirmed_student_courses.csv"
TARGET_PATH = DATA_DIR / "114_2_course_data.xlsx"
RULES_PATH = DATA_DIR / "rules" / "EE_112_rules.json"
ICS_PATH = ROOT / "final_schedule.ics"
GEMINI_API_KEY_PATH = ROOT / "private" / "gemini_api_key.txt"
HISTORICAL_COURSE_PATHS = [
    DATA_DIR / "111-113 _course_data.xlsx",
    DATA_DIR / "114_1_course_data.xlsx",
    DATA_DIR / "114_2_course_data.xlsx",
]

DAY_LABELS = [
    ("M", "一"),
    ("T", "二"),
    ("W", "三"),
    ("R", "四"),
    ("F", "五"),
    ("S", "六"),
    ("U", "日"),
]
DAY_LABELS_EN = [
    ("M", "Mon"),
    ("T", "Tue"),
    ("W", "Wed"),
    ("R", "Thu"),
    ("F", "Fri"),
    ("S", "Sat"),
    ("U", "Sun"),
]
PERIODS = ["1", "2", "3", "4", "n", "5", "6", "7", "8", "9", "a", "b", "c", "d"]
PERIOD_LABELS = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "n": "n",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "a": "a",
    "b": "b",
    "c": "c",
    "d": "d",
}
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
SLOT_COLORS = [
    ("#8ec5ff", "#1d3557"),
    ("#ffb86b", "#3b2a18"),
    ("#7dd3fc", "#123044"),
    ("#c4b5fd", "#2b2350"),
    ("#86efac", "#173a28"),
    ("#fca5a5", "#4a1f24"),
    ("#fde68a", "#3b3215"),
    ("#93c5fd", "#182f56"),
    ("#f0abfc", "#3a1f48"),
    ("#5eead4", "#153c38"),
    ("#fdba74", "#42240f"),
    ("#a7f3d0", "#153829"),
]
TIMETABLE_COLOR_MODE_MAP = {
    "依課程": "course",
    "依學分": "credits",
    "依系所": "department",
}
TIMETABLE_COLOR_MODE_LABELS = {
    "zh": {"course": "依課程", "credits": "依學分", "department": "依系所"},
    "en": {"course": "By course", "credits": "By credits", "department": "By department"},
}
DEPARTMENT_COLOR_ORDER = [
    "EE",
    "EECS",
    "CS",
    "GE",
    "GEC",
    "PE",
    "LANG",
    "MATH",
    "PHYS",
    "CHE",
    "CHEM",
    "BMES",
    "AIA",
    "ANTH",
    "OTHER",
]


def ensure_language_state() -> None:
    st.session_state.setdefault("ui_language", "zh")


def ui_language() -> str:
    ensure_language_state()
    return str(st.session_state.get("ui_language") or "zh")


def is_english() -> bool:
    return ui_language() == "en"


def txt(zh: str, en: str) -> str:
    return en if is_english() else zh


def day_labels() -> list[tuple[str, str]]:
    return DAY_LABELS_EN if is_english() else [("M", "週一"), ("T", "週二"), ("W", "週三"), ("R", "週四"), ("F", "週五"), ("S", "週六"), ("U", "週日")]


def color_mode_options() -> list[str]:
    labels = TIMETABLE_COLOR_MODE_LABELS["en" if is_english() else "zh"]
    return [labels["course"], labels["credits"], labels["department"]]


def color_mode_value(label: str) -> str:
    active = TIMETABLE_COLOR_MODE_LABELS["en" if is_english() else "zh"]
    for value, localized_label in active.items():
        if label == localized_label:
            return value
    return "course"


def read_private_gemini_key() -> str:
    if not GEMINI_API_KEY_PATH.exists():
        return ""
    return GEMINI_API_KEY_PATH.read_text(encoding="utf-8", errors="ignore").strip()


def ensure_gemini_api_key() -> str:
    existing = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if existing.strip():
        return existing.strip()
    private_key = read_private_gemini_key()
    if private_key:
        os.environ["GEMINI_API_KEY"] = private_key
    return private_key


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #0f141c;
          --sidebar: #272833;
          --sidebar-deep: #121721;
          --panel: #181d27;
          --panel-soft: #202633;
          --panel-raised: #f8fafc;
          --ink: #f7f9fc;
          --ink-dark: #172033;
          --muted: #a6b0c2;
          --muted-dark: #637083;
          --line: #3a4251;
          --line-soft: #2a313d;
          --accent: #ff4d57;
          --accent-2: #ff9f1c;
          --green: #42d392;
          --amber: #ffd166;
        }
        html, body,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
          background: var(--bg) !important;
          color: var(--ink) !important;
        }
        [data-testid="stHeader"] {
          border-bottom: 0;
          height: 2.75rem !important;
        }
        #MainMenu,
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stDeployButton"] {
          display: none !important;
          visibility: hidden !important;
          height: 0 !important;
        }
        header,
        [data-testid="stHeader"] {
          background: transparent !important;
          color: var(--ink) !important;
          display: block !important;
          visibility: visible !important;
          opacity: 1 !important;
          z-index: 999999 !important;
        }
        [data-testid="stToolbar"] {
          visibility: visible !important;
          opacity: 1 !important;
        }
        [data-testid="stHeader"] button,
        header button,
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapsedControl"] *,
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] *,
        button[kind="header"] {
          display: inline-flex !important;
          visibility: visible !important;
          opacity: 1 !important;
          pointer-events: auto !important;
        }
        .block-container {
          padding-top: 2.2rem;
          padding-bottom: 3rem;
          max-width: 1440px;
        }
        h1, h2, h3, h4, h5, h6, p, li, label, span {
          color: var(--ink);
          letter-spacing: 0;
        }
        div[data-testid="stMarkdownContainer"] {
          color: var(--ink) !important;
        }
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        section[data-testid="stSidebar"] > div,
        div[data-testid="stSidebar"] > div {
          background: var(--sidebar) !important;
          color: var(--ink) !important;
          border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] *,
        div[data-testid="stSidebar"] * {
          color: var(--ink) !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] label p,
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] label p {
          color: #f7f9fc !important;
          opacity: 1 !important;
        }
        section[data-testid="stSidebar"] input,
        div[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        div[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        div[data-testid="stSidebar"] [data-baseweb="select"] > div {
          background: #0d1118 !important;
          color: #ffffff !important;
          border-color: #3b4350 !important;
          box-shadow: none !important;
        }
        section[data-testid="stSidebar"] input:disabled,
        div[data-testid="stSidebar"] input:disabled {
          color: #b8c2d2 !important;
          -webkit-text-fill-color: #b8c2d2 !important;
          opacity: 1 !important;
        }
        section[data-testid="stSidebar"] input::placeholder,
        div[data-testid="stSidebar"] input::placeholder,
        textarea::placeholder,
        input::placeholder {
          color: #7c8798 !important;
          opacity: 1 !important;
        }
        div[data-testid="stSidebar"] .stButton > button {
          background: #151b26 !important;
          color: #fff !important;
          border: 1px solid #565d6d !important;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
          border-color: #ffffff !important;
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"],
        div[data-testid="stSidebar"] [data-testid="stExpander"] {
          background: #151b26 !important;
          border-color: #485266 !important;
        }
        .sidebar-title {
          font-size: .8rem;
          color: #aeb7c8;
          font-weight: 800;
          margin: 1.2rem 0 .45rem;
          text-transform: uppercase;
        }
        .recent-list {
          display: flex;
          flex-direction: column;
          gap: .15rem;
          margin-top: .25rem;
        }
        .recent-item {
          color: #eef2ff;
          font-size: .84rem;
          line-height: 1.35;
          padding: .38rem .2rem;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          border-bottom: 1px solid rgba(255,255,255,.04);
        }
        .app-hero {
          border-bottom: 1px solid var(--line);
          padding: 0 0 1rem 0;
          margin-bottom: 1.15rem;
        }
        .app-title {
          font-size: 2.2rem;
          font-weight: 800;
          color: var(--ink);
          margin: 0;
        }
        .app-subtitle {
          color: var(--muted);
          margin-top: .25rem;
          font-size: .98rem;
        }
        .chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: .45rem;
          margin-top: .75rem;
        }
        .chip {
          display: inline-flex;
          align-items: center;
          border: 1px solid var(--line);
          background: #f8fafc;
          color: #172033;
          padding: .25rem .55rem;
          border-radius: 999px;
          font-size: .8rem;
          line-height: 1.2;
        }
        div[data-testid="stButton"] > button {
          background: #151b26 !important;
          color: #f8fafc !important;
          border: 1px solid #485266 !important;
          border-radius: 8px;
          min-height: 2.55rem;
        }
        div[data-testid="stButton"] > button:hover {
          border-color: #f8fafc !important;
          color: #ffffff !important;
        }
        div[data-testid="stButton"] > button:disabled {
          color: #7e8797 !important;
          background: #131821 !important;
          border-color: #2d3441 !important;
        }
        div[data-testid="stExpander"] {
          background: #111720;
          border: 1px solid var(--line) !important;
          border-radius: 8px;
          color: var(--ink);
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] li,
        div[data-testid="stExpander"] td,
        div[data-testid="stExpander"] th {
          color: var(--ink);
        }
        div[data-testid="stChatMessage"] {
          background: transparent;
        }
        div[data-testid="stChatInput"],
        div[data-testid="stChatInput"] > div:first-child {
          background: transparent !important;
        }
        div[data-testid="stChatInput"] > div > div {
          background: #f8fafc !important;
          border: 1px solid #ccd5e3 !important;
          border-radius: 8px !important;
          box-shadow: none !important;
        }
        div[data-testid="stChatInput"] [data-baseweb="textarea"],
        div[data-testid="stChatInput"] [data-baseweb="textarea"] > div,
        div[data-testid="stChatInput"] textarea {
          background: #f8fafc !important;
          color: #172033 !important;
          border: 0 !important;
          box-shadow: none !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
          color: #64748b !important;
          opacity: 1 !important;
        }
        div[data-testid="stChatInput"] button {
          background: #e8eef7 !important;
          color: #172033 !important;
          border: 1px solid #d7e0ec !important;
        }
        div[data-baseweb="tab-list"] {
          border-bottom: 1px solid var(--line);
        }
        div[data-baseweb="tab"] {
          color: #c3cada;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
          color: #ff5964;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .65rem;
          margin-bottom: .75rem;
        }
        .metric-card {
          background: var(--panel-raised);
          border: 1px solid #dbe3ef;
          border-radius: 8px;
          padding: .75rem .85rem;
        }
        .metric-label {
          color: #566274;
          font-size: .78rem;
          margin-bottom: .2rem;
        }
        .metric-value {
          color: #101827;
          font-size: 1.45rem;
          font-weight: 760;
        }
        .metric-note {
          color: #64748b;
          font-size: .74rem;
          margin-top: .2rem;
        }
        .status-good { color: var(--green); }
        .status-warn { color: var(--amber); }
        .course-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: .65rem;
        }
        .course-card {
          border: 1px solid #dbe3ef;
          background: #f8fafc;
          border-radius: 8px;
          padding: .75rem;
          min-height: 128px;
        }
        .course-code {
          color: var(--accent);
          font-size: .78rem;
          font-weight: 740;
          margin-bottom: .25rem;
        }
        .course-title {
          color: #172033;
          font-weight: 760;
          margin-bottom: .45rem;
        }
        .course-meta {
          color: #475569;
          font-size: .82rem;
          line-height: 1.45;
        }
        .course-reason {
          color: #5c6677;
          font-size: .76rem;
          line-height: 1.45;
          margin-top: .45rem;
        }
        .timetable {
          width: 100%;
          min-width: 760px;
          border-collapse: collapse;
          table-layout: fixed;
          font-size: .76rem;
          background: #0f141c !important;
        }
        .timetable th, .timetable td {
          border: 1px solid #2f3744;
          vertical-align: top;
          padding: .38rem;
          background: #111720 !important;
          color: #e8eef7 !important;
        }
        .timetable th {
          background: #202838 !important;
          color: #f8fafc !important;
          text-align: center;
          font-weight: 760;
        }
        .timetable tr:nth-child(even) td {
          background: #151b26 !important;
        }
        .period-cell {
          width: 4.2rem;
          color: #f8fafc !important;
          background: #202838 !important;
          font-weight: 700;
        }
        .slot-card {
          border-left: 4px solid var(--slot-accent, #8ec5ff);
          background: var(--slot-bg, #202633) !important;
          border-top: 1px solid color-mix(in srgb, var(--slot-accent, #8ec5ff) 38%, #334055);
          border-right: 1px solid color-mix(in srgb, var(--slot-accent, #8ec5ff) 28%, #334055);
          border-bottom: 1px solid color-mix(in srgb, var(--slot-accent, #8ec5ff) 28%, #334055);
          border-radius: 6px;
          padding: .3rem .35rem;
          margin-bottom: .25rem;
          color: #f7f9fc !important;
          line-height: 1.28;
          box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
        }
        .slot-code {
          font-weight: 760;
          font-size: .72rem;
          color: #ffffff !important;
        }
        .slot-name {
          font-size: .72rem;
          color: #e8eef7 !important;
        }
        .slot-time {
          color: #b8c2d2 !important;
          font-size: .68rem;
          margin-top: .15rem;
        }
        .empty-note {
          border: 1px dashed #536174;
          color: #c3cada;
          background: #111720;
          padding: .8rem;
          border-radius: 8px;
        }
        .compact-reply {
          background: #f8fafc;
          color: #172033;
          border: 1px solid #dbe3ef;
          border-radius: 8px;
          padding: .75rem .85rem;
          line-height: 1.55;
        }
        .compact-reply * {
          color: #172033;
        }
        .chat-user {
          background: #1b202b;
          color: #ffffff;
          border-radius: 8px;
          padding: .8rem 1rem;
          border: 1px solid #252d3b;
          line-height: 1.5;
        }
        .small-muted {
          color: var(--muted);
          font-size: .8rem;
        }
        .main-split-layout-anchor {
          height: 0;
          overflow: hidden;
        }
        .resizable-main-layout {
          display: grid !important;
          grid-template-columns: minmax(360px, var(--left-width, 52%)) minmax(460px, 1fr) !important;
          gap: 1.45rem !important;
          align-items: start;
          position: relative;
        }
        .resizable-main-layout > div[data-testid="column"] {
          width: auto !important;
          min-width: 0 !important;
        }
        .resizable-main-layout .splitter-handle {
          position: absolute;
          top: 0;
          bottom: 0;
          width: 16px;
          border-left: 1px solid rgba(166, 176, 194, .45);
          cursor: col-resize;
          z-index: 20;
        }
        .resizable-main-layout .splitter-handle::after {
          content: "";
          position: absolute;
          left: 5px;
          top: 46%;
          width: 5px;
          height: 46px;
          border-radius: 999px;
          background: rgba(166, 176, 194, .45);
        }
        .resizable-main-layout.is-dragging,
        .resizable-main-layout.is-dragging * {
          cursor: col-resize !important;
          user-select: none !important;
        }
        .timetable-wrap {
          overflow-x: auto;
          padding-bottom: .3rem;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) {
          width: 100%;
          border-collapse: collapse;
          background: #111720 !important;
          color: #f7f9fc !important;
          margin: .8rem 0 1rem;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) th,
        div[data-testid="stMarkdownContainer"] table:not(.timetable) td {
          background: #111720 !important;
          color: #f7f9fc !important;
          border: 1px solid #334055 !important;
          padding: .55rem .65rem !important;
          vertical-align: top;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) th {
          background: #202838 !important;
          color: #ffffff !important;
          font-weight: 760;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) tr:nth-child(even) td {
          background: #151b26 !important;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) code {
          color: #32d583 !important;
          background: rgba(50,213,131,.12) !important;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) tr,
        div[data-testid="stExpander"] table:not(.timetable) tr {
          background: #111720 !important;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) *,
        div[data-testid="stExpander"] table:not(.timetable) * {
          color: #f7f9fc !important;
          text-shadow: none !important;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) td div,
        div[data-testid="stExpander"] table:not(.timetable) td div,
        div[data-testid="stMarkdownContainer"] table:not(.timetable) td span,
        div[data-testid="stExpander"] table:not(.timetable) td span {
          color: #f7f9fc !important;
          background: transparent !important;
        }
        div[data-testid="stMarkdownContainer"] table:not(.timetable) td div[style*="background"],
        div[data-testid="stExpander"] table:not(.timetable) td div[style*="background"],
        div[data-testid="stMarkdownContainer"] table:not(.timetable) td span[style*="background"],
        div[data-testid="stExpander"] table:not(.timetable) td span[style*="background"] {
          color: #f7f9fc !important;
          background: #202633 !important;
          border: 1px solid #334055 !important;
        }
        /* Readability guard: keep contrast stable even when Streamlit's theme menu changes. */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] :where(h1, h2, h3, h4, h5, h6, p, li, label, span, small, div),
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] :where(h1, h2, h3, h4, h5, h6, p, li, label, span, small, div) {
          color: #f7f9fc !important;
        }
        section[data-testid="stSidebar"],
        aside,
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
          background: #272833 !important;
          color: #f7f9fc !important;
        }
        section[data-testid="stSidebar"] :where(h1, h2, h3, h4, h5, h6, p, li, label, span, small, div),
        [data-testid="stSidebar"] :where(h1, h2, h3, h4, h5, h6, p, li, label, span, small, div) {
          color: #f7f9fc !important;
          opacity: 1 !important;
        }
        .chip,
        .chip *,
        .metric-card,
        .metric-card *,
        .course-card,
        .course-card *,
        .compact-reply,
        .compact-reply *,
        .timetable th,
        .timetable th *,
        .period-cell,
        .period-cell *,
        div[data-testid="stChatInput"] *,
        div[data-testid="stAlert"],
        div[data-testid="stAlert"] * {
          color: #172033 !important;
          text-shadow: none !important;
        }
        .metric-card .metric-label,
        .metric-card .metric-note,
        .course-card .course-meta,
        .course-card .course-reason,
        .compact-reply .small-muted {
          color: #566274 !important;
        }
        .timetable,
        .timetable *,
        .slot-card,
        .slot-card * {
          color: #f7f9fc !important;
          text-shadow: none !important;
        }
        .timetable thead th,
        .timetable thead th *,
        .timetable th.period-cell,
        .timetable th.period-cell * {
          color: #f8fafc !important;
          -webkit-text-fill-color: #f8fafc !important;
          opacity: 1 !important;
        }
        .slot-time,
        .slot-card .slot-time {
          color: #b8c2d2 !important;
        }
        .metric-card .status-good,
        .metric-card .status-good * {
          color: #0f9f64 !important;
        }
        .metric-card .status-warn,
        .metric-card .status-warn * {
          color: #9a6700 !important;
        }
        div[data-testid="stAlert"] {
          background: #f8fafc !important;
          border: 1px solid #dbe3ef !important;
        }
        div[data-testid="stAlert"] svg {
          color: #172033 !important;
          fill: currentColor !important;
        }
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] *,
        [data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] div[data-testid="stCaptionContainer"] * {
          color: #aeb7c8 !important;
        }
        input,
        textarea,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] * {
          color: #172033 !important;
          -webkit-text-fill-color: #172033 !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] [data-baseweb="select"] * {
          color: #f7f9fc !important;
          -webkit-text-fill-color: #f7f9fc !important;
        }
        [data-baseweb="popover"] *,
        [role="listbox"] *,
        [data-baseweb="menu"] * {
          color: #172033 !important;
          -webkit-text-fill-color: #172033 !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
          background: #f8fafc !important;
          color: #172033 !important;
          border: 1px solid #ccd5e3 !important;
          border-radius: 8px !important;
          box-shadow: none !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] *,
        div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
          color: #172033 !important;
          fill: currentColor !important;
          -webkit-text-fill-color: #172033 !important;
          opacity: 1 !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [data-baseweb="menu"],
        [data-baseweb="popover"] ul {
          background: #f8fafc !important;
          border-color: #dbe3ef !important;
        }
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="popover"] [role="option"] *,
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] li * {
          background: #f8fafc !important;
          color: #172033 !important;
          -webkit-text-fill-color: #172033 !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] li:hover {
          background: #e8eef7 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
          background: #0d1118 !important;
          border-color: #3b4350 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] * {
          color: #f7f9fc !important;
          -webkit-text-fill-color: #f7f9fc !important;
        }
        code {
          color: #32d583 !important;
          background: rgba(248,250,252,.08) !important;
          border-radius: 4px;
          padding: .05rem .22rem;
        }
        .compact-reply code,
        .course-card code,
        .metric-card code,
        .slot-card code,
        .chip code,
        div[data-testid="stAlert"] code {
          color: #07845f !important;
          background: #eef6f0 !important;
        }
        @media (max-width: 900px) {
          .metric-grid, .course-grid {
            grid-template-columns: 1fr;
          }
          .resizable-main-layout {
            display: block !important;
          }
          .resizable-main-layout .splitter-handle {
            display: none;
          }
          .timetable {
            font-size: .68rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def h(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def field(course: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = course.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def course_code(course: dict[str, Any]) -> str:
    return field(course, "code", "normalized_course_code", "raw_course_code")


def course_name(course: dict[str, Any]) -> str:
    if is_english():
        return field(course, "course_name_en", "name_en", "course_name_zh", "name_zh", "name")
    return field(course, "course_name_zh", "name_zh", "course_name_en", "name")


def course_time(course: dict[str, Any]) -> str:
    return field(course, "time", "class_time", "schedule", "上課時間")


def course_teacher(course: dict[str, Any]) -> str:
    return field(course, "teacher", "instructor", "teacher_zh", "授課教師")


def course_credits(course: dict[str, Any]) -> str:
    return field(course, "credits", "credit", "學分")


def course_label(course: dict[str, Any]) -> str:
    return f"{course_code(course)} {course_name(course)}".strip()


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def clean_review_text(text: object, limit: int = 420) -> str:
    cleaned = str(text or "").strip().strip("'\"")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:limit].strip()


def translate_review_comment_for_english(text: object) -> str:
    """Return a concise English note for Chinese review evidence."""
    cleaned = clean_review_text(text)
    if not cleaned:
        return "No written comment was available."
    if not contains_cjk(cleaned):
        return cleaned

    notes: list[str] = []
    if "涼度" in cleaned:
        notes.append("The review includes an explicit workload/easiness rating.")
    if "甜度" in cleaned:
        notes.append("The review includes an explicit grading-generosity rating.")
    if "作業" in cleaned:
        notes.append("Homework is discussed, often as textbook or chapter-based exercises.")
    if "考試" in cleaned or "段考" in cleaned or "期中" in cleaned or "期末" in cleaned:
        notes.append("Exams are discussed, including how closely they relate to homework or textbook problems.")
    if "課本" in cleaned or "投影片" in cleaned or "板書" in cleaned:
        notes.append("The teaching materials or lecture style are mentioned.")
    if "老師人很好" in cleaned or "問問題" in cleaned or "互動" in cleaned:
        notes.append("The instructor is described as approachable or willing to answer questions.")
    if "錄音" in cleaned or "遠距" in cleaned:
        notes.append("The review mentions online/recorded lecture quality.")
    if "沒有考試" in cleaned:
        notes.append("The course is described as having no exams.")
    if not notes:
        notes.append("This Chinese review comments on workload, grading, assignments, exams, or teaching style.")
    return " ".join(dict.fromkeys(notes))


def english_review_course_title(result: dict[str, Any]) -> str:
    zh_name = str(result.get("course_name") or "Course").strip()
    for section in result.get("course_sections") or []:
        if not isinstance(section, dict):
            continue
        en_name = str(section.get("course_name_en") or "").strip()
        if en_name:
            return f"{en_name} ({zh_name})" if zh_name and zh_name != en_name else en_name
    return zh_name or "Course"


def english_reason_text(reason: object) -> str:
    text = str(reason or "").strip()
    lower = text.lower()
    if not text:
        return ""
    if lower.startswith("missing required course"):
        code = text.split(":", 1)[-1].strip()
        return f"Fills a missing required course requirement: {code}."
    if lower.startswith("missing alternative requirement"):
        name = text.split(":", 1)[-1].strip()
        return f"Fills an alternative required-course requirement: {name}."
    if "補 ee112 必修缺口" in lower:
        return "Fills a missing EE112 required-course requirement."
    if "補機率替代必修" in text:
        return "Fills the probability alternative required-course requirement."
    if "必選實驗" in text or "實驗課" in text and "EE-first" in text:
        return "Helps satisfy the required lab-elective requirement."
    if "EE-first" in text or "電機實驗課" in text:
        return "Adds one EE lab course under the default EE-first planning strategy."
    if "GE/GEC" in text or "通識" in text:
        return "Fills remaining credits with GE/GEC general education courses."
    if "體育" in text or lower.startswith("optional physical education"):
        return "Optional physical education course for schedule balance; it does not add graduation credits."
    if "user requested" in lower:
        return text
    if "required_lab_electives" in lower:
        return "Helps satisfy the required lab-elective requirement."
    if "other_electives" in lower:
        return "Fills remaining elective credits."
    return text


def english_warning_text(warning: object) -> str:
    text = str(warning or "").strip()
    if not text:
        return ""
    if "Courses currently in progress are treated as expected-to-pass" in text:
        return "In-progress courses are treated as expected-to-pass for planning."
    if "目前規劃低於正常最低學分" in text or "low-credit-load" in text:
        return "The plan is below the normal minimum credit load and may require approval."
    if "正常學分" in text:
        return "The plan is within the normal credit-load range."
    if "初始課表採用 EE-first" in text:
        return "The initial schedule uses an EE-first strategy, then fills remaining credits with suitable electives."
    if "其餘選修補學分" in text:
        return "Remaining credits are filled with GE/GEC courses first; other non-technical fillers are limited."
    if "體育課" in text or "physical education course" in text:
        return "A PE course was added for balance; it contributes 0 graduation credits."
    if "EE2255" in text:
        return "EE2255 is temporarily treated as satisfied by completed related electronics courses."
    if "MVP" in text or "official graduation audit" in text:
        return "This MVP is not an official graduation audit; final eligibility must be confirmed by the department office."
    if "Courses currently in progress are only counted" in text:
        return "If any in-progress course is not passed, the graduation progress must be recalculated."
    if "Online course reviews" in text:
        return "Online course reviews are subjective soft references and may be biased or outdated."
    return text


def format_time_slot_constraint(slots: list[str]) -> str:
    slot_set = {str(slot).strip() for slot in slots if str(slot).strip()}
    day_codes = [code for code, _label in DAY_LABELS]
    labels: list[str] = []
    consumed: set[str] = set()
    for period in PERIODS:
        period_slots = {f"{day}{period}" for day in day_codes}
        if period_slots and period_slots.issubset(slot_set):
            labels.append(PERIOD_TIMES.get(period, period))
            consumed.update(period_slots)
    remaining = [slot for slot in slots if str(slot).strip() not in consumed]
    labels.extend(str(slot) for slot in remaining[: max(0, 10 - len(labels))])
    return ", ".join(labels)


def course_department(course: dict[str, Any]) -> str:
    match = re.match(r"[A-Za-z]+", course_code(course).strip())
    return match.group(0).upper() if match else "OTHER"


def normalized_credit_key(course: dict[str, Any]) -> str:
    credits = course_credits(course).strip()
    try:
        return f"{float(credits):g}"
    except (TypeError, ValueError):
        return credits or "unknown"


def stable_slot_color_key(course: dict[str, Any], color_mode: str = "course") -> str:
    if color_mode == "credits":
        return f"credits:{normalized_credit_key(course)}"
    if color_mode == "department":
        return f"department:{course_department(course)}"
    return course_code(course) or course_name(course) or course_label(course)


def slot_color_index(course: dict[str, Any], color_mode: str = "course") -> int:
    if color_mode == "department":
        department = course_department(course)
        if department in DEPARTMENT_COLOR_ORDER:
            return DEPARTMENT_COLOR_ORDER.index(department) % len(SLOT_COLORS)
    if color_mode == "credits":
        credit_key = normalized_credit_key(course)
        credit_order = ["0", "1", "2", "3", "4", "5", "unknown"]
        if credit_key in credit_order:
            return credit_order.index(credit_key) % len(SLOT_COLORS)
    key = stable_slot_color_key(course, color_mode)
    return sum((index + 1) * ord(char) for index, char in enumerate(key)) % len(SLOT_COLORS)


def slot_color_style(course: dict[str, Any], color_mode: str = "course") -> str:
    border, background = SLOT_COLORS[slot_color_index(course, color_mode)]
    return f"--slot-accent:{border};--slot-bg:{background};"


def default_student_path() -> Path:
    if DEFAULT_STUDENT_PATH.exists():
        return DEFAULT_STUDENT_PATH
    return DATA_DIR / "student_courses.xlsx"


def agent_params(student_path: str, intent_provider: str, model: str) -> dict[str, Any]:
    return {
        "student_path": str(Path(student_path).expanduser()),
        "target_path": str(TARGET_PATH),
        "rules_path": str(RULES_PATH),
        "use_llm_intent": intent_provider in {"ollama", "gemini"},
        "intent_provider": intent_provider,
        "model": model,
    }


def initial_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "kind": "welcome",
            "summary": txt(
                "嗨，我是 NTHU COPILOT。請先用下方輸入欄左側的上傳按鈕放入修課紀錄截圖；我會先 OCR，等你確認修課紀錄正確後才開始排課。",
                "Hi, I am NTHU COPILOT. Please upload your course-record screenshot with the upload button in the chat input first. I will run OCR and start planning only after you confirm the records.",
            ),
            "content": "",
        }
    ]


def ensure_agent(student_path: str, intent_provider: str, model: str) -> CoursePlanningAgent:
    params = agent_params(student_path, intent_provider, model)
    if st.session_state.get("agent_params") != params:
        st.session_state.agent = CoursePlanningAgent(
            student_path=params["student_path"],
            target_path=params["target_path"],
            rules_path=params["rules_path"],
            use_llm=False,
            use_llm_intent=params["use_llm_intent"],
            model=params["model"],
            intent_provider=params["intent_provider"],
        )
        st.session_state.agent_params = params
        st.session_state.agent_loaded_params = None
        st.session_state.load_info = None
        st.session_state.messages = initial_messages()
        st.session_state.last_result = {}
        st.session_state.calendar_result = None
        st.session_state.last_latency = None
    return st.session_state.agent


def load_agent_once(agent: CoursePlanningAgent) -> dict[str, Any]:
    params = st.session_state.get("agent_params")
    if st.session_state.get("agent_loaded_params") == params and st.session_state.get("load_info"):
        return st.session_state.load_info
    with st.spinner(txt("正在載入修課紀錄與 114-2 課程資料...", "Loading course records and 114-2 course data...")):
        info = agent.load()
    st.session_state.load_info = info
    st.session_state.agent_loaded_params = dict(params)
    return info


def reset_chat() -> None:
    st.session_state.pop("agent_params", None)
    st.session_state.pop("agent", None)
    st.session_state.pop("load_info", None)
    st.session_state.pop("agent_loaded_params", None)
    st.session_state.messages = initial_messages()
    st.session_state.last_result = {}
    st.session_state.calendar_result = None
    st.session_state.ocr_flow_state = "need_upload"
    st.session_state.ocr_confirmed_for_chat = False
    st.session_state.pop("ocr_preview", None)
    st.session_state.pop("ocr_editor_records", None)
    st.session_state.pop("ocr_pending_action", None)
    st.session_state.pop("ocr_show_table", None)
    st.session_state.pop("quick_prompt_choices", None)


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def ensure_ocr_flow_state() -> None:
    st.session_state.setdefault("messages", initial_messages())
    st.session_state.setdefault("ocr_flow_state", "need_upload")
    st.session_state.setdefault("ocr_confirmed_for_chat", False)
    st.session_state.setdefault("ocr_show_table", False)


def ocr_ready_for_chat() -> bool:
    ensure_ocr_flow_state()
    return bool(st.session_state.get("ocr_confirmed_for_chat"))


def looks_like_ocr_confirmation(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "").lower())
    if not normalized:
        return False
    phrases = [
        "沒問題", "沒有問題", "正確", "都對", "ok", "okay", "確認", "確認了",
        "沒問題了", "可以了", "就是這樣", "沒錯", "對了", "correct",
    ]
    return any(phrase in normalized for phrase in phrases)


def get_chat_input_text_and_files(value: Any) -> tuple[str, list[Any]]:
    if value is None:
        return "", []
    if isinstance(value, str):
        return value.strip(), []
    text = getattr(value, "text", None)
    files = getattr(value, "files", None)
    if text is None and isinstance(value, dict):
        text = value.get("text", "")
        files = value.get("files", [])
    return str(text or "").strip(), list(files or [])


def lock_chat_text_until_upload() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stChatInput"] textarea {
          pointer-events: none !important;
          caret-color: transparent !important;
          user-select: none !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
          color: #172033 !important;
          opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mount_resizable_splitter() -> None:
    components.html(
        """
        <script>
        (() => {
          const STORAGE_KEY = "nthu-copilot-main-split";
          const MIN_RATIO = 0.30;
          const MAX_RATIO = 0.72;

          function clamp(value) {
            return Math.max(MIN_RATIO, Math.min(MAX_RATIO, value));
          }

          function getParentDocument() {
            try {
              return window.parent.document;
            } catch (error) {
              return null;
            }
          }

          function findMainLayout(doc) {
            const anchor = doc.querySelector("#main-split-layout-anchor");
            const blocks = Array.from(doc.querySelectorAll('div[data-testid="stHorizontalBlock"]'));
            const candidates = blocks.filter((block) => {
              const columns = block.querySelectorAll(':scope > div[data-testid="column"]');
              return columns.length === 2;
            });
            if (!anchor) {
              return candidates.find((block) => block.textContent.includes("課表儀表板")) || candidates[0];
            }
            const anchorTop = anchor.getBoundingClientRect().top;
            return candidates.find((block) => block.getBoundingClientRect().top >= anchorTop - 4);
          }

          function setup() {
            const doc = getParentDocument();
            if (!doc) return;
            const block = findMainLayout(doc);
            if (!block || block.dataset.nthuResizableMounted === "1") return;

            block.dataset.nthuResizableMounted = "1";
            block.classList.add("resizable-main-layout");

            const handle = doc.createElement("div");
            handle.className = "splitter-handle";
            handle.title = "拖曳調整聊天與課表寬度";
            block.appendChild(handle);

            const saved = Number.parseFloat(window.localStorage.getItem(STORAGE_KEY) || "0.52");

            function placeHandle() {
              const firstColumn = block.querySelector(':scope > div[data-testid="column"]');
              if (!firstColumn) return;
              const blockRect = block.getBoundingClientRect();
              const columnRect = firstColumn.getBoundingClientRect();
              handle.style.left = `${columnRect.right - blockRect.left + 8}px`;
            }

            function applyRatio(ratio) {
              const safeRatio = clamp(ratio);
              block.style.setProperty("--left-width", `${safeRatio * 100}%`);
              window.localStorage.setItem(STORAGE_KEY, String(safeRatio));
              window.requestAnimationFrame(placeHandle);
            }

            applyRatio(Number.isFinite(saved) ? saved : 0.52);

            let dragging = false;
            function clientX(event) {
              if (event.touches && event.touches[0]) return event.touches[0].clientX;
              return event.clientX;
            }
            function move(event) {
              if (!dragging) return;
              const rect = block.getBoundingClientRect();
              applyRatio((clientX(event) - rect.left) / Math.max(1, rect.width));
              event.preventDefault();
            }
            function stop() {
              dragging = false;
              block.classList.remove("is-dragging");
              doc.removeEventListener("mousemove", move);
              doc.removeEventListener("mouseup", stop);
              doc.removeEventListener("touchmove", move);
              doc.removeEventListener("touchend", stop);
            }
            function start(event) {
              dragging = true;
              block.classList.add("is-dragging");
              doc.addEventListener("mousemove", move);
              doc.addEventListener("mouseup", stop);
              doc.addEventListener("touchmove", move, { passive: false });
              doc.addEventListener("touchend", stop);
              event.preventDefault();
            }

            handle.addEventListener("mousedown", start);
            handle.addEventListener("touchstart", start, { passive: false });
            window.addEventListener("resize", placeHandle);
          }

          setup();
          setTimeout(setup, 300);
          setTimeout(setup, 1000);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def current_plan() -> dict[str, Any]:
    if not ocr_ready_for_chat():
        return {}
    result = st.session_state.get("last_result") or {}
    if result.get("recommended_courses"):
        return result
    agent = st.session_state.get("agent")
    if agent and agent.last_recommendation:
        return agent.last_recommendation
    return {}


def user_prompt_history(limit: int = 12) -> list[str]:
    messages = st.session_state.get("messages") or []
    prompts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip()
        if text:
            prompts.append(text)
    return prompts[-limit:][::-1]


def render_recent_prompts() -> None:
    prompts = user_prompt_history()
    st.markdown('<div class="sidebar-title">Recents</div>', unsafe_allow_html=True)
    if not prompts:
        st.caption("還沒有對話紀錄。")
        return
    items = ["<div class='recent-list'>"]
    for prompt in prompts:
        short = prompt if len(prompt) <= 28 else prompt[:27] + "..."
        items.append(f"<div class='recent-item' title='{h(prompt)}'>{h(short)}</div>")
    items.append("</div>")
    st.markdown("\n".join(items), unsafe_allow_html=True)


def save_uploaded_screenshot(uploaded_file: Any) -> Path:
    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    image_path = DATA_DIR / f"course_screenshot{suffix}"
    image_path.write_bytes(uploaded_file.getbuffer())
    return image_path


def ocr_records_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records or [])
    for column in SAVE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[SAVE_COLUMNS].fillna("")


def apply_ocr_records(records: list[dict[str, Any]]) -> None:
    df = ocr_records_dataframe(records)
    DEFAULT_STUDENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_STUDENT_PATH, index=False, encoding="utf-8-sig")
    st.session_state.pop("agent_params", None)
    st.session_state.calendar_result = None
    st.session_state.last_result = {}


def count_course_code_like(text: str) -> int:
    return len(re.findall(r"\b[A-Z]{2,6}\s*\d{4}\b", text or "", flags=re.IGNORECASE))


@st.cache_data(show_spinner=False, ttl=300)
def _kernelspec_json() -> dict[str, Any]:
    commands = [
        ["jupyter", "kernelspec", "list", "--json"],
        [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
    ]
    for command in commands:
        executable = shutil.which(command[0]) if command[0] == "jupyter" else command[0]
        if not executable:
            continue
        try:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=12,
            )
        except Exception:
            continue
        if completed.returncode != 0:
            continue
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


@st.cache_data(show_spinner=False, ttl=300)
def discover_ocr_kernel_pythons() -> list[dict[str, str]]:
    kernels = (_kernelspec_json().get("kernelspecs") or {})
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for name, item in kernels.items():
        spec = (item or {}).get("spec") or {}
        display_name = str(spec.get("display_name") or name)
        label = f"{name} {display_name}".lower()
        argv = spec.get("argv") or []
        if not argv:
            continue
        if "ocr" not in label and "ocr" not in str(argv[0]).lower():
            continue
        python_exe = Path(str(argv[0])).expanduser()
        if not python_exe.is_absolute():
            resolved = shutil.which(str(argv[0]))
            if not resolved:
                continue
            python_exe = Path(resolved)
        if not python_exe.exists():
            continue
        key = str(python_exe.resolve())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "name": str(name),
                "display_name": display_name,
                "python": key,
            }
        )
    return candidates


def run_ocr_in_kernel_python(
    python_exe: str,
    kernel_label: str,
    image_path: Path,
    output_path: Path,
    backend_order: list[str],
) -> dict[str, Any]:
    command = [
        python_exe,
        str(ROOT / "ocr_preprocess_demo.py"),
        str(image_path),
        "--output",
        str(output_path),
        "--backend-order",
        ",".join(backend_order),
    ]
    env = os.environ.copy()
    if DEFAULT_GOOGLE_VISION_KEY_PATH.exists():
        env.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(DEFAULT_GOOGLE_VISION_KEY_PATH))
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except Exception as exc:
        return {
            "image_path": str(image_path),
            "output_path": str(output_path),
            "backend": "",
            "execution_env": f"OCR kernel: {kernel_label}",
            "success": False,
            "error": str(exc),
        }

    text = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    backend_match = re.search(r"OCR backend used:\s*([^\r\n]+)", completed.stdout)
    seconds_match = re.search(r"OCR seconds:\s*([0-9.]+)", completed.stdout)
    backend = backend_match.group(1).strip() if backend_match else ""
    script_reported_failure = text.startswith("OCR failed.") or not backend
    error_tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-8:])
    return {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "backend": backend,
        "execution_env": f"OCR kernel: {kernel_label}",
        "ocr_seconds": float(seconds_match.group(1)) if seconds_match else round(perf_counter() - started, 3),
        "text_length": len(text),
        "course_code_like_count": count_course_code_like(text),
        "success": completed.returncode == 0 and not script_reported_failure,
        "error": "" if completed.returncode == 0 and not script_reported_failure else error_tail,
    }


def run_ocr_auto(image_path: Path, output_path: Path, backend_order: list[str]) -> dict[str, Any]:
    kernel_errors: list[str] = []
    for kernel in discover_ocr_kernel_pythons():
        kernel_result = run_ocr_in_kernel_python(
            python_exe=kernel["python"],
            kernel_label=kernel["display_name"],
            image_path=image_path,
            output_path=output_path,
            backend_order=backend_order,
        )
        if kernel_result.get("success"):
            return kernel_result
        if kernel_result.get("error"):
            kernel_errors.append(f"{kernel['display_name']}: {kernel_result['error']}")

    current_result = run_ocr(
        image_path=str(image_path),
        output_path=str(output_path),
        backend_order=backend_order,
    )
    current_result["execution_env"] = "Streamlit Python"
    if current_result.get("success"):
        return current_result

    kernel_errors.append(str(current_result.get("error") or "Streamlit Python OCR failed."))
    current_result["error"] = " | ".join(error for error in kernel_errors if error)
    current_result["execution_env"] = "OCR kernel / Streamlit Python 都沒有成功"
    return current_result


def default_ocr_backend_order() -> list[str]:
    return ["google_vision", "paddleocr", "tesseract", "easyocr"]


def parse_uploaded_ocr(uploaded_file: Any, backend_order: list[str] | None = None) -> dict[str, Any]:
    image_path = save_uploaded_screenshot(uploaded_file)
    output_path = cache_path_for_image(image_path)
    order = backend_order or default_ocr_backend_order()
    ocr_info = run_ocr_auto(image_path, output_path, order)
    parsed = parse_course_screenshot_from_cache(
        image_path=str(image_path),
        student_path=str(DEFAULT_STUDENT_PATH),
        target_path=str(TARGET_PATH),
        historical_course_paths=[str(path) for path in HISTORICAL_COURSE_PATHS],
        force_reparse=True,
    )
    records = parsed.get("ocr_confirmed_student_record") or []
    st.session_state.ocr_preview = {
        "image_path": str(image_path),
        "ocr_info": ocr_info,
        "parsed": parsed,
    }
    st.session_state.ocr_editor_records = ocr_records_dataframe(records).to_dict("records")
    st.session_state.ocr_editor_version = int(st.session_state.get("ocr_editor_version", 0)) + 1
    st.session_state.pop("ocr_pending_action", None)
    st.session_state.ocr_show_table = True
    st.session_state.ocr_flow_state = "confirming"
    st.session_state.ocr_confirmed_for_chat = False
    st.session_state.messages.append(
        {
            "role": "assistant",
            "summary": txt(
                "OCR 已完成，請檢查下方修課紀錄表格。若有錯可以直接改表格或新增列；確認無誤後輸入「沒問題了」或按「確認修課紀錄」。",
                "OCR is complete. Please check the course-record table below. Edit errors or add rows directly, then confirm the records.",
            ),
            "content": "",
        }
    )
    return st.session_state.ocr_preview


def normalize_editor_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    clean_df = df.copy()
    for column in SAVE_COLUMNS:
        if column not in clean_df.columns:
            clean_df[column] = ""
    clean_df = clean_df[SAVE_COLUMNS].fillna("")
    keep_mask = (
        clean_df["normalized_course_code"].astype(str).str.strip().ne("")
        | clean_df["raw_course_code"].astype(str).str.strip().ne("")
        | clean_df["course_name_zh"].astype(str).str.strip().ne("")
    )
    return clean_df[keep_mask].to_dict("records")


def ocr_editor_dataframe(edited_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if edited_df is not None:
        return ocr_records_dataframe(normalize_editor_records(edited_df))
    return ocr_records_dataframe(st.session_state.get("ocr_editor_records") or [])


def update_ocr_editor_records(df: pd.DataFrame) -> None:
    st.session_state.ocr_editor_records = normalize_editor_records(df)
    st.session_state.ocr_editor_version = int(st.session_state.get("ocr_editor_version", 0)) + 1


def add_ocr_assistant_message(summary: str) -> None:
    st.session_state.messages.append({"role": "assistant", "summary": summary, "content": ""})


def norm_lookup_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def course_row_label(row: pd.Series | dict[str, Any]) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default="": default
    code = getter("normalized_course_code", "") or getter("raw_course_code", "")
    name = (getter("course_name_en", "") or getter("course_name_zh", "")) if is_english() else (getter("course_name_zh", "") or getter("course_name_en", ""))
    return f"{code} {name}".strip() or txt("這門課", "this course")


def row_value(row: pd.Series | dict[str, Any], *keys: str) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default="": default
    for key in keys:
        value = getter(key, "")
        if value not in (None, ""):
            return str(value).strip()
    return ""


def raw_section_suffix(value: Any) -> str:
    raw = clean_raw_course_code(value)
    raw = re.sub(r"^\d{5}", "", raw)
    return re.sub(r"\s+", "", raw).upper()


@st.cache_data(show_spinner=False, ttl=900)
def current_offering_reference_rows(target_path: str) -> list[dict[str, Any]]:
    try:
        df = load_target_courses(target_path).fillna("")
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "raw_course_code": str(row.get("raw_course_code", "")),
                "raw_suffix": raw_section_suffix(row.get("raw_course_code", "")),
                "normalized_course_code": normalize_ocr_code(row.get("normalized_course_code", "")),
                "course_name_zh": str(row.get("course_name_zh", "")),
                "teacher": str(row.get("teacher", "")),
                "time": str(row.get("time", "")),
                "classroom": str(row.get("classroom", "")),
                "credits": str(row.get("credits", "")),
            }
        )
    return rows


def enrich_candidate_for_display(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    result = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    if row_value(result, "teacher") and row_value(result, "time"):
        return result

    raw_suffix = raw_section_suffix(row_value(result, "raw_course_code"))
    normalized = normalize_ocr_code(row_value(result, "normalized_course_code", "code"))
    references = current_offering_reference_rows(str(TARGET_PATH))
    matches = []
    if raw_suffix:
        matches = [item for item in references if item.get("raw_suffix") == raw_suffix]
    if not matches and normalized:
        matches = [item for item in references if item.get("normalized_course_code") == normalized]
    if not matches:
        return result

    if raw_suffix and len(matches) == 1:
        ref = matches[0]
        result.setdefault("display_reference_note", "114-2 同 section 參考")
        result.setdefault("display_teacher", ref.get("teacher", ""))
        result.setdefault("display_time", ref.get("time", ""))
        result.setdefault("display_classroom", ref.get("classroom", ""))
        return result

    teachers = list(dict.fromkeys(item.get("teacher", "") for item in matches if item.get("teacher")))
    times = list(dict.fromkeys(item.get("time", "") for item in matches if item.get("time")))
    classrooms = list(dict.fromkeys(item.get("classroom", "") for item in matches if item.get("classroom")))
    if teachers or times or classrooms:
        result.setdefault("display_reference_note", "114-2 同課號參考")
        result.setdefault("display_teacher", " / ".join(teachers[:4]))
        result.setdefault("display_time", " / ".join(times[:4]))
        result.setdefault("display_classroom", " / ".join(classrooms[:4]))
    return result


def format_ocr_candidate(row: pd.Series | dict[str, Any], index: int | None = None) -> str:
    row = enrich_candidate_for_display(row)
    code = row_value(row, "normalized_course_code", "code")
    raw = row_value(row, "raw_course_code")
    name = row_value(row, "course_name_en", "name", "course_name_zh", "name_zh") if is_english() else row_value(row, "course_name_zh", "name_zh", "course_name_en", "name")
    title = " ".join(part for part in [code or raw or txt("未標示課號", "No course code"), name] if part)
    if index is not None:
        title = f"{index}. {title}"
    teacher = row_value(row, "teacher", "instructor")
    time = row_value(row, "time", "class_time", "schedule")
    classroom = row_value(row, "classroom")
    reference_note = row_value(row, "display_reference_note")
    reference_teacher = row_value(row, "display_teacher")
    reference_time = row_value(row, "display_time")
    reference_classroom = row_value(row, "display_classroom")
    headline_bits = [
        title,
        teacher or (f"{reference_teacher} ({txt('參考', 'reference')})" if reference_teacher else ""),
        time or (f"{reference_time} ({txt('參考', 'reference')})" if reference_time else ""),
    ]
    title = " · ".join(part for part in headline_bits if part)
    if not (teacher or reference_teacher):
        teacher_detail = txt("資料未提供（歷史課程檔沒有教師欄，且 114-2 未找到同課參考）", "Not provided; no instructor field in historical data and no 114-2 reference found")
    elif teacher:
        teacher_detail = teacher
    else:
        teacher_detail = f"{reference_teacher} ({reference_note})"
    if not (time or reference_time):
        time_detail = txt("資料未提供", "Not provided")
    elif time:
        time_detail = time
    else:
        time_detail = f"{reference_time} ({reference_note})"
    classroom_detail = classroom or (
        f"{reference_classroom} ({reference_note})" if reference_classroom else ""
    )
    detail_items = [
        (txt("學期", "Term"), row_value(row, "term")),
        (txt("原始課號", "Raw code"), raw),
        (txt("老師", "Instructor"), teacher_detail),
        (txt("時間", "Time"), time_detail),
        (txt("教室", "Classroom"), classroom_detail),
        (txt("學分", "Credits"), row_value(row, "credits", "credit")),
        (txt("狀態", "Status"), row_value(row, "status")),
        (txt("成績", "Grade"), row_value(row, "grade")),
    ]
    separator = "; " if is_english() else "；"
    colon = ": " if is_english() else "："
    details = separator.join(f"{label}{colon}{h(value)}" for label, value in detail_items if value)
    return f"<b>{h(title)}</b>" + (f"<br><span class='small-muted'>{details}</span>" if details else "")


def format_ocr_candidate_list(rows: list[dict[str, Any]]) -> str:
    return "<br><br>".join(format_ocr_candidate(row, index + 1) for index, row in enumerate(rows))


def format_ocr_row_list(df: pd.DataFrame, rows: list[int]) -> str:
    return "<br><br>".join(format_ocr_candidate(df.loc[row_idx], index + 1) for index, row_idx in enumerate(rows))


def select_candidate_position(text: str, candidates: list[dict[str, Any]]) -> tuple[int, bool]:
    value = str(text or "").strip()
    if value.isdigit():
        position = int(value) - 1
        if 0 <= position < len(candidates):
            return position, False
        return -1, False
    query = norm_lookup_text(value)
    query_code = normalize_ocr_code(value) or clean_raw_course_code(value)
    matches: list[int] = []
    for index, row in enumerate(candidates):
        row = enrich_candidate_for_display(row)
        code = normalize_ocr_code(row_value(row, "normalized_course_code", "code"))
        raw = clean_raw_course_code(row_value(row, "raw_course_code"))
        fields = [
            row_value(row, "normalized_course_code", "code"),
            row_value(row, "raw_course_code"),
            row_value(row, "course_name_zh", "name_zh"),
            row_value(row, "course_name_en", "name"),
            row_value(row, "teacher", "instructor"),
            row_value(row, "time", "class_time", "schedule"),
            row_value(row, "display_teacher"),
            row_value(row, "display_time"),
            row_value(row, "display_classroom"),
        ]
        if (query_code and query_code in {code, raw}) or (query and any(query in norm_lookup_text(field) for field in fields)):
            matches.append(index)
    if len(matches) == 1:
        return matches[0], False
    if len(matches) > 1:
        return -1, True
    return -1, False


def select_ocr_row_position(text: str, df: pd.DataFrame, rows: list[int]) -> tuple[int, bool]:
    candidates = [df.loc[row_idx].to_dict() for row_idx in rows]
    return select_candidate_position(text, candidates)


def extract_status_from_text(text: str) -> str:
    value = str(text or "").lower()
    if any(token in value for token in ["in_progress", "inprogress", "ongoing", "進行中", "修課中", "正在修", "目前修"]):
        return "in_progress"
    if any(token in value for token in ["completed", "complete", "done", "finished", "已修完", "修完", "完成", "已修"]):
        return "completed"
    return ""


def extract_grade_from_text(text: str) -> str:
    value = str(text or "").lower()
    if any(token in value for token in ["fail", "failed", "不及格", "沒過", "未通過"]):
        return "fail"
    if any(token in value for token in ["pass", "passed", "及格", "通過", "過了"]):
        return "pass"
    return ""


def find_ocr_rows(query: str, df: pd.DataFrame) -> list[int]:
    if df is None or df.empty:
        return []
    cleaned_query = str(query or "").strip()
    q = norm_lookup_text(cleaned_query)
    q_code = normalize_ocr_code(cleaned_query)
    matches: list[int] = []
    for idx, row in df.fillna("").iterrows():
        norm = normalize_ocr_code(row.get("normalized_course_code", "")) or normalize_ocr_code(row.get("raw_course_code", ""))
        raw = clean_raw_course_code(row.get("raw_course_code", ""))
        fields = [
            row.get("normalized_course_code", ""),
            row.get("raw_course_code", ""),
            row.get("course_name_zh", ""),
            row.get("course_name_en", ""),
            row.get("teacher", ""),
        ]
        if (q_code and q_code in {norm, raw}) or (q and any(q in norm_lookup_text(field) for field in fields)):
            matches.append(int(idx))
    return matches


def clean_correction_query(query: str) -> str:
    text = str(query or "")
    text = re.sub(
        r"\b(pass|passed|fail|failed|completed|complete|in_progress|inprogress|ongoing|done)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"及格|不及格|通過|沒過|未通過|已修完畢|已修完|修完|完成|進行中|修課中|正在修|狀態|成績|status|grade", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def set_pending_action(action: dict[str, Any], message: str) -> str:
    st.session_state.ocr_pending_action = action
    st.session_state.ocr_show_table = False
    return message


def apply_grade_update(text: str, df: pd.DataFrame, row_idx: int | None = None, query: str = "") -> tuple[pd.DataFrame, str]:
    grade = extract_grade_from_text(text)
    if row_idx is None:
        rows = find_ocr_rows(clean_correction_query(_extract_query_after_keywords(query or text, ["grade", "成績"])), df)
        if not rows:
            return df, "我在目前 OCR 表格裡找不到這門課。你可以直接改表格，或輸入更完整的課名/課號。"
        if len(rows) > 1:
            labels = format_ocr_row_list(df, rows[:6])
            return df, set_pending_action({"type": "grade_select", "rows": rows, "grade": grade}, f"找到多筆可能的課，請輸入要修改的編號、課號或原始課號：<br><br>{labels}")
        row_idx = rows[0]
    if not grade:
        return df, set_pending_action({"type": "grade_value", "row_idx": int(row_idx)}, f"{course_row_label(df.loc[row_idx])} 的成績要設為 `pass` 還是 `fail`？")
    term = str(df.at[row_idx, "term"]).replace(".0", "")
    if term == CURRENT_IN_PROGRESS_TERM:
        df.at[row_idx, "status"] = "in_progress"
        df.at[row_idx, "grade"] = ""
        return df, f"{course_row_label(df.loc[row_idx])} 是目前進行中的學期，我保留為 `in_progress`，不填成績。"
    df.at[row_idx, "status"] = "completed"
    df.at[row_idx, "grade"] = grade
    return df, f"已把 {course_row_label(df.loc[row_idx])} 的成績改成 `{grade}`。"


def apply_status_update(text: str, df: pd.DataFrame, row_idx: int | None = None, query: str = "") -> tuple[pd.DataFrame, str]:
    status = extract_status_from_text(text)
    if row_idx is None:
        rows = find_ocr_rows(clean_correction_query(_extract_query_after_keywords(query or text, ["status", "狀態"])), df)
        if not rows:
            return df, "我在目前 OCR 表格裡找不到這門課。你可以直接改表格，或輸入更完整的課名/課號。"
        if len(rows) > 1:
            labels = format_ocr_row_list(df, rows[:6])
            return df, set_pending_action({"type": "status_select", "rows": rows, "status": status}, f"找到多筆可能的課，請輸入要修改的編號、課號或原始課號：<br><br>{labels}")
        row_idx = rows[0]
    if not status:
        return df, set_pending_action({"type": "status_value", "row_idx": int(row_idx)}, f"{course_row_label(df.loc[row_idx])} 的狀態要設為 `completed` 還是 `in_progress`？")
    term = str(df.at[row_idx, "term"]).replace(".0", "")
    if term == CURRENT_IN_PROGRESS_TERM:
        status = "in_progress"
    df.at[row_idx, "status"] = status
    if status == "in_progress":
        df.at[row_idx, "grade"] = ""
    elif not str(df.at[row_idx, "grade"]).strip():
        df.at[row_idx, "grade"] = "pass"
    return df, f"已把 {course_row_label(df.loc[row_idx])} 的狀態改成 `{status}`。"


def search_missing_course_candidates(term: str, query: str) -> pd.DataFrame:
    warnings: list[str] = []
    candidates = _search_historical_paths_for_term(term, query, [str(path) for path in HISTORICAL_COURSE_PATHS], warnings)
    if candidates.empty and DEFAULT_STUDENT_PATH.exists():
        student_df = _safe_load_student(DEFAULT_STUDENT_PATH, warnings)
        candidates = _search_sources_for_term(term, query, student_df, pd.DataFrame())
    return candidates


def add_missing_course_record(df: pd.DataFrame, row: dict[str, Any], term: str, grade: str) -> tuple[pd.DataFrame, str]:
    status = "in_progress" if term == CURRENT_IN_PROGRESS_TERM else "completed"
    final_grade = "" if status == "in_progress" else (grade or "pass")
    record = _record_from_selected_historical(row, term, status, final_grade)
    code = str(record.get("normalized_course_code") or "")
    if code and "normalized_course_code" in df.columns:
        df = df[df["normalized_course_code"].astype(str) != code]
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    return df, f"已補上 {record.get('term')} {record.get('normalized_course_code')} {record.get('course_name_zh')}。"


def apply_missing_course(text: str, df: pd.DataFrame, term: str = "", query: str = "", grade: str = "") -> tuple[pd.DataFrame, str]:
    term = term or _semester_display_to_code(text)
    course_query = clean_correction_query(query or _clean_manual_course_query(text))
    grade = grade or extract_grade_from_text(text)
    if not term:
        return df, set_pending_action({"type": "missing_term", "query": course_query, "grade": grade}, f"你要補上的 `{course_query or '這門課'}` 是哪一學期？例如 `113第二學期` 或 `114第一學期`。")
    if term == TARGET_SEMESTER:
        return df, "114 第二學期是目標排課學期，不能加進歷史修課紀錄。你確認 OCR 後可以用「我想修某某課」加入課表。"
    if not course_query:
        return df, set_pending_action({"type": "missing_query", "term": term, "grade": grade}, f"請輸入要補上的課名或課號。")
    candidates = search_missing_course_candidates(term, course_query)
    if candidates.empty:
        return df, set_pending_action({"type": "missing_query", "term": term, "grade": grade}, f"我在 `{term}` 找不到 `{course_query}`。請換一個課名/課號，或直接在表格新增。")
    if len(candidates) > 1:
        rows = candidates.head(6).to_dict("records")
        labels = format_ocr_candidate_list(rows)
        return df, set_pending_action({"type": "missing_select", "term": term, "grade": grade, "candidates": rows}, f"找到多筆候選，請輸入要補上的編號、課號或原始課號：<br><br>{labels}")
    selected = candidates.iloc[0].to_dict()
    if term != CURRENT_IN_PROGRESS_TERM and not grade:
        return df, set_pending_action({"type": "missing_grade", "term": term, "candidate": selected}, f"找到這門課：<br><br>{format_ocr_candidate(selected)}<br><br>這門課成績是 `pass` 還是 `fail`？")
    return add_missing_course_record(df, selected, term, grade)


def handle_pending_ocr_action(text: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    pending = st.session_state.get("ocr_pending_action")
    if not pending:
        return df, "", False
    action_type = pending.get("type")
    st.session_state.pop("ocr_pending_action", None)
    if action_type == "grade_value":
        new_df, message = apply_grade_update(text, df, row_idx=int(pending["row_idx"]))
        return new_df, message, True
    if action_type == "status_value":
        new_df, message = apply_status_update(text, df, row_idx=int(pending["row_idx"]))
        return new_df, message, True
    if action_type in {"grade_select", "status_select"}:
        rows = list(pending.get("rows") or [])
        selection, ambiguous = select_ocr_row_position(text, df, rows)
        if selection < 0 or selection >= len(rows):
            detail = format_ocr_row_list(df, rows[:6])
            hint = "這個課號有多筆可能結果，請改輸入左邊的編號。" if ambiguous else "請輸入候選清單中的編號、課號或原始課號。"
            return df, set_pending_action(pending, f"{hint}<br><br>{detail}"), True
        if action_type == "grade_select":
            return (*apply_grade_update(pending.get("grade", "") or text, df, row_idx=int(rows[selection])), True)
        return (*apply_status_update(pending.get("status", "") or text, df, row_idx=int(rows[selection])), True)
    if action_type == "missing_term":
        term = _semester_display_to_code(text)
        if not term:
            return df, set_pending_action(pending, "我還是看不出學期，請輸入例如 `113第二學期` 或 `114第一學期`。"), True
        new_df, message = apply_missing_course("", df, term=term, query=pending.get("query", ""), grade=pending.get("grade", ""))
        return new_df, message, True
    if action_type == "missing_query":
        new_df, message = apply_missing_course(text, df, term=pending.get("term", ""), grade=pending.get("grade", ""))
        return new_df, message, True
    if action_type == "missing_select":
        candidates = list(pending.get("candidates") or [])
        selection, ambiguous = select_candidate_position(text, candidates)
        if selection < 0 or selection >= len(candidates):
            detail = format_ocr_candidate_list(candidates[:6])
            hint = "這個課號有多筆候選，請改輸入左邊的編號。" if ambiguous else "請輸入候選清單中的編號、課號或原始課號。"
            return df, set_pending_action(pending, f"{hint}<br><br>{detail}"), True
        term = pending.get("term", "")
        grade = pending.get("grade", "")
        if term != CURRENT_IN_PROGRESS_TERM and not grade:
            return df, set_pending_action({"type": "missing_grade", "term": term, "candidate": candidates[selection]}, f"你選的是：<br><br>{format_ocr_candidate(candidates[selection])}<br><br>這門課成績是 `pass` 還是 `fail`？"), True
        new_df, message = add_missing_course_record(df, candidates[selection], term, grade)
        return new_df, message, True
    if action_type == "missing_grade":
        grade = extract_grade_from_text(text)
        if not grade:
            return df, set_pending_action(pending, "請輸入 `pass` 或 `fail`。"), True
        new_df, message = add_missing_course_record(df, pending.get("candidate", {}), pending.get("term", ""), grade)
        return new_df, message, True
    return df, "", False


def apply_ocr_text_correction(text: str, edited_df: pd.DataFrame | None) -> str:
    df = ocr_editor_dataframe(edited_df)
    pending_df, pending_message, handled = handle_pending_ocr_action(text, df)
    if handled:
        update_ocr_editor_records(pending_df)
        st.session_state.ocr_show_table = not bool(st.session_state.get("ocr_pending_action"))
        return pending_message
    if any(token in text for token in ["漏", "缺", "少", "補", "新增", "加"]):
        new_df, message = apply_missing_course(text, df)
        update_ocr_editor_records(new_df)
        st.session_state.ocr_show_table = not bool(st.session_state.get("ocr_pending_action"))
        return message
    if "成績" in text or "grade" in text.lower():
        new_df, message = apply_grade_update(text, df)
        update_ocr_editor_records(new_df)
        st.session_state.ocr_show_table = not bool(st.session_state.get("ocr_pending_action"))
        return message
    if "狀態" in text or "status" in text.lower():
        new_df, message = apply_status_update(text, df)
        update_ocr_editor_records(new_df)
        st.session_state.ocr_show_table = not bool(st.session_state.get("ocr_pending_action"))
        return message
    return "我還在確認 OCR 修課紀錄。你可以說 `漏了電磁學`、`修正電子學一成績`、`修正某課狀態`，或直接改下方表格；確認後輸入 `沒問題了`。"


def confirm_ocr_records(records: list[dict[str, Any]]) -> None:
    apply_ocr_records(records)
    st.session_state.ocr_editor_records = records
    st.session_state.ocr_flow_state = "ready"
    st.session_state.ocr_confirmed_for_chat = True
    st.session_state.messages.append(
        {
            "role": "assistant",
            "summary": txt(
                f"修課紀錄已確認，共 {len(records)} 筆。現在可以開始排課了，例如：`幫我排 114 第二學期課表，20 學分`。",
                f"Course records confirmed: {len(records)} rows. You can start planning now, for example: `Plan a 20-credit 114-2 schedule.`",
            ),
            "content": "",
        }
    )


def render_ocr_gate_panel() -> pd.DataFrame | None:
    ensure_ocr_flow_state()
    state = st.session_state.get("ocr_flow_state")
    if state == "need_upload":
        st.info(txt("請先用下方輸入欄左側的上傳按鈕放入修課紀錄截圖。確認 OCR 結果前，我不會開始排課。", "Please upload your course-record screenshot with the upload button in the chat input. I will not start planning until you confirm the OCR result."))
        return None

    preview = st.session_state.get("ocr_preview") or {}
    parsed = preview.get("parsed") or {}
    ocr_info = preview.get("ocr_info") or {}
    records = st.session_state.get("ocr_editor_records") or parsed.get("ocr_confirmed_student_record") or []
    if not records:
        st.warning(txt("OCR 沒有解析到可確認的修課紀錄。你可以在下方表格手動新增列，至少填課號或課名。", "OCR did not find confirmable course records. You can manually add rows below; at least enter a course code or course name."))

    if ocr_info:
        if ocr_info.get("success"):
            st.success(
                txt(
                    f"OCR 完成：{ocr_info.get('backend', '')}，{ocr_info.get('course_code_like_count', 0)} 個疑似課號。",
                    f"OCR complete: {ocr_info.get('backend', '')}; {ocr_info.get('course_code_like_count', 0)} course-code-like tokens.",
                )
            )
        else:
            st.error(txt("OCR 沒有成功。可以改用表格手動新增，或確認 OCR kernel / Google Vision 設定。", "OCR failed. You can add rows manually or check the OCR kernel / Google Vision settings."))
            if ocr_info.get("error"):
                st.caption(str(ocr_info.get("error")))
        if ocr_info.get("execution_env"):
            st.caption(txt(f"執行環境：{ocr_info.get('execution_env')}", f"Execution environment: {ocr_info.get('execution_env')}"))

    for warning in (parsed.get("warnings") or [])[:4]:
        st.warning(str(warning))

    if not st.session_state.get("ocr_show_table", True):
        st.caption(txt("目前正在等待你補充 OCR 修正資訊；完成新增或修改後，這裡會再顯示最後確認表。", "Waiting for OCR correction details. The final confirmation table will return after the add/edit step is complete."))
        return ocr_records_dataframe(records)

    st.markdown("#### " + txt("修課紀錄確認", "Course Record Confirmation"))
    st.caption(txt("如果有錯，直接改表格；如果漏課，用表格最下面新增一列。確認後按按鈕，或在輸入欄打「沒問題了」。", "Edit the table directly if anything is wrong. If a course is missing, add a row at the bottom. Confirm with the button when done."))
    edited_df = st.data_editor(
        ocr_records_dataframe(records),
        key=f"ocr_confirmation_editor_{int(st.session_state.get('ocr_editor_version', 0))}",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_order=[
            "term",
            "normalized_course_code",
            "raw_course_code",
            "course_name_zh",
            "credits",
            "status",
            "grade",
            "teacher",
            "time",
            "classroom",
            "course_name_en",
            "academic_year",
            "semester",
        ],
        column_config={
            "term": txt("學期", "Term"),
            "normalized_course_code": txt("課號", "Code"),
            "raw_course_code": txt("原始課號", "Raw code"),
            "course_name_zh": txt("課名", "Course name"),
            "credits": txt("學分", "Credits"),
            "status": st.column_config.SelectboxColumn(txt("狀態", "Status"), options=["completed", "in_progress", ""]),
            "grade": st.column_config.SelectboxColumn(txt("成績", "Grade"), options=["pass", "fail", ""]),
        },
    )
    col_confirm, col_reset = st.columns([1, 1])
    with col_confirm:
        if st.button(txt("確認修課紀錄", "Confirm Course Records"), type="primary", use_container_width=True):
            records_to_save = normalize_editor_records(edited_df)
            confirm_ocr_records(records_to_save)
            rerun()
    with col_reset:
        if st.button(txt("重新上傳截圖", "Upload Again"), use_container_width=True):
            st.session_state.ocr_flow_state = "need_upload"
            st.session_state.ocr_confirmed_for_chat = False
            st.session_state.pop("ocr_preview", None)
            st.session_state.pop("ocr_editor_records", None)
            st.session_state.pop("ocr_show_table", None)
            rerun()
    return edited_df


def handle_preflight_chat_input(text: str, files: list[Any], edited_df: pd.DataFrame | None) -> bool:
    """Return True when the caller should rerun."""
    ensure_ocr_flow_state()
    if files:
        st.session_state.messages.append({"role": "user", "content": txt(f"上傳修課紀錄截圖：{files[0].name}", f"Uploaded course-record screenshot: {files[0].name}")})
        with st.spinner(txt("正在執行 OCR，第一次可能會比較久...", "Running OCR. The first run may take a while...")):
            parse_uploaded_ocr(files[0])
        return True

    if st.session_state.get("ocr_flow_state") == "need_upload":
        if text:
            st.session_state.messages.append({"role": "user", "content": text})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "summary": txt(
                        "我先不開始排課。請先用輸入欄左側的上傳按鈕放入修課紀錄截圖，完成 OCR 確認後再輸入排課需求。",
                        "I will not start planning yet. Please upload your course-record screenshot first, confirm the OCR result, then enter a planning request.",
                    ),
                    "content": "",
                }
            )
            return True
        return False

    if text and looks_like_ocr_confirmation(text) and not st.session_state.get("ocr_pending_action"):
        st.session_state.messages.append({"role": "user", "content": text})
        records_to_save = normalize_editor_records(edited_df if edited_df is not None else pd.DataFrame(st.session_state.get("ocr_editor_records") or []))
        confirm_ocr_records(records_to_save)
        return True

    if text:
        st.session_state.messages.append({"role": "user", "content": text})
        message = apply_ocr_text_correction(text, edited_df)
        add_ocr_assistant_message(message)
        return True
    return False


def render_ocr_uploader() -> None:
    with st.expander(txt("OCR 修課截圖", "OCR Course Screenshot"), expanded=False):
        st.caption(txt("上傳修課紀錄截圖，系統會先產生 OCR 文字，再轉成 agent 可讀的修課紀錄。", "Upload a course-record screenshot. The system will generate OCR text and convert it into agent-readable course records."))
        uploaded = st.file_uploader(txt("上傳 PNG/JPG", "Upload PNG/JPG"), type=["png", "jpg", "jpeg"], key="ocr_upload")
        st.caption(txt("按「套用」以前不會覆蓋目前資料；套用後會更新 data/ocr_confirmed_student_courses.csv。", "Data is not overwritten until you apply it. Applying updates data/ocr_confirmed_student_courses.csv."))
        if DEFAULT_GOOGLE_VISION_KEY_PATH.exists():
            st.caption(txt(f"已找到 Google Vision 金鑰：{DEFAULT_GOOGLE_VISION_KEY_PATH.relative_to(ROOT)}", f"Google Vision key found: {DEFAULT_GOOGLE_VISION_KEY_PATH.relative_to(ROOT)}"))
        else:
            st.caption(txt("如果要用 Google Vision，請放 private/google_vision_key.json 或設定 GOOGLE_APPLICATION_CREDENTIALS。", "To use Google Vision, place private/google_vision_key.json or set GOOGLE_APPLICATION_CREDENTIALS."))
        ocr_kernels = discover_ocr_kernel_pythons()
        if ocr_kernels:
            st.caption(txt("自動 fallback OCR kernel：", "Auto fallback OCR kernel: ") + " / ".join(kernel["display_name"] for kernel in ocr_kernels[:2]))
        else:
            st.caption(txt("目前沒找到名稱含 OCR 的 Jupyter kernel；若 Streamlit Python 沒 OCR 套件，請先安裝到同一環境。", "No Jupyter kernel with OCR in its name was found. If Streamlit Python lacks OCR packages, install them in the same environment."))
        backend = st.selectbox(
            "OCR backend",
            [
                "google_vision,paddleocr,tesseract,easyocr",
                "paddleocr,tesseract,easyocr",
                "tesseract,easyocr,paddleocr",
                "easyocr,paddleocr,tesseract",
            ],
            index=0,
        )
        if uploaded is not None:
            upload_key = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("ocr_upload_key") != upload_key:
                st.session_state.ocr_upload_key = upload_key
                st.session_state.pop("ocr_preview", None)
            st.image(uploaded.getvalue(), caption=txt("待辨識截圖", "Screenshot to OCR"), use_container_width=True)
            if st.button(txt("執行 OCR 預覽", "Run OCR Preview"), use_container_width=True):
                image_path = save_uploaded_screenshot(uploaded)
                output_path = cache_path_for_image(image_path)
                backend_order = [item.strip() for item in backend.split(",") if item.strip()]
                with st.spinner(txt("正在執行 OCR，第一次可能會比較久...", "Running OCR. The first run may take a while...")):
                    ocr_info = run_ocr_auto(image_path, output_path, backend_order)
                    parsed = parse_course_screenshot_from_cache(
                        image_path=str(image_path),
                        student_path=str(DEFAULT_STUDENT_PATH),
                        target_path=str(TARGET_PATH),
                        historical_course_paths=[
                            str(DATA_DIR / "111-113 _course_data.xlsx"),
                            str(DATA_DIR / "114_1_course_data.xlsx"),
                            str(DATA_DIR / "114_2_course_data.xlsx"),
                        ],
                        force_reparse=True,
                    )
                st.session_state.ocr_preview = {
                    "image_path": str(image_path),
                    "ocr_info": ocr_info,
                    "parsed": parsed,
                }

        preview = st.session_state.get("ocr_preview") or {}
        parsed = preview.get("parsed") or {}
        records = parsed.get("ocr_confirmed_student_record") or []
        if preview:
            ocr_info = preview.get("ocr_info") or {}
            if ocr_info.get("success"):
                st.success(
                    txt(
                        f"OCR 完成：{ocr_info.get('backend', '')}，{ocr_info.get('course_code_like_count', 0)} 個疑似課號。",
                        f"OCR complete: {ocr_info.get('backend', '')}; {ocr_info.get('course_code_like_count', 0)} course-code-like tokens.",
                    )
                )
                if ocr_info.get("execution_env"):
                    st.caption(txt(f"執行環境：{ocr_info.get('execution_env')}", f"Execution environment: {ocr_info.get('execution_env')}"))
            else:
                st.error(txt("OCR 沒有成功，請確認 JupyterHub 是否安裝 google-cloud-vision，或改用 PaddleOCR/Tesseract/EasyOCR。", "OCR failed. Check whether JupyterHub has google-cloud-vision installed, or use PaddleOCR/Tesseract/EasyOCR."))
                if ocr_info.get("error"):
                    st.caption(str(ocr_info.get("error")))
            for warning in (parsed.get("warnings") or [])[:3]:
                st.warning(str(warning))
        if records:
            st.caption(txt(f"辨識到 {len(records)} 筆修課紀錄", f"Detected {len(records)} course records"))
            st.dataframe(
                ocr_records_dataframe(records)[
                    ["term", "normalized_course_code", "course_name_zh", "credits", "status", "grade"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            if st.button(txt("套用並覆蓋修課紀錄", "Apply and Overwrite Course Records"), use_container_width=True):
                apply_ocr_records(records)
                st.success(txt("已更新 data/ocr_confirmed_student_courses.csv，正在重建對話。", "Updated data/ocr_confirmed_student_courses.csv. Rebuilding the chat."))
                rerun()
        elif preview:
            st.info(txt("沒有辨識到可套用的修課紀錄。建議截圖包含課號欄位，或先在 OCR kernel 安裝/確認 OCR backend。", "No applicable course records were detected. Use a screenshot with course-code columns, or install/check the OCR backend in the OCR kernel."))


def low_credit_required(plan: dict[str, Any]) -> bool:
    policy = plan.get("semester_credit_policy") or {}
    return bool(policy.get("low_credit_load_application_required", False))


def overload_required(plan: dict[str, Any]) -> bool:
    policy = plan.get("semester_credit_policy") or {}
    return bool(policy.get("overload_application_required", False))


def strip_chat_history_section(markdown_text: str) -> str:
    lines = str(markdown_text or "").splitlines()
    cleaned: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped == "### 聊天記錄":
            skipping = True
            continue
        if skipping and stripped.startswith("### "):
            skipping = False
        if not skipping:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def simple_course_names(courses: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for course in courses or []:
        code = course.get("code") or course.get("normalized_course_code") or ""
        name = (course.get("course_name_en") or course.get("name_en") or course.get("course_name_zh") or course.get("name_zh") or "") if is_english() else (course.get("course_name_zh") or course.get("name_zh") or course.get("course_name_en") or "")
        label = f"`{code}` {name}".strip()
        if label:
            labels.append(label)
    return "<br>".join(labels) if labels else txt("無", "None")


def missing_alternative_text(alternative_status: dict[str, Any]) -> str:
    labels: list[str] = []
    for item in (alternative_status or {}).values():
        if item.get("satisfied"):
            continue
        name = item.get("name_zh") or item.get("label") or "替代必修群組"
        options = []
        for option in item.get("missing_options") or item.get("options") or []:
            code = option.get("code", "")
            option_name = option.get("name_zh", "")
            options.append(f"`{code}` {option_name}".strip())
        detail = " / ".join(options) if options else "尚未完成"
        labels.append(f"{name}：{detail}")
    return "<br>".join(labels)


def simplified_graduation_markdown(result: dict[str, Any]) -> str:
    completed = simple_course_names(result.get("completed_required", []))
    in_progress = simple_course_names(result.get("in_progress_counted_with_warning", []))
    missing_required = simple_course_names(result.get("missing_required", []))
    missing_alt = missing_alternative_text(result.get("alternative_requirements_status", {}))
    unfinished_parts = []
    if in_progress != "無":
        unfinished_parts.append(f"修課中暫時計入，尚未正式完成：<br>{in_progress}")
    if missing_required != "無":
        unfinished_parts.append(f"尚缺必修：<br>{missing_required}")
    if missing_alt:
        unfinished_parts.append(f"尚缺替代必修：<br>{missing_alt}")
    unfinished = "<br><br>".join(unfinished_parts) if unfinished_parts else "無"

    lab = result.get("lab_elective_status", {})
    credits = result.get("credit_summary", {})
    lab_text = (
        f"{lab.get('credits_counted', 0):g}/{lab.get('min_credits', 0):g} 學分，"
        f"{lab.get('course_count', 0)}/{lab.get('min_courses', 0)} 門"
    )
    credit_text = (
        f"正式完成 {credits.get('completed_credits_official', 0):g} 學分；"
        f"若修課中皆通過，規劃採計 {credits.get('planning_credits_with_in_progress', 0):g} 學分"
    )
    return "\n".join(
        [
            "### 畢業進度檢查",
            "",
            "已依照 **EE112 入學年度規則** 進行檢查。",
            "",
            "| 項目 | 結果 |",
            "|---|---|",
            f"| 已完成 | {completed} |",
            f"| 未完成 | {unfinished} |",
            f"| 必選實驗進度 | {lab_text} |",
            f"| 學分摘要 | {credit_text} |",
        ]
    )


def english_result_markdown(result: dict[str, Any]) -> str:
    intent = result.get("intent", "")
    if intent == "check_graduation":
        completed = len(result.get("completed_required") or [])
        in_progress = len(result.get("in_progress_counted_with_warning") or [])
        missing = len(result.get("missing_required") or [])
        lab = result.get("lab_elective_status", {})
        credits = result.get("credit_summary", {})
        return "\n".join(
            [
                "### Graduation Progress",
                "",
                "Checked against the EE112 requirement rules.",
                "",
                "| Item | Result |",
                "|---|---|",
                f"| Completed required courses | {completed} |",
                f"| In-progress courses counted for planning | {in_progress} |",
                f"| Missing required courses | {missing} |",
                f"| Required lab electives | {lab.get('credits_counted', 0):g}/{lab.get('min_credits', 0):g} credits, {lab.get('course_count', 0)}/{lab.get('min_courses', 0)} courses |",
                f"| Credits | Completed {credits.get('completed_credits_official', 0):g}; planning count {credits.get('planning_credits_with_in_progress', 0):g} |",
            ]
        )

    if intent == "search_course_options":
        candidates = result.get("candidate_courses") or []
        slots = ", ".join(result.get("query_time_slots") or []) or "not specified"
        lines = ["### Available Course Options", "", f"Time slots: `{slots}`", ""]
        if result.get("category_filter") == "general_education":
            lines.extend(["Filter: GE/GEC general education courses", ""])
        if not candidates:
            lines.append("No addable candidate course was found for this time range.")
            return "\n".join(lines)
        lines.extend(["| # | Code | Course | Instructor | Credits | Time |", "|---:|---|---|---|---:|---|"])
        for index, course in enumerate(candidates, start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        h(course_code(course)),
                        h(course_name(course)),
                        h(course_teacher(course) or "N/A"),
                        h(course_credits(course) or "0"),
                        f"`{h(course_time(course) or 'TBA')}`",
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    if intent == "compare_course_teachers":
        course = english_review_course_title(result)
        lines = ["### Teacher Review Summary", "", f"Course: **{h(course)}**", ""]
        summaries = result.get("teacher_summaries") or []
        if not summaries:
            lines.append("No reliable review evidence was found.")
            return "\n".join(lines)
        lines.extend(["| Teacher | Reviews | Coolness | Sweetness |", "|---|---:|---:|---:|"])
        for summary in summaries:
            lines.append(
                f"| {h(summary.get('teacher_name') or 'N/A')} | {int(summary.get('review_count') or 0)} | "
                f"{h(summary.get('avg_coolness') if summary.get('avg_coolness') is not None else '')} | "
                f"{h(summary.get('avg_sweetness') if summary.get('avg_sweetness') is not None else '')} |"
            )
        if result.get("best_teacher"):
            lines.extend(["", f"Current best match from available samples: **{h(result.get('best_teacher'))}**."])
        evidence_lines: list[str] = []
        for summary in summaries:
            if int(summary.get("review_count") or 0) <= 0:
                continue
            teacher = summary.get("teacher_name") or "N/A"
            for item in (summary.get("evidence") or [])[:2]:
                if not isinstance(item, dict):
                    continue
                title = clean_review_text(item.get("title") or "Review source", limit=80)
                url = str(item.get("url") or "").strip().strip("'\"")
                source = clean_review_text(item.get("source") or "", limit=60)
                comment = item.get("short_comment") or item.get("snippet") or ""
                translated = translate_review_comment_for_english(comment)
                link = f"[{h(title)}]({url})" if url else h(title)
                prefix = f"- **{h(teacher)}**"
                if source:
                    prefix += f" ({h(source)})"
                evidence_lines.append(f"{prefix}: {link}")
                evidence_lines.append(f"  - English note: {h(translated)}")
        if evidence_lines:
            lines.extend(["", "#### Review Evidence", ""])
            lines.extend(evidence_lines)
        lines.append("")
        lines.append("Reviews are subjective soft references and are not official course-quality data.")
        return "\n".join(lines)

    courses = result.get("recommended_courses") or []
    if courses:
        total = result.get("total_credits", 0)
        lines = ["### Recommended Schedule", "", f"Total credits: **{total:g}**", ""]
        lines.extend(["| Code | Course | Instructor | Credits | Time | Reason |", "|---|---|---|---:|---|---|"])
        for course in courses:
            reason = english_reason_text(field(course, "recommendation_reason", "reason", "requirement_code"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        h(course_code(course)),
                        h(course_name(course)),
                        h(course_teacher(course) or "N/A"),
                        h(course_credits(course) or "0"),
                        f"`{h(course_time(course) or 'TBA')}`",
                        h(reason),
                    ]
                )
                + " |"
            )
        warnings = result.get("warnings") or []
        if warnings:
            lines.extend(["", "#### Warnings"])
            for warning in warnings[:6]:
                translated = english_warning_text(warning)
                if translated:
                    lines.append(f"- {h(translated)}")
        return "\n".join(lines)

    if intent == "confirm_final":
        return "### Final Schedule Confirmed\n\nThe current schedule has been confirmed and can be exported as an ICS calendar file."

    if intent == "update_constraints":
        return "### Preferences Updated\n\nYour persistent scheduling constraints have been updated."

    return strip_chat_history_section(result.get("agent_explanation") or "")


def display_explanation(result: dict[str, Any]) -> str:
    if is_english():
        return english_result_markdown(result)
    if result.get("intent") == "check_graduation":
        return simplified_graduation_markdown(result)
    return strip_chat_history_section(result.get("agent_explanation") or "")


def summarize_result(result: dict[str, Any], elapsed: float) -> str:
    intent = result.get("intent", "")
    courses = result.get("recommended_courses") or []
    warnings = result.get("warnings") or []
    latency = f"{elapsed:.1f}s"

    if intent == "check_graduation":
        completed_count = len(result.get("completed_required") or [])
        in_progress_count = len(result.get("in_progress_counted_with_warning") or [])
        missing_count = len(result.get("missing_required") or [])
        missing_alt_count = sum(
            1
            for item in (result.get("alternative_requirements_status") or {}).values()
            if not item.get("satisfied")
        )
        unfinished_count = in_progress_count + missing_count + missing_alt_count
        lab = result.get("lab_elective_status", {})
        credits = result.get("credit_summary", {})
        if is_english():
            return (
                f"Graduation progress updated: **{completed_count}** required courses completed, "
                f"**{unfinished_count}** unfinished items. Required labs: "
                f"**{lab.get('credits_counted', 0):g}/{lab.get('min_credits', 0):g} credits**; "
                f"planning credits: **{credits.get('planning_credits_with_in_progress', 0):g}**. `{latency}`"
            )
        return (
            f"畢業進度已更新：已完成 **{completed_count}** 門必修，"
            f"未完成 **{unfinished_count}** 項。必選實驗 "
            f"**{lab.get('credits_counted', 0):g}/{lab.get('min_credits', 0):g} 學分**；"
            f"規劃採計 **{credits.get('planning_credits_with_in_progress', 0):g} 學分**。`{latency}`"
        )

    if intent == "search_course_options":
        candidates = result.get("candidate_courses") or []
        slots = ", ".join(result.get("query_time_slots") or [])
        if candidates:
            names = "、".join(course_label(course) for course in candidates[:3])
            if is_english():
                names = ", ".join(course_label(course) for course in candidates[:3])
                return f"Found **{len(candidates)}** candidate courses. Time slots: `{slots}`. Top matches: {names}. `{latency}`"
            return f"找到 **{len(candidates)}** 門候選課。時段：`{slots}`。前幾門是：{names}。`{latency}`"
        if is_english():
            return f"No addable candidate course was found for this time range. Time slots: `{slots}`. `{latency}`"
        return f"這個時段目前沒有可加入的候選課。時段：`{slots}`。我把未列入原因放在完整說明裡。`{latency}`"

    if intent == "compare_course_teachers":
        course = english_review_course_title(result) if is_english() else result.get("course_name") or "這門課"
        summaries = result.get("teacher_summaries") or []
        teachers = result.get("discovered_teacher_names") or [
            summary.get("teacher_name") for summary in summaries if summary.get("teacher_name")
        ]
        review_count = sum(int(summary.get("review_count") or 0) for summary in summaries)
        if result.get("best_teacher") and review_count:
            if is_english():
                return (
                    f"Checked teacher reviews for {course}. Found {review_count} review samples; "
                    f"{result.get('best_teacher')} currently best matches the preference. `{latency}`"
                )
            return (
                f"已查詢 {course} 的教師評價，共找到 {review_count} 筆心得；"
                f"目前樣本中 {result.get('best_teacher')} 較符合偏好。`{latency}`"
            )
        teacher_text = "、".join(str(teacher) for teacher in teachers if teacher) or "未找到教師名單"
        if is_english():
            teacher_text = ", ".join(str(teacher) for teacher in teachers if teacher) or "no teacher list found"
            return (
                f"Checked instructors for {course}: {teacher_text}. No reliable review sample was found, "
                f"so I will not judge who is cooler or sweeter. `{latency}`"
            )
        return (
            f"已查詢 {course} 的開課教師：{teacher_text}。目前沒有找到足夠可靠的心得樣本，"
            f"所以不判斷誰比較涼或甜。`{latency}`"
        )

    if courses:
        total = result.get("total_credits", 0)
        low_text = "需要低修申請" if low_credit_required(result) else "正常學分"
        names = "、".join(course_code(course) for course in courses[:7])
        warning_text = f"提醒 {min(len(warnings), 3)} 則。" if warnings else "沒有主要提醒。"
        if is_english():
            low_text = "low-credit approval needed" if low_credit_required(result) else "normal credit load"
            names = ", ".join(course_code(course) for course in courses[:7])
            warning_text = f"{min(len(warnings), 3)} warning(s)." if warnings else "No major warnings."
            return f"Schedule updated: **{len(courses)} courses / {total:g} credits**, status: **{low_text}**. Courses: {names}. {warning_text} `{latency}`"
        return f"已更新課表：**{len(courses)} 門 / {total:g} 學分**，狀態：**{low_text}**。課程：{names}。{warning_text} `{latency}`"

    if intent == "confirm_final":
        if is_english():
            return f"Current schedule confirmed. You can export the ICS file. `{latency}`"
        return f"已確認目前課表，可以匯出 ICS。`{latency}`"

    if intent == "update_constraints":
        if is_english():
            return f"Persistent scheduling constraints updated. `{latency}`"
        return f"已更新你的持續性排課限制。`{latency}`"

    explanation = strip_chat_history_section(result.get("agent_explanation") or "")
    if not explanation:
        if is_english():
            return f"Done. `{latency}`"
        return f"已處理完成。`{latency}`"
    return explanation[:260] + ("..." if len(explanation) > 260 else "")


def submit_message(message: str) -> None:
    text = message.strip()
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    started = perf_counter()
    with st.spinner(txt("NTHU COPILOT 正在更新課表...", "NTHU COPILOT is updating your schedule...")):
        result = st.session_state.agent.chat(text)
    elapsed = perf_counter() - started
    st.session_state.last_latency = elapsed
    st.session_state.last_result = result
    st.session_state.messages.append(
        {
            "role": "assistant",
            "summary": summarize_result(result, elapsed),
            "content": display_explanation(result),
            "intent": result.get("intent", ""),
            "candidate_courses": result.get("candidate_courses") or [],
            "result": result,
            "elapsed": elapsed,
        }
    )


def render_header(parser_label: str = "") -> None:
    parser_chip = f'<span class="chip">{h(parser_label)}</span>' if parser_label else ""
    st.markdown(
        f"""
        <div class="app-hero">
          <div class="app-title">NTHU COPILOT</div>
          <div class="app-subtitle">{h(txt("互動式課程規劃助理。先上傳修課紀錄截圖並確認 OCR，再開始排課。", "Interactive course-planning assistant. Upload and confirm your OCR course records before planning."))}</div>
          <div class="chip-row">
            <span class="chip">{h(txt("114 第二學期", "Spring 2026"))}</span>
            <span class="chip">{h(txt("EE112 畢業規則", "EE112 requirements"))}</span>
            <span class="chip">{h(txt("OCR 修課紀錄", "OCR course records"))}</span>
            <span class="chip">{h(txt("ICS 匯出", "ICS export"))}</span>
            {parser_chip}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_prompts() -> None:
    if is_english():
        pool = [
            ("20 credits", "Plan a 20-credit 114-2 schedule."),
            ("18 credits", "Plan an 18-credit 114-2 schedule."),
            ("Graduation gaps", "What graduation requirements am I still missing?"),
            ("No 8am", "Avoid 8am classes."),
            ("No labs", "I do not want lab courses."),
            ("Fri 10-12", "What courses are available Friday from 10:00 to 12:00?"),
            ("Wed afternoon", "What courses are available Wednesday afternoon from 15:00 to 17:00?"),
            ("Lighter", "Plan 16 to 25 credits, not too hard, and consider online reviews."),
            ("Probability", "Which Probability teacher is easier?"),
            ("No Linear Alg", "I do not want Linear Algebra."),
        ]
    else:
        pool = [
            ("排 20 學分", "幫我排 114 第二學期課表，20 學分"),
            ("排 18 學分", "幫我排 114 第二學期課表，18 學分"),
            ("畢業缺什麼", "我想知道畢業還缺什麼"),
            ("不要早八", "我不要早八"),
            ("不要實驗課", "我不要實驗課"),
            ("週五 10-12", "星期五早上十點到十二點有什麼課可以選"),
            ("週三下午", "星期三下午三點到五點有什麼課可以選"),
            ("想輕鬆", "幫我排 16 到 25 學分，不要太硬，可以參考網路評價"),
            ("機率評價", "機率老師哪個比較涼"),
            ("線代不要", "我不要線性代數"),
        ]
    if "quick_prompt_choices" not in st.session_state:
        st.session_state.quick_prompt_choices = random.sample(pool, k=min(5, len(pool)))
    prompts = list(st.session_state.quick_prompt_choices)
    if current_plan().get("recommended_courses"):
        prompts[-1] = ("Confirm", "Finalize the schedule.") if is_english() else ("確認課表", "我決定好了")
    columns = st.columns(len(prompts))
    for index, (label, prompt) in enumerate(prompts):
        with columns[index]:
            if st.button(label, use_container_width=True, key=f"quick_{index}"):
                submit_message(prompt)
                rerun()


def render_chat_messages() -> None:
    messages = st.session_state.get("messages") or initial_messages()
    for index, message in enumerate(messages):
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if role == "assistant":
                result = message.get("result")
                if not isinstance(result, dict):
                    last_result = st.session_state.get("last_result")
                    if index == len(messages) - 1 and isinstance(last_result, dict) and last_result:
                        result = last_result
                if message.get("kind") == "welcome" or (index == 0 and not result):
                    summary = initial_messages()[0]["summary"]
                    content = ""
                    intent = ""
                elif isinstance(result, dict) and result:
                    elapsed = message.get("elapsed")
                    if not isinstance(elapsed, (int, float)):
                        elapsed = st.session_state.get("last_latency")
                    summary = summarize_result(result, float(elapsed or 0.0))
                    content = display_explanation(result)
                    intent = str(result.get("intent") or message.get("intent") or "")
                else:
                    summary = message.get("summary") or message.get("content") or ""
                    content = str(message.get("content") or "").strip()
                    intent = str(message.get("intent") or "")
                st.markdown(f'<div class="compact-reply">{summary}</div>', unsafe_allow_html=True)
                if content and content != summary:
                    should_expand = intent in {
                        "search_course_options",
                        "compare_course_teachers",
                        "check_graduation",
                    }
                    should_expand = should_expand or (index == len(messages) - 1 and len(content) < 1800)
                    with st.expander(txt("完整說明", "Full details"), expanded=should_expand):
                        st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-user">{h(message.get("content", ""))}</div>', unsafe_allow_html=True)


def constraint_labels() -> list[str]:
    agent = st.session_state.get("agent")
    if not agent:
        return []
    constraints = agent._constraints_as_preferences()
    labels: list[str] = []
    if constraints.get("exclude_lab_courses"):
        labels.append(txt("不要實驗課", "No lab courses"))
    slots = constraints.get("exclude_time_slots") or []
    if slots:
        labels.append(txt("避開 ", "Avoid ") + format_time_slot_constraint(slots))
    excluded = constraints.get("exclude_course_codes") or []
    if excluded:
        labels.append(txt(f"排除 {len(excluded)} 門課", f"Exclude {len(excluded)} courses"))
    if constraints.get("prefer_theory_ee_courses"):
        labels.append(txt("偏好電機理論課", "Prefer EE theory courses"))
    return labels


def render_status_panel(plan: dict[str, Any]) -> None:
    courses = plan.get("recommended_courses") or []
    total_credits = plan.get("total_credits", 0)
    credit_range = plan.get("target_credit_range") or {}
    range_text = ""
    if credit_range:
        range_text = txt(f"目標 {credit_range.get('min', '')}-{credit_range.get('max', '')}", f"Target {credit_range.get('min', '')}-{credit_range.get('max', '')}")
    low = low_credit_required(plan)
    overload = overload_required(plan)
    status = txt("低修", "Low") if low else txt("超修", "Over") if overload else txt("正常", "Normal")
    status_class = "status-warn" if (low or overload) else "status-good"
    latency = st.session_state.get("last_latency")
    latency_text = f"{latency:.1f}s" if isinstance(latency, (int, float)) else txt("尚未送出", "not submitted yet")

    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">{h(txt("目前總學分", "Current credits"))}</div>
            <div class="metric-value">{h(f"{total_credits:g}" if isinstance(total_credits, (int, float)) else total_credits)}</div>
            <div class="metric-note">{h(range_text or txt("尚未設定目標", "No target set"))}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">{h(txt("課程數", "Courses"))}</div>
            <div class="metric-value">{len(courses)}</div>
            <div class="metric-note">{h(txt("含 0 學分課程", "Includes 0-credit courses"))}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">{h(txt("狀態", "Status"))}</div>
            <div class="metric-value {status_class}">{status}</div>
            <div class="metric-note">{h(txt("上次回應", "Last response"))} {h(latency_text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = constraint_labels()
    with st.expander(txt("目前限制", "Current constraints"), expanded=bool(labels)):
        if labels:
            st.markdown(" ".join(f'<span class="chip">{h(label)}</span>' for label in labels), unsafe_allow_html=True)
        else:
            st.caption(txt("尚未設定持續性限制。", "No persistent constraints yet."))

    warnings = plan.get("warnings") or []
    if warnings:
        with st.expander(txt("提醒", "Warnings"), expanded=True):
            for warning in warnings[:6]:
                st.warning(str(warning))


def render_course_cards(courses: list[dict[str, Any]]) -> None:
    if not courses:
        st.markdown(
            f'<div class="empty-note">{h(txt("目前還沒有產生課表。先輸入排課需求，或按左側快捷按鈕。", "No schedule has been generated yet. Enter a planning request or use the quick buttons on the left."))}</div>',
            unsafe_allow_html=True,
        )
        return
    cards = ['<div class="course-grid">']
    for course in courses:
        meta = [
            f"{course_credits(course)} {txt('學分', 'credits')}" if course_credits(course) else "",
            course_teacher(course),
            course_time(course),
        ]
        reason = field(course, "recommendation_reason", "reason", "requirement_code", "review_note")
        cards.append(
            "<div class='course-card'>"
            f"<div class='course-code'>{h(course_code(course) or txt('未標示課號', 'No course code'))}</div>"
            f"<div class='course-title'>{h(course_name(course) or txt('未命名課程', 'Untitled course'))}</div>"
            f"<div class='course-meta'>{h(' · '.join(item for item in meta if item))}</div>"
            f"<div class='course-reason'>{h(reason)}</div>"
            "</div>"
        )
    cards.append("</div>")
    st.markdown("\n".join(cards), unsafe_allow_html=True)


def timetable_slot_map(courses: list[dict[str, Any]], color_mode: str = "course") -> dict[tuple[str, str], list[dict[str, str]]]:
    slot_map: dict[tuple[str, str], list[dict[str, str]]] = {}
    for course in courses:
        slots = sorted(parse_time_slots(course_time(course)))
        for slot in slots:
            if len(slot) < 2:
                continue
            day = slot[:1].upper()
            period = slot[1:].lower()
            slot_map.setdefault((period, day), []).append(
                {
                    "code": course_code(course),
                    "name": course_name(course),
                    "time": course_time(course),
                    "style": slot_color_style(course, color_mode),
                }
            )
    return slot_map


def render_timetable(courses: list[dict[str, Any]], color_mode: str = "course") -> None:
    if not courses:
        st.markdown(f'<div class="empty-note">{h(txt("還沒有課程可顯示。", "No courses to display yet."))}</div>', unsafe_allow_html=True)
        return
    slot_map = timetable_slot_map(courses, color_mode)
    rows = [
        "<div class='timetable-wrap'><table class='timetable'>",
        f"<thead><tr><th class='period-cell'>{h(txt('節次', 'Period'))}</th><th>{h(txt('時間', 'Time'))}</th>"
        + "".join(f"<th>{h(label)}</th>" for _, label in day_labels())
        + "</tr></thead><tbody>",
    ]
    for period in PERIODS:
        rows.append(f"<tr><td class='period-cell'>{h(PERIOD_LABELS.get(period, period))}</td><td>{h(PERIOD_TIMES.get(period, ''))}</td>")
        for day_code, _ in day_labels():
            items = slot_map.get((period, day_code), [])
            if not items:
                rows.append("<td></td>")
                continue
            cards = []
            for item in items:
                cards.append(
                    f"<div class='slot-card' style='{h(item.get('style', ''))}'>"
                    f"<div class='slot-code'>{h(item['code'])}</div>"
                    f"<div class='slot-name'>{h(item['name'])}</div>"
                    f"<div class='slot-time'>{h(item['time'])}</div>"
                    "</div>"
                )
            rows.append("<td>" + "".join(cards) + "</td>")
        rows.append("</tr>")
    rows.append("</tbody></table></div>")
    st.markdown("\n".join(rows), unsafe_allow_html=True)


def render_candidate_options(result: dict[str, Any]) -> None:
    candidates = result.get("candidate_courses") or []
    if not candidates:
        return
    st.subheader(txt("候選課程", "Candidate Courses"))
    for index, course in enumerate(candidates[:6], start=1):
        label = course_label(course)
        meta = " · ".join(
            item
            for item in [
                f"{course_credits(course)} {txt('學分', 'credits')}" if course_credits(course) else "",
                course_teacher(course),
                course_time(course),
            ]
            if item
        )
        with st.expander(f"{index}. {label}", expanded=index <= 2):
            st.caption(meta)
            if st.button(txt(f"選第 {index} 個", f"Choose #{index}"), key=f"choose_candidate_{index}"):
                submit_message(f"Choose option {index}" if is_english() else f"選第 {index} 個")
                rerun()


def export_current_schedule(courses: list[dict[str, Any]]) -> None:
    submit_message("Finalize the schedule." if is_english() else "我決定好了")
    st.session_state.calendar_result = export_schedule_to_ics(courses, output_path=str(ICS_PATH))


def render_export_panel(courses: list[dict[str, Any]]) -> None:
    if st.button(txt("確認並匯出 ICS", "Confirm and Export ICS"), disabled=not bool(courses), use_container_width=True):
        export_current_schedule(courses)
        rerun()

    calendar_result = st.session_state.get("calendar_result")
    if calendar_result and calendar_result.get("ics_path"):
        ics_path = Path(calendar_result["ics_path"])
        st.success(txt(f"已匯出 {ics_path.name}", f"Exported {ics_path.name}"))
        if ics_path.exists():
            st.download_button(
                txt("下載 ICS", "Download ICS"),
                data=ics_path.read_bytes(),
                file_name=ics_path.name,
                mime="text/calendar",
                use_container_width=True,
            )


def render_language_switch() -> None:
    ensure_language_state()
    _, language_col = st.columns([0.78, 0.22])
    with language_col:
        current = ui_language()
        selected = st.radio(
            txt("語言", "Language"),
            ["zh", "en"],
            index=0 if current == "zh" else 1,
            format_func=lambda value: "中文" if value == "zh" else "English",
            horizontal=True,
            label_visibility="collapsed",
            key="ui_language_choice",
        )
    if selected != current:
        st.session_state.ui_language = selected
        st.session_state.pop("quick_prompt_choices", None)
        messages = st.session_state.get("messages") or []
        if len(messages) <= 1:
            st.session_state.messages = initial_messages()
        rerun()


def main() -> None:
    st.set_page_config(page_title="NTHU COPILOT", layout="wide", initial_sidebar_state="expanded")
    ensure_language_state()
    inject_css()
    ensure_ocr_flow_state()
    gemini_key = ensure_gemini_api_key()
    confirmed_for_chat = ocr_ready_for_chat()
    student_path = str(default_student_path())

    with st.sidebar:
        st.header(txt("設定", "Settings"))
        intent_label = st.selectbox(
            "Intent parser",
            ["gemini", "rule", "ollama"],
            index=0,
            format_func=lambda value: {"gemini": "Gemini", "rule": txt("規則", "Rule"), "ollama": "Ollama"}[value],
            help=txt(
                "Gemini 只負責理解你的需求；排課、查衝堂與 PTT 評價仍由本地工具執行。",
                "Gemini only parses your request. Planning, conflict checks, and review lookup still run through local tools.",
            ),
        )
        intent_provider = intent_label
        if intent_provider == "gemini":
            model = st.selectbox(
                txt("Gemini 模型", "Gemini model"),
                ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
                index=0,
            )
            if gemini_key:
                st.caption(txt("已從環境變數或 private/gemini_api_key.txt 讀取 Gemini key。", "Gemini key loaded from environment or private/gemini_api_key.txt."))
            else:
                st.warning(txt("尚未找到 Gemini API key。請放到 private/gemini_api_key.txt 或設定 GEMINI_API_KEY。", "Gemini API key not found. Put it in private/gemini_api_key.txt or set GEMINI_API_KEY."))
        elif intent_provider == "ollama":
            st.caption(txt("Ollama 會跑本機模型，通常比 Gemini 慢。", "Ollama runs a local model and is usually slower than Gemini."))
            model = st.selectbox(
                txt("Ollama 模型", "Ollama model"),
                ["phi4-mini:latest", "llama3.1:8b", "mistral:latest"],
                index=0,
            )
        else:
            st.caption(txt("使用規則 parser，速度最快，但口語理解較弱。", "Rule parser is fastest but less flexible for natural language."))
            model = ""
        if st.button(txt("重開對話", "Reset chat"), use_container_width=True):
            reset_chat()
            rerun()

    parser_label = {
        "gemini": f"Parser: Gemini ({model or 'gemini-2.5-flash'})",
        "ollama": f"Parser: Ollama ({model or 'phi4-mini:latest'})",
        "rule": txt("Parser: 規則", "Parser: Rule"),
    }.get(intent_provider, txt("Parser: 規則", "Parser: Rule"))

    if confirmed_for_chat:
        agent = ensure_agent(student_path, intent_provider, model)
        try:
            load_agent_once(agent)
        except Exception as exc:
            st.error(txt("資料載入失敗，請確認 OCR 確認後的修課紀錄、114_2_course_data.xlsx 和 EE_112_rules.json 都存在。", "Failed to load data. Check confirmed OCR records, 114_2_course_data.xlsx, and EE_112_rules.json."))
            st.exception(exc)
            st.stop()

    render_language_switch()
    render_header(parser_label)
    st.markdown('<div id="main-split-layout-anchor" class="main-split-layout-anchor"></div>', unsafe_allow_html=True)
    mount_resizable_splitter()
    chat_col, plan_col = st.columns([1.2, 1], gap="large")

    with chat_col:
        st.subheader(txt("聊天", "Chat"))
        if confirmed_for_chat:
            render_quick_prompts()
        render_chat_messages()
        edited_df = None if confirmed_for_chat else render_ocr_gate_panel()
        placeholder = txt(
            "上傳修課截圖，或確認後輸入排課需求...",
            "Upload a course screenshot, or enter a planning request after confirmation...",
        ) if not confirmed_for_chat else txt(
            "輸入排課需求，或上傳新截圖重新 OCR...",
            "Enter a planning request, or upload a new screenshot to rerun OCR...",
        )
        if st.session_state.get("ocr_flow_state") == "need_upload":
            lock_chat_text_until_upload()
        chat_value = st.chat_input(
            placeholder,
            accept_file=True,
            file_type=["png", "jpg", "jpeg"],
        )
        text, files = get_chat_input_text_and_files(chat_value)
        if files:
            with st.spinner(txt("正在執行 OCR，第一次可能會比較久...", "Running OCR. The first run may take a while...")):
                st.session_state.messages.append({"role": "user", "content": txt(f"上傳修課紀錄截圖：{files[0].name}", f"Uploaded course-record screenshot: {files[0].name}")})
                parse_uploaded_ocr(files[0])
            rerun()
        elif not confirmed_for_chat:
            if handle_preflight_chat_input(text, files, edited_df):
                rerun()
        elif text:
            submit_message(text)
            rerun()
        if confirmed_for_chat:
            render_candidate_options(st.session_state.get("last_result") or {})

    with plan_col:
        st.subheader(txt("課表儀表板", "Schedule Dashboard"))
        plan = current_plan()
        courses = plan.get("recommended_courses") or []
        render_status_panel(plan)
        render_export_panel(courses)

        tab_courses, tab_timetable, tab_raw = st.tabs([txt("課程卡片", "Course Cards"), txt("週課表", "Weekly Timetable"), txt("資料", "Data")])
        with tab_courses:
            render_course_cards(courses)
        with tab_timetable:
            _, color_mode_col = st.columns([1, 0.46])
            with color_mode_col:
                color_mode_label = st.selectbox(
                    txt("分色", "Color by"),
                    color_mode_options(),
                    index=2,
                    key=f"timetable_color_mode_v2_{ui_language()}",
                )
            render_timetable(courses, color_mode_value(color_mode_label))
        with tab_raw:
            if plan:
                st.json(
                    {
                        "total_credits": plan.get("total_credits"),
                        "course_count": len(courses),
                        "low_credit": low_credit_required(plan),
                        "overload": overload_required(plan),
                        "warnings": (plan.get("warnings") or [])[:6],
                    }
                )
            else:
                st.caption(txt("尚未產生 structured result。", "No structured result yet."))


if __name__ == "__main__":
    main()
