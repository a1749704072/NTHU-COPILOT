from __future__ import annotations

import csv
import html
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    from trace_utils import langsmith_trace
except ImportError:  # pragma: no cover - package style import
    from .trace_utils import langsmith_trace


PTT_BOARD_URL = "https://www.ptt.cc/bbs/NTHU_Course/index.html"
PTT_BASE_URL = "https://www.ptt.cc"
LOCAL_CACHE_PATH = Path(__file__).resolve().parent / "data" / "course_reviews_sample.csv"
PTT_RAG_SEED_PATH = Path(__file__).resolve().parent / "data" / "ptt_rag_seed_urls.txt"
SUPPORTED_SOURCES = ("local_cache", "ptt_rag", "ptt", "web")
REVIEW_WARNINGS = [
    "Online course reviews are subjective and should only be used as soft reference.",
    "Small sample sizes may be biased, incomplete, or outdated.",
    "Reviews must not override graduation requirements, time conflicts, prerequisites, official rules, or 11420 course availability.",
    "If no reliable review is found, the system must not guess.",
]


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _strip_tags(fragment: str) -> str:
    parser = _TextExtractor()
    parser.feed(fragment)
    return html.unescape(parser.text())


def _fetch_url(url: str, timeout: int = 10) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 CoursePlanningAgent/0.2",
            "Cookie": "over18=1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


@lru_cache(maxsize=256)
def _fetch_url_cached(url: str, timeout: int = 10) -> str:
    return _fetch_url(url, timeout=timeout)


def _parse_index_entries(index_html: str) -> tuple[list[dict], str]:
    entries: list[dict] = []
    pattern = re.compile(
        r'<div class="r-ent">(?P<block>.*?)</div>\s*</div>',
        flags=re.DOTALL,
    )
    for match in pattern.finditer(index_html):
        block = match.group("block")
        title_match = re.search(r'<div class="title">\s*<a href="(?P<href>[^"]+)">(?P<title>.*?)</a>', block, re.DOTALL)
        if not title_match:
            continue
        date_match = re.search(r'<div class="date">\s*(?P<date>.*?)\s*</div>', block, re.DOTALL)
        entries.append(
            {
                "title": _strip_tags(title_match.group("title")),
                "url": urllib.parse.urljoin(PTT_BASE_URL, title_match.group("href")),
                "date_hint": _strip_tags(date_match.group("date")) if date_match else "",
            }
        )

    previous_url = ""
    for href, label in re.findall(r'<a class="btn wide" href="([^"]+)">(.*?)</a>', index_html, flags=re.DOTALL):
        if "上頁" in _strip_tags(label):
            previous_url = urllib.parse.urljoin(PTT_BASE_URL, href)
            break
    return entries, previous_url


def _parse_article(article_html: str) -> dict:
    title = ""
    author = ""
    date = ""
    meta_values = [_strip_tags(item) for item in re.findall(r'<span class="article-meta-value">(.*?)</span>', article_html, re.DOTALL)]
    if len(meta_values) >= 1:
        author = meta_values[0]
    if len(meta_values) >= 3:
        title = meta_values[2]
    if len(meta_values) >= 4:
        date = meta_values[3]

    main_match = re.search(r'<div id="main-content"[^>]*>', article_html, flags=re.DOTALL)
    body_html = article_html[main_match.end() :] if main_match else article_html
    body_html = re.split(r'</body>|</html>', body_html, maxsplit=1, flags=re.IGNORECASE | re.DOTALL)[0]
    body_html = re.sub(r'<div class="article-metaline.*?</div>', "", body_html, flags=re.DOTALL)
    body_html = re.sub(r'<span class="f2">.*?</span>', "", body_html, flags=re.DOTALL)
    body = _strip_tags(body_html)
    body = re.split(r"※ 發信站|--\s*$", body, maxsplit=1)[0].strip()
    return {"title": title, "author": author, "date": date, "content": body}


def _keyword_match(text: str, keywords: list[str], match_all: bool = False) -> bool:
    normalized = text.lower()
    cleaned = [keyword.lower().strip() for keyword in keywords if keyword and keyword.strip()]
    if not cleaned:
        return True
    if match_all:
        return all(keyword in normalized for keyword in cleaned)
    return any(keyword in normalized for keyword in cleaned)


def _ptt_search_url(query: str) -> str:
    return f"{PTT_BASE_URL}/bbs/NTHU_Course/search?{urllib.parse.urlencode({'q': query})}"


def _search_queries(course_name: str, teacher_name: str) -> list[str]:
    queries: list[str] = []
    for query in (
        f"{course_name} {teacher_name}",
        teacher_name,
        course_name,
    ):
        query = _clean(query)
        if query:
            queries.append(query)
    for value in (course_name, teacher_name):
        for part in re.split(r"[\s,，、/／;；()（）-]+", _clean(value)):
            if len(part.strip()) >= 2:
                queries.append(part.strip())
    return list(dict.fromkeys(queries))


def _query_matches_article(article_text: str, title: str, course_name: str, teacher_name: str) -> bool:
    return _strict_course_teacher_match(article_text, title, course_name, teacher_name)


def _search_ptt_entries_by_query(query: str, max_pages: int, timeout: int) -> tuple[list[dict], int, list[str]]:
    entries: list[dict] = []
    warnings: list[str] = []
    checked = 0
    next_url = _ptt_search_url(query)
    for _ in range(max_pages):
        try:
            index_html = _fetch_url(next_url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            warnings.append(f"PTT search failed for query '{query}': {exc}")
            break
        page_entries, previous_url = _parse_index_entries(index_html)
        checked += len(page_entries)
        entries.extend(page_entries)
        if not previous_url:
            break
        next_url = previous_url
    return entries, checked, warnings


def _parse_score(raw_value: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None
    if "★" in value:
        count = value.count("★")
        return float(count) if count else None
    chinese = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    if value[0] in chinese:
        return float(chinese[value[0]])
    number_match = re.search(r"([0-5](?:\.\d+)?)", value)
    if number_match:
        score = float(number_match.group(1))
        if 0 <= score <= 5:
            return score
    return None


def _parse_optional_score(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        score = float(text)
    except ValueError:
        return _parse_score(text)
    return score if 0 <= score <= 5 else None


def extract_review_scores(article_text: str) -> dict:
    score_patterns = {
        "coolness": r"(?:涼度|涼爽度|爽度)\s*[:：]?\s*([★☆]+|[0-5](?:\.\d+)?|[一二三四五])",
        "sweetness": r"(?:甜度|給分甜度|甜)\s*[:：]?\s*([★☆]+|[0-5](?:\.\d+)?|[一二三四五])",
    }
    result: dict[str, float | None] = {"coolness": None, "sweetness": None}
    for key, pattern in score_patterns.items():
        match = re.search(pattern, article_text, flags=re.IGNORECASE)
        if match:
            result[key] = _parse_score(match.group(1))
    return result


def _keyword_heuristic_scores(article_text: str) -> dict:
    """Use only explicit qualitative evidence as a soft fallback when no numeric score exists."""
    text = re.sub(r"\s+", "", article_text)
    cool_positive = ("很涼", "偏涼", "蠻涼", "超涼", "涼課", "輕鬆", "loading低", "作業少", "不點名")
    cool_negative = ("很硬", "偏硬", "超硬", "爆肝", "loading重", "作業多", "很累", "考試難")
    sweet_positive = ("很甜", "偏甜", "蠻甜", "超甜", "甜課", "給分甜", "高分", "調分", "好過")
    sweet_negative = ("給分硬", "很殺", "偏殺", "低分", "不好過", "當人", "很雷")

    def score(positive_terms: tuple[str, ...], negative_terms: tuple[str, ...]) -> float | None:
        positive = sum(1 for term in positive_terms if term in text)
        negative = sum(1 for term in negative_terms if term in text)
        if not positive and not negative:
            return None
        raw = 3.0 + min(1.4, positive * 0.7) - min(1.4, negative * 0.7)
        return max(1.0, min(5.0, round(raw, 1)))

    return {
        "coolness": score(cool_positive, cool_negative),
        "sweetness": score(sweet_positive, sweet_negative),
    }


def _review_scores_with_fallback(article_text: str) -> tuple[dict, str]:
    explicit = extract_review_scores(article_text)
    heuristic = _keyword_heuristic_scores(article_text)
    merged = {
        "coolness": explicit.get("coolness") if explicit.get("coolness") is not None else heuristic.get("coolness"),
        "sweetness": explicit.get("sweetness") if explicit.get("sweetness") is not None else heuristic.get("sweetness"),
    }
    source = "explicit" if any(value is not None for value in explicit.values()) else "keyword_heuristic"
    if all(value is None for value in merged.values()):
        source = "none"
    return merged, source


def _short_comment(article_text: str, max_chars: int = 160) -> str:
    useful_lines = []
    for line in article_text.splitlines():
        text = line.strip()
        if not text:
            continue
        if any(token in text for token in ["涼", "甜", "作業", "考試", "老師", "心得", "評分"]):
            useful_lines.append(text)
        if len(" ".join(useful_lines)) >= max_chars:
            break
    comment = " ".join(useful_lines) or "No short comment extracted."
    return comment[:max_chars]


def _empty_result(course_name: str, teacher_name: str, sources_used: list[str], warnings: list[str] | None = None) -> dict:
    return {
        "course_name": course_name,
        "teacher_name": teacher_name,
        "sources_used": sources_used,
        "review_count": 0,
        "avg_coolness": None,
        "avg_sweetness": None,
        "evidence": [],
        "warnings": list(dict.fromkeys(REVIEW_WARNINGS + (warnings or []))),
    }


def _summarize_evidence(course_name: str, teacher_name: str, sources_used: list[str], evidence: list[dict], warnings: list[str]) -> dict:
    coolness_scores = [float(item["coolness"]) for item in evidence if item.get("coolness") is not None]
    sweetness_scores = [float(item["sweetness"]) for item in evidence if item.get("sweetness") is not None]
    if not evidence:
        warnings.append("No reliable matching review was found. The system must not infer review scores.")
    elif any(item.get("score_source") == "keyword_heuristic" for item in evidence):
        warnings.append("Some coolness/sweetness scores are keyword heuristics from matched PTT evidence, not explicit numeric ratings.")
    return {
        "course_name": course_name,
        "teacher_name": teacher_name,
        "sources_used": list(dict.fromkeys(sources_used)),
        "review_count": len(evidence),
        "avg_coolness": round(statistics.mean(coolness_scores), 2) if coolness_scores else None,
        "avg_sweetness": round(statistics.mean(sweetness_scores), 2) if sweetness_scores else None,
        "evidence": evidence,
        "warnings": list(dict.fromkeys(REVIEW_WARNINGS + warnings)),
    }


def _load_ptt_rag_seed_urls(path: Path = PTT_RAG_SEED_PATH) -> list[str]:
    if not path.exists():
        return []
    urls: list[str] = []
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    except OSError:
        return []
    return list(dict.fromkeys(urls))


def _chunk_text(text: str, chunk_chars: int = 900, overlap: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", _clean(text))
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
        start += max(1, chunk_chars - overlap)
    return chunks


def _rag_terms(course_name: str, teacher_name: str) -> list[str]:
    terms = []
    for value in (course_name, teacher_name):
        text = _clean(value)
        if text:
            terms.append(text.lower())
            terms.extend(part.lower() for part in re.split(r"[\s,，、/／;；()（）-]+", text) if len(part.strip()) >= 2)
    return list(dict.fromkeys(terms))


def _rag_score(text: str, course_name: str, teacher_name: str) -> float:
    lowered = text.lower()
    score = 0.0
    if course_name and course_name.lower() in lowered:
        score += 8.0
    if teacher_name and teacher_name.lower() in lowered:
        score += 10.0
    for term in _rag_terms(course_name, teacher_name):
        if term in lowered:
            score += 1.0
    if any(token in lowered for token in ("心得", "涼度", "甜度", "老師", "課名")):
        score += 1.0
    return score


def _contains_query_term(text: str, query: str) -> bool:
    query = _clean(query).lower()
    if not query:
        return False
    lowered = text.lower()
    if query in lowered:
        return True
    parts = [part.lower() for part in re.split(r"[\s,，、/／;；()（）-]+", query) if len(part.strip()) >= 2]
    return any(part in lowered for part in parts)


def _first_review_block(text: str, max_chars: int = 1200) -> str:
    return _clean(text)[:max_chars]


def _has_explicit_teacher_header(text: str, teacher_name: str) -> bool:
    teacher_name = _clean(teacher_name)
    if not teacher_name:
        return False
    first_block = _first_review_block(text)
    patterns = (
        rf"(?:老師|授課老師|教師|教授)\s*[:：]?\s*{re.escape(teacher_name)}",
        rf"{re.escape(teacher_name)}\s*(?:老師|教授)",
    )
    return any(re.search(pattern, first_block, flags=re.IGNORECASE) for pattern in patterns)


def _has_explicit_course_header(text: str, course_name: str) -> bool:
    course_name = _clean(course_name)
    if not course_name:
        return False
    first_block = _first_review_block(text)
    patterns = (
        rf"(?:課名|課程|科目)\s*[:：]?\s*{re.escape(course_name)}",
    )
    return any(re.search(pattern, first_block, flags=re.IGNORECASE) for pattern in patterns)


def _strict_course_teacher_match(article_text: str, title: str, course_name: str, teacher_name: str) -> bool:
    """Keep teacher/course review evidence from drifting to a nearby article."""
    course_name = _clean(course_name)
    teacher_name = _clean(teacher_name)
    title = _clean(title)
    if not course_name or not teacher_name:
        haystack = f"{title}\n{article_text}"
        if course_name:
            return _contains_query_term(haystack, course_name)
        if teacher_name:
            return _contains_query_term(haystack, teacher_name)
        return False

    title_has_course = _contains_query_term(title, course_name)
    title_has_teacher = _contains_query_term(title, teacher_name)
    first_block = _first_review_block(article_text)
    header_has_course = _has_explicit_course_header(first_block, course_name)
    header_has_teacher = _has_explicit_teacher_header(first_block, teacher_name)

    if title_has_course and title_has_teacher:
        return True
    if title_has_course and header_has_teacher:
        return True
    if title_has_teacher and header_has_course:
        return True
    return header_has_course and header_has_teacher


def _rag_evidence_matches_article(article_text: str, title: str, course_name: str, teacher_name: str) -> bool:
    return _strict_course_teacher_match(article_text, title, course_name, teacher_name)


@langsmith_trace("reviews.search_ptt_rag_reviews", run_type="retriever")
def search_ptt_rag_reviews(
    course_name: str = "",
    teacher_name: str = "",
    seed_urls: list[str] | None = None,
    max_results: int = 5,
    timeout: int = 10,
) -> dict:
    """Retrieve relevant chunks from seeded PTT articles, then extract review evidence.

    This is a lightweight RAG adapter: PTT article text is fetched, chunked,
    retrieved by course/teacher query terms, and only retrieved evidence is used
    for score extraction. It does not invent reviews when no chunk matches.
    """
    urls = list(dict.fromkeys(seed_urls or _load_ptt_rag_seed_urls()))
    if not urls:
        return _empty_result(
            course_name,
            teacher_name,
            ["ptt_rag"],
            [f"No PTT RAG seed URLs were found at {PTT_RAG_SEED_PATH}."],
        )

    evidence_candidates: list[dict] = []
    warnings: list[str] = ["PTT RAG retrieves seeded PTT articles and uses only matched chunks as evidence."]
    for url in urls:
        try:
            article = _parse_article(_fetch_url_cached(url, timeout=timeout))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            warnings.append(f"PTT RAG seed article could not be fetched: {url} ({exc})")
            continue

        article_text = f"{article.get('title', '')}\n{article.get('content', '')}"
        if not _rag_evidence_matches_article(article.get("content", ""), article.get("title", ""), course_name, teacher_name):
            continue
        chunks = _chunk_text(article_text)
        scored_chunks = [
            (_rag_score(chunk, course_name, teacher_name), chunk)
            for chunk in chunks
        ]
        scored_chunks = [(score, chunk) for score, chunk in scored_chunks if score > 0]
        if not scored_chunks:
            continue
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        best_score, best_chunk = scored_chunks[0]
        full_text_score = _rag_score(article_text, course_name, teacher_name)
        scores, score_source = _review_scores_with_fallback(article_text if full_text_score > 0 else best_chunk)
        evidence_candidates.append(
            {
                "source": "PTT RAG NTHU_Course",
                "title": article.get("title") or "PTT NTHU_Course article",
                "url": url,
                "date": article.get("date", ""),
                "snippet": best_chunk[:240],
                "short_comment": best_chunk[:160],
                "coolness": scores.get("coolness"),
                "sweetness": scores.get("sweetness"),
                "score_source": score_source,
                "retrieval_score": round(best_score, 2),
            }
        )

    evidence_candidates.sort(
        key=lambda item: (
            item.get("sweetness") is None and item.get("coolness") is None,
            -float(item.get("retrieval_score") or 0),
        )
    )
    return _summarize_evidence(
        course_name,
        teacher_name,
        ["ptt_rag"],
        evidence_candidates[:max_results],
        warnings,
    )


def _normalize_sources(sources: list[str] | None) -> list[str]:
    if not sources:
        return ["ptt_rag", "ptt"]
    normalized: list[str] = []
    for source in sources:
        value = _clean(source).lower()
        if value in {"ptt_rag", "rag", "ptt-rag", "ptt rag"}:
            normalized.append("ptt_rag")
        elif value in {"ptt", "ptt_nthu_course"}:
            normalized.append("ptt_rag")
            normalized.append("ptt")
        elif value in {"local", "cache", "local_cache", "csv"}:
            normalized.append("local_cache")
        elif value in {"web", "dcard", "google", "blog", "forum", "website", "online"}:
            normalized.append("web")
    return list(dict.fromkeys(normalized)) or ["local_cache"]


def _match_cache_row(row: dict, course_name: str, teacher_name: str) -> bool:
    course_query = course_name.lower().strip()
    teacher_query = teacher_name.lower().strip()
    row_course = _clean(row.get("course_name")).lower()
    row_teacher = _clean(row.get("teacher_name")).lower()
    row_text = " ".join(
        _clean(row.get(key)).lower()
        for key in ("course_name", "teacher_name", "title", "snippet")
    )
    course_ok = not course_query or course_query in row_course or row_course in course_query or course_query in row_text
    teacher_ok = not teacher_query or teacher_query in row_teacher or teacher_query in row_text
    return course_ok and teacher_ok


def _search_local_cache_reviews(course_name: str, teacher_name: str, max_results: int = 10, cache_path: Path = LOCAL_CACHE_PATH) -> dict:
    if not cache_path.exists():
        return _empty_result(
            course_name,
            teacher_name,
            ["local_cache"],
            [f"Local review cache was not found at {cache_path}."],
        )

    evidence: list[dict] = []
    warnings: list[str] = []
    try:
        with cache_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not _match_cache_row(row, course_name, teacher_name):
                    continue
                evidence.append(
                    {
                        "source": _clean(row.get("source")) or "Local review cache",
                        "title": _clean(row.get("title")) or f"{_clean(row.get('course_name'))} review",
                        "url": _clean(row.get("url")),
                        "date": _clean(row.get("date")),
                        "snippet": _clean(row.get("snippet")),
                        "short_comment": _clean(row.get("snippet")),
                        "coolness": _parse_optional_score(row.get("coolness")),
                        "sweetness": _parse_optional_score(row.get("sweetness")),
                    }
                )
                if len(evidence) >= max_results:
                    break
    except OSError as exc:
        return _empty_result(course_name, teacher_name, ["local_cache"], [f"Local review cache could not be read: {exc}"])

    return _summarize_evidence(course_name, teacher_name, ["local_cache"], evidence, warnings)


@lru_cache(maxsize=128)
@langsmith_trace("reviews.search_live_ptt_reviews", run_type="retriever")
def search_ptt_course_reviews(
    course_name: str = "",
    teacher_name: str = "",
    max_pages: int = 5,
    timeout: int = 10,
    sleep_seconds: float = 0.15,
) -> dict:
    entries_checked = 0
    article_candidates: list[dict] = []
    warnings: list[str] = [
        "PTT is one review source, not an official course-quality source.",
        "PTT live search uses the NTHU_Course board search page and then verifies article content.",
    ]

    try:
        for query in _search_queries(course_name, teacher_name):
            entries, checked, query_warnings = _search_ptt_entries_by_query(
                query=query,
                max_pages=max(1, min(max_pages, 3)),
                timeout=timeout,
            )
            entries_checked += checked
            warnings.extend(query_warnings)
            article_candidates.extend(entries)

        # Fallback: recent-board scan catches articles that PTT search does not return.
        keywords = [course_name, teacher_name]
        next_url = PTT_BOARD_URL
        for _ in range(max(1, min(max_pages, 2))):
            index_html = _fetch_url(next_url, timeout=timeout)
            entries, previous_url = _parse_index_entries(index_html)
            for entry in entries:
                entries_checked += 1
                if _keyword_match(entry["title"], keywords, match_all=False):
                    article_candidates.append(entry)
            if not previous_url:
                break
            next_url = previous_url
            if sleep_seconds:
                time.sleep(sleep_seconds)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        result = _empty_result(course_name, teacher_name, ["ptt"], [f"PTT source search failed or was unavailable: {exc}"])
        result["source"] = "PTT NTHU_Course"
        result["entries_checked"] = entries_checked
        return result

    evidence: list[dict] = []
    unique_candidates: list[dict] = []
    seen_urls: set[str] = set()
    for candidate in article_candidates:
        url = candidate.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        unique_candidates.append(candidate)

    for candidate in unique_candidates[:30]:
        try:
            article = _parse_article(_fetch_url(candidate["url"], timeout=timeout))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue
        haystack = f"{candidate['title']}\n{article.get('title', '')}\n{article.get('content', '')}"
        if not _query_matches_article(article.get("content", ""), article.get("title", candidate["title"]), course_name, teacher_name):
            continue
        scores, score_source = _review_scores_with_fallback(haystack)
        snippet = _short_comment(article.get("content", ""))
        evidence.append(
            {
                "source": "PTT NTHU_Course",
                "title": article.get("title") or candidate["title"],
                "url": candidate["url"],
                "date": article.get("date") or candidate.get("date_hint", ""),
                "snippet": snippet,
                "short_comment": snippet,
                "coolness": scores.get("coolness"),
                "sweetness": scores.get("sweetness"),
                "score_source": score_source,
            }
        )

    result = _summarize_evidence(course_name, teacher_name, ["ptt"], evidence[:5], warnings)
    result["source"] = "PTT NTHU_Course"
    result["entries_checked"] = entries_checked
    return result


def _search_web_reviews(course_name: str, teacher_name: str, requested_source: str = "web") -> dict:
    label = "Dcard/web source" if requested_source == "dcard" else "Generic web source"
    return _empty_result(
        course_name,
        teacher_name,
        [requested_source],
        [
            f"{label} search is not enabled in this NTHU COPILOT environment. "
            "Add verified rows to data/course_reviews_sample.csv or enable a real search API before using this source."
        ],
    )


@langsmith_trace("reviews.search_course_reviews", run_type="retriever")
def search_course_reviews(
    course_name: str,
    teacher_name: str = "",
    sources: list[str] | None = None,
    max_results: int = 10,
    timeout: int = 10,
) -> dict:
    normalized_sources = _normalize_sources(sources)
    evidence: list[dict] = []
    warnings: list[str] = []
    sources_used: list[str] = []

    for source in normalized_sources:
        if source == "local_cache":
            result = _search_local_cache_reviews(course_name, teacher_name, max_results=max_results)
        elif source == "ptt_rag":
            result = search_ptt_rag_reviews(course_name, teacher_name, max_results=max_results, timeout=timeout)
        elif source == "ptt":
            result = search_ptt_course_reviews(course_name, teacher_name, max_pages=max(1, min(max_results, 5)), timeout=timeout)
        elif source == "web":
            result = _search_web_reviews(course_name, teacher_name, requested_source="web")
        else:
            result = _empty_result(course_name, teacher_name, [source], [f"Unsupported review source: {source}"])

        sources_used.extend(result.get("sources_used", [source]))
        warnings.extend(result.get("warnings", []))
        evidence.extend(result.get("evidence", []))
        if len(evidence) >= max_results:
            evidence = evidence[:max_results]
            break

    return _summarize_evidence(course_name, teacher_name, sources_used, evidence[:max_results], warnings)


@langsmith_trace("reviews.compare_teachers_for_course")
def compare_teachers_for_course(
    course_name: str,
    teacher_names: list[str],
    preference: str = "coolness",
    sources: list[str] | None = None,
    max_pages: int = 5,
    timeout: int = 10,
) -> dict:
    summaries = [
        search_course_reviews(
            course_name=course_name,
            teacher_name=teacher,
            sources=sources,
            max_results=max_pages,
            timeout=timeout,
        )
        for teacher in teacher_names
    ]
    score_key = "avg_sweetness" if preference == "sweetness" else "avg_coolness"

    def sort_key(summary: dict) -> tuple[float, int]:
        score = summary.get(score_key)
        return (float(score) if score is not None else -1.0, int(summary.get("review_count") or 0))

    ranked = sorted(summaries, key=sort_key, reverse=True)
    best = ranked[0] if ranked and sort_key(ranked[0])[0] >= 0 else None
    return {
        "course_name": course_name,
        "preference": preference,
        "score_key": score_key,
        "best_teacher": best.get("teacher_name") if best else None,
        "teacher_summaries": ranked,
        "sources_used": list(dict.fromkeys(source for summary in summaries for source in summary.get("sources_used", []))),
        "warnings": list(
            dict.fromkeys(
                REVIEW_WARNINGS
                + [
                    "Teacher comparison is based only on available review evidence and must not be treated as an objective ranking.",
                    "Always show review_count and source links/snippets when discussing review results.",
                ]
            )
        ),
    }
