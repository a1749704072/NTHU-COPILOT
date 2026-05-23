from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

MAX_TEXT = 1200
MAX_LIST_ITEMS = 8
MAX_DICT_ITEMS = 40
MAX_DEPTH = 4


def _short_text(value: str, limit: int = MAX_TEXT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def _summarize_dataframe(value: Any) -> dict[str, Any]:
    columns = [str(column) for column in getattr(value, "columns", [])]
    rows, cols = getattr(value, "shape", (None, None))
    return {
        "_type": "DataFrame",
        "rows": rows,
        "columns": cols,
        "column_names": columns[:30],
    }


def _summarize_series(value: Any) -> dict[str, Any]:
    return {
        "_type": "Series",
        "name": str(getattr(value, "name", "")),
        "length": len(value) if hasattr(value, "__len__") else None,
    }


def _summarize_object(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"_type": value.__class__.__name__}
    for attr in (
        "student_path",
        "target_path",
        "rules_path",
        "use_llm",
        "use_llm_intent",
        "intent_provider",
        "model",
    ):
        if hasattr(value, attr):
            summary[attr] = _sanitize(getattr(value, attr), depth=1)
    return summary


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        return _short_text(repr(value), 240)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _short_text(value)
    if isinstance(value, Path):
        return str(value)
    type_name = value.__class__.__name__
    if type_name == "DataFrame":
        return _summarize_dataframe(value)
    if type_name == "Series":
        return _summarize_series(value)
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            str(key): _sanitize(item_value, depth + 1)
            for key, item_value in items[:MAX_DICT_ITEMS]
        }
        if len(items) > MAX_DICT_ITEMS:
            result["_truncated_items"] = len(items) - MAX_DICT_ITEMS
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [_sanitize(item, depth + 1) for item in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            result.append(f"<truncated {len(items) - MAX_LIST_ITEMS} items>")
        return result
    if value.__class__.__module__.startswith("builtins"):
        return _short_text(repr(value), 240)
    return _summarize_object(value)


def _sanitize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return _sanitize(inputs)


def _sanitize_outputs(outputs: Any) -> Any:
    return _sanitize(outputs)


def _tracing_requested() -> bool:
    return any(
        str(os.getenv(name, "")).strip()
        for name in (
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
        )
    )


def langsmith_trace(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """Optional LangSmith decorator with small, safe payloads.

    The app must continue to run even when langsmith is not installed or no API
    key is configured, so this returns a no-op decorator unless tracing is
    clearly requested by environment variables.
    """

    def identity(func: F) -> F:
        return func

    if not _tracing_requested():
        return identity

    try:
        from langsmith import traceable
    except Exception:
        return identity

    try:
        return traceable(
            name=name,
            run_type=run_type,
            process_inputs=_sanitize_inputs,
            process_outputs=_sanitize_outputs,
        )
    except TypeError:
        return identity

