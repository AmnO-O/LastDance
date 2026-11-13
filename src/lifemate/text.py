"""Small, dependency-free text helpers shared by the planner, caller and scorer."""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable


def fold(value: Any) -> str:
    """Lower-case Vietnamese text without diacritics (đ -> d), whitespace collapsed."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFD", str(value).lower()).replace("đ", "d")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def tokens(value: Any) -> set[str]:
    """Word tokens of folded text; punctuation is a separator."""
    return set(re.findall(r"[^\W_]+", fold(value)))


def token_f1(a: Any, b: Any) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def normalize_space(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def compact_json(value: Any, ensure_ascii: bool = False) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=(",", ":"))


def truncate(value: str, limit: int) -> str:
    value = value or ""
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(1, limit - 1)] + "…"


def compact_value(value: Any, max_items: int = 8, max_string: int = 260) -> Any:
    """Copy of a tool result with long lists/strings reduced for prompt use."""
    if isinstance(value, dict):
        return {str(k): compact_value(v, max_items, max_string) for k, v in value.items()}
    if isinstance(value, list):
        return [compact_value(v, max_items, max_string) for v in value[:max_items]]
    if isinstance(value, str):
        return truncate(value, max_string)
    return value


def extract_json_values(text: str) -> list[Any]:
    """Extract every top-level JSON object/array from free-form model output.

    The scanner is string/escape aware, so braces inside a quoted argument do not
    terminate an object early. Code fences and prose around the JSON are ignored.
    """
    if not text:
        return []
    out: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        start = i
        stack: list[str] = []
        in_string = False
        escaped = False
        while i < n:
            c = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "{":
                    stack.append("}")
                elif c == "[":
                    stack.append("]")
                elif c in "}]":
                    if not stack:
                        break
                    want = stack.pop()
                    if c != want:
                        break
                    if not stack:
                        i += 1
                        candidate = text[start:i]
                        try:
                            out.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            pass
                        break
            i += 1
        else:
            break
        if not stack:
            continue
        # Malformed candidate: resume just after its opening character.
        i = start + 1
    return out


def first_json_object(text: str) -> dict[str, Any] | None:
    for value in extract_json_values(text):
        if isinstance(value, dict):
            return value
    return None


def first_json_value(text: str) -> Any:
    values = extract_json_values(text)
    return values[0] if values else None


def contains_any(text: str, words: Iterable[str]) -> bool:
    folded = " " + fold(text) + " "
    return any(" " + fold(w) + " " in folded for w in words if w)


def unique_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
