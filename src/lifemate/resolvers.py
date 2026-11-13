"""Deterministic argument compiler: references, IDs, dates, money, enums and types.

The LLM is responsible for *what* to call. This module is responsible for making
the arguments executable and schema-valid. It never reads an environment file; it
only uses the context dictionary, prior tool results recorded by the runtime, and
the catalog.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .catalog import Catalog, ToolSpec
from .text import fold, token_f1, tokens


class ResolutionError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        value: Any = None,
        xref: str | None = None,
        kind: str = "value",
    ):
        super().__init__(message)
        self.field_name = field_name
        self.value = value
        self.xref = xref
        self.kind = kind


class UnresolvedReference(ResolutionError):
    def __init__(self, message: str, *, field_name: str, value: Any, xref: str):
        super().__init__(message, field_name=field_name, value=value, xref=xref, kind="unresolved_ref")


EMPTY = (None, "", [], {})


def is_empty(value: Any) -> bool:
    return value in EMPTY


# ----------------------------------------------------------------------------------
# Prior tool results / references
# ----------------------------------------------------------------------------------


@dataclass
class ToolRecord:
    tool: str
    arguments: dict[str, Any]
    result: Any
    step: int = 0
    historical: bool = False

    @property
    def app(self) -> str:
        return self.tool.split(".", 1)[0] if "." in self.tool else self.tool


@dataclass
class ResolutionContext:
    current_time: str
    timezone_name: str = "Asia/Ho_Chi_Minh"
    user_profile: dict[str, Any] = field(default_factory=dict)
    location: dict[str, Any] = field(default_factory=dict)
    catalog: Catalog | None = None
    records: list[ToolRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.now = _parse_now(self.current_time, self.timezone_name)
        self.tz = self.now.tzinfo or timezone(timedelta(hours=7))

    def add_record(self, tool: str, arguments: dict[str, Any], result: Any, step: int = 0, historical: bool = False) -> ToolRecord:
        record = ToolRecord(tool=tool, arguments=dict(arguments or {}), result=result, step=step, historical=historical)
        self.records.append(record)
        return record

    @property
    def last(self) -> ToolRecord | None:
        return self.records[-1] if self.records else None

    def copy(self) -> "ResolutionContext":
        return ResolutionContext(
            current_time=self.current_time,
            timezone_name=self.timezone_name,
            user_profile=dict(self.user_profile),
            location=dict(self.location),
            catalog=self.catalog,
            records=list(self.records),
        )


def _parse_now(current_time: str, timezone_name: str) -> datetime:
    value = (current_time or "").strip()
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                try:
                    parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
                except Exception:
                    parsed = parsed.replace(tzinfo=timezone(timedelta(hours=7)))
            return parsed
        except ValueError:
            pass
    try:
        return datetime.now(ZoneInfo(timezone_name))
    except Exception:
        return datetime.now(timezone(timedelta(hours=7)))


def _format_offset(dt: datetime) -> str:
    off = dt.utcoffset()
    if off is None:
        off = timedelta(hours=7)
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def format_datetime(dt: datetime, tz: Any = None) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz or timezone(timedelta(hours=7)))
    elif tz is not None:
        dt = dt.astimezone(tz)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + _format_offset(dt)


# ----------------------------------------------------------------------------------
# Vietnamese numbers / money
# ----------------------------------------------------------------------------------

_DIGIT_WORDS = {
    "khong": 0,
    "mot": 1,
    "hai": 2,
    "ba": 3,
    "bon": 4,
    "tu": 4,
    "nam": 5,
    "lam": 5,
    "sau": 6,
    "bay": 7,
    "tam": 8,
    "chin": 9,
}
_SCALES = {"nghin": 10**3, "ngan": 10**3, "trieu": 10**6, "ty": 10**9, "ti": 10**9}
_MONEY_KEYS = re.compile(r"(amount|price|cost|fee|salary|budget|total|balance|money|spent|limit|payment|value|income|expense|debt|charge|deposit|threshold)", re.I)


def _words_to_under_1000(parts: list[str]) -> int:
    total = 0
    i = 0
    while i < len(parts):
        token = parts[i]
        if token in ("linh", "le", "va"):
            i += 1
            continue
        if token in _DIGIT_WORDS:
            value = _DIGIT_WORDS[token]
            if i + 1 < len(parts) and parts[i + 1] == "tram":
                total += value * 100
                i += 2
                continue
            if i + 1 < len(parts) and parts[i + 1] in ("muoi", "muoi"):
                total += value * 10
                i += 2
                if i < len(parts) and parts[i] in _DIGIT_WORDS:
                    total += _DIGIT_WORDS[parts[i]]
                    i += 1
                continue
            total += value
            i += 1
            continue
        if token in ("muoi", "muoi"):
            total += 10
            i += 1
            if i < len(parts) and parts[i] in _DIGIT_WORDS:
                total += _DIGIT_WORDS[parts[i]]
                i += 1
            continue
        if token == "tram":
            total *= 100
            i += 1
            continue
        i += 1
    return total


def _words_to_number(parts: list[str]) -> float | None:
    parts = [p for p in parts if p not in ("vnd", "dong", "d", "va")]
    if not parts:
        return None
    if parts and parts[-1] in ("ruoi", "ruoi"):
        parts = parts[:-1]
        base = _words_to_number(parts)
        if base is None:
            return None
        # "ba trăm rưỡi" = 350; "ba triệu rưỡi" = 3.5 triệu.
        for tok in reversed(parts):
            if tok in _SCALES:
                return float(base) + 0.5 * _SCALES[tok]
        if "tram" in parts:
            return float(base) + 50
        return float(base) + 0.5
    scale_positions = [(i, _SCALES[p]) for i, p in enumerate(parts) if p in _SCALES]
    if not scale_positions:
        value = _words_to_under_1000(parts)
        return float(value) if value else None
    # Recursively parse the largest scale; Vietnamese groups scales in descending order.
    i, scale = max(scale_positions, key=lambda x: x[1])
    left = _words_to_number(parts[:i])
    right = _words_to_number(parts[i + 1 :])
    if left is None:
        left = 1.0
    return float(left) * scale + (float(right) if right is not None else 0.0)


def parse_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    folded = fold(text).replace("−", "-")
    # A single numeric expression, with optional unit.
    m = re.search(r"([-+]?\d[\d.,]*)\s*(k|nghin|ngan|tr|trieu|ty|ti|dong|vnd|d)?\b", folded)
    if m:
        raw_num = m.group(1)
        unit = m.group(2) or ""
        num = _parse_numeric_literal(raw_num)
        if num is None:
            return None
        multiplier = {
            "k": 1_000,
            "nghin": 1_000,
            "ngan": 1_000,
            "tr": 1_000_000,
            "trieu": 1_000_000,
            "ty": 1_000_000_000,
            "ti": 1_000_000_000,
            "dong": 1,
            "vnd": 1,
            "d": 1,
        }.get(unit, 1)
        num *= multiplier
        if "ruoi" in folded:
            num += 0.5 * multiplier if multiplier > 1 else 0.5
        return _clean_number(num)
    # Word-number phrase.
    parts = re.findall(r"[a-z]+", folded)
    word_value = _words_to_number(parts)
    if word_value is not None:
        return _clean_number(word_value)
    return None


def _parse_numeric_literal(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    sign = -1.0 if raw.startswith("-") else 1.0
    raw = raw.lstrip("+-")
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        if raw.count(",") > 1:
            raw = raw.replace(",", "")
        else:
            left, right = raw.split(",", 1)
            raw = left + ("." + right if len(right) <= 2 else right)
    elif "." in raw:
        if raw.count(".") > 1:
            raw = raw.replace(".", "")
        else:
            left, right = raw.split(".", 1)
            if len(right) == 3 and 0 < len(left) <= 3:
                raw = left + right
    try:
        return sign * float(raw)
    except ValueError:
        return None


def _clean_number(num: float | int) -> float | int:
    if isinstance(num, float) and abs(num - round(num)) < 1e-9:
        return int(round(num))
    return num


def parse_money(value: Any, field_name: str = "") -> int | float | None:
    num = parse_number(value)
    if num is None:
        return None
    text = fold(value)
    has_unit = bool(re.search(r"(k|nghin|ngan|tr|trieu|ty|ti|dong|vnd)\b", text))
    if not has_unit and _MONEY_KEYS.search(field_name or "") and 0 < abs(num) < 1000:
        # Colloquial Vietnamese money: "ba trăm rưỡi", "chuyển 500" -> 350k/500k.
        num = num * 1000
    return _clean_number(num)


# ----------------------------------------------------------------------------------
# Vietnamese dates / times
# ----------------------------------------------------------------------------------

_WEEKDAYS = {
    "thu hai": 0,
    "thu 2": 0,
    "thu ba": 1,
    "thu 3": 1,
    "thu tu": 2,
    "thu 4": 2,
    "thu nam": 3,
    "thu 5": 3,
    "thu sau": 4,
    "thu 6": 4,
    "thu bay": 5,
    "thu 7": 5,
    "chu nhat": 6,
    "cn": 6,
}
_PERIODS = ("sang", "trua", "chieu", "toi", "dem")


def _date_from_text(text: str, now: datetime) -> date | None:
    folded = fold(text)
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", folded)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", folded)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else now.year
        if year < 100:
            year += 2000
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if not m.group(3) and candidate < now.date():
            try:
                candidate = candidate.replace(year=year + 1)
            except ValueError:
                pass
        return candidate

    base = now.date()
    if re.search(r"\bhom kia\b", folded):
        return base - timedelta(days=2)
    if re.search(r"\bhom qua\b", folded):
        return base - timedelta(days=1)
    if re.search(r"\b(hom nay|ngay hom nay)\b", folded):
        return base

    for label, weekday in sorted(_WEEKDAYS.items(), key=lambda x: -len(x[0])):
        match = re.search(r"(?<![a-z])" + re.escape(label) + r"(?![a-z])", folded)
        if not match:
            continue
        # Week qualifiers apply to the named calendar week. Exclude the
        # weekday itself so "sau" in "thu sau" cannot mean "next".
        before, after = folded[:match.start()], folded[match.end():]
        for qualifier, offset in ((r"tuan (?:sau|toi)", 7), (r"tuan nay", 0)):
            if re.search(r"\b" + qualifier + r"\s*$", before) or re.match(r"\s+" + qualifier + r"\b", after):
                return base + timedelta(days=weekday - base.weekday() + offset)
        delta = (weekday - base.weekday()) % 7
        if delta == 0 and re.match(r"\s+(?:toi|sau)\b", after):
            delta = 7
        return base + timedelta(days=delta)

    if re.search(r"\bcuoi tuan\b", folded):
        delta = (5 - now.weekday()) % 7
        return base + timedelta(days=delta)
    if re.search(r"\bdau tuan sau\b", folded):
        delta = (0 - now.weekday()) % 7
        if delta == 0:
            delta = 7
        return base + timedelta(days=delta)
    if re.search(r"\btuan sau\b", folded):
        return base + timedelta(days=7)
    if re.search(r"\btuan nay\b", folded):
        return base

    if re.search(r"\b(ngay mot|mot nua)\b", folded):
        return base + timedelta(days=2)
    if re.search(r"\b(ngay mai|mai)\b", folded):
        return base + timedelta(days=1)
    if re.search(r"\bngay kia\b", folded):
        return base + timedelta(days=2)

    m = re.search(r"\bngay\s+(\d{1,2})\b", folded)
    if m:
        day = int(m.group(1))
        try:
            candidate = date(now.year, now.month, day)
        except ValueError:
            return None
        if candidate < base:
            month = now.month + 1
            year = now.year
            if month > 12:
                month = 1
                year += 1
            try:
                candidate = date(year, month, day)
            except ValueError:
                return None
        return candidate
    return None


def _time_from_text(text: str, now: datetime) -> time | None:
    folded = fold(text)
    hour: int | None = None
    minute = 0

    m = re.search(r"\b(\d{1,2})\s*(?::|h|gio)\s*(\d{1,2})?\b", folded)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
    else:
        # "sáu rưỡi", "tám giờ", "mười lăm giờ"
        m = re.search(r"\b((?:[a-z]+\s+){0,4}?)(gio)\b", folded)
        if m:
            prefix = m.group(1).strip().split()
            # Keep only the tail that can form an hour.
            tail: list[str] = []
            for tok in reversed(prefix):
                if tok in _DIGIT_WORDS or tok in ("muoi", "tram", "linh", "le"):
                    tail.insert(0, tok)
                else:
                    break
            if tail:
                value = _words_to_under_1000(tail)
                if 0 <= value <= 23:
                    hour = value
            after = folded[m.end() :]
            m2 = re.match(r"\s*(\d{1,2})\b", after)
            if m2:
                minute = int(m2.group(1))
        else:
            m = re.search(r"\b(\d{1,2})\s+ruoi\b", folded)
            if m:
                hour, minute = int(m.group(1)), 0
            else:
                m = re.search(r"\b([a-z]+)\s+ruoi\b", folded)
                if m and m.group(1) in _DIGIT_WORDS:
                    hour, minute = _DIGIT_WORDS[m.group(1)], 0

    if hour is None:
        return None
    if hour > 23 or minute > 59:
        return None

    has_period = any(re.search(r"\b" + p + r"\b", folded) for p in _PERIODS)
    if has_period:
        if re.search(r"\b(sang)\b", folded) and hour == 12:
            hour = 0
        elif re.search(r"\b(trua|chieu|toi|dem)\b", folded) and 0 < hour < 12:
            hour += 12

    if re.search(r"\b(kem)\b", folded):
        m = re.search(r"kem\s+(\d{1,2})", folded)
        if m:
            total = hour * 60 + minute - int(m.group(1))
            total %= 24 * 60
            hour, minute = total // 60, total % 60
    if re.search(r"\b(hon)\b", folded):
        m = re.search(r"hon\s+(\d{1,2})", folded)
        if m:
            total = hour * 60 + minute + int(m.group(1))
            hour, minute = (total // 60) % 24, total % 60
    if "ruoi" in folded:
        minute += 30
        if minute >= 60:
            hour = (hour + 1) % 24
            minute -= 60
    try:
        return time(hour, minute)
    except ValueError:
        return None


def _datetime_from_text(text: str, now: datetime, tz: Any) -> datetime | None:
    folded = fold(text)
    # Relative offsets from now.
    m = re.search(r"\b(?:sau|trong)\s+(\d+(?:[.,]\d+)?)\s*(tieng|gio|phut)\b", folded)
    if m:
        amount = _parse_numeric_literal(m.group(1)) or 0
        if m.group(2) == "phut":
            return now + timedelta(minutes=amount)
        return now + timedelta(hours=amount)
    m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(tieng|gio|phut)\s+(?:nua|toi)\b", folded)
    if m:
        amount = _parse_numeric_literal(m.group(1)) or 0
        if m.group(2) == "phut":
            return now + timedelta(minutes=amount)
        return now + timedelta(hours=amount)

    # Explicit ISO first.
    iso = (text or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}T", iso):
        try:
            parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=tz)
            return parsed.astimezone(tz)
        except ValueError:
            pass
    d = _date_from_text(text, now)
    t = _time_from_text(text, now)
    if d is not None and t is not None:
        return datetime.combine(d, t).replace(tzinfo=tz)
    if d is not None:
        return None  # a date alone is not a valid datetime argument
    if t is not None:
        candidate = datetime.combine(now.date(), t).replace(tzinfo=tz)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate
    return None


def _date_value(text: str, now: datetime) -> date | None:
    iso = (text or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", iso)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return _date_from_text(text, now)


def _time_value(text: str, now: datetime) -> time | None:
    iso = (text or "").strip()
    if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", iso):
        parts = [int(x) for x in iso.split(":")[:2]]
        try:
            return time(parts[0], parts[1])
        except ValueError:
            return None
    return _time_from_text(text, now)


# ----------------------------------------------------------------------------------
# Enums / booleans
# ----------------------------------------------------------------------------------

_SYNONYMS: dict[str, set[str]] = {
    "on": {"on", "bat", "mo", "enable", "enabled", "true", "1"},
    "off": {"off", "tat", "dong", "disable", "disabled", "false", "0", "khoa"},
    "cool": {"cool", "lam lanh", "lanh", "dieu hoa"},
    "heat": {"heat", "suoi", "nong"},
    "dry": {"dry", "kho", "hut am"},
    "fan": {"fan", "quat", "gio"},
    "bike": {"bike", "xe may", "motorbike", "xe 2 banh", "2 banh"},
    "car4": {"car4", "xe 4 cho", "o to 4 cho", "bon cho", "4 cho"},
    "car7": {"car7", "xe 7 cho", "o to 7 cho", "bay cho", "7 cho"},
    "cash": {"cash", "tien mat", "tm"},
    "ewallet": {"ewallet", "vi", "vi dien tu", "vi điện tử"},
    "bank": {"bank", "ngan hang", "chuyen khoan", "tai khoan"},
    "once": {"once", "mot lan", "1 lan", "khong lap"},
    "daily": {"daily", "hang ngay", "moi ngay", "ngay"},
    "weekly": {"weekly", "hang tuan", "moi tuan", "tuan"},
    "weekdays": {"weekdays", "ngay thuong", "thu 2 den thu 6", "trong tuan"},
    "weekends": {"weekends", "cuoi tuan", "thu 7 chu nhat"},
    "walk": {"walk", "di bo", "bo"},
    "transit": {"transit", "cong cong", "xe buyt"},
    "economy": {"economy", "pho thong", "hang pho thong"},
    "business": {"business", "thuong gia", "hang thuong gia"},
    "vacuum": {"vacuum", "hut bui", "hut"},
    "mop": {"mop", "lau", "lau nha"},
    "both": {"both", "ca hai", "hut va lau"},
    "away": {"away", "ra ngoai", "vang nha"},
    "home": {"home", "o nha", "trong nha"},
    "pause": {"pause", "tam dung", "dung", "ngung"},
    "resume": {"resume", "tiep tuc", "chay tiep", "phat tiep"},
    "next": {"next", "tiep", "bai tiep", "chuyen bai"},
    "previous": {"previous", "lui", "bai truoc", "quay lai"},
    "volume": {"volume", "am luong", "to nho"},
    "archive": {"archive", "luu tru", "luu"},
    "delete": {"delete", "xoa", "bo"},
    "open": {"open", "mo", "chua xong", "dang lam"},
    "done": {"done", "xong", "hoan thanh", "da xong"},
    "low": {"low", "thap", "thap uu tien"},
    "normal": {"normal", "binh thuong", "vua"},
    "high": {"high", "cao", "uu tien cao", "gap"},
}

_ENUM_ALIAS: dict[str, str] = {}
for _canonical, _aliases in _SYNONYMS.items():
    for _alias in _aliases:
        _ENUM_ALIAS[fold(_alias)] = _canonical


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    folded = fold(value)
    if folded in {"true", "yes", "co", "bat", "mo", "on", "1", "dung", "phai"}:
        return True
    if folded in {"false", "no", "khong", "tat", "dong", "off", "0", "sai", "khong phai"}:
        return False
    return None


def coerce_enum(value: Any, enum: list[Any]) -> Any:
    if value in enum:
        return value
    folded = fold(value)
    for candidate in enum:
        if fold(candidate) == folded:
            return candidate
    alias = _ENUM_ALIAS.get(folded)
    for candidate in enum:
        if alias and fold(candidate) == fold(alias):
            return candidate
    # Token overlap / string similarity for small enums.
    best = None
    best_score = 0.0
    for candidate in enum:
        score = max(token_f1(value, candidate), SequenceMatcher(None, folded, fold(candidate)).ratio())
        if score > best_score:
            best, best_score = candidate, score
    if best is not None and best_score >= 0.78:
        return best
    raise ResolutionError(
        f"{value!r} is not one of {enum}; choose exactly one of these values",
        value=value,
        kind="enum",
    )


# ----------------------------------------------------------------------------------
# Symbolic/entity resolution
# ----------------------------------------------------------------------------------

_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$", re.I)
_PHONE_RE = re.compile(r"^(?:\+84|0)\d{8,10}$")


def _looks_like_opaque_id(value: Any) -> bool:
    """True for IDs that need no entity fuzzy match (c_08, VOV3, JV567871861, 19001234)."""
    text = str(value).strip()
    if not text or any(ch.isspace() for ch in text):
        return False
    if _ID_RE.match(text) or _PHONE_RE.match(text.replace(" ", "")):
        return True
    # Dataset IDs are not always snake_case: keep digit-bearing opaque tokens.
    return len(text) >= 2 and any(ch.isdigit() for ch in text) and all(
        ch.isalnum() or ch in "_-.:" for ch in text
    )

_REF_EXACT = re.compile(r"^\{\{\s*(\$[^{}]+?)\s*\}\}$")
_REF_INLINE = re.compile(r"\{\{\s*(\$[^{}]+?)\s*\}\}")
_REF_PLAIN = re.compile(r"^\$[A-Za-z0-9_.\[\]:\-]+$")
_SYMBOLIC = re.compile(r"^@([a-zA-Z_]+)\((.*)\)$", re.S)

_FIELD_XREF_HINTS: dict[str, str] = {
    "contact_id": "contacts.contacts",
    "event_id": "calendar.events",
    "device_id": "smart_home.devices",
    "scene_id": "smart_home.scenes",
    "alarm_id": "alarm.alarms",
    "reminder_id": "reminders.reminders",
    "bill_id": "bill_payment.bills",
    "beneficiary_id": "banking.beneficiaries",
    "from_account_id": "banking.accounts",
    "account_id": "banking.accounts",
    "chat_id": "chat_ott.chats",
    "track_id": "music.tracks",
    "playlist_id": "music.playlists",
    "episode_id": "podcast_radio.episodes",
    "show_id": "podcast_radio.shows",
    "book_id": "library.books",
    "course_id": "learning.courses",
    "reward_id": "loyalty.rewards",
    "merchant_id": "food_delivery.restaurants",
    "restaurant_id": "food_delivery.restaurants",
    "order_id": "food_delivery.orders",
    "booking_id": "flights.bookings",
    "flight_id": "flights.flights",
    "ride_id": "ride_hailing.rides",
    "lock_id": "door_lock.locks",
    "appliance_id": "appliances.appliances",
    "schedule_id": "medication.schedules",
    "note_id": "notes.notes",
    "task_id": "tasks.tasks",
    "article_id": "news.articles",
    "alert_id": "home_security.alerts",
    "email_id": "email.emails",
    "program_id": "tv.programs",
    "meeting_id": "video_meeting.meetings",
    "chore_id": "household_chores.chores",
    "rule_id": "home_automation.rules",
    "subscription_id": "subscriptions.subscriptions",
    "tracking_id": "package_tracking.packages",
    "bike_id": "bike_rental.bikes",
}

_READ_HINTS: dict[str, tuple[str, str | None]] = {
    "contacts.contacts": ("contacts.search", "query"),
    "calendar.events": ("calendar.list_events", None),
    "smart_home.devices": ("smart_home.get_status", None),
    "smart_home.scenes": ("smart_home.get_status", None),
    "alarm.alarms": ("alarm.list", None),
    "reminders.reminders": ("reminders.list", None),
    "bill_payment.bills": ("bill_payment.list_bills", None),
    "banking.accounts": ("banking.get_balance", None),
    "banking.beneficiaries": ("banking.list_beneficiaries", None),
    "chat_ott.chats": ("chat_ott.list_chats", None),
    "music.tracks": ("music.list_tracks", None),
    "music.playlists": ("music.list_tracks", None),
    "door_lock.locks": ("door_lock.list_locks", None),
    "appliances.appliances": ("appliances.list", None),
    "learning.courses": ("learning.list_courses", None),
    "library.books": ("library.list_books", None),
    "medication.schedules": ("medication.list", None),
    "notes.notes": ("notes.list", None),
    "tasks.tasks": ("tasks.list", None),
    "news.articles": ("news.headlines", None),
    "home_security.alerts": ("home_security.list_alerts", None),
    "email.emails": ("email.list_inbox", None),
    "tv.programs": ("tv.guide", None),
    "video_meeting.meetings": ("video_meeting.list", None),
    "household_chores.chores": ("household_chores.list", None),
    "home_automation.rules": ("home_automation.list_rules", None),
    "subscriptions.subscriptions": ("subscriptions.list", None),
    "package_tracking.packages": ("package_tracking.list", None),
    "bike_rental.bikes": ("bike_rental.list_nearby", None),
    "food_delivery.restaurants": ("food_delivery.list_restaurants", None),
    "food_delivery.orders": ("food_delivery.order_history", None),
    "flights.bookings": ("flights.my_bookings", None),
    "flights.flights": ("flights.search", "date"),
    "ride_hailing.rides": ("ride_hailing.get_ride", None),
    "loyalty.rewards": ("loyalty.list_rewards", None),
    "podcast_radio.episodes": ("podcast_radio.search", "query"),
    "podcast_radio.shows": ("podcast_radio.search", "query"),
}


def xref_for_field(field_name: str, prop: dict[str, Any]) -> str | None:
    if prop.get("x-ref"):
        return str(prop["x-ref"])
    return _FIELD_XREF_HINTS.get(str(field_name))


def read_hint_for_xref(
    catalog: Catalog | None,
    xref: str,
    installed_apps: dict[str, Any] | Iterable[str] | None = None,
) -> tuple[str, str | None] | None:
    hint = _READ_HINTS.get(xref)
    if catalog is None:
        return hint

    def available(tool: ToolSpec, query_arg: str | None) -> bool:
        supplied = {query_arg} if query_arg else set()
        return (
            tool.kind == "read"
            and supplied <= tool.properties.keys()
            and set(tool.required) <= supplied
            and (installed_apps is None or catalog.is_installed(tool.name, installed_apps))
        )

    if hint:
        spec = catalog.tools.get(hint[0])
        if spec and available(spec, hint[1]):
            return hint

    app_id, separator, _ = xref.partition(".")
    if not separator:
        return None

    candidates = []
    for tool in catalog.read_tools(app_id):
        query_arg = next(
            (key for key in ("query", "name", "q", "keyword", "text") if key in tool.properties),
            None,
        )
        if available(tool, query_arg):
            candidates.append((tool, query_arg))
    if not candidates:
        return None

    def rank(candidate: tuple[ToolSpec, str | None]) -> tuple[int, int]:
        tool, _ = candidate
        preferred = ("search", "list", "get", "status", "nearby", "recent", "inbox")
        return len(tool.required), int(not any(word in tool.short_name for word in preferred))

    tool, query_arg = min(candidates, key=rank)
    return tool.name, query_arg


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


# Display keys used for fuzzy entity matching. Deliberately identity-only:
# including relation/phone/email makes the score depend on incidental fields
# and hides genuine ambiguity (two contacts both named "Lan").
_MATCH_KEYS = ("name", "title", "label", "room")
_FALLBACK_MATCH_KEYS = ("merchant", "code", "type", "provider")


def _record_id(record: dict[str, Any]) -> str | None:
    for key in ("id", "merchant_id", "bike_id", "tracking_id"):
        value = record.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    for key, value in record.items():
        if key.endswith("_id") and isinstance(value, (str, int)) and str(value):
            return str(value)
    return None


def _known_ids(ctx: ResolutionContext) -> set[str]:
    """Every id-like value seen so far in results or arguments.

    Gold trajectories sometimes reference an id that no read tool returned in
    this conversation (e.g. ``contact_id`` inside an SMS result, or
    ``merchant_id`` on a loyalty program).  Such ids are legitimate; other
    opaque-looking strings are not.
    """
    known: set[str] = set()
    for record in ctx.records:
        values: list[Any] = [record.result, record.arguments]
        for value in values:
            for obj in _iter_dicts(value):
                for key, item in obj.items():
                    if (key == "id" or key.endswith("_id")) and isinstance(item, str) and item:
                        known.add(item)
    return known


def _xref_app(xref: str) -> str:
    return xref.split(".", 1)[0]


def _entity_candidates(xref: str, ctx: ResolutionContext) -> list[tuple[str, str, str]]:
    """(id, best display text, source tool) for all records of the x-ref app."""
    app = _xref_app(xref)
    out: list[tuple[str, str, str]] = []
    for record in ctx.records:
        if record.app != app:
            continue
        for obj in _iter_dicts(record.result):
            rid = _record_id(obj)
            if not rid:
                continue
            displays = [str(obj[k]) for k in _MATCH_KEYS if isinstance(obj.get(k), str) and str(obj[k]).strip()]
            if not displays:
                displays = [
                    str(obj[k])
                    for k in _FALLBACK_MATCH_KEYS
                    if isinstance(obj.get(k), str) and str(obj[k]).strip()
                ]
            display = " | ".join(displays)
            out.append((rid, display, record.tool))
    return out


def resolve_entity(value: Any, xref: str, ctx: ResolutionContext, field_name: str = "") -> str:
    text = str(value).strip()
    if not text:
        raise ResolutionError("empty entity reference", field_name=field_name, value=value)
    candidates = _entity_candidates(xref, ctx)
    # Exact opaque id first: it may be a known id that no read of this app
    # returned (SMS contact_id, loyalty merchant_id, tracking_id, ...).
    if _looks_like_opaque_id(text) and text in _known_ids(ctx):
        return text
    if not candidates:
        raise UnresolvedReference(
            f"cannot resolve {field_name or 'entity'} {text!r}: no {xref} data has been read yet. "
            f"Call the read/search tool for {xref.split('.', 1)[0]} first.",
            field_name=field_name,
            value=value,
            xref=xref,
        )
    # Exact ID / name first.
    folded = fold(text)
    exact = [(rid, display, tool) for rid, display, tool in candidates if fold(rid) == folded or any(fold(part.strip()) == folded for part in display.split("|"))]
    if exact:
        ids = {x[0] for x in exact}
        if len(ids) == 1:
            return exact[-1][0]
    if _looks_like_opaque_id(text):
        known = {rid for rid, _, _ in candidates}
        if text in known:
            return text
        raise UnresolvedReference(
            f"{field_name or 'entity'} id {text!r} is not in the current {xref} records; "
            "read the app again and copy an exact id",
            field_name=field_name,
            value=value,
            xref=xref,
        )
    best_score = 0.0
    best: list[tuple[str, str, str]] = []
    for rid, display, tool in candidates:
        score = max(token_f1(text, display), SequenceMatcher(None, folded, fold(display)).ratio())
        if score > best_score + 1e-9:
            best_score, best = score, [(rid, display, tool)]
        elif abs(score - best_score) < 1e-9:
            best.append((rid, display, tool))
    if best and best_score >= 0.6:
        ids = {x[0] for x in best}
        if len(ids) == 1:
            return best[-1][0]
        raise ResolutionError(
            f"{field_name or 'entity'} {text!r} matches several records: "
            + ", ".join(sorted(ids)[:8])
            + "; use an exact id",
            field_name=field_name,
            value=value,
            xref=xref,
            kind="ambiguous_ref",
        )
    raise UnresolvedReference(
        f"cannot resolve {field_name or 'entity'} {text!r} in {xref}; call its read/search tool and copy the exact id",
        field_name=field_name,
        value=value,
        xref=xref,
    )


def _resolve_path(expr: str, ctx: ResolutionContext) -> Any:
    raw = expr.strip()
    if raw.startswith("{{") and raw.endswith("}}"):
        raw = raw[2:-2].strip()
    if raw.startswith("$"):
        raw = raw[1:]
    current_records = [r for r in ctx.records if not r.historical]
    record: ToolRecord | None = None
    rest = ""
    m = re.match(r"^tool:([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)(.*)$", raw)
    if m:
        tool_name, rest = m.group(1), m.group(2)
        matches = [r for r in current_records if r.tool == tool_name]
        if not matches:
            raise ResolutionError(f"no prior result for {tool_name!r}", value=expr, kind="ref")
        record = matches[-1]
    else:
        m = re.match(r"^(step|s|result)?\s*(\d+)(.*)$", raw)
        if m:
            index = int(m.group(2)) - 1
            rest = m.group(3)
            if index < 0 or index >= len(current_records):
                raise ResolutionError(f"reference step {index + 1} has not run yet", value=expr, kind="ref")
            record = current_records[index]
        else:
            m = re.match(r"^(last|l)(.*)$", raw)
            if m:
                rest = m.group(2)
                if not current_records:
                    raise ResolutionError("no prior tool result for $last", value=expr, kind="ref")
                record = current_records[-1]
            else:
                raise ResolutionError(f"unsupported reference {expr!r}", value=expr, kind="ref")
    current: Any = record.result
    rest = rest.strip()
    if rest.startswith("."):
        rest = rest[1:]
    if not rest:
        return current
    # Tokenise dotted and [n] path segments.
    parts: list[str] = []
    for segment in rest.split("."):
        pos = 0
        while pos < len(segment):
            if segment[pos] == "[":
                end = segment.find("]", pos)
                if end < 0:
                    raise ResolutionError(f"bad reference path {expr!r}", value=expr, kind="ref")
                parts.append(segment[pos + 1 : end])
                pos = end + 1
            else:
                nxt = segment.find("[", pos)
                if nxt < 0:
                    parts.append(segment[pos:])
                    break
                parts.append(segment[pos:nxt])
                pos = nxt
        if not segment:
            continue
    for part in parts:
        if part == "":
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise ResolutionError(f"reference {expr!r} has no item {part!r}", value=expr, kind="ref")
        elif isinstance(current, dict):
            if part not in current:
                raise ResolutionError(f"reference {expr!r} has no key {part!r}", value=expr, kind="ref")
            current = current[part]
        else:
            raise ResolutionError(f"reference {expr!r} cannot descend into {type(current).__name__}", value=expr, kind="ref")
    return current


def resolve_references(value: Any, ctx: ResolutionContext) -> Any:
    if isinstance(value, str):
        m = _REF_EXACT.match(value)
        if m:
            return _resolve_path(m.group(1), ctx)
        if _REF_PLAIN.match(value.strip()):
            return _resolve_path(value.strip(), ctx)
        if _REF_INLINE.search(value):
            def repl(match: re.Match[str]) -> str:
                resolved = _resolve_path(match.group(1), ctx)
                return str(resolved)
            return _REF_INLINE.sub(repl, value)
        return value
    if isinstance(value, list):
        return [resolve_references(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: resolve_references(v, ctx) for k, v in value.items()}
    return value


def _temporal_kind(key: str, prop: dict[str, Any]) -> str | None:
    fmt = prop.get("format")
    if fmt in {"date", "date-time", "time"}:
        return fmt
    if fmt or prop.get("type") not in (None, "string"):
        return None
    desc = fold(prop.get("description") or "")
    if "iso-8601" in desc or "iso 8601" in desc or "date-time" in desc:
        return "date-time"
    if "yyyy-mm-dd" in desc:
        return "date"
    key_f = fold(key)
    if key_f in {"date", "date_from", "date_to", "birthday", "due_date", "start_date", "end_date"}:
        return "date"
    if key_f in {"start", "end", "time", "schedule_time", "valid_until", "since", "deadline", "due", "at", "timestamp", "checkin_time", "checkout_time"}:
        return "date-time"
    return None


def _resolve_symbolic(value: str, prop: dict[str, Any], key: str, ctx: ResolutionContext) -> Any:
    m = _SYMBOLIC.match(value.strip())
    if not m:
        return value
    kind, inner = m.group(1).lower(), m.group(2).strip()
    if kind == "ref":
        return _resolve_path(inner, ctx)
    if kind in {"time", "date", "datetime"}:
        if kind == "time":
            t = _time_value(inner, ctx.now)
            if t is None:
                raise ResolutionError(f"cannot parse time {inner!r}", field_name=key, value=value, kind="time")
            return t.strftime("%H:%M")
        if kind == "date":
            d = _date_value(inner, ctx.now)
            if d is None:
                raise ResolutionError(f"cannot parse date {inner!r}", field_name=key, value=value, kind="date")
            return d.isoformat()
        dt = _datetime_from_text(inner, ctx.now, ctx.tz)
        if dt is None:
            raise ResolutionError(f"cannot parse date-time {inner!r}", field_name=key, value=value, kind="datetime")
        return format_datetime(dt, ctx.tz)
    if kind == "amount":
        amount = parse_money(inner, key)
        if amount is None:
            raise ResolutionError(f"cannot parse amount {inner!r}", field_name=key, value=value, kind="amount")
        return amount
    if kind == "enum":
        return inner
    if kind in {"bool", "boolean"}:
        b = coerce_bool(inner)
        if b is None:
            raise ResolutionError(f"cannot parse boolean {inner!r}", field_name=key, value=value, kind="bool")
        return b
    # everything else is an entity name, e.g. @contact(Sếp Dũng), @event(chạy bộ)
    xref = xref_for_field(key, prop)
    if not xref:
        raise ResolutionError(f"{value!r} needs an id field with a known x-ref", field_name=key, value=value, kind="symbolic")
    return resolve_entity(inner, xref, ctx, key)


def _looks_like_phone(value: Any) -> bool:
    return isinstance(value, str) and bool(_PHONE_RE.match(value.replace(" ", "")))


def _apply_phone_alternatives(args: dict[str, Any], tool: ToolSpec) -> dict[str, Any]:
    """Move a name/number between ``contact_id`` and ``phone_number`` alternatives only."""
    if not tool.one_of:
        return args
    alternatives = [group[0] for group in tool.one_of if len(group) == 1]
    if len(alternatives) < 2:
        return args
    phone_fields = [alt for alt in alternatives if "phone" in alt.lower()]
    entity_fields = [alt for alt in alternatives if tool.prop(alt).get("x-ref")]
    if not phone_fields or not entity_fields:
        return args
    # A name in a phone-number field -> the entity-id alternative.
    for phone_field in phone_fields:
        value = args.get(phone_field)
        if is_empty(value) or not isinstance(value, str) or not any(ch.isalpha() for ch in value):
            continue
        for entity_field in entity_fields:
            if is_empty(args.get(entity_field)):
                args[entity_field] = value
                args.pop(phone_field, None)
                break
    # An explicit phone number in an entity-id field -> the phone-number alternative.
    for entity_field in entity_fields:
        value = args.get(entity_field)
        if not _looks_like_phone(value):
            continue
        for phone_field in phone_fields:
            if is_empty(args.get(phone_field)):
                args[phone_field] = value
                args.pop(entity_field, None)
                break
    return args


def _coerce_integer(value: Any, field_name: str) -> int:
    num = parse_number(value)
    if num is None:
        raise ResolutionError(
            f"{field_name} must be an integer (got {value!r})",
            field_name=field_name,
            value=value,
            kind="type",
        )
    if isinstance(num, float) and abs(num - round(num)) > 1e-9:
        raise ResolutionError(
            f"{field_name} must be an integer, not {num}",
            field_name=field_name,
            value=value,
            kind="type",
        )
    return int(round(num))


def coerce_value(
    value: Any,
    prop: dict[str, Any],
    key: str,
    ctx: ResolutionContext,
    tool: ToolSpec | None = None,
) -> Any:
    value = resolve_references(value, ctx)
    if isinstance(value, str):
        value = _resolve_symbolic(value, prop, key, ctx)

    if is_empty(value):
        return None

    enum = prop.get("enum")
    if enum:
        return coerce_enum(value, list(enum))

    typ = prop.get("type")
    temporal = _temporal_kind(key, prop)
    if temporal == "date":
        d = _date_value(str(value), ctx.now) if isinstance(value, str) else (value if isinstance(value, date) else None)
        if isinstance(value, datetime):
            d = value.date()
        if d is None:
            raise ResolutionError(f"{key} must be a date like YYYY-MM-DD (got {value!r})", field_name=key, value=value, kind="date")
        return d.isoformat()
    if temporal == "date-time":
        if isinstance(value, datetime):
            return format_datetime(value, ctx.tz)
        dt = _datetime_from_text(str(value), ctx.now, ctx.tz)
        if dt is None:
            raise ResolutionError(
                f"{key} must be an ISO date-time such as 2026-11-29T09:00:00+07:00 (got {value!r})",
                field_name=key,
                value=value,
                kind="datetime",
            )
        return format_datetime(dt, ctx.tz)
    if temporal == "time":
        if isinstance(value, time):
            return value.strftime("%H:%M")
        t = _time_value(str(value), ctx.now)
        if t is None:
            raise ResolutionError(f"{key} must be HH:MM (got {value!r})", field_name=key, value=value, kind="time")
        return t.strftime("%H:%M")

    if typ == "boolean":
        b = coerce_bool(value)
        if b is None:
            raise ResolutionError(f"{key} must be boolean (got {value!r})", field_name=key, value=value, kind="type")
        return b
    if typ == "integer":
        return _coerce_integer(value, key)
    if typ == "number":
        num = parse_number(value)
        if num is None:
            raise ResolutionError(f"{key} must be a number (got {value!r})", field_name=key, value=value, kind="type")
        return num
    if typ == "array":
        if not isinstance(value, list):
            if isinstance(value, str):
                value = [x.strip() for x in re.split(r"[,;]", value) if x.strip()]
            else:
                value = [value]
        item_schema = prop.get("items") or {}
        out = []
        for item in value:
            item = resolve_references(item, ctx)
            if isinstance(item, str) and _SYMBOLIC.match(item.strip()):
                item = _resolve_symbolic(item, item_schema, key, ctx)
            item_enum = item_schema.get("enum")
            if item_enum:
                item = coerce_enum(item, list(item_enum))
            item_type = item_schema.get("type")
            if item_type == "integer":
                item = _coerce_integer(item, key)
            elif item_type == "number":
                item = parse_number(item)
                if item is None:
                    raise ResolutionError(
                        f"{key} must contain only numbers (got {value!r})",
                        field_name=key,
                        value=value,
                        kind="type",
                    )
            elif item_type == "boolean":
                item = coerce_bool(item)
                if item is None:
                    raise ResolutionError(
                        f"{key} must contain only booleans (got {value!r})",
                        field_name=key,
                        value=value,
                        kind="type",
                    )
            elif item_type == "string" and not isinstance(item, str):
                item = str(item)
            out.append(item)
        return out
    if typ == "object":
        return value
    # string or untyped
    if typ == "string":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if not isinstance(value, str):
            return str(value)
    return value


def resolve_arguments(
    tool: ToolSpec,
    arguments: dict[str, Any] | None,
    ctx: ResolutionContext,
) -> dict[str, Any]:
    """Compile model arguments to schema-shaped values, dropping unknown/empty args."""
    if not isinstance(arguments, dict):
        arguments = {}
    props = tool.properties
    args = {k: v for k, v in arguments.items() if k in props}
    args = _apply_phone_alternatives(args, tool)
    out: dict[str, Any] = {}
    for key, prop in props.items():
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("@") and not _SYMBOLIC.match(stripped) and "(" not in stripped:
                # @phone-style shorthand: treat as an entity/text hint.
                value = stripped[1:]
            elif _SYMBOLIC.match(stripped):
                value = _resolve_symbolic(stripped, prop, key, ctx)
            else:
                value = resolve_references(value, ctx)
        if is_empty(value):
            continue
        xref = xref_for_field(key, prop)
        if xref and not (prop.get("format") or _temporal_kind(key, prop)):
            value = resolve_entity(value, xref, ctx, key)
        try:
            value = coerce_value(value, prop, key, ctx, tool)
        except ResolutionError as exc:
            if exc.field_name is None:
                exc.field_name = key
            if exc.value is None:
                exc.value = args.get(key)
            raise
        if is_empty(value):
            continue
        out[key] = value
    return out


def missing_requirement_params(tool: ToolSpec, args: dict[str, Any]) -> list[str]:
    """Required params still missing; for x-one-of, return the first viable alternative name."""
    missing = [k for k in tool.required if is_empty(args.get(k))]
    if missing:
        return missing
    if tool.one_of and not any(all(not is_empty(args.get(k)) for k in group) for group in tool.one_of):
        first_by_len = sorted(tool.one_of, key=len)[0]
        return [first_by_len[0]] if first_by_len else []
    return []
