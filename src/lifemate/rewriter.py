"""Multi-candidate ASR rewriter.

The transcriber may produce several hypotheses for the same spoken utterance.
This stage asks the model to combine them into one clean Vietnamese request and
then applies a conservative guard: if the rewrite changes a number, a negation,
or invents unsupported content, the stage falls back to the production top-1
candidate.  The rewriter is therefore allowed to help but never to make the
downstream planner/caller worse than the raw top-1 transcript.

The stage never sees tool catalogs or app schemas.  It may receive a compact
reference block with conversation context/history, used only to disambiguate a
name or word that already appears in the candidates; the rewrite itself must
stay fully supported by the candidates.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Sequence

from .llm import ChatBackend, GenerationSettings
from .resolvers import parse_number
from .text import fold, normalize_space


REWRITER_SYSTEM = """You are a Vietnamese sentence selector and normalizer for a personal assistant.
Input: a spoken request — an audio file with the user's latest request in Vietnamese, possibly mixed with common English words.
You receive N text transcripts of that same spoken request. The transcripts may differ because of speech recognition errors.
Your job is to produce one clean, complete Vietnamese sentence that best represents what the speaker said.

CANDIDATE PRIORITY
- Candidate [1] is the production transcript and the default answer. Start from [1].
- Candidates [2]..[N] are secondary evidence. Use them only to repair a word or phrase that is clearly wrong in [1].
- When candidates disagree on a contiguous phrase, take the whole phrase from one candidate. Never splice words from different candidates into the same phrase.
- If candidates disagree on the value of a number, date, time, or amount, keep [1]'s spoken form exactly; do not convert it to digits or guess a value.

RULES
- Compare all candidates, using [1] as the anchor (see CANDIDATE PRIORITY). Prefer content supported by most candidates.
- Duplicate candidates are allowed and count as agreement.
- If candidates disagree on a name, number, date, time, amount, negation, or condition:
  - for numbers, dates, times, and amounts, follow CANDIDATE PRIORITY and keep [1]'s spoken form;
  - otherwise prefer the value supported by the majority;
  - if there is no majority, keep [1]'s value unless it is clearly garbled;
  - if still unsure, keep the most conservative/verbatim form.
- Do not invent a new value.
- Do not merge incompatible candidates. You may combine parts only when they agree or the combination is clearly supported by the candidates.
- Preserve names, numbers, dates, times, amounts, negations, conditions, and intent.
- Keep person names exactly as heard. Do not replace a name with "N" or any other placeholder. If a name is unclear, keep the closest heard form; do not invent a new name.
- Preserve emails, URLs, handles, IDs, plate numbers, and model codes exactly. In an address or ID, "a còng" = @, "chấm" = ., "gạch dưới" = _, "gạch ngang" = -; never translate these into words like "kèm".
- Do not add information that is not present in any candidate.
- Do not remove information that is present in the chosen content.
- Keep pronouns and honorifics: "tôi", "mình", "anh", "chị", "em", "chú", "bác", ...
- Keep colloquial and regional wording. Preserve English words such as “review” and “proposal” as spoken. Capitalize clear proper names. Do not formalize, paraphrase, or translate.
- Do not answer the request, summarize it, explain it, or resolve references.
- Do not resolve relative dates/times; keep "mai", "hôm nay", "ngày kia", "tuần sau".
- The optional CONTEXT block gives time, place, and user profile. Use it only to choose between candidate forms. Never copy a fact from it that no candidate contains.
- The optional PREVIOUS TURNS block gives earlier conversation. Use it only to disambiguate a name, number, or word already heard in the candidates. Do not resolve pronouns, do not continue the conversation, and do not import a name, number, or request from it.
- Context and history are reference only. The answer must remain fully supported by the candidates.
- Do not reorder clauses or merge separate requests.
- Normalize casing and punctuation: comma for clause boundaries, period for a statement, question mark for a question.
- Prefer one sentence with commas. Use a period only when the input clearly contains separate sentences.
- Normalize spoken numbers, dates, times, and amounts to standard digits only when the value is unambiguous and all candidates agree on it. Otherwise keep [1]'s spoken form.
- If all candidates are identical, only fix punctuation/casing.
- If all candidates are garbled, keep the most conservative/coherent version; do not guess.
- N is only the number of candidates. It is not a placeholder and must never appear in your answer.
- Output only the final sentence. No explanation, no label, no JSON, no code fence.
- Do not include the "CLEAN:" prefix in your answer.

CANDIDATE FORMAT
[1] first candidate
[2] second candidate
...

EXAMPLES
CANDIDATES:
[1] "đặt báo thức sáu giờ sáng"
[2] "đặt báo thức sáu giờ sáng mai"
[3] "báo thức sáu giờ sáng mai"
CLEAN: "Đặt báo thức 6 giờ sáng mai."

CANDIDATES:
[1] "đừng nhắc tôi mua sữa"
[2] "nhắc tôi mua sữa"
[3] "đừng nhắc tôi mua xữa"
CLEAN: "Đừng nhắc tôi mua sữa."

CANDIDATES:
[1] "ĐẶT XE BỐN CHỖ TỪ CÔNG TY ĐI KHÁM RĂNG CHIỀU NAY ĐẶT NGAY BÂY GIỜ NHÉ"
[2] "đặt xe bốn chổ từ công ty đi khám răng chiều nay đặt ngay bây giờ nhé"
[3] "đặt xe bốn chỗ từ công ty đi khám răng chiều nay đặt ngay bây giờ nhé"
CLEAN: "Đặt xe 4 chỗ từ công ty đi khám răng chiều nay, đặt ngay bây giờ nhé."

CANDIDATES:
[1] "mở bài hát mới nhất của liam shaw"
[2] "mở bài hát mới nhất của ly am sau"
[3] "mở bài hát mới nhất của liam shaw"
CLEAN: "Mở bài hát mới nhất của Liam Shaw."

CANDIDATES:
[1] "chị ơi nhắc em review proposal trước deadline chiều mai nha"
[2] "chị ơi nhắc em review pro po zồ trước đét lai chiều mai nha"
CLEAN: "Chị ơi, nhắc em review proposal trước deadline chiều mai nha."

CANDIDATES:
[1] "xe bốn ba a năm sáu bảy chấm tám chín có dính phạt nguội chi không rửa"
[2] "xe 43a 567.89 có dính phạt nguội chi không rứa"
[3] "xe 43a 567 chấm 89 có dính phạt nguội tri không rứa"
CLEAN: "Xe 43A 567.89 có dính phạt nguội chi không rứa?"

CANDIDATES:
[1] "để em coi tuyến nào nhanh nhất đặt sẵn taxi bốn chỗ đón anh ở trạm xuống"
[2] "em coi tuyến nào em rất đặc sẵn taxi bốn chỗ đón anh ở trạm xuống"
CLEAN: "Để em coi tuyến nào nhanh nhất, đặt sẵn taxi bốn chỗ đón anh ở trạm xuống."

CANDIDATES:
[1] "soạn sẵn email nháp gửi finance a còng example chấm test với nội dung đó nha"
[2] "soạn sẵn email nhét với nội dung đó nha"
CLEAN: "Soạn sẵn email nháp gửi finance@example.test với nội dung đó nha."
"""

CANDIDATE_HEADER = "CANDIDATES (same spoken utterance, different recognition hypotheses):"
CONTEXT_HEADER = "CONTEXT (reference only; do not copy facts that are not in the candidates):"
HISTORY_HEADER = "PREVIOUS TURNS (reference only; do not continue the conversation):"
CLEAN_LABEL = "CLEAN SENTENCE:"

_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:CLEAN(?:\s+SENTENCE)?|WRITTEN(?:\s+SENTENCE)?|REWRITTEN|OUTPUT|ANSWER)"
    r"\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?|\n?\s*```\s*$")
_ECHO_RE = re.compile(r"^\s*\[\d+\]\s*")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.UNICODE)
_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?\b")
_TIME_RE = re.compile(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b")

# Number words are matched with their diacritics so "từ" (from) is not read
# as "tư" (four), "năm" (year) as "năm" (five), or "bà" as "ba".  ASR output
# in this project is diacritic-preserving even when it is all uppercase.
_NUMBER_WORDS = frozenset(
    """
    không một hai ba bốn tư năm lăm sáu bảy tám chín
    mười mươi trăm nghìn ngàn triệu tỷ tỉ rưỡi linh lẻ
    """.split()
)
_NUMBER_WORDS_FOLDED = frozenset(fold(word) for word in _NUMBER_WORDS)
_DIGIT_WORDS = {
    "không": 0,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
}
_SCALE_WORDS = {
    "k": 1_000,
    "nghìn": 1_000,
    "ngàn": 1_000,
    "tr": 1_000_000,
    "triệu": 1_000_000,
    "tỷ": 1_000_000_000,
    "tỉ": 1_000_000_000,
}

# A single number word is treated as a real number only when it is followed by
# one of these units/classifiers.  This keeps "một lúc" or "một hôm" from being
# forced through the numeric guard.
_NUMBER_UNITS = frozenset(
    """
    giờ phút giây ngày tháng năm tuần đêm tiếng lần
    triệu nghìn ngàn tỷ tỉ rưỡi
    đồng vnd tệ usd euro yên bảng
    lít km mét kg gam phần trăm
    chỗ người cân hộp chiếc cái con vé phòng
    """.split()
)
# "năm" is both the digit five and the year unit.  Inside a number phrase it
# continues the number only before a tens/hundreds/scale word; otherwise it is
# a trailing unit ("hai năm" = 2 years, not 25).
_CONTINUATION_WORDS = frozenset(
    {"mươi", "trăm", "chục", "nghìn", "ngàn", "triệu", "tỷ", "tỉ"}
)
_PHONE_RUN = 4
# A single number word after one of these prefixes is an ordinal/date, so the
# rewriter may normalise "thứ tư" to "thứ 4" and "tháng năm" to "tháng 5".
_NUMBER_PREFIXES = frozenset({"thứ", "ngày", "mùng", "mồng", "tháng", "quý"})

# Diacritic-aware matching avoids the ``dùng`` (use) vs ``đừng`` (don't)
# collision that a folded comparison would create.  The unaccented variants
# are accepted for older/unaccented transcripts, except "dung" itself.
_NEGATION_RE = re.compile(
    r"(?<!\w)(?:không|khong|hông|hổng|hok|đừng|khỏi|khoi|chẳng|chang|chưa|chua|thôi|thoi)(?!\w)",
    re.IGNORECASE,
)

# Raw lowercase surface forms.  Folded matching would confuse "sữa" (milk)
# with "sửa" (edit) and "bát" (bowl) with "bật" (turn on).
_ACTION_WORDS = frozenset(
    """
    đặt gửi chuyển mua xóa lưu gọi nhắc mở tắt đổi hủy đăng tải chia tạo cập nhật
    kiếm tìm thêm bỏ sửa xem đọc nghe phát ghi dánh quay chụp bật khóa
    """.split()
)

_STOPWORDS = frozenset(
    """
    a a oi nhe nha nha hen nhi u vang da roi thi ma la va cua tu voi de da dang se
    giup dum gium ho luon cho
    """.split()
)


# ----------------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------------


def normalize_candidates(
    candidates: Sequence[str] | str,
    *,
    max_candidates: int = 5,
    max_chars: int = 1200,
) -> list[str]:
    """Clean candidate strings and keep at most ``max_candidates`` of them.

    Empty hypotheses are dropped.  Duplicates are intentionally preserved: they
    are the model's only explicit signal that several decoders agree.  Order is
    preserved, so ``candidates[0]`` stays the production top-1 fallback.
    """
    if isinstance(candidates, str):
        candidates = [candidates]
    out: list[str] = []
    for value in candidates:
        text = normalize_space(value)
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        out.append(text)
        if len(out) >= max(1, int(max_candidates)):
            break
    return out


def render_candidate_block(candidates: Sequence[str]) -> str:
    """Render candidates as unambiguous ``[i] "text"`` lines."""
    lines: list[str] = []
    for index, text in enumerate(candidates, 1):
        # JSON quoting keeps newlines/quotes inside one line and makes the
        # candidate boundary obvious to the model.
        lines.append(f"[{index}] {json.dumps(text, ensure_ascii=False)}")
    return "\n".join(lines)


# Only these context fields can help disambiguate a heard word.  In particular
# the installed-app catalog is intentionally excluded: it is large and cannot
# change which candidate is the better transcript.
_CONTEXT_KEYS = ("current_time", "timezone", "location", "user_profile")


def render_context_block(context: dict[str, Any] | None) -> str:
    """Render a small, whitelisted view of the sample context."""
    if not isinstance(context, dict):
        return ""
    lines: list[str] = []
    for key in _CONTEXT_KEYS:
        value = context.get(key)
        if value in (None, "", {}, []):
            continue
        lines.append(
            f"{key}: {json.dumps(value, ensure_ascii=False, separators=(', ', ': '))}"
        )
    return "\n".join(lines)


def render_history_block(
    history: Sequence[dict[str, Any]] | None,
    *,
    max_turns: int = 8,
    max_chars: int = 400,
) -> str:
    """Render prior dialogue text only, dropping tool calls and tool results."""
    if not history:
        return ""
    lines: list[str] = []
    for message in list(history)[-max(1, int(max_turns)):]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        text = normalize_space(str(message.get("text") or message.get("audio_text") or ""))
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def rewriter_messages(
    candidates: Sequence[str],
    *,
    context: dict[str, Any] | None = None,
    history: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    parts: list[str] = []
    context_block = render_context_block(context)
    if context_block:
        parts.append(f"{CONTEXT_HEADER}\n{context_block}")
    history_block = render_history_block(history)
    if history_block:
        parts.append(f"{HISTORY_HEADER}\n{history_block}")
    parts.append(f"{CANDIDATE_HEADER}\n{render_candidate_block(candidates)}")
    user = "\n\n".join(parts) + f"\n\n{CLEAN_LABEL}"
    return [
        {"role": "system", "content": REWRITER_SYSTEM},
        {"role": "user", "content": user},
    ]


# ----------------------------------------------------------------------------------
# Output cleanup
# ----------------------------------------------------------------------------------


def clean_rewriter_output(raw: str) -> str:
    """Strip labels, fences, wrapping quotes and collapsed whitespace."""
    text = str(raw or "").replace("\ufeff", "").strip()
    if not text:
        return ""
    # A single fenced block is common even though the prompt forbids it.
    text = _FENCE_RE.sub("", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    # Drop a leading label line, or a label prefix on the first content line.
    while lines and _LABEL_RE.match(lines[0]):
        lines[0] = _LABEL_RE.sub("", lines[0], count=1).strip()
        if not lines[0]:
            lines.pop(0)
    if not lines:
        return ""
    text = " ".join(lines)
    # Remove repeated labels such as "CLEAN: CLEAN: ...".
    text = _LABEL_RE.sub("", text).strip()
    # A model sometimes wraps the whole answer in quotes.
    if len(text) >= 2 and text[0] in "\"'“”" and text[-1] in "\"'“”":
        text = text[1:-1].strip()
        # The label may have been inside the quotes.
        text = _LABEL_RE.sub("", text).strip()
    return normalize_space(text)


# ----------------------------------------------------------------------------------
# Numeric mentions
# ----------------------------------------------------------------------------------


def _literal_value(token: str, scale: str = "") -> int | None:
    text = token + (f" {scale}" if scale else "")
    value = parse_number(text)
    if value is None:
        return None
    return int(round(float(value)))


def _number_values(text: str) -> Counter[int]:
    """Extract strong numeric mentions, canonicalising words and digits.

    Dates (``15/11``) and times (``9:15``) are extracted first so their
    separators cannot be mistaken for decimal separators.  Plain numbers,
    currency amounts, and spoken number phrases are extracted afterwards.
    """
    lowered = str(text or "").lower()
    if not lowered.strip():
        return Counter()

    values: Counter[int] = Counter()

    def _take_date(match: re.Match[str]) -> str:
        for group in match.groups():
            if group:
                values[int(group)] += 1
        return " "

    def _take_time(match: re.Match[str]) -> str:
        values[int(match.group(1))] += 1
        values[int(match.group(2))] += 1
        return " "

    lowered = _DATE_RE.sub(_take_date, lowered)
    lowered = _TIME_RE.sub(_take_time, lowered)

    tokens = _NUMBER_TOKEN_RE.findall(lowered)

    # A long run of digit words ("không chín năm không...") is a phone number
    # or ID, not a year expression.  Emit it as one value and remove the run
    # so the "năm = year" rule below cannot split it.
    clean_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] in _DIGIT_WORDS:
            end = index
            while end < len(tokens) and tokens[end] in _DIGIT_WORDS:
                end += 1
            if end - index >= _PHONE_RUN:
                digits = "".join(str(_DIGIT_WORDS[word]) for word in tokens[index:end])
                values[int(digits)] += 1
                index = end
                continue
        clean_tokens.append(tokens[index])
        index += 1
    tokens = clean_tokens

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token[0].isdigit():
            literal = token
            next_index = index + 1
            if next_index < len(tokens) and tokens[next_index] in _SCALE_WORDS:
                value = _literal_value(literal, tokens[next_index])
                next_index += 1
            else:
                value = _literal_value(literal)
            if value is not None:
                values[value] += 1
            index = next_index
            continue

        if token in _NUMBER_WORDS:
            start = index
            index += 1
            trailing_unit = False
            while index < len(tokens) and tokens[index] in _NUMBER_WORDS:
                if tokens[index] == "năm" and (
                    index + 1 >= len(tokens)
                    or tokens[index + 1] not in _CONTINUATION_WORDS
                ):
                    trailing_unit = True
                    break
                index += 1
            span = tokens[start:index]
            # A lone number word is only trusted in a numeric context.
            if len(span) == 1 and not (
                (index < len(tokens) and tokens[index] in _NUMBER_UNITS)
                or (start > 0 and tokens[start - 1] in _NUMBER_PREFIXES)
            ):
                if trailing_unit:
                    index += 1
                continue
            # ASR often reads a number digit by digit ("hai sáu" = 26,
            # "không chín..." for a phone number).  A span made only of digit
            # words is a digit string, not an arithmetic sum.
            if len(span) >= 2 and all(word in _DIGIT_WORDS for word in span):
                digits = "".join(str(_DIGIT_WORDS[word]) for word in span)
                values[int(digits)] += 1
            else:
                value = parse_number(" ".join(span))
                if value is not None:
                    values[int(round(float(value)))] += 1
            if trailing_unit:
                index += 1
            continue

        index += 1

    return values


# ----------------------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteGuardConfig:
    min_length_ratio: float = 0.5
    max_length_ratio: float = 2.0
    min_content_recall: float = 0.75
    min_content_precision: float = 0.9
    fuzzy_token_ratio: float = 0.72
    max_output_chars: int = 800
    require_numbers: bool = True
    require_negations: bool = True
    reject_single_support_content: bool = True


@dataclass(frozen=True)
class GuardVerdict:
    ok: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons)}


def _content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for token in _WORD_RE.findall(str(text or "").lower()):
        folded = fold(token)
        if not folded or folded in _STOPWORDS or folded in _NUMBER_WORDS_FOLDED:
            continue
        # Negations are compared as a class by the expansion guard, so the
        # exact particle ("đừng" vs "không") must not hurt token matching.
        if _NEGATION_RE.fullmatch(token):
            continue
        out.append(token)
    return out


def _token_score(left: str, right: str) -> float:
    if left == right or fold(left) == fold(right):
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _greedy_matches(
    source: Sequence[str],
    target: Sequence[str],
    threshold: float,
) -> int:
    used: set[int] = set()
    matched = 0
    for token in source:
        best_index = -1
        best_score = 0.0
        for index, other in enumerate(target):
            if index in used:
                continue
            score = _token_score(token, other)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score >= threshold:
            used.add(best_index)
            matched += 1
    return matched


def _best_candidate(
    candidates: Sequence[str],
    output_tokens: Sequence[str],
    threshold: float = 0.72,
) -> str:
    """The candidate the rewrite is closest to (content-token recall)."""
    best = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        candidate_tokens = _content_tokens(candidate)
        if not candidate_tokens:
            score = 1.0 if not output_tokens else 0.0
        else:
            score = _greedy_matches(candidate_tokens, output_tokens, threshold) / len(candidate_tokens)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _medoid(candidates: Sequence[str], threshold: float = 0.72) -> str:
    """Candidate with the highest average agreement with the other candidates."""
    if len(candidates) <= 1:
        return candidates[0]
    token_sets = [_content_tokens(candidate) for candidate in candidates]
    best_index = 0
    best_score = -1.0
    for index, tokens in enumerate(token_sets):
        total = 0.0
        for other_index, other in enumerate(token_sets):
            if index == other_index:
                continue
            if not tokens and not other:
                total += 1.0
            elif not tokens or not other:
                total += 0.0
            else:
                total += _greedy_matches(tokens, other, threshold) / max(len(tokens), len(other))
        if total > best_score:
            best_index = index
            best_score = total
    return candidates[best_index]


def _support_count(token: str, candidates: Sequence[str], threshold: float) -> int:
    support = 0
    for candidate in candidates:
        if any(_token_score(token, other) >= threshold for other in _content_tokens(candidate)):
            support += 1
    return support


def candidate_agreement(
    candidates: Sequence[str],
    threshold: float = 0.72,
) -> float:
    """Highest directional content-token recall between any candidate pair.

    Directional recall is used instead of symmetric F1 because a complete
    candidate and a truncated version of it should still count as agreement.
    The value is in ``[0, 1]``; 1.0 means every content word of one candidate
    is present in another.
    """
    if len(candidates) < 2:
        return 1.0
    token_sets = [_content_tokens(candidate) for candidate in candidates]
    best = 0.0
    for left_index in range(len(token_sets)):
        for right_index in range(left_index + 1, len(token_sets)):
            left = token_sets[left_index]
            right = token_sets[right_index]
            if not left and not right:
                score = 1.0
            elif not left or not right:
                score = 0.0
            else:
                score = max(
                    _greedy_matches(left, right, threshold) / len(left),
                    _greedy_matches(right, left, threshold) / len(right),
                )
            best = max(best, score)
    return best


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(str(text or "")))


def _ordered_words(text: str) -> tuple[str, ...]:
    # Preserve diacritics and punctuation inside values such as 1.500 or A-12.
    return tuple(re.findall(r"[^\W_]+(?:[.,/-][^\W_]+)*", text.casefold()))


def guard_rewrite(
    candidates: Sequence[str],
    output: str,
    config: RewriteGuardConfig | None = None,
) -> GuardVerdict:
    """Conservative check that the rewrite did not damage the request."""
    cfg = config or RewriteGuardConfig()
    reasons: list[str] = []

    candidates = [candidate for candidate in candidates if normalize_space(candidate)]
    if not candidates:
        return GuardVerdict(False, ("no_candidates",))

    text = normalize_space(output)
    if not text:
        return GuardVerdict(False, ("empty_output",))
    if len(text) > cfg.max_output_chars:
        return GuardVerdict(False, (f"output_too_long:{len(text)}",))
    if _ECHO_RE.match(text) or CANDIDATE_HEADER.lower() in text.lower():
        return GuardVerdict(False, ("echoed_prompt",))

    candidate_word_counts = sorted(len(_WORD_RE.findall(candidate)) for candidate in candidates)
    # Use the median candidate length, not top-1: a truncated top-1 must not
    # make a complete rewrite from another hypothesis look like a runaway.
    baseline_words = max(1, candidate_word_counts[len(candidate_word_counts) // 2])
    output_words = len(_WORD_RE.findall(text))
    ratio = output_words / baseline_words
    if ratio < cfg.min_length_ratio:
        reasons.append(f"output_too_short:{ratio:.2f}")
    elif ratio > cfg.max_length_ratio:
        reasons.append(f"output_too_long_ratio:{ratio:.2f}")
    if reasons:
        return GuardVerdict(False, tuple(reasons))

    # Bag-of-words checks cannot distinguish which recipient owns an amount
    # or which action a negation modifies. Reject unsupported permutations.
    def order_without_polite_ending(value: str) -> tuple[str, ...]:
        order = _ordered_words(value)
        if order and order[-1] in {"nha", "nhé", "ạ"}:
            return order[:-1]
        return order

    output_order = order_without_polite_ending(text)
    candidate_orders = [order_without_polite_ending(candidate) for candidate in candidates]
    if output_order not in candidate_orders and any(
        Counter(order) == Counter(output_order) for order in candidate_orders
    ):
        return GuardVerdict(False, ("unsupported_word_order",))

    output_tokens = _content_tokens(text)
    reference = _best_candidate(candidates, output_tokens, cfg.fuzzy_token_ratio)
    reference_tokens = _content_tokens(reference)
    union_tokens = _content_tokens(" ".join(candidates))

    # 1. The rewrite must keep most of the candidate it is closest to.
    if reference_tokens:
        recall = _greedy_matches(reference_tokens, output_tokens, cfg.fuzzy_token_ratio) / len(reference_tokens)
        if recall < cfg.min_content_recall:
            reasons.append(f"low_content_recall:{recall:.2f}")

    # 2. The rewrite must not invent words that appear in no candidate.
    if output_tokens:
        precision = _greedy_matches(output_tokens, union_tokens, cfg.fuzzy_token_ratio) / len(output_tokens)
        if precision < cfg.min_content_precision:
            reasons.append(f"invented_content:{precision:.2f}")

    # 3. With three or more hypotheses, a content word supported by only one
    #    candidate and absent from the medoid is probably a decoder artefact.
    #    Keep it only when it is also in the production top-1.
    if cfg.reject_single_support_content and len(candidates) >= 3:
        medoid_tokens = _content_tokens(_medoid(candidates, cfg.fuzzy_token_ratio))
        top1_tokens = _content_tokens(candidates[0])
        for token in output_tokens:
            if any(_token_score(token, other) >= cfg.fuzzy_token_ratio for other in medoid_tokens):
                continue
            if any(_token_score(token, other) >= cfg.fuzzy_token_ratio for other in top1_tokens):
                continue
            if _support_count(token, candidates, cfg.fuzzy_token_ratio) <= 1:
                reasons.append(f"unsupported_content:{fold(token)}")

    # 4. Numbers, dates, times, and amounts must survive.
    if cfg.require_numbers:
        reference_numbers = _number_values(reference)
        output_numbers = _number_values(text)
        union_numbers = Counter()
        for candidate in candidates:
            for value, count in _number_values(candidate).items():
                union_numbers[value] = max(union_numbers[value], count)
        missing = reference_numbers - output_numbers
        invented = {value for value in output_numbers if value not in union_numbers}
        for value in sorted(missing):
            reasons.append(f"missing_number:{value}")
        for value in sorted(invented):
            reasons.append(f"invented_number:{value}")

    # 5. A negation/cancellation supported by a strict majority must survive,
    #    and the rewrite must not invent one.  The whole class is compared at
    #    once so "đừng" -> "không" is allowed; only the presence of a negation
    #    is preserved, not the exact particle.
    if cfg.require_negations:
        support = sum(1 for candidate in candidates if _has_negation(candidate))
        output_negated = _has_negation(text)
        if support * 2 > len(candidates) and not output_negated:
            reasons.append("missing_negation")
        elif output_negated and support == 0:
            reasons.append("invented_negation")
        elif output_negated and len(candidates) >= 3 and support * 2 <= len(candidates):
            reasons.append("unsupported_negation")

    # 6. An action word added from a single hypothesis can change what the
    #    assistant does; require majority support unless it is the top-1 word.
    #    Surface matching keeps "sữa" from being read as the verb "sửa".
    if len(candidates) >= 3:
        top1_surface = set(_WORD_RE.findall(candidates[0].lower()))
        for token in output_tokens:
            surface = token.lower()
            if surface not in _ACTION_WORDS:
                continue
            count = _support_count(token, candidates, cfg.fuzzy_token_ratio)
            if count == 1 and surface not in top1_surface:
                reasons.append(f"unsupported_action:{surface}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_reasons = tuple(reason for reason in reasons if not (reason in seen or seen.add(reason)))
    return GuardVerdict(not unique_reasons, unique_reasons)


# ----------------------------------------------------------------------------------
# Rewriter
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteResult:
    text: str
    raw_text: str = ""
    candidates: tuple[str, ...] = ()
    fallback: bool = False
    skipped: bool = False
    reasons: tuple[str, ...] = ()
    seconds: float = 0.0

    @property
    def changed(self) -> bool:
        if not self.candidates:
            return False
        return fold(self.text) != fold(self.candidates[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "candidates": list(self.candidates),
            "fallback": self.fallback,
            "skipped": self.skipped,
            "reasons": list(self.reasons),
            "seconds": round(self.seconds, 4),
            "changed": self.changed,
        }


class Rewriter:
    """Prompt-only or fine-tuned multi-candidate rewriter."""

    def __init__(
        self,
        backend: ChatBackend,
        *,
        max_candidates: int = 5,
        max_candidate_chars: int = 1200,
        max_new_tokens: int = 160,
        seed: int = 13,
        skip_single: bool = True,
        min_candidate_agreement: float = 0.70,
        guard: RewriteGuardConfig | None = None,
    ):
        self.backend = backend
        self.max_candidates = max(1, int(max_candidates))
        self.max_candidate_chars = max(40, int(max_candidate_chars))
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.seed = int(seed)
        self.skip_single = bool(skip_single)
        self.min_candidate_agreement = float(min_candidate_agreement)
        self.guard = guard or RewriteGuardConfig()

    def rewrite(
        self,
        candidates: Sequence[str] | str,
        *,
        context: dict[str, Any] | None = None,
        history: Sequence[dict[str, Any]] | None = None,
    ) -> RewriteResult:
        started = time.perf_counter()
        normalized = normalize_candidates(
            candidates,
            max_candidates=self.max_candidates,
            max_chars=self.max_candidate_chars,
        )
        if not normalized:
            return RewriteResult(
                text="",
                fallback=True,
                reasons=("no_candidates",),
                seconds=time.perf_counter() - started,
            )

        # A single input can still opt out. Multiple agreeing inputs should
        # get punctuation and casing even when there is nothing to select.
        if self.skip_single and len(normalized) == 1:
            return RewriteResult(
                text=normalized[0],
                candidates=tuple(normalized),
                skipped=True,
                reasons=("single_candidate",),
                seconds=time.perf_counter() - started,
            )

        # Two badly corrupted transcripts contain no reliable information to
        # merge.  Asking the LLM to make them fluent only adds hallucinated
        # words, so keep the production top-1 instead.  Directional recall
        # keeps this gate from firing on a complete candidate plus a truncated
        # copy of it.
        if len(normalized) > 1 and self.min_candidate_agreement > 0:
            agreement = candidate_agreement(normalized, self.guard.fuzzy_token_ratio)
            if agreement < self.min_candidate_agreement:
                return RewriteResult(
                    text=normalized[0],
                    candidates=tuple(normalized),
                    fallback=True,
                    reasons=(f"low_candidate_agreement:{agreement:.2f}",),
                    seconds=time.perf_counter() - started,
                )

        messages = rewriter_messages(normalized, context=context, history=history)
        try:
            raw = self.backend.complete(
                messages,
                GenerationSettings(
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    seed=self.seed,
                ),
            )
        except Exception as exc:  # never let a rewriter failure break the sample
            return RewriteResult(
                text=normalized[0],
                candidates=tuple(normalized),
                fallback=True,
                reasons=(f"backend_error:{type(exc).__name__}",),
                seconds=time.perf_counter() - started,
            )

        cleaned = clean_rewriter_output(raw)
        verdict = guard_rewrite(normalized, cleaned, self.guard)
        if verdict.ok and len(normalized) > 1:
            agreed_order = _ordered_words(normalized[0])
            if all(_ordered_words(candidate) == agreed_order for candidate in normalized):
                if _ordered_words(cleaned) != agreed_order:
                    verdict = GuardVerdict(False, ("agreement_requires_formatting_only",))
        if not verdict.ok:
            return RewriteResult(
                text=normalized[0],
                raw_text=raw,
                candidates=tuple(normalized),
                fallback=True,
                reasons=verdict.reasons,
                seconds=time.perf_counter() - started,
            )
        return RewriteResult(
            text=cleaned,
            raw_text=raw,
            candidates=tuple(normalized),
            reasons=verdict.reasons,
            seconds=time.perf_counter() - started,
        )


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit rate used to compare top-1 and oracle candidates."""
    ref = _WORD_RE.findall(fold(reference))
    hyp = _WORD_RE.findall(fold(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, hyp_token in enumerate(hyp, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_token != hyp_token),
            )
        previous = current
    return previous[-1] / len(ref)


def oracle_candidate(reference: str, candidates: Sequence[str]) -> str:
    """The candidate with the lowest WER against a reference transcript."""
    usable = [candidate for candidate in candidates if normalize_space(candidate)]
    if not usable:
        return ""
    return min(usable, key=lambda candidate: word_error_rate(reference, candidate))


def oracle_word_error_rate(reference: str, candidates: Sequence[str]) -> float:
    """Best achievable WER by selecting one of the candidates.

    Compare this with ``word_error_rate(reference, candidates[0])``.  If the
    gap is small, a selector has no headroom and only a single-candidate
    normalizer is worth training.
    """
    chosen = oracle_candidate(reference, candidates)
    return word_error_rate(reference, chosen) if chosen else 1.0


def unique_candidate_count(candidates: Sequence[str]) -> int:
    """How many genuinely different hypotheses a candidate set contains."""
    fingerprints = {
        " ".join(_WORD_RE.findall(fold(candidate)))
        for candidate in candidates
        if normalize_space(candidate)
    }
    return len(fingerprints)
