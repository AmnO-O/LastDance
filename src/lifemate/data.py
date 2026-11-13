"""Data loading, transcript preparation and deterministic dev splits."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from .text import unique_preserve


class MissingTranscript(RuntimeError):
    pass


class Transcriber(Protocol):
    def transcribe(self, path: Path, cache_key: str = "") -> str: ...


def find_data_root(start: str | Path | None = None) -> Path:
    """Find the directory that contains split folders such as ``train/`` and ``public_test/``."""
    start_path = Path(start or Path.cwd()).resolve()
    candidates = [
        start_path,
        start_path / "train+public_test",
        start_path / "train+public_test" / "train+public_test",
        *start_path.parents[:5],
        *[p / "train+public_test" for p in start_path.parents[:5]],
        *[p / "train+public_test" / "train+public_test" for p in start_path.parents[:5]],
    ]
    for path in candidates:
        if (
            (path / "train" / "train_tools_catalog.json").is_file()
            or (path / "public_test" / "public_test_tools_catalog.json").is_file()
        ):
            return path
    # Common layout used by the released package: <root>/<split>/zaic_tools
    for path in candidates:
        if (path / "zaic_tools" / "__init__.py").is_file() and path.name in {"train", "public_test"}:
            return path.parent
    raise FileNotFoundError(
        "Could not find the data root. Pass --root pointing to the folder that contains train/ and public_test/."
    )


def split_dir(root: str | Path, split: str) -> Path:
    return Path(root) / split


def catalog_path(root: str | Path, split: str) -> Path:
    return split_dir(root, split) / f"{split}_tools_catalog.json"


def jsonl_path(root: str | Path, split: str) -> Path:
    return split_dir(root, split) / f"{split}.jsonl"


@dataclass
class Sample:
    raw: dict[str, Any]
    split: str
    root: Path | None = None

    @property
    def id(self) -> str:
        return str(self.raw.get("id") or "")

    @property
    def context(self) -> dict[str, Any]:
        return self.raw.get("context") or {}

    @property
    def installed_apps(self) -> dict[str, Any]:
        return self.context.get("installed_apps") or {}

    @property
    def current_time(self) -> str:
        return str(self.context.get("current_time") or "")

    @property
    def timezone(self) -> str:
        return str(self.context.get("timezone") or "Asia/Ho_Chi_Minh")

    @property
    def user_profile(self) -> dict[str, Any]:
        return self.context.get("user_profile") or {}

    @property
    def location(self) -> dict[str, Any]:
        return self.context.get("location") or {}

    @property
    def audio(self) -> str:
        return str(self.raw.get("audio") or "")

    @property
    def audio_text(self) -> str:
        return str(self.raw.get("audio_text") or "")

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self.raw.get("history") or [])

    @property
    def label(self) -> dict[str, Any]:
        return self.raw.get("label") or {}

    @property
    def gold_calls(self) -> list[dict[str, Any]]:
        return list((self.label or {}).get("tool_calls") or [])

    @property
    def gold_apps(self) -> list[str]:
        """Target apps, including the app of ``sys.ask_missing_info.arguments.tool_name``."""
        apps: list[str] = []
        for call in self.gold_calls:
            name = str(call.get("name") or "")
            if name == "sys.ask_missing_info":
                target = str((call.get("arguments") or {}).get("tool_name") or "")
            else:
                target = name
            if "." in target:
                apps.append(target.split(".", 1)[0])
        return unique_preserve(apps)

    @property
    def has_label(self) -> bool:
        return bool(self.label)

    def audio_path(self, relative: str | None = None) -> Path:
        rel = relative if relative is not None else self.audio
        if not rel:
            raise FileNotFoundError(f"sample {self.id} has no audio path")
        if self.root is None:
            raise FileNotFoundError("Sample has no root; audio path cannot be resolved")
        rel_path = Path(rel)
        if rel_path.is_absolute():
            return rel_path
        return Path(self.root) / self.split / rel_path


@dataclass
class PreparedSample:
    sample: Sample
    request_text: str
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.sample.id


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_ids(path: str | Path) -> list[str]:
    """Read a fixed id list from JSON (list / {"ids": [...]}) or one id per line."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return unique_preserve(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict) and isinstance(value.get("ids"), list):
        return unique_preserve(str(item).strip() for item in value["ids"] if str(item).strip())
    ids: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("id"):
                ids.append(str(row["id"]))
            continue
        ids.append(line)
    return unique_preserve(ids)


def find_validation_ids(root: str | Path | None = None) -> Path | None:
    """Locate the shared validation id file used for every dev run (prioritizing 200, fallback 100)."""
    root_path = Path(root) if root else None
    filenames = ["validation_200_ids.txt", "validation_100_ids.txt"]
    candidates: list[Path] = []
    if root_path is not None:
        for fname in filenames:
            candidates.extend(
                [
                    root_path / "train" / fname,
                    root_path / fname,
                    root_path / "public_test" / fname,
                ]
            )
    for fname in filenames:
        candidates.append(Path(fname))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_samples(
    root: str | Path,
    split: str,
    ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Sample]:
    root = Path(root)
    wanted = set(ids) if ids is not None else None
    out: list[Sample] = []
    for raw in load_jsonl(jsonl_path(root, split)):
        if wanted is not None and str(raw.get("id")) not in wanted:
            continue
        out.append(Sample(raw=raw, split=split, root=root))
        if limit is not None and len(out) >= limit:
            break
    if wanted is not None:
        found = {s.id for s in out}
        missing = wanted - found
        if missing:
            raise KeyError(f"unknown sample ids: {sorted(missing)[:10]}")
        order = {sid: i for i, sid in enumerate(ids)}
        out.sort(key=lambda s: order.get(s.id, 10**9))
    return out


def _transcribe_relative(
    sample: Sample,
    relative: str,
    transcriber: Transcriber | None,
    turn: str = "current",
) -> str:
    if not relative:
        return ""
    if transcriber is None:
        raise MissingTranscript(f"{sample.id}: no transcript for {relative}")
    path = sample.audio_path(relative)
    return transcriber.transcribe(path, cache_key=f"{sample.id}:{turn}")


def prepare_sample(
    sample: Sample,
    transcriber: Transcriber | None = None,
    require_text: bool = True,
) -> PreparedSample:
    """Fill transcript text for the final request and every previous user turn.

    Training files already contain ``audio_text``; public/private files require an
    ASR implementation or a supplied transcript JSONL.
    """
    request_text = sample.audio_text
    if not request_text:
        if require_text:
            request_text = _transcribe_relative(sample, sample.audio, transcriber, "current")
        elif transcriber is not None and sample.audio:
            try:
                request_text = _transcribe_relative(sample, sample.audio, transcriber, "current")
            except Exception:
                request_text = ""
    if require_text and not request_text:
        raise MissingTranscript(f"{sample.id}: empty final transcript")

    history: list[dict[str, Any]] = []
    user_turn = 0
    for msg in sample.history:
        item = dict(msg)
        if item.get("role") == "user" and not item.get("text"):
            text = str(item.get("audio_text") or "")
            if not text and item.get("audio"):
                if require_text:
                    text = _transcribe_relative(sample, str(item["audio"]), transcriber, str(user_turn))
                elif transcriber is not None:
                    try:
                        text = _transcribe_relative(sample, str(item["audio"]), transcriber, str(user_turn))
                    except Exception:
                        text = ""
            item["text"] = text
        if item.get("role") == "user":
            user_turn += 1
        history.append(item)
    return PreparedSample(sample=sample, request_text=request_text, history=history)


def read_transcript_file(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read ``{id: text}`` or ``{id: {"current": ..., "history": [...]}}`` JSON/JSONL."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        raw: dict[str, Any] = {}
        for row in load_jsonl(path):
            sid = str(row.get("id") or "")
            if sid:
                raw[sid] = row
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for sid, value in raw.items():
        if isinstance(value, str):
            out[str(sid)] = {"current": value, "history": []}
        elif isinstance(value, dict):
            out[str(sid)] = {
                "current": str(value.get("current") or value.get("audio_text") or value.get("text") or ""),
                "history": [str(x) if not isinstance(x, dict) else str(x.get("text") or "") for x in value.get("history", [])],
            }
    return out


class FixedTranscripts:
    """Transcriber backed by an external transcript mapping (e.g. an ASR cache)."""

    def __init__(self, mapping: dict[str, dict[str, Any]]):
        self.mapping = mapping

    def text_for(self, sample_id: str, turn: str = "current") -> str:
        bucket = self.mapping.get(sample_id) or {}
        if turn == "current":
            return str(bucket.get("current") or "")
        try:
            idx = int(turn)
        except ValueError:
            return ""
        history = bucket.get("history") or []
        return str(history[idx]) if 0 <= idx < len(history) else ""

    def transcribe(self, path: Path, cache_key: str = "") -> str:
        sample_id, _, turn = (cache_key or "").partition(":")
        return self.text_for(sample_id or path.parent.name, turn or "current")


def stratified_dev_ids(samples: list[Sample], n: int = 100, seed: int = 13) -> list[str]:
    """Deterministic coverage-oriented dev split.

    First allocate by the number of target apps (proportionally, with at least one
    sample from every non-empty shape such as 4-app and 5-app requests), then spread
    each allocation across primary apps. This is for local evaluation only.
    """
    rng = random.Random(seed)
    by_size: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        apps = sample.gold_apps
        by_size[len(apps)][apps[0] if apps else "__none__"].append(sample.id)
    for bucket in by_size.values():
        for values in bucket.values():
            rng.shuffle(values)

    sizes = sorted(by_size)
    if not sizes:
        return []
    if n <= 0:
        return []

    # Largest-remainder allocation by target-app count, with a floor of one for
    # every shape that exists (so rare 4/5-app requests are represented).
    total = len(samples)
    quotas: dict[int, int] = {}
    for size in sizes:
        quotas[size] = 1 if by_size[size] else 0
    remaining = max(0, n - sum(quotas.values()))
    weights = {size: sum(len(v) for v in by_size[size].values()) for size in sizes}
    exact = {size: remaining * weights[size] / total for size in sizes}
    for size in sizes:
        quotas[size] += int(exact[size])
    left = n - sum(quotas.values())
    for size in sorted(sizes, key=lambda s: (-(exact[s] - int(exact[s])), s))[:left]:
        quotas[size] += 1
    # Do not exceed what is available.
    for size in sizes:
        quotas[size] = min(quotas[size], weights[size])
    left = n - sum(quotas.values())
    for size in sorted(sizes, key=lambda s: (-weights[s], s)):
        take = min(left, weights[size] - quotas[size])
        if take > 0:
            quotas[size] += take
            left -= take
        if left <= 0:
            break

    selected: list[str] = []
    for size in sizes:
        count = quotas[size]
        buckets = by_size[size]
        keys = sorted(buckets)
        rng.shuffle(keys)
        # Take one per primary app before taking a second from any bucket.
        cursor = {k: 0 for k in keys}
        while count > 0:
            progressed = False
            for key in keys:
                if count <= 0:
                    break
                pos = cursor[key]
                if pos < len(buckets[key]):
                    selected.append(buckets[key][pos])
                    cursor[key] = pos + 1
                    count -= 1
                    progressed = True
            if not progressed:
                break
    rng.shuffle(selected)
    return selected[:n]


def dataset_stats(samples: Iterable[Sample]) -> dict[str, Any]:
    samples = list(samples)
    app_counts = Counter(len(s.gold_apps) for s in samples)
    call_counts = Counter(len(s.gold_calls) for s in samples)
    primary = Counter(s.gold_apps[0] if s.gold_apps else "__none__" for s in samples)
    return {
        "samples": len(samples),
        "installed_apps_mean": (sum(len(s.installed_apps) for s in samples) / len(samples)) if samples else 0,
        "target_app_counts": dict(sorted(app_counts.items())),
        "call_counts": dict(sorted(call_counts.items())),
        "primary_apps": dict(primary.most_common()),
        "empty": sum(1 for s in samples if not s.gold_calls),
        "ask": sum(1 for s in samples if any(c.get("name") == "sys.ask_missing_info" for c in s.gold_calls)),
    }
