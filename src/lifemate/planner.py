"""App-level planner: one model call, ranked installed-app shortlist."""
from __future__ import annotations

import json
import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

from .catalog import Catalog
from .data import PreparedSample
from .llm import ChatBackend, GenerationSettings
from .prompts import planner_messages
from .text import extract_json_values, fold, token_f1, tokens, unique_preserve


@dataclass
class PlanStep:
    app: str
    goal: str
    condition: str | None
    source_index: int


@dataclass
class PlanDecision:
    apps: list[str] = field(default_factory=list)
    raw_text: str = ""
    parsed: bool = False
    error: str | None = None
    fallback: bool = False
    steps: list[PlanStep] = field(default_factory=list)
    steps_parsed: bool | None = None
    apps_source: str | None = None
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = {
            "apps": self.apps,
            "parsed": self.parsed,
            "error": self.error,
            "fallback": self.fallback,
        }
        if self.steps_parsed is not None:
            result.update(steps=[asdict(step) for step in self.steps], steps_parsed=self.steps_parsed,
                          apps_source=self.apps_source, notes=self.notes)
        if self.metrics:
            result["metrics"] = self.metrics
        return result


def _normalise_app(token: Any, catalog: Catalog, installed: set[str]) -> str | None:
    if not isinstance(token, str):
        return None
    text = str(token).strip().strip("\"'`")
    if not text:
        return None
    low = fold(text)
    if low in installed:
        return low
    # Tolerate "weather (Thời tiết)" or "app: weather".
    text = re.sub(r"^(app|ứng dụng|ung dung)\s*[:=-]\s*", "", text, flags=re.I)
    low = fold(text)
    if low in installed:
        return low
    alias = catalog.app_name_map()
    if low in alias and alias[low] in installed:
        return alias[low]
    # Try the first token / id-like substring.
    for match in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text):
        key = fold(match)
        if key in installed:
            return key
        if key in alias and alias[key] in installed:
            return alias[key]
    return None


def parse_plan(
    text: str,
    catalog: Catalog,
    installed_apps: dict[str, Any] | Iterable[str],
    top_k: int = 4,
) -> PlanDecision:
    installed = set(catalog.installed_app_ids(installed_apps))
    raw_apps: list[Any] | None = None
    parsed = False
    explicit_empty = False
    for value in extract_json_values(text or ""):
        if isinstance(value, dict):
            if "apps" in value:
                raw_apps = value.get("apps")
                if isinstance(raw_apps, str):
                    raw_apps = [raw_apps]
                if raw_apps is None:
                    raw_apps = []
                if not isinstance(raw_apps, list):
                    raw_apps = None
                    continue
                explicit_empty = not raw_apps or bool(value.get("no_app"))
                parsed = True
                break
            if "app" in value:
                raw_apps = [value.get("app")]
                parsed = True
                break
        elif isinstance(value, list):
            raw_apps = value
            parsed = True
            break
    if raw_apps is None:
        # Last-resort recovery: scan the model output for installed ids.
        raw_apps = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text or "")
    apps: list[str] = []
    for token in raw_apps or []:
        app = _normalise_app(token, catalog, installed)
        if app:
            apps.append(app)
    apps = unique_preserve(apps)
    if explicit_empty:
        apps = []
    return PlanDecision(apps=apps[: max(0, top_k)], raw_text=text or "", parsed=parsed and (parsed or explicit_empty))


_PLAN_REF = re.compile(r"\b(?:P|step\s+)(\d+)\b", re.IGNORECASE)
_LEADING_APPS = re.compile(r'^\s*(?:```(?:json)?\s*)?\{\s*"apps"\s*:\s*', re.IGNORECASE)


def _recover_leading_apps(text: str) -> list | None:
    """Recover only a complete apps array at the start of an incomplete plan."""
    match = _LEADING_APPS.match(text)
    if not match:
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(text, match.end())
    except ValueError:
        return None
    tail = text[end:].lstrip()
    if isinstance(value, list) and (not tail or tail[0] in ",}"):
        return value
    return None


def parse_roadmap(text, catalog, installed_apps, top_k=4) -> PlanDecision:
    """Parse independent shortlist and roadmap fields in the experimental format."""
    values = extract_json_values(text or "")
    objects = [v for v in values if isinstance(v, dict) and "steps" in v]
    payload = next((v for v in reversed(objects) if "apps" in v or "app" in v),
                   objects[-1] if objects else None)
    repaired = False
    if payload is None and not values and (text or "").lstrip().startswith("{"):
        try:
            candidate = json.loads(text.rstrip() + "}")
        except ValueError:
            pass
        else:
            if isinstance(candidate, dict) and "apps" in candidate and isinstance(candidate.get("steps"), list):
                payload = candidate
                repaired = True
    decision = parse_plan(json.dumps(payload, ensure_ascii=False) if payload is not None else text,
                          catalog, installed_apps, top_k)
    if repaired:
        decision.notes.extend(("invalid_json", "repaired_missing_object_close"))
    if payload is None and not decision.parsed:
        recovered = _recover_leading_apps(text or "")
        if recovered:
            decision = parse_plan(json.dumps({"apps": recovered}), catalog, installed_apps, top_k)
            if decision.apps:
                decision.notes.append("recovered_leading_apps")
            else:
                decision.parsed = False
    if not decision.parsed:
        # Ignore IDs scanned from incomplete text; Planner uses request-based fallback.
        decision.apps = []
    decision.raw_text = text
    decision.steps_parsed = False
    decision.apps_source = ("recovered" if "recovered_leading_apps" in decision.notes else
                            "model" if decision.parsed else None)
    if payload is None:
        decision.notes.append("missing_steps_object" if values else "invalid_json")
        return decision
    installed = set(catalog.installed_app_ids(installed_apps))
    raw_apps, raw_steps = payload.get("apps"), payload.get("steps")
    if isinstance(raw_apps, list):
        decision.notes.extend(f"invalid_app_entry:{i}" for i, app in enumerate(raw_apps, 1)
                              if not isinstance(app, str))
    if not decision.parsed:
        decision.notes.append("invalid_apps")

    if not isinstance(raw_steps, list):
        decision.notes.append("invalid_steps")
        return decision
    kept = {}
    invalid_step = False
    for index, item in enumerate(raw_steps, 1):
        if (not isinstance(item, list) or len(item) not in (2, 3)
                or not isinstance(item[0], str) or not isinstance(item[1], str)
                or not 0 < len(" ".join(item[1].split())) <= 100
                or (len(item) == 3 and item[2] is not None
                    and (not isinstance(item[2], str) or len(item[2]) > 100))):
            decision.notes.append(f"invalid_step:{index}")
            invalid_step = True
            continue
        app = _normalise_app(item[0], catalog, installed)
        if app is None:
            decision.notes.append(f"unavailable_step_app:{index}")
            continue
        goal = " ".join(item[1].split())
        condition = (" ".join((item[2] or "").split()) or None) if len(item) == 3 else None
        if condition and any(int(ref) not in kept for ref in _PLAN_REF.findall(condition)):
            decision.notes.append(f"unresolved_step_dependency:{index}")
            continue
        if len(kept) >= 4:
            decision.notes.append(f"step_cap:{index}")
            continue
        kept[index] = PlanStep(app, goal, condition, index)

    renumbered = {source: i for i, source in enumerate(kept, 1)}
    decision.steps = [
        replace(step, condition=_PLAN_REF.sub(lambda m: f"P{renumbered[int(m[1])]}", step.condition))
        if step.condition else step for step in kept.values()
    ]
    decision.steps_parsed = not invalid_step
    step_apps = unique_preserve(step.app for step in decision.steps)
    if step_apps and not decision.apps:
        if raw_apps == []:
            decision.notes.append("apps_empty_with_steps")
        decision.apps = step_apps[:max(0, top_k)]
        decision.apps_source = "derived"
        decision.parsed = True
    elif step_apps and set(step_apps) != set(decision.apps):
        decision.notes.append("apps_steps_mismatch")
    if not decision.apps and raw_apps != []:
        decision.parsed = False
    return decision


class Planner:
    def __init__(
        self,
        backend: ChatBackend,
        catalog: Catalog,
        *,
        top_k: int = 4,
        max_new_tokens: int = 120,
        max_history_chars: int = 3200,
        seed: int = 13,
        include_steps: bool = False,
        collect_metrics: bool = False,
    ):
        self.backend = backend
        self.catalog = catalog
        self.top_k = int(top_k)
        self.max_new_tokens = int(max_new_tokens)
        self.max_history_chars = int(max_history_chars)
        self.seed = int(seed)
        self.include_steps = include_steps
        self.collect_metrics = collect_metrics

    def plan(self, prepared: PreparedSample) -> PlanDecision:
        messages = planner_messages(
            prepared,
            self.catalog,
            top_k=self.top_k,
            max_history_chars=self.max_history_chars,
            include_steps=self.include_steps,
        )
        started = time.perf_counter()
        try:
            raw = self.backend.complete(
                messages,
                GenerationSettings(max_new_tokens=self.max_new_tokens, temperature=0.0, seed=self.seed),
            )
        except Exception as exc:  # the runtime should still be able to answer with a fallback
            decision = PlanDecision(error=f"{type(exc).__name__}: {exc}",
                                    steps_parsed=False if self.include_steps else None)
            if not self.include_steps:
                return decision
        else:
            parser = parse_roadmap if self.include_steps else parse_plan
            decision = parser(raw, self.catalog, prepared.sample.installed_apps, self.top_k)
        if self.collect_metrics:
            decision.metrics = {**getattr(self.backend, "last_generation", {}),
                                "seconds": time.perf_counter() - started}
        if not decision.parsed:
            decision.fallback = True
            decision.apps = lexical_fallback(prepared, self.catalog, self.top_k)
            if self.include_steps:
                decision.apps_source = "lexical"
        return decision


class GoldPlanner:
    """Debug-only oracle planner used to isolate caller errors from planner errors."""

    def __init__(self, catalog: Catalog, top_k: int = 4):
        self.catalog = catalog
        self.top_k = int(top_k)

    def plan(self, prepared: PreparedSample) -> PlanDecision:
        installed = set(self.catalog.installed_app_ids(prepared.sample.installed_apps))
        apps = [a for a in prepared.sample.gold_apps if a in installed]
        return PlanDecision(apps=apps[: self.top_k], raw_text="", parsed=True)


class EmptyPlanner:
    def __init__(self, catalog: Catalog | None = None):
        self.catalog = catalog

    def plan(self, prepared: PreparedSample) -> PlanDecision:
        return PlanDecision(apps=[], raw_text="", parsed=True)


def lexical_fallback(prepared: PreparedSample, catalog: Catalog, top_k: int) -> list[str]:
    """Cheap recovery when a prompt-only planner emits unparseable text.

    This is a safety net, not the intended router: it only runs when parsing fails.
    """
    request = fold(prepared.request_text)
    request_tokens = tokens(request)
    scored: list[tuple[float, str]] = []
    for app_id in catalog.installed_app_ids(prepared.sample.installed_apps):
        app = catalog.apps[app_id]
        haystack = " ".join([app.id, app.name_vi, app.description] + [t.name + " " + t.description for t in app.tools])
        hay_tokens = tokens(haystack)
        overlap = len(request_tokens & hay_tokens)
        if not overlap:
            continue
        score = token_f1(request, haystack) + 0.02 * overlap
        scored.append((score, app_id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [app_id for _, app_id in scored[: max(0, top_k)]]
