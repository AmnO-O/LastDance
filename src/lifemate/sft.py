"""Build stepwise supervised examples from labeled train trajectories.

One model serves both stages used by the runtime:

* planner prompt  -> ``{"apps": [...]}``
* caller prompt   -> one action object (call / ask / finish)

The caller examples replay the *gold* trace, so each prompt is exactly the
state the runtime would be in after the previous gold action.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .catalog import Catalog
from .data import PreparedSample, Sample, prepare_sample
from .prompts import caller_messages, planner_messages
from .resolvers import ToolRecord
from .text import fold

# Free-text fields are composed by the model (SMS body, note title, ...), so
# they are deliberately absent from the prompt and must not be filtered.
_COMPOSED_FIELDS = {
    "content",
    "body",
    "subject",
    "title",
    "note",
    "text",
    "message",
    "description",
}
_DATE_LIKE = re.compile(
    r"^\d{4}-\d\d-\d\d([T ]\d\d:\d\d(:\d\d)?([+-]\d\d:\d\d)?)?$"
)
_TIME_LIKE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def _compact(value: Any) -> str:
    """Diacritic-free, punctuation-free form so ``1900 1234`` matches ``19001234``."""
    return re.sub(r"[^a-z0-9]", "", fold(value))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass
class SFTExample:
    task: str  # "planner" | "caller"
    sample_id: str
    step: int
    messages: list[dict[str, str]]
    target: str

    def as_row(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "sample_id": self.sample_id,
            "step": self.step,
            "messages": self.messages,
            "target": self.target,
        }


def planner_example(
    prepared: PreparedSample,
    catalog: Catalog,
    *,
    top_k: int = 4,
    max_history_chars: int = 3200,
) -> SFTExample:
    apps = prepared.sample.gold_apps[:top_k]
    return SFTExample(
        task="planner",
        sample_id=prepared.id,
        step=0,
        messages=planner_messages(prepared, catalog, top_k=top_k, max_history_chars=max_history_chars),
        target=_json({"apps": apps}),
    )


def caller_ask_target(name: str, arguments: dict[str, Any]) -> str:
    tool = str(arguments.get("tool_name") or "")
    params = [str(p) for p in (arguments.get("params") or [])]
    return _json({"action": "ask", "tool": tool, "params": params})


def caller_steps(
    prepared: PreparedSample,
    catalog: Catalog,
    *,
    top_k: int = 4,
    max_steps: int = 10,
    max_history_chars: int = 10000,
    schema_style: str = "dsl",
    include_finish: bool = True,
    initial_focus: list[str] | None = None,
) -> list[SFTExample]:
    sample = prepared.sample
    installed = set(catalog.installed_app_ids(sample.installed_apps))
    focus = (sample.gold_apps[:top_k] if initial_focus is None else
             list(dict.fromkeys(app for app in initial_focus if app in installed))[:top_k])
    results = list(sample.label.get("tool_results") or [])
    trace: list[ToolRecord] = []
    examples: list[SFTExample] = []
    ended_with_ask = False

    for index, call in enumerate(sample.gold_calls):
        name = str(call.get("name") or "")
        arguments = call.get("arguments") or {}
        if not name:
            continue
        if name == "sys.ask_missing_info":
            target = caller_ask_target(name, arguments)
            ended_with_ask = True
        else:
            target = _json({"action": "call", "tool": name, "arguments": arguments})
        examples.append(
            SFTExample(
                task="caller",
                sample_id=prepared.id,
                step=index,
                messages=caller_messages(
                    prepared,
                    catalog,
                    focus,
                    trace,
                    notes=(),
                    step=index,
                    max_steps=max_steps,
                    max_history_chars=max_history_chars,
                    schema_style=schema_style,
                ),
                target=target,
            )
        )
        if ended_with_ask:
            break
        if initial_focus is not None:
            tool = catalog.tools.get(name)
            if tool and tool.app != "sys" and tool.app not in focus:
                focus.append(tool.app)
        trace.append(
            ToolRecord(
                tool=name,
                arguments=arguments,
                result=results[index] if index < len(results) else None,
                step=index + 1,
            )
        )

    if include_finish and not ended_with_ask:
        examples.append(
            SFTExample(
                task="caller",
                sample_id=prepared.id,
                step=len(sample.gold_calls),
                messages=caller_messages(
                    prepared,
                    catalog,
                    focus,
                    trace,
                    notes=(),
                    step=len(sample.gold_calls),
                    max_steps=max_steps,
                    max_history_chars=max_history_chars,
                    schema_style=schema_style,
                ),
                target=_json({"action": "finish"}),
            )
        )
    return examples


def build_examples(
    catalog: Catalog,
    samples: Iterable[Sample],
    *,
    tasks: tuple[str, ...] = ("planner", "caller"),
    top_k: int = 4,
    max_steps: int = 10,
    max_history_chars: int = 10000,
    schema_style: str = "dsl",
    include_finish: bool = True,
    derivable_only: bool = False,
    stats: dict[str, int] | None = None,
) -> list[SFTExample]:
    out: list[SFTExample] = []
    dropped = 0
    for sample in samples:
        prepared = prepare_sample(sample)
        if "planner" in tasks:
            out.append(planner_example(prepared, catalog, top_k=top_k))
        if "caller" in tasks:
            for example in caller_steps(
                prepared,
                catalog,
                top_k=top_k,
                max_steps=max_steps,
                max_history_chars=max_history_chars,
                schema_style=schema_style,
                include_finish=include_finish,
            ):
                if derivable_only and not caller_target_derivable(example, catalog):
                    dropped += 1
                    continue
                out.append(example)
    if stats is not None:
        stats["dropped_nonderivable"] = dropped
    return out


def caller_target_derivable(example: SFTExample, catalog: Catalog) -> bool:
    """Whether every literal target argument is visible in the caller prompt.

    ~2% of gold caller steps ask for a value that the model cannot see: a
    contact id from a different environment snapshot, a spoken phone number, a
    unit conversion. Training on those teaches invention, so the SFT entry
    point filters them out. Composed free text and computed dates are expected
    to be absent and are kept.
    """
    try:
        target = json.loads(example.target)
    except json.JSONDecodeError:
        return True
    if target.get("action") != "call":
        return True
    arguments = target.get("arguments") or {}
    if not arguments:
        return True
    tool = catalog.tools.get(str(target.get("tool") or ""))
    prompt = _compact(" ".join(str(m.get("content") or "") for m in example.messages))
    for name, value in arguments.items():
        if not isinstance(value, str) or len(value) < 2:
            continue
        if name in _COMPOSED_FIELDS or _DATE_LIKE.match(value) or _TIME_LIKE.match(value):
            continue
        if _compact(value) not in prompt:
            return False
    return True
