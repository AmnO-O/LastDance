"""Stepwise caller: turns the focused catalog into one executable action per turn."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .catalog import Catalog
from .data import PreparedSample
from .llm import ChatBackend, GenerationSettings
from .prompts import caller_messages
from .resolvers import ToolRecord
from .text import extract_json_values, first_json_object, fold


@dataclass
class CallerAction:
    action: str = "invalid"
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    params: list[str] = field(default_factory=list)
    app: str | None = None
    message: str | None = None
    raw_text: str = ""
    messages: list[dict[str, str]] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.action}
        if self.tool:
            out["tool"] = self.tool
        if self.arguments:
            out["arguments"] = self.arguments
        if self.params:
            out["params"] = self.params
        if self.app:
            out["app"] = self.app
        if self.message:
            out["message"] = self.message
        return out


def _action_name(value: Any) -> str:
    text = fold(value).replace(" ", "_")
    return {
        "tool": "call",
        "invoke": "call",
        "execute": "call",
        "use_tool": "call",
        "done": "finish",
        "end": "finish",
        "stop": "finish",
        "complete": "finish",
        "need_tool": "need_app",
        "need": "need_app",
        "switch_app": "need_app",
        "request_app": "need_app",
        "missing_info": "ask",
        "ask_missing_info": "ask",
        "ask_info": "ask",
    }.get(text, text if text in {"call", "finish", "need_app", "ask", "invalid"} else "invalid")


def _as_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def parse_caller_action(
    text: str,
    catalog: Catalog,
    focus_apps: Iterable[str] | None = None,
    installed_apps: dict[str, Any] | None = None,
) -> CallerAction:
    candidates = [v for v in extract_json_values(text or "") if isinstance(v, dict)]
    obj = candidates[0] if candidates else first_json_object(text or "")
    focus = list(focus_apps or [])
    if isinstance(obj, dict):
        raw_action = obj.get("action") or obj.get("type") or obj.get("op") or ""
        action = _action_name(raw_action)
        tool = obj.get("tool") or obj.get("tool_name") or obj.get("name") or obj.get("function")
        if isinstance(tool, dict):  # OpenAI tool-call shape
            tool = tool.get("name")
        arguments = _as_arguments(obj.get("arguments") or obj.get("args") or obj.get("parameters"))
        params = obj.get("params") or obj.get("missing_params") or obj.get("missing")
        if isinstance(params, str):
            params = [params]
        if not isinstance(params, list):
            params = []
        app = obj.get("app") or obj.get("app_id") or obj.get("class")
        # A model may express the ask via the sys tool while naming the real
        # target in ``tool_name``; the runtime needs the target tool, not sys.
        target_tool = obj.get("tool_name") or obj.get("target_tool")
        if isinstance(target_tool, str) and target_tool.strip():
            if str(tool or "").strip() in {"sys.ask_missing_info", "ask_missing_info"}:
                tool = target_tool.strip()
                if action == "call":
                    action = "ask"
        # Models frequently put the tool name in "action" instead of "call".
        # Treat a dotted/known action value as the intended tool call.
        action_text = str(raw_action).strip()
        if action == "invalid" and action_text and "." in action_text:
            resolved_action = catalog.resolve_tool(action_text, focus)
            if resolved_action is not None:
                action = "call"
                tool = resolved_action.name
        elif action == "call" and action_text and "." in action_text:
            resolved_action = catalog.resolve_tool(action_text, focus)
            if resolved_action is not None:
                tool = resolved_action.name

        if action == "invalid":
            if tool and arguments is not None and ("arguments" in obj or "args" in obj or "parameters" in obj):
                action = "call"
            elif tool and params:
                action = "ask"
            elif app:
                action = "need_app"
            elif any(k in obj for k in ("finish", "done")) and obj.get("finish") is not False:
                action = "finish"
            elif fold(text).strip() in {"finish", "done", "stop", "{}"}:
                action = "finish"
        if tool is not None:
            tool = str(tool).strip()
            if tool and "." not in tool and focus:
                resolved = catalog.resolve_tool(tool, focus)
                if resolved is not None:
                    tool = resolved.name
        if action == "need_app":
            if not app and tool and "." in str(tool):
                app = str(tool).split(".", 1)[0]
            return CallerAction(
                action="need_app",
                tool=str(tool) if tool else None,
                arguments=arguments,
                params=[str(x) for x in params],
                app=str(app) if app else None,
                raw_text=text,
            )
        if action == "ask":
            return CallerAction(
                action="ask",
                tool=str(tool) if tool else None,
                arguments=arguments,
                params=[str(x) for x in params],
                app=str(app) if app else None,
                raw_text=text,
            )
        if action == "call":
            return CallerAction(
                action="call",
                tool=str(tool) if tool else None,
                arguments=arguments,
                params=[str(x) for x in params],
                app=str(app) if app else None,
                raw_text=text,
            )
        if action == "finish":
            return CallerAction(action="finish", raw_text=text)
        return CallerAction(action="invalid", message="response did not contain a valid action", raw_text=text)
    # A model that says "finish" in prose is harmless; anything else gets a repair turn.
    if fold(text).strip() in {"finish", "done", "stop", "end"}:
        return CallerAction(action="finish", raw_text=text)
    return CallerAction(action="invalid", message="could not parse a JSON action", raw_text=text)


class Caller:
    def __init__(
        self,
        backend: ChatBackend,
        catalog: Catalog,
        *,
        max_new_tokens: int = 320,
        max_history_chars: int = 10000,
        max_steps: int = 12,
        seed: int = 13,
        schema_style: str = "dsl",
        include_fallback_descriptions: bool = False,
        capture_prompts: bool = False,
        collect_metrics: bool = False,
        discovery_prompt: str | None = None,
    ):
        self.backend = backend
        self.catalog = catalog
        self.max_new_tokens = int(max_new_tokens)
        self.max_history_chars = int(max_history_chars)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.schema_style = str(schema_style)
        self.include_fallback_descriptions = bool(include_fallback_descriptions)
        self.capture_prompts = capture_prompts
        self.collect_metrics = collect_metrics
        self.discovery_prompt = discovery_prompt

    def next_action(
        self,
        prepared: PreparedSample,
        focus_apps: Iterable[str],
        trace: Iterable[ToolRecord],
        notes: Iterable[str] = (),
        step: int = 0,
        roadmap: str = "",
        discovery: bool = False,
        discovered_tools: Iterable[str] = (),
    ) -> CallerAction:
        focus = list(focus_apps)
        messages = caller_messages(
            prepared,
            self.catalog,
            focus_apps=focus,
            trace=trace,
            notes=notes,
            step=step,
            max_steps=self.max_steps,
            max_history_chars=self.max_history_chars,
            schema_style=self.schema_style,
            include_fallback_descriptions=self.include_fallback_descriptions,
            roadmap=roadmap,
            discovery=discovery,
            discovered_tools=discovered_tools,
        )
        started = time.perf_counter()
        if discovery and self.discovery_prompt is not None:
            messages[0]["content"] = self.discovery_prompt
        try:
            raw = self.backend.complete(
                messages,
                GenerationSettings(max_new_tokens=self.max_new_tokens, temperature=0.0, seed=self.seed),
            )
        except Exception as exc:
            action = CallerAction(action="invalid", message=f"{type(exc).__name__}: {exc}")
        else:
            action = parse_caller_action(
                raw,
                self.catalog,
                focus_apps=focus,
                installed_apps=prepared.sample.installed_apps,
            )
        if self.capture_prompts:
            action.messages = messages
        if self.collect_metrics:
            action.metrics = {**getattr(self.backend, "last_generation", {}),
                              "seconds": time.perf_counter() - started}
        return action
