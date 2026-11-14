"""Per-sample Search-First loop: one ToolEnv, caller step by step.

The agent has no planner stage.  It starts with the schemas passed as
``initial_apps`` (empty in production) and discovers every other tool through
``sys.search_tools``.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .caller import Caller, CallerAction
from .catalog import Catalog, ToolSpec
from .data import PreparedSample, Sample, Transcriber, prepare_sample
from .discovery import SEARCH_TOOL, search_tools
from .resolvers import (
    ResolutionContext,
    ResolutionError,
    ToolRecord,
    UnresolvedReference,
    is_empty,
    missing_requirement_params,
    read_hint_for_xref,
    resolve_arguments,
)


@dataclass
class AgentConfig:
    top_k: int = 4
    max_steps: int = 10
    max_reprompts: int = 2
    max_auto_reads_per_step: int = 2
    auto_read: bool = True
    auto_ask: bool = True
    stop_after_ask: bool = True
    max_history_chars: int = 10000
    max_notes: int = 8
    tool_discovery: bool = False
    max_searches: int = 1
    verify_finish: bool = False
    preopen_tools: int = 0


# A successful identical call is normally a loop. A few write tools are
# explicitly repeatable, though: redeeming the same reward twice buys a second
# voucher (gold does this when the user says "redeem as many as possible").
REPEATABLE_WRITE_TOOLS = frozenset({"loyalty.redeem"})



@dataclass
class RunResult:
    sample_id: str
    calls: list[dict[str, Any]]
    state_hash: str
    focus_apps: list[str] = field(default_factory=list)
    trace: list[ToolRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_outputs: list[dict[str, Any]] = field(default_factory=list)
    transcript: str = ""
    error: str | None = None
    finished: bool = False
    seconds: float = 0.0
    asr: dict[str, Any] | None = None

    def submission(self) -> dict[str, Any]:
        return {"id": self.sample_id, "tool_calls": self.calls, "state_hash": self.state_hash}

    def to_dict(self, include_trace: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.sample_id,
            "calls": self.calls,
            "state_hash": self.state_hash,
            "focus_apps": self.focus_apps,
            "error": self.error,
            "finished": self.finished,
            "seconds": round(self.seconds, 3),
            "notes": self.notes,
            "transcript": self.transcript,
            "raw_outputs": self.raw_outputs,
        }
        if self.asr is not None:
            out["asr"] = self.asr
        if include_trace:
            out["trace"] = [
                {"step": r.step, "tool": r.tool, "arguments": r.arguments, "result": r.result}
                for r in self.trace
            ]
        return out


def load_tool_env_class(root: str | Path, split: str):
    """Import the split's own ``zaic_tools`` package (train and public have separate copies)."""
    split_path = (Path(root) / split).resolve()
    package_dir = split_path / "zaic_tools"
    if not (package_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"no zaic_tools package at {package_dir}")
    split_str = str(split_path)
    # Always move the requested split to the front. In a train -> public ->
    # train sequence the public path would otherwise stay ahead of train, and
    # the cached-module flush below would import the public package again.
    while split_str in sys.path:
        sys.path.remove(split_str)
    sys.path.insert(0, split_str)
    existing = sys.modules.get("zaic_tools")
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "")).resolve()
        if package_dir.resolve() not in existing_file.parents:
            for name in list(sys.modules):
                if name == "zaic_tools" or name.startswith("zaic_tools."):
                    del sys.modules[name]
            importlib.invalidate_caches()
    module = importlib.import_module("zaic_tools")
    return module.ToolEnv


def make_tool_env(root: str | Path, split: str, sample_id: str):
    tool_env_cls = load_tool_env_class(root, split)
    return tool_env_cls(split, sample_id, root=str(root))


def _clean_entity(value: Any) -> str:
    text = str(value).strip()
    m = re.match(r"^@[a-zA-Z_]+\((.*)\)$", text, re.S)
    if m:
        text = m.group(1).strip()
    return text.strip(" \"'`")


class LifeMateAgent:
    def __init__(
        self,
        catalog: Catalog,
        caller: Caller,
        *,
        initial_apps: Iterable[str] = (),
        transcriber: Transcriber | None = None,
        config: AgentConfig | None = None,
    ):
        self.catalog = catalog
        self.initial_apps = tuple(initial_apps)
        self.caller = caller
        self.transcriber = transcriber
        self.config = config or AgentConfig()

    # -- public -----------------------------------------------------------------------
    def run(self, sample: Sample, env: Any) -> RunResult:
        started = time.perf_counter()
        calls: list[dict[str, Any]] = []
        state_hash = ""
        try:
            prepared = prepare_sample(sample, self.transcriber, require_text=True)
        except Exception as exc:
            try:
                state_hash = env.get_state_hash()
                calls = env.get_tool_calls()
            except Exception:
                pass
            return RunResult(
                sample_id=sample.id,
                calls=calls,
                state_hash=state_hash,
                error=f"transcript: {type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started,
            )

        ctx = ResolutionContext(
            current_time=sample.current_time,
            timezone_name=sample.timezone,
            user_profile=sample.user_profile,
            location=sample.location,
            catalog=self.catalog,
        )
        self._seed_history(ctx, prepared)

        installed_apps = set(self.catalog.installed_app_ids(sample.installed_apps))
        focus = [app for app in self.initial_apps if app in installed_apps]
        focus = list(dict.fromkeys(focus))[: self.config.top_k]
        trace: list[ToolRecord] = []
        notes: list[str] = []
        raw_outputs: list[dict[str, Any]] = []
        seen_auto_reads: set[tuple[str, str]] = set()
        finished = False
        fatal: str | None = None
        reprompts = 0
        discovery = self.config.tool_discovery
        opened = ({
            schema["name"] for schema in self.catalog.full_schemas(focus, sample.installed_apps)
            if not schema["name"].startswith("sys.")
        } if discovery else None)
        discovered: list[str] = []
        if opened is not None and self.config.preopen_tools:
            discovered = [
                tool.name for tool in search_tools(
                    self.catalog, sample.installed_apps, prepared.request_text,
                    opened=opened, limit=self.config.preopen_tools,
                )
            ]
            opened.update(discovered)
        searches = 0
        finish_checked = False

        for step in range(self.config.max_steps):
            visible_notes = notes[-self.config.max_notes :]
            action = self.caller.next_action(
                prepared,
                focus,
                trace,
                notes=visible_notes,
                step=step,
                **({"discovery": True, "discovered_tools": tuple(discovered)} if discovery else {}),
            )
            raw_outputs.append({"step": step + 1, "action": action.as_dict(), "raw": action.raw_text})
            if step == 0 and self.config.preopen_tools:
                raw_outputs[-1]["preopened_tools"] = list(discovered)
            if action.metrics:
                raw_outputs[-1]["metrics"] = action.metrics
            if action.messages is not None:
                raw_outputs[-1].update(
                    messages=action.messages, focus_apps=list(focus),
                    notes=visible_notes, trace_count=len(trace),
                )
            if action.action == "finish":
                if self.config.verify_finish and not finish_checked and step + 1 < self.config.max_steps:
                    finish_checked = True
                    started_check = time.perf_counter()
                    try:
                        check = self.caller.check_finish(prepared, trace)
                    except Exception as exc:
                        check = {"missing": "", "error": f"{type(exc).__name__}: {exc}"}
                    raw_outputs[-1]["finish_check"] = {**check, "seconds": time.perf_counter() - started_check}
                    if check["missing"]:
                        notes.append("Before finishing, recheck this possible unfinished request: " + check["missing"])
                        continue
                finished = True
                break
            if action.action == "invalid":
                reprompts += 1
                notes.append("invalid model action: " + str(action.message or "unparseable"))
                if reprompts > self.config.max_reprompts:
                    break
                continue

            if action.action == "need_app":
                if discovery:
                    notes.append("App loading is disabled. Use sys.search_tools to find a tool by capability.")
                    reprompts += 1
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                app = (action.app or "").strip()
                if not app and action.tool and "." in action.tool:
                    app = action.tool.split(".", 1)[0]
                if app not in self.catalog.apps:
                    reprompts += 1
                    notes.append(f"requested unknown app {app!r}")
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                if app not in installed_apps:
                    reprompts += 1
                    notes.append(f"app {app!r} is not installed in this sample")
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                # Loading a schema is progress, not a failed attempt. Counting
                # it would let three legitimate need_app calls exhaust the
                # budget before the caller gets to act. Re-loading an app that
                # is already in focus is a loop and does consume the budget.
                if app in focus:
                    reprompts += 1
                    notes.append(f"schemas for app {app!r} are already loaded")
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                focus.append(app)
                notes.append(f"loaded full schemas for app {app!r}")
                continue

            if action.action == "ask":
                target_spec = self._resolve_spec(action.tool, focus)
                if target_spec is None:
                    reprompts += 1
                    notes.append(f"ask target is not a known tool: {action.tool!r}")
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                if not self.catalog.is_installed(target_spec.name, sample.installed_apps):
                    reprompts += 1
                    notes.append(f"ask target {target_spec.name!r} is not installed")
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                if opened is not None and target_spec.name not in opened:
                    notes.append("Ask target schema is not open. Use sys.search_tools before asking about that tool.")
                    reprompts += 1
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                # An explicit ask may target an optional-but-needed argument:
                # gold asks for tv.control.volume even though only `action`
                # is required. Trust the caller's explicit params when they
                # are real properties and are not already provided.
                provided = action.arguments or {}
                raw_requested = [str(p) for p in action.params or []]
                requested = [
                    p for p in raw_requested if p in target_spec.properties and p not in provided
                ]
                if raw_requested:
                    params = self._ordered_params(target_spec, requested)
                else:
                    missing = missing_requirement_params(target_spec, provided)
                    params = missing
                if not params:
                    reprompts += 1
                    notes.append(
                        f"ask action for {target_spec.name} names no missing required param; "
                        "call the tool or finish instead"
                    )
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                ok = self._call_tool(
                    env,
                    self.catalog.tools["sys.ask_missing_info"],
                    {"tool_name": target_spec.name, "params": params},
                    ctx,
                    trace,
                    notes,
                    step=len(trace) + 1,
                )
                if ok:
                    if self.config.stop_after_ask:
                        finished = True
                        break
                    reprompts = 0
                else:
                    reprompts += 1
                    if reprompts > self.config.max_reprompts:
                        break
                continue

            if action.action != "call":
                reprompts += 1
                notes.append(f"unsupported action {action.action!r}")
                if reprompts > self.config.max_reprompts:
                    break
                continue

            if opened is not None and action.tool == SEARCH_TOOL:
                query = (action.arguments or {}).get("query")
                if searches >= self.config.max_searches or step + 1 >= self.config.max_steps or not (
                    isinstance(query, str) and 0 < len(query.strip()) <= 200
                ):
                    notes.append(f"Search requires a nonempty query up to 200 characters, remaining search quota (max {self.config.max_searches}), and a remaining action turn.")
                    reprompts += 1
                    if reprompts > self.config.max_reprompts:
                        break
                    continue
                searches += 1
                started_search = time.perf_counter()
                matches = search_tools(self.catalog, sample.installed_apps, query, opened=opened)
                newly_found = [tool.name for tool in matches]
                discovered.extend(newly_found)
                opened.update(newly_found)
                trace.append(ToolRecord(tool=SEARCH_TOOL, arguments={"query": query.strip()},
                                        result={"ok": True, "tools": newly_found,
                                                "source": "catalog metadata"}, step=len(trace) + 1))
                raw_outputs[-1]["discovery_seconds"] = time.perf_counter() - started_search
                if not matches:
                    notes.append("No matching tool. Use an open tool if appropriate, or finish.")
                reprompts = 0
                continue
            spec = self._resolve_spec(action.tool, focus)
            if spec is None:
                reprompts += 1
                notes.append(f"unknown or uninstalled tool {action.tool!r}")
                if reprompts > self.config.max_reprompts:
                    break
                continue
            if not self.catalog.is_installed(spec.name, sample.installed_apps):
                reprompts += 1
                notes.append(f"tool {spec.name!r} is not installed")
                if reprompts > self.config.max_reprompts:
                    break
                continue
            if opened is not None and spec.name not in opened:
                notes.append(f"Tool {spec.name!r} is not open. Use sys.search_tools to find the required capability.")
                reprompts += 1
                if reprompts > self.config.max_reprompts:
                    break
                continue
            if opened is None and spec.app not in focus and spec.app != "sys":
                # Lazy schema loading: the caller used the compact fallback directly.
                focus.append(spec.app)

            resolved: dict[str, Any] | None = None
            ambiguous_param: str | None = None
            raw_arguments = dict(action.arguments or {})
            for attempt in range(self.config.max_auto_reads_per_step + 1):
                try:
                    resolved = resolve_arguments(spec, raw_arguments, ctx)
                    break
                except UnresolvedReference as exc:
                    if attempt < self.config.max_auto_reads_per_step and self.config.auto_read and exc.xref:
                        if self._auto_read(exc, sample, env, ctx, trace, notes, focus, seen_auto_reads, opened):
                            continue
                    notes.append(str(exc))
                    resolved = None
                    break
                except ResolutionError as exc:
                    notes.append(str(exc))
                    resolved = None
                    # Two records match the spoken name.  The contract is to ask
                    # which one, not to guess; gold does this for contacts.
                    if exc.kind == "ambiguous_ref" and exc.field_name:
                        if self._is_required_field(spec, exc.field_name):
                            ambiguous_param = str(exc.field_name)
                        else:
                            raw_arguments.pop(exc.field_name, None)
                            if attempt < self.config.max_auto_reads_per_step:
                                continue
                    break
            if ambiguous_param and self.config.auto_ask:
                ok = self._call_tool(
                    env,
                    self.catalog.tools["sys.ask_missing_info"],
                    {"tool_name": spec.name, "params": [ambiguous_param]},
                    ctx,
                    trace,
                    notes,
                    step=len(trace) + 1,
                )
                if ok:
                    if self.config.stop_after_ask:
                        finished = True
                        break
                    reprompts = 0
                else:
                    reprompts += 1
                    if reprompts > self.config.max_reprompts:
                        break
                continue
            if resolved is None:
                reprompts += 1
                if reprompts > self.config.max_reprompts:
                    break
                continue

            missing = missing_requirement_params(spec, resolved)
            if missing:
                if self.config.auto_ask:
                    ok = self._call_tool(
                        env,
                        self.catalog.tools["sys.ask_missing_info"],
                        {"tool_name": spec.name, "params": missing},
                        ctx,
                        trace,
                        notes,
                        step=len(trace) + 1,
                    )
                    if ok:
                        if self.config.stop_after_ask:
                            finished = True
                            break
                        reprompts = 0
                    else:
                        reprompts += 1
                        if reprompts > self.config.max_reprompts:
                            break
                    continue
                notes.append(f"{spec.name} is missing required params: {missing}")
                reprompts += 1
                if reprompts > self.config.max_reprompts:
                    break
                continue

            call_key = (spec.name, json.dumps(resolved, ensure_ascii=False, sort_keys=True))
            same_calls = [
                record
                for record in trace
                if record.tool == spec.name
                and json.dumps(record.arguments, ensure_ascii=False, sort_keys=True) == call_key[1]
            ]
            if same_calls:
                last = same_calls[-1]
                # A rejected call changed nothing, so one identical retry is
                # legitimate (transient failure, stale read). Successful calls
                # stay guarded: the duplicate check is the loop breaker that
                # stops the caller burning the step and latency budget. The
                # allowlist is the narrow exception for writes whose identical
                # repetition has a distinct effect (buying another voucher).
                repeatable_success = (
                    spec.name in REPEATABLE_WRITE_TOOLS
                    and isinstance(last.result, dict)
                    and last.result.get("ok") is not False
                )
                retry_after_failure = (
                    len(same_calls) == 1
                    and isinstance(last.result, dict)
                    and last.result.get("ok") is False
                )
                # A successful read becomes stale if a later successful write
                # changed the state; refreshing it is legitimate. Consecutive
                # identical reads with no intervening write stay blocked.
                read_refresh = False
                if (
                    spec.kind == "read"
                    and isinstance(last.result, dict)
                    and last.result.get("ok") is not False
                ):
                    last_index = next(
                        index
                        for index in range(len(trace) - 1, -1, -1)
                        if trace[index] is last
                    )
                    read_refresh = any(
                        self._is_successful_write(record)
                        for record in trace[last_index + 1 :]
                    )
                if not repeatable_success and not retry_after_failure and not read_refresh:
                    reprompts += 1
                    notes.append(
                        f"duplicate call skipped: {spec.name} already ran with these exact arguments; "
                        "read its result in the trace and choose the next action or finish"
                    )
                    if reprompts > self.config.max_reprompts:
                        break
                    continue

            ok = self._call_tool(env, spec, resolved, ctx, trace, notes, step=len(trace) + 1)
            if ok:
                reprompts = 0
            else:
                reprompts += 1
                if reprompts > self.config.max_reprompts:
                    break

        try:
            calls = env.get_tool_calls()
            state_hash = env.get_state_hash()
        except Exception as exc:
            fatal = f"environment: {type(exc).__name__}: {exc}"
        return RunResult(
            sample_id=sample.id,
            calls=calls,
            state_hash=state_hash,
            focus_apps=focus,
            trace=trace,
            notes=notes,
            raw_outputs=raw_outputs,
            transcript=prepared.request_text,
            error=fatal,
            finished=finished,
            seconds=time.perf_counter() - started,
        )

    # -- helpers ----------------------------------------------------------------------
    def _seed_history(self, ctx: ResolutionContext, prepared: PreparedSample) -> None:
        for msg in prepared.history:
            if msg.get("role") != "assistant":
                continue
            calls = msg.get("tool_calls") or []
            results = msg.get("tool_results") or []
            for i, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "")
                args = call.get("arguments") or {}
                result = results[i] if i < len(results) else None
                if name:
                    ctx.add_record(name, args if isinstance(args, dict) else {}, result, step=-1, historical=True)

    def _resolve_spec(self, tool_name: str | None, focus: Iterable[str]) -> ToolSpec | None:
        if not tool_name:
            return None
        return self.catalog.resolve_tool(str(tool_name), list(focus))

    @staticmethod
    def _is_required_field(spec: ToolSpec, field_name: str) -> bool:
        if field_name in spec.required:
            return True
        return any(field_name in group for group in spec.one_of)

    @staticmethod
    def _ordered_params(spec: ToolSpec, params: Iterable[str]) -> list[str]:
        requested = set(str(p) for p in params or [])
        ordered = [p for p in spec.required if p in requested]
        ordered += [p for p in spec.properties if p in requested and p not in ordered]
        return ordered

    def _is_successful_write(self, record: ToolRecord) -> bool:
        spec = self.catalog.tools.get(record.tool)
        if spec is None or spec.kind != "write":
            return False
        return not (isinstance(record.result, dict) and record.result.get("ok") is False)

    def _call_tool(
        self,
        env: Any,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ResolutionContext,
        trace: list[ToolRecord],
        notes: list[str],
        step: int,
    ) -> bool:
        try:
            check = env.validate(spec.name, arguments)
        except Exception as exc:
            notes.append(f"validate {spec.name} failed: {type(exc).__name__}: {exc}")
            return False
        if not check.get("ok"):
            notes.append(f"schema rejected {spec.name}: {check.get('error')}")
            return False
        try:
            result = env.call(spec.name, arguments)
        except Exception as exc:  # the runtime normally never raises
            notes.append(f"call {spec.name} failed: {type(exc).__name__}: {exc}")
            return False
        record = ctx.add_record(spec.name, arguments, result, step=step)
        trace.append(record)
        if isinstance(result, dict) and result.get("ok") is False:
            notes.append(f"{spec.name} was rejected by the tool environment: {result.get('error')}")
            return False
        return True

    def _auto_read(
        self,
        exc: UnresolvedReference,
        sample: Sample,
        env: Any,
        ctx: ResolutionContext,
        trace: list[ToolRecord],
        notes: list[str],
        focus: list[str],
        seen: set[tuple[str, str]],
        opened: set[str] | None = None,
    ) -> bool:
        xref = exc.xref or ""
        hint = read_hint_for_xref(self.catalog, xref, sample.installed_apps)
        if not hint:
            notes.append(f"no read tool known for {xref}; cannot resolve {exc.value!r}")
            return False
        tool_name, query_arg = hint
        if opened is not None and tool_name not in opened:
            notes.append(f"The lookup needed for {exc.field_name or 'this action'} is not open. Use sys.search_tools.")
            return False
        args: dict[str, Any] = {}
        if query_arg:
            args[query_arg] = _clean_entity(exc.value)
        key = (tool_name, json.dumps(args, ensure_ascii=False, sort_keys=True))
        if key in seen:
            notes.append(f"auto-read already attempted: {tool_name} {args}")
            return False
        seen.add(key)
        spec = self.catalog.tools[tool_name]
        if opened is None and spec.app not in focus and spec.app != "sys":
            focus.append(spec.app)
        notes.append(f"auto-read {tool_name} {args} to resolve {exc.field_name or 'entity'} {exc.value!r}")
        return self._call_tool(env, spec, args, ctx, trace, notes, step=len(trace) + 1)
