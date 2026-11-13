"""Typed, read-only view of a split's tool catalog plus compact prompt renderers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Prompt-only details checked against the released simulator.
RESULT_HINTS = {
    "email.list": "emails[]: id, from_name, subject. Dùng id để đọc, trả lời hoặc sửa email.",
    "email.read": "email: thông tin email và body.",
    "email.archive": "archive: {ok, item}; delete: {ok} khi thành công.",
    "tasks.list": "tasks[]: id, title, due, done. Dùng due làm mốc hạn nếu được nhắc tới.",
    "tasks.add": "{ok, id, item}; item.due là hạn của công việc vừa tạo.",
    "reminders.list": "reminders[]: id, content, time, repeat, done.",
    "reminders.create": "{ok, id, item}; item.time là giờ nhắc đã lưu.",
    "fuel_price.get_price": "prices[]: type và price.",
}
PARAMETER_HINTS = {
    "email.archive": {
        "email_id": "Lấy từ emails[].id của email.list.",
        "action": "archive: lưu trữ; delete: xóa.",
    },
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str
    description: str
    parameters: dict[str, Any]
    returns: str
    app: str

    @property
    def short_name(self) -> str:
        return self.name.split(".", 1)[1] if "." in self.name else self.name

    @property
    def properties(self) -> dict[str, dict[str, Any]]:
        return self.parameters.get("properties") or {}

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required") or [])

    @property
    def one_of(self) -> list[list[str]]:
        return [list(group) for group in (self.parameters.get("x-one-of") or [])]

    def prop(self, name: str) -> dict[str, Any]:
        return self.properties.get(name) or {}

    def prompt_schema(self) -> dict[str, Any]:
        """The full executable schema shown to the caller (not every catalog tool)."""
        parameters = self.parameters
        hints = PARAMETER_HINTS.get(self.name)
        if hints:
            properties = {
                name: {**prop, "description": hints[name]} if name in hints else prop
                for name, prop in self.properties.items()
            }
            parameters = {**parameters, "properties": properties}
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "parameters": parameters,
            "returns": RESULT_HINTS.get(self.name, self.returns),
        }


@dataclass(frozen=True)
class AppSpec:
    id: str
    name_vi: str
    group: str
    description: str
    tools: tuple[ToolSpec, ...]

    def app_line(self, max_description: int = 260) -> str:
        desc = " ".join((self.description or "").split())
        if len(desc) > max_description:
            desc = desc[: max_description - 1] + "…"
        group = f", {self.group}" if self.group else ""
        return f"- {self.id} ({self.name_vi}{group}): {desc}"


def _sig_type(prop: dict[str, Any]) -> str:
    enum = prop.get("enum")
    if enum:
        return "|".join(str(v) for v in enum)
    typ = prop.get("type") or "any"
    if typ == "array":
        item = prop.get("items") or {}
        item_type = "|".join(str(v) for v in item.get("enum", [])) if item.get("enum") else (item.get("type") or "any")
        return f"array<{item_type}>"
    return str(typ)


def _arg_mark(tool: ToolSpec, key: str) -> str:
    """Schema mark: ``*`` required, ``~`` one-of alternative, ``?`` optional.

    A one-of member is not required on its own: providing any one complete
    group satisfies the validator. An explicitly required member stays ``*``
    even when it also appears in a group.
    """
    if key in tool.required:
        return "*"
    if any(key in group for group in tool.one_of):
        return "~"
    return "?"


def compact_tool_line(tool: ToolSpec, include_description: bool = True, required_only: bool = False) -> str:
    """One line per tool: name, required/optional args, enums, x-ref, fuzzy/format."""
    parts: list[str] = []
    for key, prop in tool.properties.items():
        required = key in tool.required
        in_group = any(key in group for group in tool.one_of)
        if required_only and not (required or in_group):
            continue
        mark = _arg_mark(tool, key)
        bit = f"{key}{mark}:{_sig_type(prop)}"
        if prop.get("x-ref"):
            bit += "@" + str(prop["x-ref"])
        if prop.get("x-match") == "fuzzy":
            bit += "~fuzzy"
        if prop.get("format"):
            bit += "[" + str(prop["format"]) + "]"
        parts.append(bit)
    line = f"{tool.name}({', '.join(parts)})"
    if tool.one_of:
        line += " one_of=" + "|".join("+".join(g) for g in tool.one_of)
    desc = " ".join((tool.description or "").split())
    if include_description and desc:
        line += " — " + (desc[:180] + ("…" if len(desc) > 180 else ""))
    return line


def _short_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_tool_schema(tool: ToolSpec, description_chars: int = 130) -> str:
    """Compact, readable schema line block for the caller prompt.

    A DSL keeps the same information as the JSON schema (required flags, enums,
    x-ref ids, one-of groups, defaults) at roughly half the tokens.
    """
    lines = [f"{tool.name} [{tool.kind}] {_short_text(tool.description, description_chars)}"]
    schema = tool.prompt_schema()
    for key, prop in (schema["parameters"].get("properties") or {}).items():
        mark = _arg_mark(tool, key)
        type_text = _sig_type(prop)
        extras: list[str] = []
        if prop.get("x-ref"):
            extras.append("id:" + str(prop["x-ref"]).split(".")[-1])
        if prop.get("format"):
            extras.append(str(prop["format"]))
        if prop.get("x-match") == "fuzzy":
            extras.append("fuzzy")
        if prop.get("default") is not None:
            extras.append("default=" + json.dumps(prop["default"], ensure_ascii=False))
        line = f"  {key}{mark}:{type_text}"
        if extras:
            line += " (" + ", ".join(extras) + ")"
        desc = _short_text(prop.get("description"), 110)
        if desc:
            line += " — " + desc
        lines.append(line)
    if tool.one_of:
        lines.append("  one_of: " + " | ".join("+".join(group) for group in tool.one_of))
    if schema["returns"]:
        lines.append("  -> " + _short_text(schema["returns"], 80))
    return "\n".join(lines)


@dataclass(frozen=True)
class Catalog:
    schema_version: str
    apps: dict[str, AppSpec]
    tools: dict[str, ToolSpec]
    source: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "") -> "Catalog":
        apps: dict[str, AppSpec] = {}
        tools: dict[str, ToolSpec] = {}
        for app_raw in data.get("apps", []):
            app_id = str(app_raw.get("id") or "")
            if not app_id:
                continue
            app_tools: list[ToolSpec] = []
            for tool_raw in app_raw.get("tools", []):
                name = str(tool_raw.get("name") or "")
                if not name:
                    continue
                tool = ToolSpec(
                    name=name,
                    kind=str(tool_raw.get("kind") or "write"),
                    description=str(tool_raw.get("description") or ""),
                    parameters=dict(tool_raw.get("parameters") or {}),
                    returns=str(tool_raw.get("returns") or ""),
                    app=str(tool_raw.get("app") or (name.split(".", 1)[0] if "." in name else app_id)),
                )
                app_tools.append(tool)
                tools[name] = tool
            apps[app_id] = AppSpec(
                id=app_id,
                name_vi=str(app_raw.get("name_vi") or app_id),
                group=str(app_raw.get("group") or ""),
                description=str(app_raw.get("description") or ""),
                tools=tuple(app_tools),
            )
        return cls(
            schema_version=str(data.get("schema_version") or ""),
            apps=apps,
            tools=tools,
            source=source,
        )

    # -- installed views ---------------------------------------------------------------
    @staticmethod
    def _installed_mapping(installed_apps: dict[str, Any] | Iterable[str]) -> dict[str, list[str]]:
        if isinstance(installed_apps, dict):
            out: dict[str, list[str]] = {}
            for app_id, value in installed_apps.items():
                if isinstance(value, dict):
                    names = list(value.get("tools") or [])
                else:
                    names = list(value or [])
                out[str(app_id)] = [str(x) for x in names]
            return out
        return {str(x): [] for x in installed_apps}

    def installed_app_ids(self, installed_apps: dict[str, Any] | Iterable[str]) -> list[str]:
        mapping = self._installed_mapping(installed_apps)
        return [app_id for app_id in mapping if app_id in self.apps]

    def installed_tools(self, installed_apps: dict[str, Any] | Iterable[str]) -> list[ToolSpec]:
        mapping = self._installed_mapping(installed_apps)
        out: list[ToolSpec] = []
        seen: set[str] = set()
        for app_id, names in mapping.items():
            app = self.apps.get(app_id)
            if app is None:
                continue
            allowed = set(names) if names else {t.name for t in app.tools}
            for tool in app.tools:
                if tool.name in allowed and tool.name not in seen:
                    seen.add(tool.name)
                    out.append(tool)
        # sys is always available even though it is not installed as an ordinary app
        if "sys.ask_missing_info" in self.tools:
            out.append(self.tools["sys.ask_missing_info"])
        return out

    def is_installed(self, tool_name: str, installed_apps: dict[str, Any] | Iterable[str]) -> bool:
        if tool_name == "sys.ask_missing_info":
            return True
        mapping = self._installed_mapping(installed_apps)
        app_id, _, short = tool_name.partition(".")
        if app_id not in mapping or not short:
            return False
        names = mapping[app_id]
        return (short in names or tool_name in names) if names else tool_name in self.tools

    def installed_tool_names(self, installed_apps: dict[str, Any] | Iterable[str]) -> set[str]:
        return {t.name for t in self.installed_tools(installed_apps)}

    # -- prompt views ------------------------------------------------------------------
    def app_directory(self, installed_apps: dict[str, Any] | Iterable[str]) -> str:
        lines = [self.apps[a].app_line() for a in self.installed_app_ids(installed_apps)]
        return "\n".join(lines) if lines else "(no installed apps)"

    def compact_directory(self, installed_apps: dict[str, Any] | Iterable[str], exclude: set[str] | None = None) -> str:
        exclude = exclude or set()
        lines = [compact_tool_line(t) for t in self.installed_tools(installed_apps) if t.name not in exclude]
        return "\n".join(lines) if lines else "(no other installed tools)"

    def full_schemas(self, app_ids: Iterable[str], installed_apps: dict[str, Any] | Iterable[str]) -> list[dict[str, Any]]:
        installed = self.installed_tool_names(installed_apps)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for app_id in app_ids:
            app = self.apps.get(app_id)
            if app is None:
                continue
            for tool in app.tools:
                if tool.name in installed and tool.name not in seen:
                    seen.add(tool.name)
                    out.append(tool.prompt_schema())
        sys_tool = self.tools.get("sys.ask_missing_info")
        if sys_tool and sys_tool.name not in seen:
            out.append(sys_tool.prompt_schema())
        return out

    def read_tools(self, app_id: str) -> list[ToolSpec]:
        app = self.apps.get(app_id)
        return [t for t in app.tools if t.kind == "read"] if app else []

    def resolve_tool(self, name: str, focus_apps: Iterable[str] | None = None) -> ToolSpec | None:
        name = (name or "").strip()
        if name in self.tools:
            return self.tools[name]
        if "." not in name and focus_apps:
            candidates = [self.tools[f"{a}.{name}"] for a in focus_apps if f"{a}.{name}" in self.tools]
            if len(candidates) == 1:
                return candidates[0]
        return None

    def app_name_map(self) -> dict[str, str]:
        """Folded aliases -> app id (id, Vietnamese name, and simple name variants)."""
        out: dict[str, str] = {}
        from .text import fold

        for app_id, app in self.apps.items():
            out[fold(app_id)] = app_id
            out[fold(app.name_vi)] = app_id
            out[fold(app.name_vi.replace(" ", "_"))] = app_id
        return out
