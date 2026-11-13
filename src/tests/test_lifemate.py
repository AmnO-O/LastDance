"""Unit and end-to-end tests for the LifeMate agent.

Run from the split directory (the one containing train.jsonl):

    $env:PYTHONPATH='src'; python -m unittest discover -s src/tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lifemate.caller import Caller, CallerAction, parse_caller_action  # noqa: E402
from lifemate.catalog import Catalog, compact_tool_line, render_tool_schema  # noqa: E402
from lifemate.cli import _build_backend  # noqa: E402
from lifemate.data import (  # noqa: E402
    PreparedSample,
    Sample,
    catalog_path,
    find_data_root,
    find_validation_ids,
    load_samples,
    read_ids,
)
from lifemate.llm import ScriptedBackend  # noqa: E402
from lifemate.planner import parse_plan, parse_roadmap  # noqa: E402
from lifemate.prompts import CALLER_SYSTEM, caller_messages, planner_messages  # noqa: E402
from lifemate.resolvers import (  # noqa: E402
    ResolutionContext,
    ResolutionError,
    UnresolvedReference,
    coerce_bool,
    coerce_enum,
    coerce_value,
    parse_money,
    resolve_arguments,
    resolve_entity,
    resolve_references,
)
from lifemate.runtime import AgentConfig, LifeMateAgent, make_tool_env  # noqa: E402
from lifemate.scoring import calls_match, score_prediction  # noqa: E402


def _catalog(data: dict[str, Any] | None = None) -> Catalog:
    if data is not None:
        return Catalog.from_dict(data)
    return Catalog.from_dict(
        {
            "schema_version": "test",
            "apps": [
                {
                    "id": "demo",
                    "name_vi": "Demo",
                    "group": "test",
                    "description": "demo app",
                    "tools": [
                        {
                            "name": "demo.do",
                            "kind": "write",
                            "description": "do the thing",
                            "parameters": {
                                "type": "object",
                                "properties": {"x": {"type": "string"}, "y": {"type": "string"}},
                                "required": ["x"],
                            },
                            "returns": "",
                            "app": "demo",
                        }
                    ],
                },
                {
                    "id": "sys",
                    "name_vi": "Hệ thống",
                    "group": "control",
                    "description": "system",
                    "tools": [
                        {
                            "name": "sys.ask_missing_info",
                            "kind": "meta",
                            "description": "report missing params",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "tool_name": {"type": "string"},
                                    "params": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["tool_name", "params"],
                            },
                            "returns": "",
                            "app": "sys",
                        }
                    ],
                },
            ],
        }
    )


def _sample(sample_id: str = "synthetic_0001", label: dict[str, Any] | None = None) -> Sample:
    return Sample(
        raw={
            "id": sample_id,
            "audio_text": "làm việc đó giúp tôi",
            "context": {
                "current_time": "2026-11-28T07:10:00+07:00",
                "timezone": "Asia/Ho_Chi_Minh",
                "location": {"lat": 16.0544, "lng": 108.2022},
                "user_profile": {"name": "Em Linh"},
                "installed_apps": {"demo": {"tools": ["demo.do"]}},
            },
            "history": [],
            "label": label or {},
        },
        split="train",
        root=None,
    )


class FakeEnv:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "demo.do" and not arguments.get("x"):
            return {"ok": False, "error": "missing x"}
        return {"ok": True}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"name": name, "arguments": dict(arguments)})
        if name == "demo.do":
            self.state["demo"] = {"x": arguments.get("x")}
        return {"ok": True, "done": True}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return [dict(call) for call in self.calls]

    def get_state_hash(self) -> str:
        return f"hash:{len(self.calls)}"


class ReplayCaller:
    """Caller oracle: replays the labeled calls one action at a time."""

    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = list(calls)
        self.index = 0

    def next_action(self, prepared: Any, focus: Any, trace: Any, notes: Any = (), step: int = 0) -> CallerAction:
        if self.index >= len(self.calls):
            return CallerAction(action="finish")
        call = self.calls[self.index]
        self.index += 1
        name = str(call.get("name") or "")
        arguments = dict(call.get("arguments") or {})
        if name == "sys.ask_missing_info":
            return CallerAction(
                action="ask",
                tool=str(arguments.get("tool_name") or ""),
                params=list(arguments.get("params") or []),
            )
        return CallerAction(action="call", tool=name, arguments=arguments)


class ResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = ResolutionContext(
            "2026-11-28T07:10:00+07:00",
            "Asia/Ho_Chi_Minh",
            catalog=_catalog(),
        )

    def test_money(self) -> None:
        cases = {
            "50 nghìn": 50000,
            "1,5 triệu": 1500000,
            "ba trăm rưỡi": 350000,
            "một trăm tám mươi nghìn": 180000,
            "500k": 500000,
            "3.4 triệu": 3400000,
            "2tr": 2000000,
            "một trăm linh năm nghìn": 105000,
            "ba triệu rưỡi": 3500000,
        }
        for text, expected in cases.items():
            self.assertEqual(parse_money(text, "amount"), expected, text)

    def test_weekday_qualifiers_and_relative_offsets(self) -> None:
        date_prop = {"type": "string", "format": "date"}
        datetime_prop = {"type": "string", "format": "date-time"}
        cases = [
            ("2026-11-11", "thứ sáu tuần sau", "2026-11-20"),
            ("2026-11-13", "thứ sáu", "2026-11-13"),
            ("2026-10-05", "thứ sáu tuần sau", "2026-10-16"),  # train_0843
            ("2026-12-04", "thứ sáu tuần sau", "2026-12-11"),  # train_1435
            ("2026-11-13", "tuần sau thứ ba", "2026-11-17"),
            ("2026-11-15", "thứ sáu tuần này", "2026-11-13"),
        ]
        for today, phrase, expected in cases:
            with self.subTest(today=today, phrase=phrase):
                ctx = ResolutionContext(today + "T09:00:00+07:00", "Asia/Ho_Chi_Minh")
                self.assertEqual(coerce_value(phrase, date_prop, "date", ctx), expected)
        for phrase, expected in (("30 phút nữa", "07:40"), ("2 giờ nữa", "09:10")):
            with self.subTest(phrase=phrase):
                self.assertEqual(coerce_value(phrase, datetime_prop, "time", self.ctx),
                                 f"2026-11-28T{expected}:00+07:00")

    def test_vietnamese_datetime(self) -> None:
        prop = {
            "type": "string",
            "description": "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00",
        }
        cases = {
            "9 giờ sáng mai": "2026-11-29T09:00:00+07:00",
            "6 rưỡi sáng mai": "2026-11-29T06:30:00+07:00",
            "3 giờ chiều mai": "2026-11-29T15:00:00+07:00",
            "15:30": "2026-11-28T15:30:00+07:00",
            "sau 2 tiếng": "2026-11-28T09:10:00+07:00",
        }
        for text, expected in cases.items():
            self.assertEqual(coerce_value(text, prop, "time", self.ctx), expected, text)
        for incomplete in ("mai", "chiều nay", "thứ hai tới"):
            with self.assertRaises(Exception):
                coerce_value(incomplete, prop, "time", self.ctx)

    def test_enum_and_bool(self) -> None:
        self.assertEqual(coerce_enum("bật", ["on", "off"]), "on")
        self.assertEqual(coerce_enum("Tắt", ["on", "off"]), "off")
        self.assertTrue(coerce_bool("bật"))
        self.assertFalse(coerce_bool("không"))
        with self.assertRaises(Exception):
            coerce_enum("không rõ", ["on", "off"])

    def test_integer_array_rejects_fractional_and_unparseable_items(self) -> None:
        prop = {"type": "array", "items": {"type": "integer"}}
        self.assertEqual(
            coerce_value(["1", 2, "3.0", " 4 "], prop, "ids", self.ctx),
            [1, 2, 3, 4],
        )
        for bad in (["1.5"], ["abc"], [None], [True]):
            with self.assertRaises(ResolutionError) as caught:
                coerce_value(bad, prop, "ids", self.ctx)
            self.assertEqual(caught.exception.kind, "type", bad)
            self.assertEqual(caught.exception.field_name, "ids", bad)

    def test_references_and_entities(self) -> None:
        self.ctx.add_record(
            "contacts.search",
            {"query": "Sếp Dũng"},
            {"ok": True, "contacts": [{"id": "c_08", "name": "Sếp Dũng"}]},
            step=1,
        )
        self.assertEqual(resolve_references("$1.contacts[0].id", self.ctx), "c_08")
        self.assertEqual(resolve_entity("Sếp Dũng", "contacts.contacts", self.ctx, "contact_id"), "c_08")

    def test_two_contacts_with_same_name_are_ambiguous(self) -> None:
        self.ctx.add_record(
            "contacts.search",
            {"query": "Lan"},
            {
                "ok": True,
                "contacts": [
                    {"id": "c_05", "name": "Lan", "relation": "vợ", "phone": "09876195982"},
                    {"id": "c_10", "name": "Lan", "relation": "đồng nghiệp", "phone": "09350661998"},
                ],
            },
            step=1,
        )
        with self.assertRaises(ResolutionError) as caught:
            resolve_entity("Lan", "contacts.contacts", self.ctx, "contact_id")
        self.assertEqual(caught.exception.kind, "ambiguous_ref")
        self.assertEqual(caught.exception.field_name, "contact_id")

    def test_unique_name_still_resolves(self) -> None:
        self.ctx.add_record(
            "contacts.search",
            {"query": "Bố"},
            {"ok": True, "contacts": [{"id": "c_02", "name": "Bố", "relation": "bố"}]},
            step=1,
        )
        self.assertEqual(resolve_entity("Bố", "contacts.contacts", self.ctx, "contact_id"), "c_02")

    def test_id_seen_in_another_app_is_accepted(self) -> None:
        # Gold sometimes references a contact_id first seen in an SMS result.
        self.ctx.add_record(
            "sms.read_recent",
            {},
            {"ok": True, "messages": [{"id": "msg_417", "contact_id": "c_778", "content": "..."}]},
            step=1,
        )
        self.assertEqual(resolve_entity("c_778", "contacts.contacts", self.ctx, "contact_id"), "c_778")

    def test_unknown_opaque_id_is_rejected(self) -> None:
        self.ctx.add_record(
            "contacts.search",
            {"query": "Lan"},
            {"ok": True, "contacts": [{"id": "c_05", "name": "Lan"}]},
            step=1,
        )
        with self.assertRaises(UnresolvedReference):
            resolve_entity("c_99", "contacts.contacts", self.ctx, "contact_id")

    def test_phone_alternative(self) -> None:
        catalog = _catalog(
            {
                "schema_version": "test",
                "apps": [
                    {
                        "id": "phone",
                        "name_vi": "Điện thoại",
                        "group": "control",
                        "description": "phone",
                        "tools": [
                            {
                                "name": "phone.call",
                                "kind": "write",
                                "description": "call",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                                        "phone_number": {"type": "string"},
                                    },
                                    "required": [],
                                    "x-one-of": [["contact_id"], ["phone_number"]],
                                },
                                "returns": "",
                                "app": "phone",
                            }
                        ],
                    },
                    {
                        "id": "sys",
                        "name_vi": "Hệ thống",
                        "group": "control",
                        "description": "system",
                        "tools": [
                            {
                                "name": "sys.ask_missing_info",
                                "kind": "meta",
                                "description": "report missing params",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "tool_name": {"type": "string"},
                                        "params": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["tool_name", "params"],
                                },
                                "returns": "",
                                "app": "sys",
                            }
                        ],
                    },
                ],
            }
        )
        ctx = ResolutionContext("2026-11-28T07:10:00+07:00", catalog=catalog)
        args = resolve_arguments(catalog.tools["phone.call"], {"contact_id": "0905123456"}, ctx)
        self.assertEqual(args, {"phone_number": "0905123456"})


def _schema_catalog() -> Catalog:
    return _catalog(
        {
            "schema_version": "test",
            "apps": [
                {
                    "id": "phone",
                    "name_vi": "Điện thoại",
                    "group": "control",
                    "description": "phone",
                    "tools": [
                        {
                            "name": "phone.call",
                            "kind": "write",
                            "description": "call",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                                    "phone_number": {"type": "string"},
                                },
                                "required": [],
                                "x-one-of": [["contact_id"], ["phone_number"]],
                            },
                            "returns": "",
                            "app": "phone",
                        }
                    ],
                },
                {
                    "id": "notes",
                    "name_vi": "Ghi chú",
                    "group": "productivity",
                    "description": "notes",
                    "tools": [
                        {
                            "name": "notes.create",
                            "kind": "write",
                            "description": "create a note",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string", "x-match": "fuzzy"},
                                    "title": {"type": "string"},
                                },
                                "required": ["content"],
                            },
                            "returns": "",
                            "app": "notes",
                        }
                    ],
                },
                {
                    "id": "mixed",
                    "name_vi": "Hỗn hợp",
                    "group": "test",
                    "description": "mixed",
                    "tools": [
                        {
                            "name": "mixed.set",
                            "kind": "write",
                            "description": "set a value",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "string"},
                                    "b": {"type": "string"},
                                },
                                "required": ["a"],
                                "x-one-of": [["a"], ["b"]],
                            },
                            "returns": "",
                            "app": "mixed",
                        }
                    ],
                },
            ],
        }
    )


class SchemaRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = _schema_catalog()

    def test_one_of_members_are_alternatives_in_fallback(self) -> None:
        line = compact_tool_line(
            self.catalog.tools["phone.call"],
            include_description=False,
            required_only=True,
        )
        self.assertIn("contact_id~:string@contacts.contacts", line)
        self.assertIn("phone_number~:string", line)
        self.assertNotIn("contact_id*", line)
        self.assertNotIn("phone_number*", line)
        self.assertIn("one_of=contact_id|phone_number", line)

    def test_one_of_members_are_alternatives_in_full_schema(self) -> None:
        text = render_tool_schema(self.catalog.tools["phone.call"])
        self.assertIn("  contact_id~:string (id:contacts)", text)
        self.assertIn("  phone_number~:string", text)
        self.assertNotIn("contact_id*", text)
        self.assertNotIn("phone_number*", text)
        self.assertIn("  one_of: contact_id | phone_number", text)

    def test_fallback_keeps_fuzzy_without_description(self) -> None:
        line = compact_tool_line(
            self.catalog.tools["notes.create"],
            include_description=False,
            required_only=True,
        )
        self.assertIn("content*:string~fuzzy", line)
        self.assertNotIn("title", line)

    def test_full_schema_keeps_fuzzy_without_description(self) -> None:
        text = render_tool_schema(self.catalog.tools["notes.create"])
        self.assertIn("content*:string (fuzzy)", text)

    def test_explicit_required_wins_over_one_of_mark(self) -> None:
        for text in (
            compact_tool_line(
                self.catalog.tools["mixed.set"],
                include_description=False,
                required_only=True,
            ),
            render_tool_schema(self.catalog.tools["mixed.set"]),
        ):
            self.assertIn("a*:", text)
            self.assertIn("b~:", text)


class PlannerParsingTests(unittest.TestCase):
    def test_incomplete_roadmap_recovers_only_complete_data(self) -> None:
        cases = (
            ('{"apps":["demo"],"steps":[["demo","inspect"]]', ["demo"], 1, "model"),
            ('{"apps":["demo"],"steps":[["demo","ins', ["demo"], 0, "recovered"),
            ('{"apps":["demo","oth', [], 0, None),
        )
        for raw, apps, step_count, source in cases:
            with self.subTest(raw=raw):
                plan = parse_roadmap(raw, _catalog(), ["demo"])
                self.assertEqual((plan.apps, len(plan.steps), plan.apps_source), (apps, step_count, source))
                self.assertIn("invalid_json", plan.notes)

    def test_roadmap_preserves_legacy_apps_and_valid_sibling_steps(self) -> None:
        catalog = _catalog()
        for raw in ('{"apps":"demo"}', '{"app":"demo"}', '["demo"]'):
            self.assertEqual(parse_roadmap(raw, catalog, ["demo"]).apps, ["demo"])
        raw = ('{"example":true} {"apps":[null,"demo"],"steps":'
               '["bad step",["demo","inspect"],["demo","update","after P2 succeeds"],'
               '["demo","unavailable dependency","after P1 succeeds"]]}')
        plan = parse_roadmap(raw, catalog, ["demo"])
        self.assertEqual(plan.apps, ["demo"])
        self.assertEqual([s.source_index for s in plan.steps], [2, 3])
        self.assertEqual(plan.steps[1].condition, "after P1 succeeds")
        self.assertFalse(plan.steps_parsed)

    def test_roadmap_fields_fall_back_independently(self) -> None:
        catalog = _catalog()
        plan = parse_roadmap('{"apps":["demo"],"steps":["demo: inspect"]}', catalog, ["demo"])
        self.assertEqual(plan.apps, ["demo"])
        self.assertEqual(plan.steps, [])
        self.assertFalse(plan.steps_parsed)
        self.assertTrue(plan.parsed)
        plan = parse_roadmap('{"apps":{},"steps":[["demo","inspect"]]}', catalog, ["demo"])
        self.assertEqual(plan.apps, ["demo"])
        self.assertEqual(plan.apps_source, "derived")
        self.assertTrue(plan.steps_parsed)

    def test_roadmap_dependency_reindexes_and_prunes_transitively(self) -> None:
        payload = {"apps": ["demo"], "steps": [
            ["demo", "inspect"], ["missing", "check"], ["demo", "adjust", "after P2 succeeds"],
            ["demo", "check result", "after P3 succeeds"], ["demo", "update", "after P1 succeeds"],
            ["demo", "notify", "after P5 succeeds"], ["demo", "archive"], ["demo", "repeat"],
        ]}
        plan = parse_roadmap(json.dumps(payload), _catalog(), ["demo"])
        self.assertEqual([s.source_index for s in plan.steps], [1, 5, 6, 7])
        self.assertEqual([s.condition for s in plan.steps], [None, "after P1 succeeds", "after P2 succeeds", None])

    def test_roadmap_is_rendered_before_the_cache_boundary(self) -> None:
        sample = _sample()
        prepared = PreparedSample(sample, "Please inspect", [])
        plain = caller_messages(prepared, _catalog(), ["demo"], [])
        roadmap = "PLANNER ROADMAP (advisory)\nP1. demo — inspect"
        rendered = caller_messages(prepared, _catalog(), ["demo"], [], roadmap=roadmap)
        self.assertEqual(plain[0]["content"], CALLER_SYSTEM)
        self.assertIn("CURRENT REQUEST\nPlease inspect\n\n" + roadmap
                      + "\n\nEXECUTION TRACE FOR THE CURRENT REQUEST\n", rendered[1]["content"])

    def setUp(self) -> None:
        self.catalog = _catalog()
        self.installed = {"demo": {"tools": ["demo.do"]}}

    def test_ranked_installed_apps(self) -> None:
        decision = parse_plan('{"apps":["demo","not_installed"]}', self.catalog, self.installed, top_k=4)
        self.assertTrue(decision.parsed)
        self.assertEqual(decision.apps, ["demo"])

    def test_empty_plan(self) -> None:
        decision = parse_plan('{"apps":[]}', self.catalog, self.installed)
        self.assertTrue(decision.parsed)
        self.assertEqual(decision.apps, [])

    def test_plain_list_and_aliases(self) -> None:
        decision = parse_plan('["Demo"]', self.catalog, self.installed)
        self.assertEqual(decision.apps, ["demo"])

    def test_fallback_is_lexical(self) -> None:
        decision = parse_plan("I cannot answer", self.catalog, self.installed)
        self.assertFalse(decision.parsed)

    def test_prompt_examples_render_after_formatting(self) -> None:
        prepared = PreparedSample(_sample(), "đặt báo thức 6 giờ sáng mai")
        messages = planner_messages(prepared, self.catalog)
        self.assertIn('{"apps": ["alarm"]}', messages[0]["content"])
        self.assertIn('{"apps": ["weather", "calendar", "contacts", "sms"]}', messages[0]["content"])


class MockBackendTests(unittest.TestCase):
    def test_mock_backend_returns_fixed_caller_action(self) -> None:
        response = '{"action":"finish"}'
        backend = _build_backend(SimpleNamespace(backend="mock", mock_response=response))
        self.assertEqual(backend.complete([{"role": "system", "content": CALLER_SYSTEM}], None), response)


class AsrDownloadTests(unittest.TestCase):
    def test_download_requests_every_model_file(self) -> None:
        import lifemate.asr as asr

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for name in asr.SherpaOnnxTranscriber.FILES:
                (model_dir / name).write_bytes(b"x")
            calls: dict[str, Any] = {}

            class FakeRecognizer:
                @staticmethod
                def from_transducer(**kwargs: Any) -> object:
                    calls["recognizer"] = kwargs
                    return object()

            fake_sherpa = ModuleType("sherpa_onnx")
            fake_sherpa.OfflineRecognizer = FakeRecognizer  # type: ignore[attr-defined]

            def fake_snapshot(**kwargs: Any) -> str:
                calls["download"] = kwargs
                return str(model_dir)

            fake_hub = ModuleType("huggingface_hub")
            fake_hub.snapshot_download = fake_snapshot  # type: ignore[attr-defined]
            with patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa, "huggingface_hub": fake_hub}):
                asr.SherpaOnnxTranscriber(decoding_method="greedy_search")

        self.assertEqual(
            set(calls["download"]["allow_patterns"]),
            set(asr.SherpaOnnxTranscriber.FILES),
        )
        self.assertEqual(
            calls["recognizer"]["tokens"], str(model_dir / "tokens.txt")
        )


class CallerParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = _catalog()

    def test_call(self) -> None:
        action = parse_caller_action(
            '{"action":"call","tool":"do","arguments":{"x":"hi"}}',
            self.catalog,
            focus_apps=["demo"],
        )
        self.assertEqual(action.action, "call")
        self.assertEqual(action.tool, "demo.do")
        self.assertEqual(action.arguments, {"x": "hi"})

    def test_call_alias_and_openai_shape(self) -> None:
        action = parse_caller_action(
            '{"tool":"do","arguments":{"x":"hi"}}',
            self.catalog,
            focus_apps=["demo"],
        )
        self.assertEqual(action.action, "call")
        self.assertEqual(action.tool, "demo.do")

    def test_tool_name_in_action_field(self) -> None:
        action = parse_caller_action(
            '{"action":"demo.do","arguments":{"x":"hi"}}',
            self.catalog,
            focus_apps=["demo"],
        )
        self.assertEqual(action.action, "call")
        self.assertEqual(action.tool, "demo.do")
        self.assertEqual(action.arguments, {"x": "hi"})

    def test_need_app_ask_finish(self) -> None:
        need = parse_caller_action('{"action":"need_app","app":"demo"}', self.catalog)
        self.assertEqual((need.action, need.app), ("need_app", "demo"))
        ask = parse_caller_action(
            '{"action":"ask","tool":"demo.do","params":["x"]}', self.catalog, focus_apps=["demo"]
        )
        self.assertEqual((ask.action, ask.tool, ask.params), ("ask", "demo.do", ["x"]))
        finish = parse_caller_action('{"action":"finish"}', self.catalog)
        self.assertEqual(finish.action, "finish")

    def test_ask_through_sys_uses_target_tool_name(self) -> None:
        action = parse_caller_action(
            '{"action":"ask","tool":"sys.ask_missing_info","tool_name":"demo.do","params":["x"]}',
            self.catalog,
            focus_apps=["demo"],
        )
        self.assertEqual((action.action, action.tool, action.params), ("ask", "demo.do", ["x"]))

    def test_invalid(self) -> None:
        action = parse_caller_action("no json here", self.catalog)
        self.assertEqual(action.action, "invalid")


class ValidationSplitTests(unittest.TestCase):
    def test_read_ids_accepts_text_json_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = root / "ids.txt"
            text.write_text("train_0002\ntrain_0001\ntrain_0002\n", encoding="utf-8")
            self.assertEqual(read_ids(text), ["train_0002", "train_0001"])

            as_json = root / "ids.json"
            as_json.write_text(json.dumps({"ids": ["a", "b"]}), encoding="utf-8")
            self.assertEqual(read_ids(as_json), ["a", "b"])

            as_jsonl = root / "ids.jsonl"
            as_jsonl.write_text('{"id": "x"}\n{"id": "y"}\n', encoding="utf-8")
            self.assertEqual(read_ids(as_jsonl), ["x", "y"])

    def test_find_validation_ids_prefers_split_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            target = root / "train" / "validation_100_ids.txt"
            target.write_text("train_0001\n", encoding="utf-8")
            self.assertEqual(find_validation_ids(root), target)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = Catalog.from_dict(
            {
                "schema_version": "test",
                "apps": [
                    {
                        "id": "notes",
                        "name_vi": "Ghi chú",
                        "group": "productivity",
                        "description": "notes",
                        "tools": [
                            {
                                "name": "notes.add",
                                "kind": "write",
                                "description": "add",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string", "x-match": "fuzzy"},
                                    },
                                    "required": ["content"],
                                },
                                "returns": "",
                                "app": "notes",
                            }
                        ],
                    }
                ],
            }
        )
        self.gold = [{"name": "notes.add", "arguments": {"content": "đi mua sữa"}}]

    def test_hash_mismatch_scores_zero(self) -> None:
        score = score_prediction(self.gold, "bad", self.gold, "good", self.catalog)
        self.assertEqual(score.score, 0.0)
        self.assertFalse(score.hash_ok)

    def test_exact_prediction(self) -> None:
        score = score_prediction(self.gold, "good", self.gold, "good", self.catalog)
        self.assertEqual(score.score, 1.0)
        self.assertTrue(score.hash_ok)

    def test_fuzzy_argument_and_partial_f1(self) -> None:
        prediction = [{"name": "notes.add", "arguments": {"content": "mua sữa"}}]
        self.assertTrue(calls_match(prediction[0], self.gold[0], self.catalog))
        score = score_prediction(prediction, "good", self.gold, "good", self.catalog)
        self.assertEqual(score.score, 1.0)

    def test_extra_call_reduces_precision(self) -> None:
        prediction = self.gold + [{"name": "notes.add", "arguments": {"content": "khác"}}]
        score = score_prediction(prediction, "good", self.gold, "good", self.catalog)
        self.assertAlmostEqual(score.score, 2 / 3, places=6)
        self.assertEqual((score.matched, score.predicted, score.gold), (1, 2, 1))

    def test_empty_gold(self) -> None:
        score = score_prediction([], "good", [], "good", self.catalog)
        self.assertEqual(score.score, 1.0)
        score = score_prediction(self.gold, "good", [], "good", self.catalog)
        self.assertEqual(score.score, 0.0)


class RuntimeSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch("lifemate.discovery._get_e5_dense_ranking", return_value=[]))

    def test_discovery_blocks_unopened_calls_and_never_executes_search(self) -> None:
        catalog = _catalog()
        backend = ScriptedBackend([
            '{"action":"need_app","app":"demo"}',
            '{"action":"call","tool":"demo.do","arguments":{"x":"hi"}}',
            '{"action":"call","tool":"sys.search_tools","arguments":{"query":"do the thing"}}',
            '{"action":"call","tool":"sys.search_tools","arguments":{"query":"demo"}}',
            '{"action":"call","tool":"demo.do","arguments":{"x":"hi"}}',
            '{"action":"finish"}',
        ])
        class FakeTranscriber:
            def transcribe(self, path: Path, cache_key: str = "") -> str:
                return {"request.wav": "ASR current request", "history.wav": "ASR history turn"}[path.name]

        sample = _sample()
        sample.root = Path(".")
        sample.raw.update(
            audio="request.wav",
            audio_text="",
            history=[{"role": "user", "audio": "history.wav"}],
        )
        result = LifeMateAgent(
            catalog, Caller(backend, catalog),
            transcriber=FakeTranscriber(),
            config=AgentConfig(max_steps=6, tool_discovery=True),
        ).run(sample, FakeEnv())
        self.assertEqual(result.transcript, "ASR current request")
        first_user_message = backend.calls[0][1]["content"]
        self.assertIn("ASR current request", first_user_message)
        self.assertIn("ASR history turn", first_user_message)
        self.assertTrue(result.finished)
        self.assertEqual(result.calls, [{"name": "demo.do", "arguments": {"x": "hi"}}])
        self.assertEqual(result.state_hash, "hash:1")
        self.assertEqual([record.tool for record in result.trace], ["sys.search_tools", "demo.do"])
        self.assertNotIn("need_app", backend.calls[0][0]["content"])
        self.assertNotIn("FALLBACK COMPACT DIRECTORY", backend.calls[0][1]["content"])
        self.assertNotIn("FOCUSED APPS", backend.calls[0][1]["content"])
        self.assertIn("demo.do", backend.calls[3][1]["content"])
        before = backend.calls[0][1]["content"]
        after = backend.calls[3][1]["content"]
        marker = "\nEXECUTION TRACE FOR THE CURRENT REQUEST\n"
        self.assertEqual(before.split(marker)[0], after.split(marker)[0])
        self.assertEqual(after.count("\nCURRENT REQUEST\n"), 1)
        self.assertIn("DISCOVERED FUNCTION SCHEMAS", after.split(marker)[1])
        self.assertNotIn("taxi.book_ride", backend.calls[0][0]["content"])
        system = backend.calls[0][0]["content"]
        self.assertIn("Ignore unrelated results", system)
        self.assertIn("out-of-scope goal", system)
        self.assertIn('{"action":"ask","tool":"alarm.create","params":["time"]}', system)

    def test_discovery_search_uses_only_installed_non_meta_tools(self) -> None:
        from lifemate.discovery import search_tools

        catalog = _catalog()
        self.assertEqual(search_tools(catalog, [], "demo"), [])
        self.assertEqual(search_tools(catalog, ["demo"], "demo", opened={"demo.do"}), [])
        self.assertEqual(search_tools(catalog, ["demo"], "zyxwvu"), [])
        self.assertEqual([tool.name for tool in search_tools(catalog, ["demo"], "demo")], ["demo.do"])

    def test_call_then_finish(self) -> None:
        catalog = _catalog()
        backend = ScriptedBackend(
            [
                '{"action":"call","tool":"demo.do","arguments":{"x":"hi"}}',
                '{"action":"finish"}',
            ]
        )
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=4),
        )
        env = FakeEnv()
        result = agent.run(_sample(), env)
        self.assertTrue(result.finished)
        self.assertEqual(result.calls, [{"name": "demo.do", "arguments": {"x": "hi"}}])
        self.assertEqual(result.state_hash, "hash:1")
        self.assertFalse(result.error)

    def test_ambiguous_contact_auto_asks_instead_of_guessing(self) -> None:
        catalog = _catalog(
            {
                "schema_version": "test",
                "apps": [
                    {
                        "id": "contacts",
                        "name_vi": "Danh bạ",
                        "group": "communication",
                        "description": "contacts",
                        "tools": [
                            {
                                "name": "contacts.search",
                                "kind": "read",
                                "description": "search contacts",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                },
                                "returns": "",
                                "app": "contacts",
                            }
                        ],
                    },
                    {
                        "id": "phone",
                        "name_vi": "Điện thoại",
                        "group": "communication",
                        "description": "phone",
                        "tools": [
                            {
                                "name": "phone.call",
                                "kind": "write",
                                "description": "call a contact",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                                        "phone_number": {"type": "string"},
                                    },
                                    "required": [],
                                    "x-one-of": [["contact_id"], ["phone_number"]],
                                },
                                "returns": "",
                                "app": "phone",
                            }
                        ],
                    },
                    {
                        "id": "sys",
                        "name_vi": "Hệ thống",
                        "group": "control",
                        "description": "system",
                        "tools": [
                            {
                                "name": "sys.ask_missing_info",
                                "kind": "meta",
                                "description": "report missing params",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "tool_name": {"type": "string"},
                                        "params": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["tool_name", "params"],
                                },
                                "returns": "",
                                "app": "sys",
                            }
                        ],
                    },
                ],
            }
        )
        backend = ScriptedBackend(
            [
                '{"action":"call","tool":"contacts.search","arguments":{"query":"Lan"}}',
                '{"action":"call","tool":"phone.call","arguments":{"contact_id":"Lan"}}',
                '{"action":"finish"}',
            ]
        )
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["contacts", "phone"],
            config=AgentConfig(top_k=4, max_steps=6),
        )
        sample = _sample()
        sample.raw["context"]["installed_apps"] = {
            "contacts": {"tools": ["contacts.search"]},
            "phone": {"tools": ["phone.call"]},
        }
        env = FakeEnv()
        env.call = lambda name, arguments: (
            env.calls.append({"name": name, "arguments": dict(arguments)})
            or (
                {
                    "ok": True,
                    "contacts": [
                        {"id": "c_05", "name": "Lan", "relation": "vợ"},
                        {"id": "c_10", "name": "Lan", "relation": "đồng nghiệp"},
                    ],
                }
                if name == "contacts.search"
                else {"ok": True}
            )
        )
        result = agent.run(sample, env)
        self.assertEqual(
            result.calls,
            [
                {"name": "contacts.search", "arguments": {"query": "Lan"}},
                {
                    "name": "sys.ask_missing_info",
                    "arguments": {"tool_name": "phone.call", "params": ["contact_id"]},
                },
            ],
        )

    def test_missing_required_auto_asks(self) -> None:
        catalog = _catalog()
        backend = ScriptedBackend(['{"action":"call","tool":"demo.do","arguments":{}}'])
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=4, auto_ask=True),
        )
        env = FakeEnv()
        result = agent.run(_sample(), env)
        self.assertTrue(result.finished)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0]["name"], "sys.ask_missing_info")
        self.assertEqual(result.calls[0]["arguments"], {"tool_name": "demo.do", "params": ["x"]})

    def test_ask_for_optional_param_is_allowed(self) -> None:
        catalog = _catalog()
        backend = ScriptedBackend(
            [
                '{"action":"ask","tool":"demo.do","arguments":{"x":"hi"},"params":["y"]}',
                '{"action":"finish"}',
            ]
        )
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=4, auto_ask=True),
        )
        env = FakeEnv()
        result = agent.run(_sample(), env)
        self.assertTrue(result.finished)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0]["name"], "sys.ask_missing_info")
        self.assertEqual(result.calls[0]["arguments"], {"tool_name": "demo.do", "params": ["y"]})

    def test_ask_for_unknown_param_is_rejected(self) -> None:
        catalog = _catalog()
        backend = ScriptedBackend(
            [
                '{"action":"ask","tool":"demo.do","arguments":{"x":"hi"},"params":["z"]}',
                '{"action":"finish"}',
            ]
        )
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=4, auto_ask=True),
        )
        env = FakeEnv()
        result = agent.run(_sample(), env)
        self.assertTrue(result.finished)
        self.assertEqual(result.calls, [])
        self.assertTrue(any("no missing required param" in note for note in result.notes))

    def test_identical_call_is_not_executed_twice(self) -> None:
        catalog = _catalog()
        repeated = '{"action":"call","tool":"demo.do","arguments":{"x":"hi"}}'
        backend = ScriptedBackend([repeated, repeated, repeated, '{"action":"finish"}'])
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=6, max_reprompts=2),
        )
        env = FakeEnv()
        result = agent.run(_sample(), env)
        self.assertEqual(result.calls, [{"name": "demo.do", "arguments": {"x": "hi"}}])
        self.assertTrue(any("duplicate call skipped" in note for note in result.notes))

    def test_repeatable_write_can_run_twice(self) -> None:
        catalog = _catalog(
            {
                "schema_version": "test",
                "apps": [
                    {
                        "id": "loyalty",
                        "name_vi": "Loyalty",
                        "group": "test",
                        "description": "loyalty app",
                        "tools": [
                            {
                                "name": "loyalty.redeem",
                                "kind": "write",
                                "description": "redeem a reward",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "a": {"type": "string"},
                                        "b": {"type": "string"},
                                    },
                                    "required": ["a", "b"],
                                },
                                "returns": "",
                                "app": "loyalty",
                            }
                        ],
                    }
                ],
            }
        )
        repeated = (
            '{"action":"call","tool":"loyalty.redeem",'
            '"arguments":{"a":"one","b":"two"}}'
        )
        backend = ScriptedBackend([repeated, repeated, '{"action":"finish"}'])
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["loyalty"],
            config=AgentConfig(top_k=4, max_steps=6, max_reprompts=2),
        )
        sample = _sample()
        sample.raw["context"]["installed_apps"] = {"loyalty": {"tools": ["loyalty.redeem"]}}
        result = agent.run(sample, FakeEnv())
        self.assertTrue(result.finished)
        self.assertEqual(len(result.calls), 2)
        self.assertFalse(any("duplicate call skipped" in note for note in result.notes))

    def test_read_can_refresh_after_successful_write(self) -> None:
        catalog = _catalog(
            {
                "schema_version": "test",
                "apps": [
                    {
                        "id": "demo",
                        "name_vi": "Demo",
                        "group": "test",
                        "description": "demo app",
                        "tools": [
                            {
                                "name": "demo.get",
                                "kind": "read",
                                "description": "read the thing",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                },
                                "returns": "",
                                "app": "demo",
                            },
                            {
                                "name": "demo.do",
                                "kind": "write",
                                "description": "do the thing",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"x": {"type": "string"}},
                                    "required": ["x"],
                                },
                                "returns": "",
                                "app": "demo",
                            },
                        ],
                    }
                ],
            }
        )
        backend = ScriptedBackend(
            [
                '{"action":"call","tool":"demo.get","arguments":{}}',
                '{"action":"call","tool":"demo.do","arguments":{"x":"hi"}}',
                '{"action":"call","tool":"demo.get","arguments":{}}',
                '{"action":"finish"}',
            ]
        )
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=6, max_reprompts=2),
        )
        sample = _sample()
        sample.raw["context"]["installed_apps"] = {"demo": {"tools": ["demo.get", "demo.do"]}}
        result = agent.run(sample, FakeEnv())
        self.assertTrue(result.finished)
        self.assertEqual(
            [call["name"] for call in result.calls],
            ["demo.get", "demo.do", "demo.get"],
        )
        self.assertFalse(any("duplicate call skipped" in note for note in result.notes))

    def test_identical_read_without_write_is_blocked(self) -> None:
        catalog = _catalog(
            {
                "schema_version": "test",
                "apps": [
                    {
                        "id": "demo",
                        "name_vi": "Demo",
                        "group": "test",
                        "description": "demo app",
                        "tools": [
                            {
                                "name": "demo.get",
                                "kind": "read",
                                "description": "read the thing",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                },
                                "returns": "",
                                "app": "demo",
                            }
                        ],
                    }
                ],
            }
        )
        repeated = '{"action":"call","tool":"demo.get","arguments":{}}'
        backend = ScriptedBackend([repeated, repeated, '{"action":"finish"}'])
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=6, max_reprompts=2),
        )
        sample = _sample()
        sample.raw["context"]["installed_apps"] = {"demo": {"tools": ["demo.get"]}}
        result = agent.run(sample, FakeEnv())
        self.assertEqual(result.calls, [{"name": "demo.get", "arguments": {}}])
        self.assertTrue(any("duplicate call skipped" in note for note in result.notes))

    def test_unknown_tool_reprompts_then_finishes(self) -> None:
        catalog = _catalog()
        backend = ScriptedBackend(
            [
                '{"action":"call","tool":"demo.nope","arguments":{}}',
                '{"action":"finish"}',
            ]
        )
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=4, max_reprompts=2),
        )
        env = FakeEnv()
        result = agent.run(_sample(), env)
        self.assertTrue(result.finished)
        self.assertEqual(result.calls, [])
        self.assertTrue(any("unknown or uninstalled tool" in note for note in result.notes))

    def test_failed_call_is_retried_once(self) -> None:
        catalog = _catalog()
        repeated = '{"action":"call","tool":"demo.do","arguments":{"x":"hi"}}'
        backend = ScriptedBackend([repeated, repeated, '{"action":"finish"}'])
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=6, max_reprompts=2),
        )
        env = FakeEnv()
        attempts = {"n": 0}

        def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            attempts["n"] += 1
            env.calls.append({"name": name, "arguments": dict(arguments)})
            if attempts["n"] == 1:
                return {"ok": False, "error": "transient"}
            return {"ok": True, "done": True}

        env.call = call
        result = agent.run(_sample(), env)
        self.assertEqual(len(result.calls), 2)
        self.assertTrue(result.finished)
        self.assertFalse(any("duplicate call skipped" in note for note in result.notes))

    def test_failed_ask_is_not_reported_as_finished(self) -> None:
        catalog = _catalog()
        backend = ScriptedBackend(['{"action":"call","tool":"demo.do","arguments":{}}'])
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=1, max_reprompts=2, auto_ask=True),
        )
        env = FakeEnv()

        def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            env.calls.append({"name": name, "arguments": dict(arguments)})
            return {"ok": False, "error": "ask rejected"}

        env.call = call
        result = agent.run(_sample(), env)
        self.assertFalse(result.finished)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0]["name"], "sys.ask_missing_info")
        self.assertTrue(any("rejected by the tool environment" in note for note in result.notes))

    def test_invalid_ask_respects_reprompt_limit(self) -> None:
        catalog = _catalog()
        bad = '{"action":"ask","tool":"demo.do","arguments":{"x":"hi"},"params":["z"]}'
        backend = ScriptedBackend([bad] * 10)
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            initial_apps=["demo"],
            config=AgentConfig(top_k=4, max_steps=10, max_reprompts=2),
        )
        result = agent.run(_sample(), FakeEnv())
        self.assertFalse(result.finished)
        self.assertEqual(result.calls, [])
        self.assertEqual(len(backend.calls), 3)

    def test_valid_need_app_does_not_consume_reprompts(self) -> None:
        catalog = _catalog()
        bad = '{"action":"ask","tool":"demo.do","arguments":{"x":"hi"},"params":["z"]}'
        backend = ScriptedBackend(
            ['{"action":"need_app","app":"demo"}', bad, bad, bad, '{"action":"finish"}']
        )
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            config=AgentConfig(top_k=4, max_steps=10, max_reprompts=2),
        )
        result = agent.run(_sample(), FakeEnv())
        self.assertFalse(result.finished)
        self.assertEqual(result.calls, [])
        self.assertEqual(len(backend.calls), 4)
        self.assertTrue(any("loaded full schemas" in note for note in result.notes))

    def test_repeated_need_app_respects_reprompt_limit(self) -> None:
        catalog = _catalog()
        need = '{"action":"need_app","app":"demo"}'
        backend = ScriptedBackend([need] * 10)
        agent = LifeMateAgent(
            catalog,
            Caller(backend, catalog),
            config=AgentConfig(top_k=4, max_steps=10, max_reprompts=2),
        )
        result = agent.run(_sample(), FakeEnv())
        self.assertFalse(result.finished)
        self.assertEqual(result.calls, [])
        self.assertEqual(len(backend.calls), 4)
        self.assertTrue(any("already loaded" in note for note in result.notes))


class RuntimeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        train_dir = Path(__file__).resolve().parents[2]
        cls.root = Path(os.environ["LIFEMATE_DATA_ROOT"]) if os.environ.get("LIFEMATE_DATA_ROOT") else find_data_root(train_dir)
        cls.catalog = Catalog.load(catalog_path(cls.root, "train"))
        cls.sample = load_samples(cls.root, "train", ids=["train_0008"])[0]

    def test_labeled_trajectory_replays_in_one_env(self) -> None:
        env = make_tool_env(self.root, "train", self.sample.id)
        agent = LifeMateAgent(
            self.catalog,
            ReplayCaller(self.sample.gold_calls),
            initial_apps=self.sample.gold_apps,
            config=AgentConfig(top_k=4, max_steps=8),
        )
        result = agent.run(self.sample, env)
        self.assertTrue(result.finished)
        self.assertFalse(result.error)
        self.assertEqual(
            [call["name"] for call in result.calls],
            [call["name"] for call in self.sample.gold_calls],
        )
        state = env.get_state()
        for app, final in (self.sample.label.get("final_state") or {}).items():
            self.assertEqual(state.get(app), final, app)
        score = score_prediction(
            result.calls,
            result.state_hash,
            self.sample.gold_calls,
            result.state_hash,
            self.catalog,
        )
        self.assertEqual(score.score, 1.0)
        self.assertTrue(score.hash_ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
