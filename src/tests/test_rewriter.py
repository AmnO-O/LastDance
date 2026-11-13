"""Unit tests for the multi-candidate ASR rewriter."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lifemate.llm import GenerationSettings, ScriptedBackend  # noqa: E402
from lifemate.rewriter import (  # noqa: E402
    CANDIDATE_HEADER,
    CLEAN_LABEL,
    CONTEXT_HEADER,
    HISTORY_HEADER,
    REWRITER_SYSTEM,
    RewriteGuardConfig,
    Rewriter,
    _number_values,
    candidate_agreement,
    clean_rewriter_output,
    guard_rewrite,
    normalize_candidates,
    oracle_candidate,
    oracle_word_error_rate,
    render_candidate_block,
    render_context_block,
    render_history_block,
    rewriter_messages,
    unique_candidate_count,
    word_error_rate,
)


class _ExplodingBackend:
    def complete(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        raise AssertionError("backend should not be called")


class _FailingBackend:
    def complete(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        raise RuntimeError("model exploded")


HOMOPHONE_CANDIDATES = [
    "à thôi khỏe ạt nữa nha chiều mình tiện đi đường ghém siêu thị mua luôn",
    "à thôi khỏi đặt nữa nha chiều mình tiện đi đường ghé siêu thị mua luôn",
    "à thôi khỏi đặt nữa nha chiều mình tiện đi đường ghé siêu thị mua luôn",
]
HOMOPHONE_CLEAN = "À thôi khỏi đặt nữa nha, chiều mình tiện đi đường ghé siêu thị mua luôn."


class CandidateFormattingTests(unittest.TestCase):
    def test_normalize_drops_empty_and_caps(self) -> None:
        self.assertEqual(
            normalize_candidates(["  một  ", "", "   ", "hai", "ba"], max_candidates=2),
            ["một", "hai"],
        )

    def test_normalize_keeps_duplicates_and_order(self) -> None:
        self.assertEqual(
            normalize_candidates(["top1", "top1", "different"]),
            ["top1", "top1", "different"],
        )

    def test_normalize_accepts_single_string(self) -> None:
        self.assertEqual(normalize_candidates(" mở đèn "), ["mở đèn"])

    def test_render_block_is_numbered_and_quoted(self) -> None:
        block = render_candidate_block(["mở đèn", 'nói "xin chào"'])
        self.assertIn('[1] "mở đèn"', block)
        self.assertIn('[2] "nói \\"xin chào\\""', block)
        self.assertEqual(len(block.splitlines()), 2)

    def test_messages_shape(self) -> None:
        messages = rewriter_messages(["a", "b"])
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], REWRITER_SYSTEM)
        self.assertIn(CANDIDATE_HEADER, messages[1]["content"])
        self.assertTrue(messages[1]["content"].rstrip().endswith(CLEAN_LABEL))
        self.assertIn('[1] "a"', messages[1]["content"])
        self.assertIn('[2] "b"', messages[1]["content"])

    def test_prompt_has_all_eight_examples_and_no_name_masking(self) -> None:
        self.assertEqual(REWRITER_SYSTEM.count("CANDIDATES:"), 8)
        self.assertIn("must never appear in your answer", REWRITER_SYSTEM)
        self.assertIn('Do not replace a name with "N"', REWRITER_SYSTEM)

    def test_prompt_states_the_input_is_a_spoken_request(self) -> None:
        self.assertIn("spoken request", REWRITER_SYSTEM)
        self.assertIn("audio file", REWRITER_SYSTEM)
        self.assertIn("possibly mixed with common English words", REWRITER_SYSTEM)
        self.assertIn("text transcripts", REWRITER_SYSTEM)
        self.assertIn("review proposal", REWRITER_SYSTEM)

    def test_prompt_has_candidate_priority_and_entity_rules(self) -> None:
        self.assertIn("CANDIDATE PRIORITY", REWRITER_SYSTEM)
        self.assertIn("Never splice words from different candidates", REWRITER_SYSTEM)
        self.assertIn("keep [1]'s spoken form", REWRITER_SYSTEM)
        self.assertIn('"a còng" = @', REWRITER_SYSTEM)
        self.assertIn("finance@example.test", REWRITER_SYSTEM)

    def test_every_prompt_example_passes_guard(self) -> None:
        examples: list[tuple[list[str], str]] = []
        for block in REWRITER_SYSTEM.split("CANDIDATES:")[1:]:
            candidates: list[str] = []
            output = ""
            for line in block.strip().splitlines():
                candidate_match = re.match(r"\[\d+\]\s+(.*)$", line)
                if candidate_match:
                    candidates.append(json.loads(candidate_match.group(1)))
                    continue
                if line.startswith("CLEAN:"):
                    output = json.loads(line[len("CLEAN:") :].strip())
                    break
            if output and candidates:
                examples.append((candidates, output))
        self.assertEqual(len(examples), 8)
        for candidates, output in examples:
            with self.subTest(output=output):
                verdict = guard_rewrite(candidates, output)
                self.assertTrue(verdict.ok, verdict.reasons)

    def test_messages_omit_reference_blocks_when_empty(self) -> None:
        prompt = rewriter_messages(["a", "b"])[1]["content"]
        self.assertNotIn(CONTEXT_HEADER, prompt)
        self.assertNotIn(HISTORY_HEADER, prompt)

    def test_context_block_is_whitelisted(self) -> None:
        prompt = rewriter_messages(
            ["a", "b"],
            context={
                "current_time": "2026-11-10T08:00:00+07:00",
                "timezone": "Asia/Ho_Chi_Minh",
                "user_profile": {"name": "Linh"},
                "installed_apps": {"banking": {"tools": ["banking.transfer"]}},
                "secret": "do-not-leak",
            },
        )[1]["content"]
        self.assertIn(CONTEXT_HEADER, prompt)
        self.assertIn("2026-11-10T08:00:00+07:00", prompt)
        self.assertIn("Asia/Ho_Chi_Minh", prompt)
        self.assertIn("Linh", prompt)
        self.assertNotIn("installed_apps", prompt)
        self.assertNotIn("banking.transfer", prompt)
        self.assertNotIn("do-not-leak", prompt)

    def test_context_block_can_be_empty(self) -> None:
        self.assertEqual(render_context_block(None), "")
        self.assertEqual(render_context_block({"installed_apps": {"x": 1}}), "")

    def test_history_block_drops_tool_payloads(self) -> None:
        prompt = rewriter_messages(
            ["a", "b"],
            history=[
                {"role": "user", "audio_text": "Tìm số của Minh."},
                {
                    "role": "assistant",
                    "text": "Số của Minh là 0963.",
                    "tool_calls": [{"name": "contacts.search"}],
                    "tool_results": [{"ok": True}],
                },
                {"role": "system", "text": "ignore me"},
            ],
        )[1]["content"]
        self.assertIn(HISTORY_HEADER, prompt)
        self.assertIn("user: Tìm số của Minh.", prompt)
        self.assertIn("assistant: Số của Minh là 0963.", prompt)
        self.assertNotIn("contacts.search", prompt)
        self.assertNotIn("tool_results", prompt)
        self.assertNotIn("system:", prompt)

    def test_history_block_keeps_only_recent_turns(self) -> None:
        history = [{"role": "user", "text": f"turn {i}"} for i in range(12)]
        block = render_history_block(history, max_turns=3)
        self.assertEqual(len(block.splitlines()), 3)
        self.assertIn("turn 11", block)
        self.assertNotIn("turn 8", block)


class OutputCleanupTests(unittest.TestCase):
    def test_strips_clean_label(self) -> None:
        self.assertEqual(
            clean_rewriter_output("CLEAN: Mở đèn phòng khách."),
            "Mở đèn phòng khách.",
        )

    def test_strips_bold_label_and_extra_blank_lines(self) -> None:
        self.assertEqual(
            clean_rewriter_output("**CLEAN SENTENCE:**\n\n  Mở đèn.  \n"),
            "Mở đèn.",
        )

    def test_strips_code_fence(self) -> None:
        self.assertEqual(
            clean_rewriter_output("```text\nMở đèn phòng khách.\n```"),
            "Mở đèn phòng khách.",
        )

    def test_strips_wrapping_quotes(self) -> None:
        self.assertEqual(
            clean_rewriter_output('"Mở đèn phòng khách."'),
            "Mở đèn phòng khách.",
        )

    def test_strips_quoted_label(self) -> None:
        self.assertEqual(
            clean_rewriter_output('"CLEAN: Mở đèn phòng khách."'),
            "Mở đèn phòng khách.",
        )

    def test_joins_wrapped_lines(self) -> None:
        self.assertEqual(
            clean_rewriter_output("Mở đèn\nphòng khách."),
            "Mở đèn phòng khách.",
        )

    def test_empty(self) -> None:
        self.assertEqual(clean_rewriter_output("   "), "")


class GuardTests(unittest.TestCase):
    def test_accepts_homophone_repair(self) -> None:
        verdict = guard_rewrite(HOMOPHONE_CANDIDATES, HOMOPHONE_CLEAN)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_accepts_identical_noop(self) -> None:
        verdict = guard_rewrite(["Mở đèn phòng khách."], "Mở đèn phòng khách.")
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_accepts_spoken_number_to_digit(self) -> None:
        verdict = guard_rewrite(
            ["đặt báo thức sáu giờ sáng mai", "đặt báo thức 6 giờ sáng mai"],
            "Đặt báo thức 6 giờ sáng mai.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_accepts_date_words_to_digits(self) -> None:
        verdict = guard_rewrite(
            ["nhắc tôi ngày mười lăm tháng mười một"],
            "Nhắc tôi ngày 15 tháng 11.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_accepts_digit_by_digit_spoken_number(self) -> None:
        verdict = guard_rewrite(
            ["bật điều hoà phòng khách hai sáu độ"],
            "Bật điều hoà phòng khách 26 độ.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_from_is_not_four(self) -> None:
        verdict = guard_rewrite(
            ["chuyển hai triệu từ tài khoản thanh toán"],
            "Chuyển 2 triệu từ tài khoản thanh toán.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_accepts_large_spoken_amount(self) -> None:
        verdict = guard_rewrite(
            ["vàng sjc trên một trăm hai mươi hai triệu thì bán"],
            "Vàng SJC trên 122 triệu thì bán.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_accepts_fifty_percent(self) -> None:
        verdict = guard_rewrite(
            ["để đèn mờ năm mươi phần trăm"],
            "Để đèn mờ 50 phần trăm.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_accepts_punctuation_and_case_only(self) -> None:
        verdict = guard_rewrite(
            ["ĐẶT XE BỐN CHỖ TỪ CÔNG TY ĐI KHÁM RĂNG CHIỀU NAY"],
            "Đặt xe bốn chỗ từ công ty đi khám răng chiều nay.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_truncated_top1_does_not_block_complete_candidate(self) -> None:
        verdict = guard_rewrite(
            ["mở đèn", "mở đèn phòng khách và bật quạt trần"],
            "Mở đèn phòng khách và bật quạt trần.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_rejects_dropped_negation(self) -> None:
        verdict = guard_rewrite(
            ["đừng nhắc tôi mua sữa", "nhắc tôi mua sữa", "đừng nhắc tôi mua sữa"],
            "Nhắc tôi mua sữa.",
        )
        self.assertFalse(verdict.ok)
        self.assertIn("missing_negation", verdict.reasons)

    def test_rejects_invented_negation(self) -> None:
        verdict = guard_rewrite(
            ["nhắc tôi mua sữa", "nhắc tôi mua sữa"],
            "Đừng nhắc tôi mua sữa.",
        )
        self.assertFalse(verdict.ok)
        self.assertIn("invented_negation", verdict.reasons)

    def test_dung_is_not_a_negation(self) -> None:
        verdict = guard_rewrite(
            ["dùng thẻ thanh toán mua vé", "dùng thẻ thanh toán mua vé"],
            "Thanh toán mua vé bằng thẻ.",
        )
        # "bằng" is a paraphrase and may be rejected, but the shared "dùng"
        # must never be mistaken for "đừng" and force a missing_negation.
        self.assertFalse(any("negation" in reason for reason in verdict.reasons))

    def test_negation_class_can_change_particle(self) -> None:
        verdict = guard_rewrite(
            ["đừng nhắc tôi mua sữa", "đừng nhắc tôi mua sữa"],
            "Không nhắc tôi mua sữa.",
        )
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_rejects_changed_amount(self) -> None:
        verdict = guard_rewrite(
            ["chuyển cho lê minh hai triệu"] * 3,
            "Chuyển cho Lê Minh 20 triệu.",
        )
        self.assertFalse(verdict.ok)
        self.assertIn("missing_number:2000000", verdict.reasons)
        self.assertIn("invented_number:20000000", verdict.reasons)

    def test_rejects_dropped_time_number(self) -> None:
        verdict = guard_rewrite(
            ["nhắc mình gọi lại cho khách lúc chín giờ kém mười lăm tối mai"] * 2,
            "Nhắc mình gọi lại cho khách lúc 9 giờ tối mai.",
        )
        self.assertFalse(verdict.ok)
        self.assertIn("missing_number:15", verdict.reasons)

    def test_rejects_changed_date(self) -> None:
        verdict = guard_rewrite(
            ["nhắc tôi ngày 15 tháng 11"] * 3,
            "Nhắc tôi ngày 15 tháng 12.",
        )
        self.assertFalse(verdict.ok)
        self.assertIn("missing_number:11", verdict.reasons)
        self.assertIn("invented_number:12", verdict.reasons)

    def test_rejects_content_from_single_hypothesis(self) -> None:
        verdict = guard_rewrite(
            [
                "mở bài hát mới nhất của liam shaw",
                "mở bài hát mới nhất của liam shaw và lưu vào playlist",
                "mở bài hát mới nhất của liam shaw",
            ],
            "Mở bài hát mới nhất của Liam Shaw và lưu vào playlist.",
        )
        self.assertFalse(verdict.ok)
        self.assertTrue(any(reason.startswith("unsupported_content:") for reason in verdict.reasons))

    def test_rejects_invented_content(self) -> None:
        verdict = guard_rewrite(
            ["mở đèn phòng khách và bật quạt trần"] * 3,
            "Mở đèn phòng khách và bật quạt trần rồi gửi email cho sếp.",
        )
        self.assertFalse(verdict.ok)
        self.assertTrue(any(reason.startswith("invented_content:") for reason in verdict.reasons))

    def test_rejects_too_short(self) -> None:
        verdict = guard_rewrite(
            ["nhắc mình gọi lại cho khách lúc chín giờ kém mười lăm tối mai"] * 2,
            "Gọi khách.",
        )
        self.assertFalse(verdict.ok)
        self.assertTrue(any(reason.startswith("output_too_short") for reason in verdict.reasons))

    def test_rejects_prompt_echo(self) -> None:
        verdict = guard_rewrite(["mở đèn"], "[1] mở đèn")
        self.assertFalse(verdict.ok)
        self.assertIn("echoed_prompt", verdict.reasons)

    def test_no_candidates_guard(self) -> None:
        verdict = guard_rewrite([], "Mở đèn.")
        self.assertFalse(verdict.ok)
        self.assertIn("no_candidates", verdict.reasons)

    def test_guard_config_can_relax_precision(self) -> None:
        config = RewriteGuardConfig(min_content_precision=0.0, max_length_ratio=3.0)
        verdict = guard_rewrite(["mở đèn"], "Mở đèn và bật quạt.", config)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_candidate_agreement_uses_directional_recall(self) -> None:
        self.assertEqual(candidate_agreement(["mở đèn", "mở đèn"]), 1.0)
        self.assertEqual(
            candidate_agreement(["mở đèn", "mở đèn phòng khách và bật quạt trần"]),
            1.0,
        )
        self.assertLess(
            candidate_agreement(
                [
                    "mở thư ngày nắng về tập ba mươi lưu lượt mặc bẫy vôch list luôn à thôi khỏi hết đi mai còn ben",
                    "mở khương ngày nắng về tập ba mươi lưu luật mặt bẫy vô quát list luôn là thôi khỏi hết đi mai còn bé trai",
                ]
            ),
            0.7,
        )


class RewriterTests(unittest.TestCase):
    def test_single_candidate_skips_backend(self) -> None:
        rewriter = Rewriter(_ExplodingBackend())
        result = rewriter.rewrite("mở đèn phòng khách")
        self.assertTrue(result.skipped)
        self.assertFalse(result.fallback)
        self.assertEqual(result.text, "mở đèn phòng khách")
        self.assertEqual(result.reasons, ("single_candidate",))

    def test_low_agreement_candidates_skip_rewrite(self) -> None:
        candidates = [
            "mở thư ngày nắng về tập ba mươi lưu lượt mặc bẫy vôch list luôn à thôi khỏi hết đi mai còn ben",
            "mở khương ngày nắng về tập ba mươi lưu luật mặt bẫy vô quát list luôn là thôi khỏi hết đi mai còn bé trai",
        ]
        rewriter = Rewriter(_ExplodingBackend())
        result = rewriter.rewrite(candidates)
        self.assertTrue(result.fallback)
        self.assertEqual(result.text, candidates[0])
        self.assertEqual(result.candidates, tuple(candidates))
        self.assertTrue(result.reasons[0].startswith("low_candidate_agreement:"))

    def test_truncated_candidate_is_not_low_agreement(self) -> None:
        backend = ScriptedBackend(["Mở đèn phòng khách và bật quạt trần."])
        result = Rewriter(backend).rewrite(
            ["mở đèn", "mở đèn phòng khách và bật quạt trần"]
        )
        self.assertFalse(result.fallback)
        self.assertEqual(result.text, "Mở đèn phòng khách và bật quạt trần.")
        self.assertEqual(len(backend.calls), 1)

    def test_identical_candidates_are_normalized(self) -> None:
        backend = ScriptedBackend(["Mở đèn."])
        result = Rewriter(backend).rewrite(["mở đèn", " mở  đèn "])
        self.assertFalse(result.skipped)
        self.assertFalse(result.fallback)
        self.assertEqual(result.text, "Mở đèn.")
        self.assertEqual(len(backend.calls), 1)

    def test_agreement_does_not_allow_rewriting_words(self) -> None:
        backend = ScriptedBackend(["Không nhắc tôi mua sữa."])
        original = "đừng nhắc tôi mua sữa"
        result = Rewriter(backend).rewrite([original, original])
        self.assertTrue(result.fallback)
        self.assertEqual(result.text, original)
        self.assertIn("agreement_requires_formatting_only", result.reasons)

    def test_guard_rejects_moving_negation_or_swapping_amounts(self) -> None:
        cases = [
            ("không bật đèn phòng khách rồi bật quạt phòng ngủ",
             "bật đèn phòng khách rồi không bật quạt phòng ngủ"),
            ("chuyển cho lan 200 nghìn và cho minh 500 nghìn",
             "chuyển cho lan 500 nghìn và cho minh 200 nghìn"),
        ]
        for source, output in cases:
            with self.subTest(output=output):
                verdict = guard_rewrite([source, source], output)
                self.assertFalse(verdict.ok)
                self.assertIn("unsupported_word_order", verdict.reasons)
                with_polite_endings = [source + " nha", source + " nhé"]
                verdict = guard_rewrite(with_polite_endings, output)
                self.assertFalse(verdict.ok)
                self.assertIn("unsupported_word_order", verdict.reasons)
                self.assertTrue(guard_rewrite(with_polite_endings, source + " nhé.").ok)

    def test_no_candidates(self) -> None:
        rewriter = Rewriter(_ExplodingBackend())
        result = rewriter.rewrite(["", "   "])
        self.assertTrue(result.fallback)
        self.assertEqual(result.text, "")
        self.assertEqual(result.reasons, ("no_candidates",))

    def test_calls_backend_and_returns_clean_output(self) -> None:
        backend = ScriptedBackend(["CLEAN: Mở đèn phòng khách."])
        rewriter = Rewriter(backend)
        result = rewriter.rewrite(["MỞ ĐÈN PHÒNG KHÁCH", "mở đèn phòng khách"])
        self.assertFalse(result.fallback)
        self.assertFalse(result.skipped)
        self.assertEqual(result.text, "Mở đèn phòng khách.")
        self.assertEqual(result.candidates, ("MỞ ĐÈN PHÒNG KHÁCH", "mở đèn phòng khách"))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][0]["content"], REWRITER_SYSTEM)
        self.assertEqual(backend.calls[0][1]["content"].count("[1]"), 1)
        self.assertEqual(backend.calls[0][1]["content"].count("[2]"), 1)

    def test_falls_back_when_guard_rejects(self) -> None:
        backend = ScriptedBackend(["Nhắc tôi mua sữa."])
        rewriter = Rewriter(backend)
        result = rewriter.rewrite(
            ["đừng nhắc tôi mua sữa", "nhắc tôi mua sữa", "đừng nhắc tôi mua sữa"]
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.text, "đừng nhắc tôi mua sữa")
        self.assertEqual(result.raw_text, "Nhắc tôi mua sữa.")
        self.assertIn("missing_negation", result.reasons)

    def test_falls_back_on_backend_error(self) -> None:
        rewriter = Rewriter(_FailingBackend())
        result = rewriter.rewrite(["mở đèn", "mở đèn phòng khách"])
        self.assertTrue(result.fallback)
        self.assertEqual(result.text, "mở đèn")
        self.assertEqual(result.reasons, ("backend_error:RuntimeError",))

    def test_skip_single_can_be_disabled(self) -> None:
        backend = ScriptedBackend(["Mở đèn phòng khách."])
        rewriter = Rewriter(backend, skip_single=False)
        result = rewriter.rewrite(["mở đèn phòng khách"])
        self.assertFalse(result.skipped)
        self.assertEqual(result.text, "Mở đèn phòng khách.")
        self.assertEqual(len(backend.calls), 1)

    def test_changed_property(self) -> None:
        backend = ScriptedBackend(["Mở đèn phòng khách."])
        result = Rewriter(backend).rewrite(["mở đèn phòng khách", "mở đèn phòng khách."])
        self.assertTrue(result.changed)
        self.assertEqual(result.as_dict()["text"], "Mở đèn phòng khách.")

    def test_duplicate_candidates_are_kept_in_prompt(self) -> None:
        backend = ScriptedBackend(["Mở đèn."])
        rewriter = Rewriter(backend, skip_single=False)
        rewriter.rewrite(["mở đèn", "mở đèn", "mở đèn"])
        prompt = backend.calls[0][1]["content"]
        self.assertIn('[1] "mở đèn"', prompt)
        self.assertIn('[3] "mở đèn"', prompt)

    def test_rewrite_forwards_context_and_history(self) -> None:
        backend = ScriptedBackend(["Mở đèn."])
        rewriter = Rewriter(backend)
        rewriter.rewrite(
            ["mở đèn", "mở đèn"],
            context={
                "timezone": "Asia/Ho_Chi_Minh",
                "installed_apps": {"smart_home": {"tools": ["smart_home.light_set"]}},
            },
            history=[{"role": "user", "text": "câu trước"}],
        )
        prompt = backend.calls[0][1]["content"]
        self.assertIn(CONTEXT_HEADER, prompt)
        self.assertIn("Asia/Ho_Chi_Minh", prompt)
        self.assertIn("câu trước", prompt)
        self.assertNotIn("installed_apps", prompt)
        self.assertNotIn("smart_home.light_set", prompt)


class NumericMentionTests(unittest.TestCase):
    def test_year_unit_is_not_digit_five(self) -> None:
        self.assertEqual(dict(_number_values("hai năm")), {2: 1})
        self.assertEqual(dict(_number_values("mười lăm năm")), {15: 1})

    def test_five_as_tens_digit(self) -> None:
        self.assertEqual(dict(_number_values("một trăm năm mươi")), {150: 1})
        self.assertEqual(dict(_number_values("hai nghìn năm trăm")), {2500: 1})

    def test_digit_by_digit_pair(self) -> None:
        self.assertEqual(dict(_number_values("hai sáu độ")), {26: 1})

    def test_year_is_not_a_number(self) -> None:
        self.assertEqual(dict(_number_values("năm 2026")), {2026: 1})
        self.assertEqual(dict(_number_values("năm nay")), {})

    def test_negation_after_year(self) -> None:
        self.assertEqual(dict(_number_values("hai năm không đổi")), {2: 1})

    def test_spoken_phone_number(self) -> None:
        self.assertEqual(
            dict(_number_values("không chín năm không năm ba sáu không ba sáu bảy")),
            {9505360367: 1},
        )

    def test_weekday_ordinals(self) -> None:
        self.assertEqual(dict(_number_values("họp vào thứ tư")), {4: 1})
        self.assertEqual(dict(_number_values("họp vào thứ năm")), {5: 1})
        self.assertEqual(dict(_number_values("họp vào tháng năm")), {5: 1})
        self.assertEqual(dict(_number_values("họp vào ngày một")), {1: 1})


class WordErrorRateTests(unittest.TestCase):
    def test_identical(self) -> None:
        self.assertEqual(word_error_rate("mở đèn phòng khách", "Mở đèn phòng khách."), 0.0)

    def test_one_substitution(self) -> None:
        self.assertAlmostEqual(word_error_rate("a b c", "a x c"), 1 / 3)

    def test_empty_reference(self) -> None:
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "mở đèn"), 1.0)

    def test_oracle_selects_best_candidate(self) -> None:
        reference = "đặt báo thức 6 giờ sáng mai"
        candidates = [
            "đặt báo thức sáu giờ sáng nay",
            "đặt báo thức 6 giờ sáng mai",
            "đặt báo thức sáu giờ sáng",
        ]
        self.assertEqual(oracle_candidate(reference, candidates), candidates[1])
        self.assertEqual(oracle_word_error_rate(reference, candidates), 0.0)

    def test_oracle_empty(self) -> None:
        self.assertEqual(oracle_candidate("a b", []), "")
        self.assertEqual(oracle_word_error_rate("a b", []), 1.0)

    def test_unique_candidate_count(self) -> None:
        self.assertEqual(unique_candidate_count(["Mở đèn", "mở đèn.", "", "bật quạt"]), 2)


if __name__ == "__main__":
    unittest.main()
