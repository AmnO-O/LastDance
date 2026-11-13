import unittest
from lifemate.sft_mix import mix_examples


class MixtureTests(unittest.TestCase):
    def test_default_counts_and_reproducibility(self):
        pools = {"gold": list(range(10)), "asr": ["noisy"], "recovery": ["fix"]}
        rows, counts = mix_examples(pools, (0.4, 0.4, 0.2), 13)
        self.assertEqual(len(rows), 25)
        self.assertEqual(counts["recovery"]["selected"], 5)
        self.assertEqual(rows, mix_examples(pools, (0.4, 0.4, 0.2), 13)[0])

    def test_missing_enabled_source_is_rejected(self):
        with self.assertRaises(ValueError):
            mix_examples({"gold": [1], "asr": [], "recovery": []}, (0.4, 0.4, 0.2), 13)

    def test_clean_baseline(self):
        rows, _ = mix_examples({"gold": [1, 2], "asr": [], "recovery": []}, (1, 0, 0), 13)
        self.assertEqual(sorted(rows), [1, 2])

    def test_encode_example_rejects_open_thinking_tag(self):
        from types import SimpleNamespace
        from lifemate.train_unsloth import encode_example

        class MockThinkingTokenizer:
            def __init__(self, template_output):
                self.template_output = template_output

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
                if "enable_thinking" in kwargs:
                    raise TypeError("unexpected keyword argument enable_thinking")
                return self.template_output

            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [4, 5]}

            eos_token_id = 99

        ex = SimpleNamespace(messages=[{"role": "user", "content": "hello"}], target='{"action":"finish"}')
        # Unclosed thinking block must raise ValueError
        unclosed = MockThinkingTokenizer("<|im_start|>assistant\n<think>\n")
        with self.assertRaises(ValueError) as ctx:
            encode_example(unclosed, ex, max_length=100)
        self.assertIn("unclosed thinking mode", str(ctx.exception))

        # Closed thinking block must be accepted
        closed = MockThinkingTokenizer("<|im_start|>assistant\n<think>\n</think>\n")
        res = encode_example(closed, ex, max_length=100)
        self.assertIsNotNone(res)
        self.assertEqual(res["input_ids"], [1, 2, 3, 4, 5, 99])

    def test_recovery_examples_accepts_step0_need_app(self):
        import json
        import tempfile
        from pathlib import Path
        from lifemate.catalog import Catalog
        from lifemate.data import Sample
        from lifemate.sft_mix import recovery_examples

        cat = Catalog.from_dict({
            "schema_version": "test",
            "apps": [{"id": "notes", "name_vi": "Ghi chú", "description": "Ghi chú", "tools": []}],
            "tools": [],
        })
        sample = Sample(
            raw={
                "id": "s1",
                "context": {"installed_apps": {"notes": {"version": "1.0", "tools": []}}},
                "audio": "a.wav",
                "audio_text": "tạo ghi chú",
                "history": [],
                "label": {"tool_calls": [], "tool_results": [], "final_state": {}},
            },
            split="train",
        )
        with tempfile.TemporaryDirectory() as tmp:
            rec_file = Path(tmp) / "recovery.jsonl"
            row = {
                "sample_id": "s1",
                "verified": True,
                "trace": [],
                "notes": [],
                "focus_apps": [],
                "target": {"action": "need_app", "app": "notes"},
            }
            rec_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows = recovery_examples([sample], cat, rec_file, max_steps=10, max_history_chars=1000)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].task, "caller")
            self.assertIn("need_app", rows[0].target)

    def test_collect_rollouts_turn_level_mixing(self):
        import json
        import tempfile
        from unittest.mock import patch
        from pathlib import Path
        from lifemate.cli import main

        class FakeEnv:
            def __init__(self, *args, **kwargs):
                self.calls = []
            def validate(self, name, args):
                return {"ok": True}
            def call(self, name, args):
                self.calls.append({"name": name, "arguments": args})
                return {"ok": True}
            def get_tool_calls(self):
                return list(self.calls)
            def get_state_hash(self):
                return "hash_test"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            train_dir.mkdir()
            cat_data = {"schema_version": "test", "apps": [{"id": "notes", "name_vi": "Ghi chú", "description": "Ghi chú", "tools": []}], "tools": []}
            (train_dir / "train_tools_catalog.json").write_text(json.dumps(cat_data), encoding="utf-8")
            samples = [
                {
                    "id": "t1",
                    "context": {"installed_apps": {"notes": {"version": "1.0", "tools": []}}},
                    "audio": "a.wav",
                    "audio_text": "gold current text",
                    "history": [{"role": "user", "text": "gold history text"}],
                    "label": {"tool_calls": [], "tool_results": [], "final_state": {}},
                },
                {
                    "id": "val1",
                    "context": {"installed_apps": {"notes": {"version": "1.0", "tools": []}}},
                    "audio": "a.wav",
                    "audio_text": "gold val text",
                    "history": [],
                    "label": {"tool_calls": [], "tool_results": [], "final_state": {}},
                }
            ]
            (train_dir / "train.jsonl").write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
            val_file = root / "val_ids.txt"
            val_file.write_text("val1\n", encoding="utf-8")
            asr_file = root / "asr.json"
            asr_data = {"t1": {"current": "asr current text", "history": ["asr history text"]}}
            asr_file.write_text(json.dumps(asr_data), encoding="utf-8")
            out_traces = root / "traces.jsonl"

            with patch("lifemate.cli.make_tool_env", side_effect=lambda *args, **kwargs: FakeEnv()):
                code = main([
                    "collect-rollouts",
                    "--root", str(root),
                    "--validation-ids", str(val_file),
                    "--transcripts", str(asr_file),
                    "--transcript-ratio", "0.5",
                    "--backend", "mock",
                    "--mock-response", '{"action":"finish"}',
                    "--save-traces", str(out_traces),
                    "--seed", "42",
                ])
            self.assertEqual(code, 0)
            self.assertTrue(out_traces.is_file())
            traces = [json.loads(line) for line in out_traces.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["id"], "t1")
            self.assertIn("transcript_sources", traces[0])
            self.assertIn(traces[0]["transcript_sources"]["current"], {"gold", "gipformer"})
            self.assertIn(traces[0]["transcript_sources"]["history"][0], {"gold", "gipformer"})
