"""Search-First SFT: verified replay, LoRA training, and Dev/Val evaluation.

Typical flow (from the repository root):

    python -m pipeline.run_search_sft prepare
    python -m pipeline.run_search_sft train --run-tag search_first
    python -m pipeline.run_search_sft dev --adapter models/caller_adapter

Only the Search-First agent is implemented here: a discovery prompt,
``sys.search_tools``, and ``max_searches=4`` / ``max_steps=12``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import random
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.common import DATA_DIR, WORK, dataset, dump, read, rows, sha, transcript_path, write_rows
from pipeline.runtime_utils import (
    CALLER_TOKENS,
    MODEL,
    PrefixCacheCaller,
    adapter_sha,
    configure,
    prepared_sample,
)

from lifemate import discovery
from lifemate.prompts import DISCOVERY_USER_TEMPLATE, SFT_SYSTEM_PROMPT
import hashlib
from lifemate.caller import Caller
from lifemate.catalog import Catalog
from lifemate.data import catalog_path, read_transcript_file
from lifemate.llm import ScriptedBackend
from lifemate.runtime import AgentConfig, LifeMateAgent, make_tool_env
from lifemate.scoring import calls_match, replay_gold, score_prediction
from lifemate.sft import SFTExample, caller_target_derivable

CONFIG = AgentConfig(tool_discovery=True, max_searches=4, max_steps=12, top_k=discovery.SEARCH_LIMIT)
SEED = 42

# Capability descriptions used by the replay teacher when a gold action needs a
# tool that is not open yet.  They describe the function, never its values.
ACTION_CAPABILITIES = {
    "contacts.search": "tra cứu thông tin liên hệ theo tên trong danh bạ",
    "bill_payment.list_bills": "xem các hóa đơn điện nước internet còn phải đóng",
    "bill_payment.pay": "thanh toán hóa đơn điện nước internet đã xác định",
    "weather.get_forecast": "xem dự báo thời tiết và khả năng mưa",
    "calendar.list_events": "tra cứu sự kiện và cuộc hẹn đã có trong lịch",
    "calendar.cancel_event": "hủy một sự kiện hoặc cuộc hẹn đã có",
    "sms.send": "gửi tin nhắn SMS qua số điện thoại hoặc danh bạ",
    "chat_ott.send_message": "gửi tin nhắn qua ứng dụng chat Zalo",
}


def inputs():
    root, val, samples = dataset()
    split = read(DATA_DIR / "split_ids.json")
    by_id = {sample.id: sample for sample in samples}
    ids = split["train"] + split["dev"] + val
    if len(set(ids)) != len(ids) or set(ids) - by_id.keys():
        raise ValueError("Split manifest contains overlap or unknown IDs")
    return root, val, split, by_id


def exact_calls(actual, expected, catalog):
    return len(actual) == len(expected) and all(
        calls_match(a, b, catalog)
        and (
            catalog.tools.get(b["name"]) is None
            or catalog.tools[b["name"]].kind != "write"
            or a["arguments"] == b["arguments"]
        )
        for a, b in zip(actual, expected)
    )


def writes(calls, catalog):
    return Counter(
        json.dumps(call, sort_keys=True, ensure_ascii=False)
        for call in calls
        if call["name"] in catalog.tools and catalog.tools[call["name"]].kind == "write"
    )


class ReplayRejected(Exception):
    """The available observation cannot reproduce this gold trajectory."""


def configure_sft_prompt():
    configure()
    pair = json.dumps([SFT_SYSTEM_PROMPT, DISCOVERY_USER_TEMPLATE, sha(REPO / "src" / "lifemate" / "catalog.py")], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(pair.encode("utf-8")).hexdigest()


class GoldCaller(Caller):
    """Scripted caller that emits the gold action and builds SFT rows from the replay."""

    def __init__(self, catalog, gold, config=CONFIG):
        super().__init__(
            ScriptedBackend(self.label),
            catalog,
            max_steps=config.max_steps,
            max_new_tokens=CALLER_TOKENS,
            capture_prompts=True,
            discovery_prompt=SFT_SYSTEM_PROMPT,
            user_template=DISCOVERY_USER_TEMPLATE,
        )
        self.config = config
        self.gold = gold
        self.target = None
        self.examples = []

    def label(self, messages, settings):
        return json.dumps(self.target, ensure_ascii=False)

    def next_action(self, prepared, focus_apps, trace, notes=(), step=0, **options):
        completed = [
            {"name": record.tool, "arguments": record.arguments}
            for record in trace
            if record.tool != discovery.SEARCH_TOOL
        ]
        if not exact_calls(completed, self.gold[: len(completed)], self.catalog):
            raise ReplayRejected("execution_diverged")
        if discovery._E5_FAILED:
            raise RuntimeError("E5 failed. Refusing to build training data from a lexical fallback.")
        if trace and trace[-1].tool == discovery.SEARCH_TOOL and len(self.examples):
            prior = json.loads(self.examples[-1].target)
            if prior.get("tool") == discovery.SEARCH_TOOL:
                next_gold = self.gold[len(completed)]
                tool = (
                    next_gold["arguments"]["tool_name"]
                    if next_gold["name"] == "sys.ask_missing_info"
                    else next_gold["name"]
                )
                if tool not in options["discovered_tools"]:
                    raise ReplayRejected("retrieval_miss")
        if len(completed) == len(self.gold):
            self.target = {"action": "finish"}
        else:
            next_gold = self.gold[len(completed)]
            asking = next_gold["name"] == "sys.ask_missing_info"
            args = next_gold["arguments"]
            tool = args["tool_name"] if asking else next_gold["name"]
            if tool not in options["discovered_tools"]:
                if sum(record.tool == discovery.SEARCH_TOOL for record in trace) >= self.config.max_searches:
                    raise ReplayRejected("search_budget")
                if step + 1 >= self.config.max_steps:
                    raise ReplayRejected("turn_budget")
                if not self.catalog.is_installed(tool, prepared.sample.installed_apps):
                    raise ReplayRejected("tool_unavailable")
                spec = self.catalog.tools[tool]
                query = ACTION_CAPABILITIES.get(tool, discovery.EXPANSIONS.get(tool) or spec.description)
                query = " ".join(query.split())[:200]
                if not query:
                    raise ReplayRejected("missing_capability_description")
                self.target = {
                    "action": "call",
                    "tool": discovery.SEARCH_TOOL,
                    "arguments": {"query": query},
                }
            elif asking:
                self.target = {"action": "ask", "tool": tool, "params": args["params"]}
            else:
                self.target = {"action": "call", "tool": tool, "arguments": args}
        action = super().next_action(prepared, focus_apps, trace, notes=notes, step=step, **options)
        example = SFTExample("caller", prepared.sample.id, step, action.messages, self.label(None, None))
        if self.target.get("tool") != discovery.SEARCH_TOOL and not caller_target_derivable(
            example, self.catalog
        ):
            raise ReplayRejected("argument_not_derivable")
        self.examples.append(example)
        return action


def mixed_sample(sample, entry, rng):
    """Sample each spoken turn independently and record its actual source."""
    history = []
    sources = {}
    asr_history = entry.get("history") or []

    def choose(key, asr, gold):
        use_asr = rng.random() < 0.2
        asr = (asr or "").strip()
        sources[key] = "gipformer" if use_asr and asr else ("gold_fallback" if use_asr else "gold")
        return asr if use_asr and asr else gold

    current = sample.audio_text or sample.raw.get("text", "")
    if sample.audio:
        current = choose("current", entry.get("current"), current)
    user_index = 0
    for turn in sample.history:
        updated = dict(turn)
        if turn.get("role") == "user":
            if turn.get("audio"):
                asr = asr_history[user_index] if user_index < len(asr_history) else None
                updated["text"] = choose(f"history:{user_index}", asr, turn.get("text") or turn.get("audio_text") or "")
            user_index += 1
        history.append(updated)
    return replace(sample, raw={**sample.raw, "audio_text": current, "history": history}), sources


def prepare(transcripts_mode, full_train=False, teacher=None):
    from transformers import AutoTokenizer

    from lifemate.train_unsloth import encode_example

    prompt_sha = configure_sft_prompt()
    root, val, split, by_id = inputs()
    if full_train:
        excluded = set(val) | set(split["dev"])
        split = {**split, "train": sorted(by_id.keys() - excluded)}
    WORK.mkdir(parents=True, exist_ok=True)
    catalog = Catalog.load(catalog_path(root, "train"))
    transcripts = read_transcript_file(transcript_path())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    reports, examples, counts = [], {"train": [], "dev": []}, Counter()
    env_factory = lambda sid: make_tool_env(root, "train", sid)
    for group, ids in split.items():
        for sid in ids:
            sample = by_id[sid]
            _, gold_calls, gold_hash = replay_gold(sample, env_factory)
            if not exact_calls(gold_calls, sample.gold_calls, catalog):
                reports.append(dict(id=sid, group=group, status="gold_replay_mismatch"))
                counts[f"{group}:gold_replay_mismatch"] += 1
                continue
            rng = random.Random(f"{SEED}:{sid}")
            config = (
                replace(CONFIG, max_steps=rng.randint(13, 20), max_searches=6)
                if full_train and group == "train"
                else CONFIG
            )
            variants = (
                ("mixed",) if transcripts_mode == "mixed" and group == "train" else
                ("gold", "gipformer")
                if transcripts_mode == "both" and group == "train"
                else (("gold" if transcripts_mode == "gold" else "gipformer"),)
            )
            for source in variants:
                caller = GoldCaller(catalog, sample.gold_calls, config=config)
                turn_sources = {}
                result = None
                replay_score = None
                try:
                    if source == "mixed":
                        incoming, turn_sources = mixed_sample(sample, transcripts.get(sid, {}), rng)
                    else:
                        incoming = (
                            prepared_sample(sample, transcripts.get(sid, {}))
                            if source == "gipformer"
                            else sample
                        )
                except ValueError:
                    status = "missing_gipformer"
                else:
                    agent = LifeMateAgent(catalog, caller, config=config)
                    try:
                        result = agent.run(incoming, env_factory(sid))
                        if discovery._E5_FAILED:
                            raise RuntimeError("Hybrid retriever did not load")
                        if not result.finished:
                            raise ReplayRejected("unfinished")
                        replay_score = score_prediction(
                            result.calls, result.state_hash, sample.gold_calls, gold_hash, catalog
                        )
                        if replay_score.score != 1.0:
                            raise ReplayRejected("score_below_1")
                        if not exact_calls(result.calls, gold_calls, catalog):
                            raise ReplayRejected("gold_calls_mismatch")
                        if any(encode_example(tokenizer, example, 4096) is None for example in caller.examples):
                            raise ReplayRejected("overlength")
                    except ReplayRejected as exc:
                        status = str(exc)
                    else:
                        status = "kept"
                        examples[group].extend(
                            dict(example.as_row(), source=source) for example in caller.examples
                        )
                gold_status = status
                if teacher is not None and group == "train" and status not in {"kept", "overlength", "missing_gipformer"}:
                    gemini_caller = Caller(
                        teacher, catalog, max_steps=config.max_steps, capture_prompts=True,
                        discovery_prompt=SFT_SYSTEM_PROMPT, user_template=DISCOVERY_USER_TEMPLATE,
                    )
                    result = LifeMateAgent(catalog, gemini_caller, config=config).run(incoming, env_factory(sid))
                    replay_score = score_prediction(result.calls, result.state_hash, sample.gold_calls, gold_hash, catalog)
                    teacher_rows = [
                        SFTExample("caller", sid, decision["step"] - 1, decision["messages"],
                                   json.dumps(decision["action"], ensure_ascii=False))
                        for decision in result.raw_outputs if decision["action"]["action"] in {"call", "ask", "finish"}
                    ]
                    clean = all(
                        record.result.get("ok") is True and
                        (record.tool != discovery.SEARCH_TOOL or record.result.get("tools"))
                        for record in result.trace
                    )
                    if (result.finished and not result.error and not result.notes and clean
                            and replay_score.score == 1.0 and writes(result.calls, catalog) == writes(gold_calls, catalog)
                            and len(teacher_rows) == len(result.raw_outputs)
                            and all(encode_example(tokenizer, row, 4096) is not None for row in teacher_rows)):
                        status = "kept"
                        examples[group].extend(dict(row.as_row(), source="gemini") for row in teacher_rows)
                    else:
                        status = "gemini_rejected"
                    source = "gemini"
                counts[f"{group}:{source}:{status}"] += 1
                reports.append(
                    dict(
                        id=sid,
                        group=group,
                        source=source,
                        status=status,
                        gold_status=gold_status,
                        actions=len(caller.examples),
                        score=replay_score.score if replay_score is not None else None,
                        turn_sources=turn_sources,
                        max_steps=config.max_steps,
                        rejected_decision=caller.target if status != "kept" else None,
                        notes=result.notes if result else [],
                    )
                )
    files = {}
    for group, group_examples in examples.items():
        # Gold and Gipformer can produce identical rendered prompts.
        unique = {
            (json.dumps(row["messages"], sort_keys=True), row["target"]): row
            for row in group_examples
        }
        kept = list(unique.values())
        path = WORK / f"{group}_examples.jsonl"
        write_rows(path, kept)
        kinds = Counter(
            "search" if action.get("tool") == discovery.SEARCH_TOOL else action["action"]
            for action in (json.loads(row["target"]) for row in kept)
        )
        files[group] = dict(count=len(kept), actions=kinds, sha256=sha(path))
    write_rows(WORK / "gold_replay_cases.jsonl", reports)
    manifest = dict(
        prompt_sha256=prompt_sha,
        model=MODEL,
        e5_model=discovery.E5_MODEL_ID,
        e5_revision=discovery.E5_REVISION,
        retriever_sha256=sha(discovery.__file__),
        max_searches=CONFIG.max_searches,
        max_steps=CONFIG.max_steps,
        top_k=discovery.SEARCH_LIMIT,
        transcripts=transcripts_mode,
        full_train=full_train,
        required_replay_score=1.0,
        gemini_model=teacher.model if teacher is not None else None,
        train_max_searches=6 if full_train else CONFIG.max_searches,
        train_max_steps=[13, 20] if full_train else [CONFIG.max_steps, CONFIG.max_steps],
        transcript_sha256=sha(transcript_path()),
        split=split,
        val_ids=val,
        counts=counts,
        examples=files,
    )
    dump(WORK / "gold_replay_manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False), files, flush=True)


def train(args):
    import torch

    from lifemate.train_unsloth import parser as trainer_parser, train_examples

    work = WORK / args.run_tag
    rank = os.environ.get("LOCAL_RANK")
    if rank is None:
        count = min(2, torch.cuda.device_count())
        if not count or args.batch % count:
            raise ValueError("Global batch must divide evenly across one or two available GPUs")
        if (work / "manifest.json").exists():
            raise ValueError(f"Use a new run tag or move the existing training output: {work}")
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={count}",
                __file__,
                *sys.argv[1:],
            ],
            check=True,
        )
        return

    manifest = read(WORK / "gold_replay_manifest.json")
    if manifest["prompt_sha256"] != configure_sft_prompt() or manifest["retriever_sha256"] != sha(discovery.__file__):
        raise ValueError("Prepared prompt or retriever differs from the current code")
    if manifest.get("required_replay_score") != 1.0:
        raise ValueError("Run prepare again to enforce the 1.0 replay score gate")
    by_group = {}
    for source, destination in (("train", "train"), ("dev", "validation")):
        path = WORK / f"{source}_examples.jsonl"
        if sha(path) != manifest["examples"][source]["sha256"]:
            raise ValueError(f"Prepared rows checksum mismatch: {path}")
        by_group[destination] = [
            SFTExample(**{key: row[key] for key in ("task", "sample_id", "step", "messages", "target")})
            for row in rows(path)
        ]
    if not all(by_group.values()):
        raise ValueError("Both train and dev need verified examples before starting GPU training")
    kinds = manifest["examples"]["train"]["actions"]
    if not all(kinds.get(kind) for kind in ("search", "call", "finish")):
        raise ValueError("Training data must cover search, call and finish")
    work.mkdir(parents=True, exist_ok=True)
    world = int(os.environ["WORLD_SIZE"])
    train_args = trainer_parser().parse_args(
        [
            "--output",
            str(work),
            "--model",
            MODEL,
            "--max-length",
            "4096",
            "--rank",
            "16",
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(SEED),
            "--gradient-accumulation",
            str(args.batch // world),
            "--max-train-steps",
            str(args.steps),
            "--save-steps",
            str(args.save_steps),
            *(["--init-adapter", str(args.init_adapter)] if args.init_adapter else []),
        ]
    )
    summary = {group: {"examples": len(group_rows)} for group, group_rows in by_group.items()}
    summary.update(
        initialization="continued_lora" if args.init_adapter else "fresh_lora",
        replay_manifest_sha256=sha(WORK / "gold_replay_manifest.json"),
        prompt_sha256=manifest["prompt_sha256"],
        global_batch=args.batch,
    )
    train_examples(train_args, by_group, summary, int(rank))


def eval_worker(rank, world, mode, ids, arms, cache_enabled):
    import torch

    from lifemate.llm import TransformersBackend

    configure()
    torch.cuda.set_device(rank)
    root, _, _, by_id = inputs()
    WORK.mkdir(parents=True, exist_ok=True)
    catalog = Catalog.load(catalog_path(root, "train"))
    transcripts = read_transcript_file(transcript_path())
    samples = {sid: prepared_sample(by_id[sid], transcripts.get(sid, {})) for sid in ids[rank::world]}
    env_factory = lambda sid: make_tool_env(root, "train", sid)
    gold = {sid: replay_gold(sample, env_factory)[2] for sid, sample in samples.items()}
    if samples:
        discovery.search_tools(catalog, next(iter(samples.values())).installed_apps, "tra cứu thông tin")
        if discovery._E5_FAILED:
            raise RuntimeError("Could not initialize hybrid search before evaluation")
    for name, adapter in arms:
        backend = TransformersBackend(
            MODEL,
            device=f"cuda:{rank}",
            dtype="float16",
            adapter_path=adapter,
            verify_adapter=False,
        )
        path = WORK / f"{mode}_{name}_rank{rank}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for sid, sample in samples.items():
                cache = PrefixCacheCaller(backend, backend) if cache_enabled else None
                env = env_factory(sid)
                caller = Caller(
                    cache or backend,
                    catalog,
                    max_new_tokens=CALLER_TOKENS,
                    max_steps=CONFIG.max_steps,
                    collect_metrics=True,
                    discovery_prompt=SFT_SYSTEM_PROMPT,
                    user_template=DISCOVERY_USER_TEMPLATE,
                )
                agent = LifeMateAgent(catalog, caller, config=CONFIG)
                torch.cuda.synchronize(rank)
                started = time.perf_counter()
                result = agent.run(sample, env)
                torch.cuda.synchronize(rank)
                elapsed = time.perf_counter() - started
                stats = cache.finish_request() if cache else {}
                if stats.get("errors") or discovery._E5_FAILED:
                    raise RuntimeError(f"{sid}: cache or hybrid retriever failed: {stats.get('errors')}")
                score = score_prediction(result.calls, result.state_hash, sample.gold_calls, gold[sid], catalog)
                # Writes are compared strictly because fuzzy scoring can hide side effects.
                predicted, expected = writes(result.calls, catalog), writes(sample.gold_calls, catalog)
                row = dict(
                    id=sid,
                    score=score.score,
                    hash_ok=score.hash_ok,
                    seconds=elapsed,
                    cache=stats,
                    wrong_writes=sum((predicted - expected).values()),
                    missing_writes=sum((expected - predicted).values()),
                    result=result.to_dict(),
                )
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
        del caller, agent, cache, backend
        torch.cuda.empty_cache()


def evaluate(args):
    import math

    import torch
    import torch.multiprocessing as mp

    _, val, split, _ = inputs()
    ids = {
        "val": val,
        "smoke-eval": split["dev"][:4],
        "dev": split["dev"],
    }[args.mode]
    arms = [("base", None)] if args.compare_base else []
    for index, path in enumerate(args.adapter):
        if not (path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing LoRA weights in {path}")
        arms.append((f"student_{index}_{path.name}", str(path.resolve())))
    if not arms:
        raise ValueError("Supply at least one --adapter path, or use --compare-base")
    world = min(2, torch.cuda.device_count())
    if not world:
        raise RuntimeError("Evaluation requires a GPU")
    WORK.mkdir(parents=True, exist_ok=True)
    mp.spawn(eval_worker, args=(world, args.mode, ids, arms, not args.no_cache), nprocs=world, join=True)
    metrics = {}
    for name, adapter in arms:
        result = [
            row
            for rank in range(world)
            for row in rows(WORK / f"{args.mode}_{name}_rank{rank}.jsonl")
        ]
        by_id = {row["id"]: row for row in result}
        if len(result) != len(ids) or set(by_id) != set(ids):
            raise RuntimeError(f"{name}: missing, duplicate or unexpected IDs")
        latency = sorted(row["seconds"] for row in result)
        hits = sum(row["cache"].get("hits", 0) for row in result)
        misses = sum(row["cache"].get("misses", 0) for row in result)
        write_rows(WORK / f"{args.mode}_{name}_traces.jsonl", (by_id[sid] for sid in ids))
        metrics[name] = dict(
            score=100 * sum(row["score"] for row in result) / len(ids),
            hashes=sum(row["hash_ok"] for row in result),
            searches=sum(
                call["tool"] == discovery.SEARCH_TOOL
                for row in result
                for call in row["result"]["trace"]
            ),
            wrong_writes=sum(row["wrong_writes"] for row in result),
            missing_writes=sum(row["missing_writes"] for row in result),
            mean_seconds=sum(latency) / len(latency),
            p95_seconds=latency[math.ceil(0.95 * len(ids)) - 1],
            max_seconds=latency[-1],
            cache_hit_rate=hits / (hits + misses) if hits + misses else None,
            over_30_ids=[row["id"] for row in result if row["seconds"] > 30],
            adapter_sha256=adapter_sha(adapter) if adapter else None,
        )
    dump(
        WORK / f"{args.mode}_report.json",
        dict(
            ids=ids,
            cache=not args.no_cache,
            metrics=metrics,
            prompt_sha256=configure_sft_prompt(),
            retriever_sha256=sha(discovery.__file__),
            traces_sha256={name: sha(WORK / f"{args.mode}_{name}_traces.jsonl") for name, _ in arms},
            live_asr=False,
        ),
    )
    print(metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "smoke-eval", "dev", "val"))
    parser.add_argument("--transcripts", choices=("gold", "gipformer", "both", "mixed"), default="both")
    parser.add_argument("--gemini-fallback", action="store_true")
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    parser.add_argument("--full-train", action="store_true", help="Prepare all eligible train IDs with a 13-20 step budget")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--run-tag", default="search_first")
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--adapter", type=Path, action="append", default=[])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--compare-base", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        teacher = None
        if args.gemini_fallback:
            from pipeline.gemini_teacher import GeminiTeacher

            teacher = GeminiTeacher(json.loads(os.environ.get("LIFEMATE_GEMINI_KEYS", "[]")), args.gemini_model)
            if not teacher.complete(
                [{"role": "system", "content": "Return one JSON object."},
                 {"role": "user", "content": 'Return {"ready":true}.'}], None,
            ):
                raise RuntimeError("Gemini preflight returned no text")
        prepare(args.transcripts, full_train=args.full_train, teacher=teacher)
    elif args.mode == "train":
        if not 0 < args.learning_rate < 1:
            parser.error("--learning-rate must be between 0 and 1")
        if min(args.steps, args.batch, args.save_steps) < 1 or args.steps % args.save_steps:
            parser.error("Use positive steps, batch, and save interval. Save interval must divide steps.")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
