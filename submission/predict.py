#!/usr/bin/env python3
"""Final Search-First LifeMate inference entrypoint.

The agent starts with zero open apps.  At each step it may call
``sys.search_tools`` (at most four searches), open the returned schemas, and
execute one action.  The final configuration is:

    prompt          discovery variant of Prompt B (SHA-256 5268...)
    adapter         A100 Search-First checkpoint-100 (SHA-256 927186...)
    max_searches    4
    max_steps       12
    retrieval top_k 6

Output: ``/result/submission.jsonl`` with one line per sample:
``{"id", "tool_calls", "state_hash", "time"}``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent


def _code_root() -> Path:
    for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
        if (candidate / "src" / "lifemate").is_dir():
            return candidate
    return SCRIPT_DIR


CODE_ROOT = _code_root()
for path in (CODE_ROOT / "src", CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline.runtime_utils import (  # noqa: E402
    CALLER_TOKENS,
    DISCOVERY_PROMPT_SHA256,
    PrefixCacheCaller,
    RETRIEVER_SHA256,
    adapter_path,
    apply_prompt_b,
    verify_adapter,
)
from pipeline.common import sha  # noqa: E402

SPLITS = ("private_test", "public_test", "train")


def find_data_root_and_split(input_path: Path) -> tuple[Path, str]:
    """Find the dataset root (the folder containing ``<split>/``) and split name."""
    # Local debugging: the caller passed the split directory itself.
    for split in SPLITS:
        if input_path.name == split and (input_path / f"{split}.jsonl").is_file():
            return input_path.parent, split

    # The mounted /data directory normally contains one or more split folders.
    for root in (input_path, *input_path.parents[:2]):
        for split in SPLITS:
            split_dir = root / split
            if (split_dir / f"{split}.jsonl").is_file() and (
                split_dir / f"{split}_tools_catalog.json"
            ).is_file():
                return root, split

    # Flat layout: <input>/<split>.jsonl, where <input> is the split directory.
    for split in SPLITS:
        if (
            (input_path / f"{split}.jsonl").is_file()
            and (input_path / f"{split}_tools_catalog.json").is_file()
        ):
            return input_path.parent, split

    for jsonl in input_path.rglob("*.jsonl"):
        split = jsonl.stem
        if split in SPLITS and (jsonl.parent / f"{split}_tools_catalog.json").is_file():
            return jsonl.parent.parent, split
    raise FileNotFoundError(
        f"Could not find private_test.jsonl, public_test.jsonl or train.jsonl under {input_path}"
    )


def first_dir(explicit: str | None, candidates) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            return path
        raise FileNotFoundError(f"Not a directory: {path}")
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def load_asr(explicit: str | None):
    from lifemate.asr import SherpaOnnxTranscriber

    asr_dir = first_dir(
        explicit,
        [
            CODE_ROOT / "models" / "asr_gipformer",
            CODE_ROOT / "saved_models" / "asr_gipformer",
            CODE_ROOT / "gip",
            Path("/code/saved_models/asr_gipformer"),
        ],
    )
    if asr_dir is None:
        raise FileNotFoundError("Could not find the Gipformer ASR directory (--asr-path).")
    return SherpaOnnxTranscriber(asr_dir, num_threads=4, lowercase=True)


def select_samples(jsonl_path: Path, ids: list[str] | None, limit: int | None) -> list[dict]:
    if ids and len(set(ids)) != len(ids):
        raise ValueError("--ids contains duplicate IDs")
    requested = set(ids or [])
    rows = []
    matched = {}
    with jsonl_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not ids:
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
            elif row["id"] in requested:
                if row["id"] in matched:
                    raise ValueError(f"Duplicate sample ID in {jsonl_path}: {row['id']}")
                matched[row["id"]] = row
    if not ids:
        return rows
    missing = [sid for sid in ids if sid not in matched]
    if missing:
        raise ValueError(f"IDs missing from {jsonl_path}: {', '.join(missing)}")
    return [matched[sid] for sid in ids]


def print_debug_trace(row: dict) -> None:
    def show(label, value):
        print(f"{label}: {json.dumps(value, ensure_ascii=False, default=str)}", flush=True)

    print(f"\n=== {row['id']} ===", flush=True)
    show("ASR", row["transcript"])
    decisions = row.get("raw_outputs", [])
    records = row.get("trace", [])
    for index, decision in enumerate(decisions):
        show(f"Caller step {decision['step']} raw output", decision["raw"])
        show("Parsed action", decision["action"])
        if "finish_check" in decision:
            show("Finish check", decision["finish_check"])
        start = decision["trace_count"]
        end = decisions[index + 1]["trace_count"] if index + 1 < len(decisions) else len(records)
        for record in records[start:end]:
            show("Tool call", {"name": record["tool"], "arguments": record["arguments"]})
            show("Tool response", record["result"])
    show("Runtime notes", row.get("notes", []))
    show("Outcome", {"finished": row.get("finished", False), "error": row.get("error")})


def main() -> None:
    parser = argparse.ArgumentParser(description="LifeMate Search-First submission predictor")
    parser.add_argument("pos_data_dir", nargs="?", default=None, help="Input dataset directory")
    parser.add_argument("pos_output_file", nargs="?", default=None, help="Output submission.jsonl")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-path", default=None)
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument("--adapter-path", default=None)
    adapters.add_argument("--no-lora", action="store_true", help="Use base Qwen without loading an adapter")
    parser.add_argument("--asr-path", default=None)
    parser.add_argument("--no-asr", action="store_true", help="Use audio_text only (local debugging)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument("--ids", nargs="+", help="Run exactly these sample IDs in the given order")
    parser.add_argument("--check-finish", action="store_true", help="Check the first finish before accepting it")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--user-template-file", type=Path)
    parser.add_argument("--caller-checklist", action="store_true", help="Try concise write and finish rules")
    parser.add_argument("--asr-corrections", action="store_true", help="Add a short ASR interpretation rule")
    parser.add_argument("--debug-trace", type=Path, help="Print caller decisions and tool results and save a JSONL trace")
    parser.add_argument("--no-cache", action="store_true", help="Disable the request-local prefix cache")
    parser.add_argument("--skip-adapter-check", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow using the HF model id")
    args = parser.parse_args()

    raw_input = args.data_dir or args.pos_data_dir or os.environ.get("DATA_DIR", "/data")
    data_root, split = find_data_root_and_split(Path(raw_input).resolve())
    jsonl_path = data_root / split / f"{split}.jsonl"
    samples = select_samples(jsonl_path, args.ids, args.limit)
    output_file = Path(
        args.output or args.pos_output_file or os.environ.get("OUTPUT_FILE", "/result/submission.jsonl")
    ).resolve()
    if args.debug_trace:
        if args.debug_trace.resolve() == output_file:
            parser.error("--debug-trace and --output must be different files")
        args.debug_trace.parent.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"data root: {data_root}")
    print(f"split: {split} ({len(samples)} selected samples)")
    print(f"output: {output_file}", flush=True)

    # Per BTC rule: the split's own zaic_tools is preferred; the bundled copy is
    # only a fallback for local runs.
    sys.path.insert(0, str(data_root / split))
    try:
        from zaic_tools import ToolEnv
    except ImportError:
        sys.path.insert(0, str(CODE_ROOT / "zaic_tools"))
        from zaic_tools import ToolEnv

    from lifemate import discovery, prompts
    from lifemate.caller import Caller
    from lifemate.catalog import Catalog
    from lifemate.data import Sample
    from lifemate.llm import AdapterView, TransformersBackend
    from lifemate.runtime import AgentConfig, LifeMateAgent

    apply_prompt_b(prompts)
    prompt_sha = hashlib.sha256(prompts.discovery_system(prompts.CALLER_SYSTEM).encode()).hexdigest()
    if prompt_sha != DISCOVERY_PROMPT_SHA256:
        raise RuntimeError(f"Search-First prompt hash mismatch: {prompt_sha}")
    retriever_sha = sha(discovery.__file__)
    if retriever_sha != RETRIEVER_SHA256:
        raise RuntimeError(f"Retriever source hash mismatch: {retriever_sha}")
    print(f"prompt sha256: {prompt_sha}")
    print(f"retriever sha256: {retriever_sha}")

    started = time.perf_counter()
    transcriber = None if args.no_asr else load_asr(args.asr_path)

    model_path = args.model_path
    if model_path is None:
        for candidate in (
            CODE_ROOT / "models" / "Qwen3.5-4B",
            CODE_ROOT / "saved_models" / "Qwen3.5-4B",
            Path("/code/saved_models/Qwen3.5-4B"),
        ):
            if (candidate / "config.json").is_file():
                model_path = str(candidate)
                break
    if model_path is None:
        if not args.allow_download:
            raise FileNotFoundError(
                "Could not find a local Qwen3.5-4B model directory. "
                "Pass --model-path or use --allow-download."
            )
        model_path = "Qwen/Qwen3.5-4B"

    adapter_dir = None
    if args.no_lora:
        print("adapter: disabled (base model)")
    else:
        adapter_dir = adapter_path(args.adapter_path)
        if args.skip_adapter_check:
            print(f"adapter: {adapter_dir} (SHA check skipped)")
        else:
            print(f"adapter sha256: {verify_adapter(adapter_dir)}")

    import torch

    has_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    device = "cuda:0" if has_gpu else "cpu"
    dtype = "float16" if has_gpu else "float32"
    backend = TransformersBackend(
        str(model_path),
        device=device,
        dtype=dtype,
        adapter_path=str(adapter_dir) if adapter_dir else None,
        verify_adapter=False,
    )
    caller_view = AdapterView(backend, enabled=not args.no_lora)
    cache = None if args.no_cache else PrefixCacheCaller(backend, caller_view)
    catalog = Catalog.load(data_root / split / f"{split}_tools_catalog.json")
    config = AgentConfig(
        top_k=discovery.SEARCH_LIMIT,
        max_steps=12,
        tool_discovery=True,
        max_searches=4,
        verify_finish=args.check_finish,
    )
    discovery_prompt = None
    if args.asr_corrections:
        discovery_prompt = prompts.discovery_system(prompts.CALLER_SYSTEM) + (
            "\nASR may omit punctuation. Follow clear corrections (e.g. à không). "
            "If ambiguity changes a write, ask before acting."
        )
        print("effective caller prompt sha256:", hashlib.sha256(discovery_prompt.encode()).hexdigest())
    if args.caller_checklist:
        discovery_prompt = discovery_prompt or prompts.discovery_system(prompts.CALLER_SYSTEM)
        discovery_prompt = discovery_prompt.replace(
            "ACTION RULES\n",
            "ACTION RULES\n"
            "- Before a write, follow the user's latest clear instruction, including corrections.\n"
            "- Before finish, match each requested action to successful tool results.\n",
            1,
        )
        print("checklist prompt sha256:", hashlib.sha256(discovery_prompt.encode()).hexdigest())
    if args.system_prompt_file:
        discovery_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    if discovery_prompt is not None:
        print("active system prompt sha256:", hashlib.sha256(discovery_prompt.encode()).hexdigest())
    user_template = args.user_template_file.read_text(encoding="utf-8") if args.user_template_file else None
    caller = Caller(
        cache or caller_view,
        catalog,
        max_new_tokens=CALLER_TOKENS,
        max_history_chars=config.max_history_chars,
        max_steps=config.max_steps,
        collect_metrics=True,
        capture_prompts=bool(args.debug_trace),
        discovery_prompt=discovery_prompt,
        user_template=user_template,
    )
    agent = LifeMateAgent(catalog, caller, config=config)
    print(f"load model: {int((time.perf_counter() - started) * 1000)} ms", flush=True)

    predict_started = time.perf_counter()
    cache_hits = cache_misses = 0

    with ExitStack() as stack:
        out = stack.enter_context(output_file.open("w", encoding="utf-8"))
        debug = stack.enter_context(args.debug_trace.open("w", encoding="utf-8")) if args.debug_trace else None
        for index, item in enumerate(samples, 1):
            sid = item["id"]
            sample_started = time.perf_counter()

            current_audio = item.get("audio")
            if current_audio and transcriber is not None:
                audio_file = data_root / split / current_audio
                current_text = transcriber.transcribe(audio_file) if audio_file.is_file() else ""
            else:
                current_text = item.get("audio_text") or item.get("text") or ""

            history = []
            for message in item.get("history") or []:
                message = dict(message)
                if (
                    message.get("role") == "user"
                    and not message.get("text")
                    and message.get("audio")
                    and transcriber is not None
                ):
                    audio_file = data_root / split / message["audio"]
                    message["text"] = (
                        transcriber.transcribe(audio_file) if audio_file.is_file() else ""
                    )
                history.append(message)

            sample = Sample(
                raw={**item, "audio_text": current_text, "history": history},
                split=split,
                root=data_root,
            )
            env = ToolEnv(split, sid, root=data_root)
            if cache is not None:
                cache.finish_request()
            started_sample = time.perf_counter()
            result = None
            run_error = None
            try:
                result = agent.run(sample, env)
            except Exception as exc:  # keep one bad sample from stopping the submission
                run_error = f"{type(exc).__name__}: {exc}"
                print(f"[WARN] {sid}: {run_error}", file=sys.stderr, flush=True)
            elapsed = time.perf_counter() - started_sample
            if cache is not None:
                stats = cache.finish_request()
                cache_hits += stats["hits"]
                cache_misses += stats["misses"]
                if stats["errors"]:
                    print(f"[WARN] {sid}: prefix cache errors: {stats['errors']}", file=sys.stderr, flush=True)

            tool_calls = env.get_tool_calls()
            state_hash = env.get_state_hash()
            submission = {
                "id": sid,
                "tool_calls": tool_calls,
                "state_hash": state_hash,
                "time": int((time.perf_counter() - sample_started) * 1000),
            }
            out.write(json.dumps(submission, ensure_ascii=False) + "\n")
            out.flush()
            if debug is not None:
                row = result.to_dict() if result is not None else {
                    "id": sid, "transcript": current_text, "error": run_error,
                    "calls": tool_calls, "state_hash": state_hash,
                }
                debug.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                debug.flush()
                print_debug_trace(row)
            print(
                f"[{index}/{len(samples)}] {sid}: {len(tool_calls)} calls, "
                f"{int(elapsed * 1000)} ms, hash={state_hash[:12]}",
                flush=True,
            )

    total = time.perf_counter() - predict_started
    print(
        f"predict: {len(samples)} samples, {int(total * 1000)} ms, "
        f"{total / max(1, len(samples)) * 1000:.1f} ms/sample",
        flush=True,
    )
    if cache is not None and cache_hits + cache_misses:
        print(f"prefix cache: {cache_hits}/{cache_hits + cache_misses} hits", flush=True)


if __name__ == "__main__":
    main()
