"""Command line interface for training-split verification, dev evaluation and submission."""
from __future__ import annotations

import argparse
import multiprocessing
import json
import os
import random
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .asr import CommandTranscriber, FasterWhisperTranscriber, SherpaOnnxTranscriber
from .caller import Caller
from .catalog import Catalog
from .data import (
    FixedTranscripts,
    dataset_stats,
    find_data_root,
    find_validation_ids,
    load_samples,
    read_ids,
    read_transcript_file,
    stratified_dev_ids,
    catalog_path,
)
from .llm import AdapterView, MockBackend, TransformersBackend
from .rewriter import Rewriter
from .runtime import AgentConfig, LifeMateAgent, RunResult, make_tool_env
from .scoring import aggregate, evaluate_predictions, replay_gold, score_prediction
from .text import unique_preserve


def _stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _load_ids(args: argparse.Namespace) -> list[str] | None:
    ids: list[str] = []
    if getattr(args, "ids", None):
        ids.extend(x.strip() for x in str(args.ids).split(",") if x.strip())
    if getattr(args, "ids_file", None):
        ids.extend(read_ids(args.ids_file))
    return unique_preserve(ids) or None


def _add_common_run_args(p: argparse.ArgumentParser, *, allow_split: bool = True) -> None:
    p.add_argument("--root", default=None, help="folder containing train/ and public_test/")
    if allow_split:
        p.add_argument("--split", default=None, help="split name (default: train for eval, public_test for predict)")
    p.add_argument("--ids", default=None, help="comma-separated sample ids")
    p.add_argument("--ids-file", default=None, help="JSON list or one id per line")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-reprompts", type=int, default=3)
    p.add_argument("--caller-max-history-chars", type=int, default=10000)
    p.add_argument("--no-auto-read", action="store_true")
    p.add_argument("--no-auto-ask", action="store_true")
    p.add_argument("--save-traces", default=None, help="write full per-sample traces to this JSONL")
    p.add_argument("--quiet", action="store_true")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["transformers", "mock"], default="transformers")
    p.add_argument("--model", default=os.environ.get("LIFEMATE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default=os.environ.get("LIFEMATE_DEVICE", "cuda"))
    p.add_argument("--dtype", default=os.environ.get("LIFEMATE_DTYPE", "bfloat16"))
    p.add_argument("--max-input-tokens", type=int, default=int(os.environ.get("LIFEMATE_MAX_INPUT_TOKENS", "15200")))
    p.add_argument("--adapter", default=os.environ.get("LIFEMATE_ADAPTER") or None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--tool-discovery", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--caller-max-new-tokens", type=int, default=320)
    p.add_argument("--mock-response", default='{"action":"finish"}',
                   help="fixed caller response when --backend mock")


def _build_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return MockBackend(args.mock_response)
    return TransformersBackend(
        args.model,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        adapter_path=getattr(args, "adapter", None),
        rewriter_adapter_path=getattr(args, "gip_rewriter_adapter", None),
        local_files_only=args.local_files_only,
    )


def _build_transcriber(args: argparse.Namespace):
    if getattr(args, "transcripts", None):
        return FixedTranscripts(read_transcript_file(args.transcripts))
    if getattr(args, "asr_command", None):
        return CommandTranscriber(args.asr_command, cache_path=getattr(args, "asr_cache", None))
    if getattr(args, "asr_model", None):
        return FasterWhisperTranscriber(
            args.asr_model,
            device=getattr(args, "asr_device", "cuda"),
            compute_type=getattr(args, "asr_compute_type", "float16"),
            cache_path=getattr(args, "asr_cache", None),
            local_files_only=getattr(args, "local_files_only", False),
        )
    return None


def _select_samples(args: argparse.Namespace, split: str):
    root = Path(args.root) if args.root else find_data_root()
    ids = _load_ids(args)
    all_samples = load_samples(root, split, ids=None if (ids or args.limit) else None)
    if ids:
        wanted = set(ids)
        samples = [s for s in all_samples if s.id in wanted]
        order = {sid: i for i, sid in enumerate(ids)}
        samples.sort(key=lambda s: order.get(s.id, 10**9))
    elif args.dev:
        chosen = stratified_dev_ids(all_samples, n=int(args.dev), seed=int(args.seed))
        wanted = set(chosen)
        order = {sid: i for i, sid in enumerate(chosen)}
        samples = [s for s in all_samples if s.id in wanted]
        samples.sort(key=lambda s: order.get(s.id, 10**9))
    else:
        samples = all_samples
    if args.limit is not None:
        samples = samples[: args.limit]
    return root, samples


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def cmd_stats(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else find_data_root()
    split = args.split or "train"
    samples = load_samples(root, split)
    stats = dataset_stats(samples)
    stats.update(
        {
            "root": str(root),
            "split": split,
            "catalog": str(catalog_path(root, split)),
            "has_labels": all(s.has_label for s in samples),
        }
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_dev_ids(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else find_data_root()
    split = args.split or "train"
    samples = load_samples(root, split)
    ids = stratified_dev_ids(samples, n=args.n, seed=args.seed)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ids, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else find_data_root()
    split = args.split or "train"
    _, samples = _select_samples(args, split)
    catalog = Catalog.load(catalog_path(root, split))
    env_factory = lambda sid: make_tool_env(root, split, sid)  # noqa: E731
    mismatches: list[dict[str, Any]] = []
    score_sum = 0.0
    for i, sample in enumerate(samples, 1):
        env, calls, target_hash = replay_gold(sample, env_factory)
        state = env.get_state()
        for app, final in (sample.label.get("final_state") or {}).items():
            if state.get(app) != final:
                mismatches.append({"id": sample.id, "app": app})
        score = score_prediction(sample.gold_calls, target_hash, sample.gold_calls, target_hash, catalog)
        score_sum += score.score
        if not args.quiet and (i % 100 == 0 or i == len(samples)):
            print(f"verified {i}/{len(samples)}", file=sys.stderr)
    summary = {
        "split": split,
        "samples": len(samples),
        "perfect_call_score": score_sum / len(samples) if samples else 0.0,
        "final_state_mismatches": len(mismatches),
        "examples": mismatches[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 1


def _run_samples(args: argparse.Namespace, samples, root: Path, split: str, catalog: Catalog, transcriber) -> list[Any]:
    return list(_iter_runs(args, samples, root, split, catalog, transcriber))


def _prepare_gip_sample(sample, finetuned, base=None, rewriter=None):
    """Replace spoken turns explicitly, even when train gold text is present."""
    turns = []

    def transcribe(relative, turn):
        path = sample.audio_path(relative)
        primary = finetuned.transcribe(path)
        if not primary:
            raise ValueError(f"{sample.id} {turn}: empty fine-tuned Gipformer transcript")
        row = {"turn": turn, "finetuned": primary}
        if rewriter is not None:
            alternate = base.transcribe(path)
            if not alternate:
                raise ValueError(f"{sample.id} {turn}: empty base Gipformer transcript")
            result = rewriter.rewrite([primary, alternate])
            row.update(base=alternate, rewrite=result.as_dict())
            primary = result.text
        turns.append(row)
        return primary

    history = []
    for index, message in enumerate(sample.history):
        item = dict(message)
        if item.get("role") == "user" and item.get("audio"):
            item["text"] = transcribe(str(item["audio"]), f"history:{index}")
            item["audio_text"] = item["text"]
        history.append(item)
    current = transcribe(sample.audio, "current") if sample.audio else (
        str(sample.raw.get("text") or sample.audio_text)
    )
    return replace(sample, raw={**sample.raw, "audio_text": current, "history": history}), turns


def _add_gip_args(parser):
    parser.add_argument("--gip-finetuned-dir", type=Path,
                        help="Decode spoken turns from WAV with this Gipformer ONNX model")
    parser.add_argument("--gip-base-dir", type=Path,
                        help="Also decode with this model and rewrite the two candidates using the base LLM")
    parser.add_argument("--gip-rewriter-adapter", type=Path,
                        help="Optional LoRA adapter directory for ASR rewriter")


def _iter_runs(args, samples, root, split, catalog, transcriber):
    ft_dir = getattr(args, "gip_finetuned_dir", None)
    base_dir = getattr(args, "gip_base_dir", None)
    if base_dir and not ft_dir:
        raise SystemExit("--gip-base-dir requires --gip-finetuned-dir")
    if ft_dir and transcriber is not None:
        raise SystemExit("Gipformer model directories cannot be combined with another ASR input")
    backend = _build_backend(args)
    finetuned = SherpaOnnxTranscriber(ft_dir) if ft_dir else None
    base = SherpaOnnxTranscriber(base_dir) if base_dir else None
    rewriter_adapter = getattr(args, "gip_rewriter_adapter", None)
    if base:
        if rewriter_adapter:
            rewriter_backend = AdapterView(backend, enabled=True, adapter_name="rewriter")
        else:
            rewriter_backend = AdapterView(backend, enabled=False)
        rewriter = Rewriter(rewriter_backend, seed=args.seed)
    else:
        rewriter = None
    config = AgentConfig(
        max_steps=args.max_steps,
        max_reprompts=args.max_reprompts,
        auto_read=not args.no_auto_read,
        auto_ask=not args.no_auto_ask,
        max_history_chars=args.caller_max_history_chars,
        tool_discovery=args.tool_discovery,
        max_searches=4,
    )
    caller = Caller(
        backend,
        catalog,
        max_new_tokens=args.caller_max_new_tokens,
        max_history_chars=args.caller_max_history_chars,
        max_steps=args.max_steps,
        seed=args.seed,
        capture_prompts=getattr(args, "command", None) == "collect-rollouts",
    )
    agent = LifeMateAgent(catalog, caller, transcriber=transcriber, config=config)
    started = time.perf_counter()
    for i, sample in enumerate(samples, 1):
        sample_started = time.perf_counter()
        env = None
        try:
            env = make_tool_env(root, split, sample.id)
            if finetuned is None:
                run = agent.run(sample, env)
            else:
                asr_started = time.perf_counter()
                prepared, turns = _prepare_gip_sample(sample, finetuned, base, rewriter)
                asr_seconds = time.perf_counter() - asr_started
                run = agent.run(prepared, env)
                run.seconds += asr_seconds
                run.asr = {"seconds": asr_seconds, "turns": turns}
        except Exception as exc:
            if args.command != "predict":
                raise
            calls, state_hash = [], ""
            if env is not None:
                try:
                    calls = env.get_tool_calls()
                    state_hash = env.get_state_hash()
                except Exception as state_error:
                    print(f"{sample.id}: could not read simulator state: {state_error}", file=sys.stderr)
            run = RunResult(
                sample_id=sample.id, calls=calls, state_hash=state_hash,
                error=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - sample_started,
            )
            print(f"{sample.id}: {run.error}", file=sys.stderr)
        if not args.quiet:
            print(
                f"[{i}/{len(samples)}] {sample.id} calls={len(run.calls)} "
                f"finished={run.finished} err={run.error or '-'} {run.seconds:.1f}s",
                file=sys.stderr,
            )
        yield run
    if not args.quiet:
        print(f"total {time.perf_counter() - started:.1f}s", file=sys.stderr)


def cmd_collect_rollouts(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else find_data_root()
    validation_path = args.validation_ids or find_validation_ids(root)
    if not validation_path:
        raise SystemExit("Provide --validation-ids to exclude the holdout.")
    held_out = set(read_ids(validation_path))
    all_samples = load_samples(root, "train")
    known = {sample.id: sample for sample in all_samples}
    if not held_out or held_out - known.keys():
        raise SystemExit("Validation IDs must be nonempty and belong to the train split.")
    ids = _load_ids(args)
    if ids and (set(ids) - known.keys() or set(ids) & held_out):
        raise SystemExit("Requested IDs must exist in train and exclude validation IDs.")
    if ids is None and (args.ids is not None or args.ids_file is not None):
        raise SystemExit("The supplied ID selection is empty.")
    samples = [known[sid] for sid in ids] if ids else [s for s in all_samples if s.id not in held_out]
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples or not all(s.has_label for s in samples):
        raise SystemExit("Select at least one labeled training sample.")
    gold_texts = {sample.id: sample.audio_text for sample in samples}
    sample_sources: dict[str, dict[str, Any]] = {}
    if args.transcripts:
        transcripts = read_transcript_file(args.transcripts)
        missing = [s.id for s in samples if not transcripts.get(s.id, {}).get("current", "").strip()]
        if missing:
            raise SystemExit(f"Missing current ASR transcripts for {len(missing)} samples, including {missing[:5]}")
        ratio = max(0.0, min(1.0, float(getattr(args, "transcript_ratio", 0.2))))
        rng = random.Random(args.seed)
        updated = []
        for sample in samples:
            bucket = transcripts[sample.id]
            history = []
            user_index = 0
            history_sources: list[str] = []
            for message in sample.history:
                message = dict(message)
                if message.get("role") == "user":
                    texts = bucket.get("history", [])
                    use_asr = (rng.random() < ratio) and user_index < len(texts) and bool(texts[user_index].strip())
                    if use_asr:
                        message["text"] = texts[user_index]
                        history_sources.append("gipformer")
                    else:
                        history_sources.append("gold")
                    user_index += 1
                history.append(message)
            use_current_asr = (rng.random() < ratio) and bool(bucket.get("current", "").strip())
            current_text = bucket["current"] if use_current_asr else sample.audio_text
            current_source = "gipformer" if use_current_asr else "gold"
            sample_sources[sample.id] = {
                "current": current_source,
                "history": history_sources,
            }
            updated.append(replace(sample, raw={**sample.raw, "audio_text": current_text, "history": history}))
        samples = updated
    catalog = Catalog.load(catalog_path(root, "train"))
    output = Path(args.save_traces)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    review_count = 0
    try:
        stream = output.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"Trace file already exists: {output}. Choose a new --save-traces path.") from None
    with stream:
        for sample, run in _rollout_runs(args, samples, root, catalog):
            _, _, gold_hash = replay_gold(sample, lambda sid: make_tool_env(root, "train", sid))
            score = score_prediction(run.calls, run.state_hash, sample.gold_calls, gold_hash, catalog)
            failed_tools = [r.step for r in run.trace if isinstance(r.result, dict) and r.result.get("ok") is False]
            review = score.score < 1 or not run.finished or bool(run.error or failed_tools) or any(
                action["action"]["action"] == "invalid" for action in run.raw_outputs
            )
            row = {**run.to_dict(), "gold_transcript": gold_texts[sample.id],
                   "transcript_sources": sample_sources.get(sample.id, {"current": "gold", "history": []}),
                   "gold_calls": sample.gold_calls, "gold_state_hash": gold_hash,
                   "score": score.as_dict(), "failed_tool_steps": failed_tools,
                   "needs_review": review}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            scores.append(score)
            review_count += int(review)
    print(json.dumps({**aggregate(scores), "needs_review": review_count, "traces": str(output)}, indent=2))
    return 0


def _rollout_worker(args, samples, root, catalog, connection):
    try:
        import torch

        torch.cuda.set_device(args.device)
        for run in _iter_runs(args, samples, root, "train", catalog, None):
            connection.send(run)
    finally:
        connection.close()


def _rollout_runs(args, samples, root, catalog):
    if args.backend == "transformers" and args.device == "cuda" and len(samples) > 1:
        import torch

        use_two = not args.single_gpu and torch.cuda.device_count() >= 2
    else:
        use_two = False
    if not use_two:
        yield from zip(samples, _iter_runs(args, samples, root, "train", catalog, None))
        return

    # Spawn independent model replicas. Only the parent scores and writes traces.
    context = multiprocessing.get_context("spawn")
    workers = []
    by_id = {sample.id: sample for sample in samples}
    try:
        for rank in range(2):
            worker_args = argparse.Namespace(**{**vars(args), "device": f"cuda:{rank}"})
            receiver, sender = context.Pipe(duplex=False)
            worker = context.Process(target=_rollout_worker, args=(
                worker_args, samples[rank::2], root, catalog, sender,
            ))
            try:
                worker.start()
            except BaseException:
                receiver.close()
                raise
            finally:
                sender.close()
            workers.append((worker, receiver))
        from multiprocessing.connection import wait

        pending = {receiver: worker for worker, receiver in workers}
        while pending:
            for receiver in wait(list(pending)):
                try:
                    run = receiver.recv()
                except EOFError:
                    worker = pending.pop(receiver)
                    worker.join()
                    if worker.exitcode:
                        raise RuntimeError(f"GPU rollout worker exited with code {worker.exitcode}. Partial traces were kept.")
                else:
                    yield by_id[run.sample_id], run
    finally:
        for worker, receiver in workers:
            receiver.close()
            if worker.is_alive():
                worker.terminate()
            worker.join()


def _ensure_validation_ids(args: argparse.Namespace) -> Path | None:
    """Use the shared validation id file unless the caller chose a subset."""
    if any((getattr(args, "ids", None), getattr(args, "ids_file", None), getattr(args, "dev", None))):
        return None
    path = find_validation_ids(args.root or find_data_root())
    if path is not None:
        args.ids_file = str(path)
    return path


def cmd_eval(args: argparse.Namespace) -> int:
    validation_file = _ensure_validation_ids(args)
    root, samples = _select_samples(args, args.split or "train")
    split = args.split or "train"
    catalog = Catalog.load(catalog_path(root, split))
    transcriber = _build_transcriber(args)
    runs = _run_samples(args, samples, root, split, catalog, transcriber)
    predictions = {r.sample_id: r.submission() for r in runs}
    if args.output:
        _write_jsonl(args.output, predictions.values())
    if args.save_traces:
        _write_jsonl(args.save_traces, (r.to_dict() for r in runs))
    env_factory = lambda sid: make_tool_env(root, split, sid)  # noqa: E731
    report = evaluate_predictions(samples, predictions, catalog, env_factory)
    report["timing"] = {
        "total_seconds": sum(r.seconds for r in runs),
        "mean_seconds": sum(r.seconds for r in runs) / (len(runs) or 1),
        "max_seconds": max((r.seconds for r in runs), default=0.0),
    }
    report["errors"] = sum(1 for r in runs if r.error)
    if args.gip_finetuned_dir is not None:
        turns = [turn for run in runs if run.asr for turn in run.asr["turns"]]
        rewrites = [turn["rewrite"] for turn in turns if "rewrite" in turn]
        report["asr"] = {
            "spoken_turns": len(turns),
            "total_seconds": sum(run.asr["seconds"] for run in runs if run.asr),
            "rewrite_attempts": len(rewrites),
            "rewrite_fallbacks": sum(item["fallback"] for item in rewrites),
        }
    if validation_file is not None:
        report["validation_ids_file"] = str(validation_file)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report.pop("details", None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    split = args.split or "public_test"
    root, samples = _select_samples(args, split)
    catalog = Catalog.load(catalog_path(root, split))
    transcriber = _build_transcriber(args)
    if transcriber is None and args.gip_finetuned_dir is None:
        # Train already has transcripts; public/private must provide one.
        if samples and not samples[0].audio_text:
            raise SystemExit(
                "This split has no audio_text. Pass --transcripts file.json, --asr-model, or --asr-command."
            )
    output = Path(args.output)
    traces = Path(args.save_traces) if args.save_traces else None
    if traces and traces.resolve() == output.resolve():
        raise SystemExit("--output and --save-traces must use different files")
    output.parent.mkdir(parents=True, exist_ok=True)
    if traces:
        traces.parent.mkdir(parents=True, exist_ok=True)
    count = empty = calls = errors = finished = 0
    seconds = 0.0
    with ExitStack() as stack:
        predictions = stack.enter_context(output.open("w", encoding="utf-8"))
        trace_stream = stack.enter_context(traces.open("w", encoding="utf-8")) if traces else None
        for run in _iter_runs(args, samples, root, split, catalog, transcriber):
            predictions.write(json.dumps(run.submission(), ensure_ascii=False) + "\n")
            predictions.flush()
            if trace_stream is not None:
                trace_stream.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")
                trace_stream.flush()
            count += 1
            empty += not run.calls
            calls += len(run.calls)
            errors += bool(run.error)
            finished += run.finished
            seconds += run.seconds
    summary = {
        "split": split,
        "samples": count,
        "output": str(output),
        "empty_predictions": empty,
        "mean_calls": calls / (count or 1),
        "errors": errors,
        "finished": finished,
        "mean_seconds": seconds / (count or 1),
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lifemate", description="ZAC 2026 LifeMate Search-First agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="print dataset statistics")
    p_stats.add_argument("--root", default=None)
    p_stats.add_argument("--split", default="train")
    p_stats.set_defaults(func=cmd_stats)

    p_dev = sub.add_parser("dev-ids", help="create a deterministic stratified dev id list")
    p_dev.add_argument("--root", default=None)
    p_dev.add_argument("--split", default="train")
    p_dev.add_argument("--n", type=int, default=100)
    p_dev.add_argument("--seed", type=int, default=13)
    p_dev.add_argument("--output", default=None)
    p_dev.set_defaults(func=cmd_dev_ids)

    p_verify = sub.add_parser("verify", help="replay labeled train calls and verify local scoring")
    p_verify.add_argument("--root", default=None)
    p_verify.add_argument("--split", default="train")
    p_verify.add_argument("--ids", default=None)
    p_verify.add_argument("--ids-file", default=None)
    p_verify.add_argument("--limit", type=int, default=None)
    p_verify.add_argument("--quiet", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_eval = sub.add_parser("eval", help="run the agent on labeled train/dev and score it")
    _add_common_run_args(p_eval)
    _add_model_args(p_eval)
    p_eval.add_argument("--dev", type=int, default=None, help="use a deterministic stratified dev subset")
    _add_gip_args(p_eval)
    p_eval.add_argument("--transcripts", default=None)
    p_eval.add_argument("--asr-model", default=None)
    p_eval.add_argument("--asr-command", default=None)
    p_eval.add_argument("--asr-cache", default=None)
    p_eval.add_argument("--asr-device", default="cuda")
    p_eval.add_argument("--asr-compute-type", default="float16")
    p_eval.add_argument("--output", default="predictions.jsonl")
    p_eval.add_argument("--report", default="eval_report.json")
    p_eval.set_defaults(func=cmd_eval)

    p_rollout = sub.add_parser("collect-rollouts", help="Collect scored training traces for recovery review")
    _add_common_run_args(p_rollout, allow_split=False)
    _add_model_args(p_rollout)
    p_rollout.add_argument("--validation-ids", help="Holdout IDs to exclude")
    p_rollout.add_argument("--transcripts", help="Gipformer JSON or JSONL for selected training IDs")
    p_rollout.add_argument("--single-gpu", action="store_true", help="Disable automatic two-GPU sharding")
    p_rollout.add_argument(
        "--transcript-ratio",
        "--asr-ratio",
        type=float,
        default=0.2,
        help="Probability of substituting Gipformer ASR per user utterance in current and history (default: 0.2)",
    )
    p_rollout.set_defaults(func=cmd_collect_rollouts, save_traces="training_rollouts.jsonl", split="train",
                           model=os.environ.get("LIFEMATE_MODEL", "Qwen/Qwen3.5-4B"))

    p_predict = sub.add_parser("predict", help="run on public/private test and write submission.jsonl")
    _add_common_run_args(p_predict)
    _add_model_args(p_predict)
    _add_gip_args(p_predict)
    p_predict.add_argument("--transcripts", default=None)
    p_predict.add_argument("--asr-model", default=None)
    p_predict.add_argument("--asr-command", default=None)
    p_predict.add_argument("--asr-cache", default=None)
    p_predict.add_argument("--asr-device", default="cuda")
    p_predict.add_argument("--asr-compute-type", default="float16")
    p_predict.add_argument("--output", default="submission.jsonl")
    p_predict.add_argument("--report", default="predict_report.json")
    p_predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    _stdout_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "dev"):
        args.dev = None
    return int(args.func(args))
