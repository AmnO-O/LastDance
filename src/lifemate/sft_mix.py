"""Reproducible caller mixtures with strict sample-level separation."""
from __future__ import annotations

import copy
import json
import random

from .caller import parse_caller_action
from .data import PreparedSample, prepare_sample, read_transcript_file, load_jsonl
from .prompts import caller_messages
from .resolvers import ToolRecord
from .sft import SFTExample, caller_steps, caller_target_derivable


def planner_focus_examples(samples, catalog, path, **options):
    """Add alternate prompts using planner output observed with gold transcripts."""
    by_id = {sample.id: sample for sample in samples}
    rows = []
    seen = set()
    for rollout in load_jsonl(path):
        sid = rollout["id"]
        sample = by_id.get(sid)
        sources = rollout.get("transcript_sources", {})
        history = sources.get("history")
        if (sample is None or sources.get("current") != "gold"
                or rollout.get("transcript") != sample.audio_text
                or not isinstance(history, list)
                or len(history) != sum(m.get("role") == "user" for m in sample.history)
                or any(source != "gold" for source in history)):
            continue
        planned = rollout.get("planned_apps")
        if not isinstance(planned, list) or not all(isinstance(app, str) for app in planned):
            raise ValueError(f"Invalid planner shortlist for {sid}")
        if sid in seen:
            raise ValueError(f"Duplicate eligible planner rollout for {sid}")
        seen.add(sid)
        prepared = prepare_sample(sample)
        baseline = caller_steps(prepared, catalog, **options)
        predicted = caller_steps(prepared, catalog, initial_focus=planned, **options)
        for gold_row, predicted_row in zip(baseline, predicted):
            if (gold_row.messages != predicted_row.messages
                    and caller_target_derivable(predicted_row, catalog)):
                rows.append(predicted_row)
    return rows


def asr_examples(samples, catalog, path, **options):
    mapping = read_transcript_file(path)
    rows = []
    for sample in samples:
        bucket = mapping.get(sample.id)
        if not bucket or not bucket.get("current"):
            continue
        prepared = prepare_sample(sample)
        history = copy.deepcopy(prepared.history)
        user_index = 0
        for message in history:
            if message.get("role") == "user":
                texts = bucket.get("history", [])
                if user_index < len(texts) and texts[user_index]:
                    message["text"] = texts[user_index]
                    message.pop("audio_text", None)
                user_index += 1
        noisy = PreparedSample(sample, bucket["current"], history)
        for row in caller_steps(noisy, catalog, **options):
            if caller_target_derivable(row, catalog):
                rows.append(row)
    return rows


def recovery_examples(samples, catalog, path, **options):
    """Load reviewed continuations; a verified flag is an assertion by the curator."""
    by_id = {s.id: s for s in samples}
    rows = []
    for line, record in enumerate(load_jsonl(path), 1):
        sid = str(record.get("sample_id") or "")
        if sid not in by_id:
            raise ValueError(f"Recovery row {line}: unknown or held-out sample {sid!r}")
        if record.get("verified") is not True:
            raise ValueError(f"Recovery row {line}: continuation must be verified by its curator")
        trace = [ToolRecord(tool=r["tool"], arguments=r["arguments"], result=r["result"],
                            step=i + 1) for i, r in enumerate(record.get("trace", []))]
        notes = record.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            raise ValueError(f"Recovery row {line}: notes must be strings")
        focus = record.get("focus_apps", [])
        installed = set(catalog.installed_app_ids(by_id[sid].installed_apps))
        if not isinstance(focus, list) or any(a not in installed for a in focus):
            raise ValueError(f"Recovery row {line}: invalid app focus")
        target = record["target"]
        target_text = json.dumps(target, ensure_ascii=False)
        action = parse_caller_action(target_text, catalog, focus)
        if action.action not in {"call", "ask", "finish", "need_app"}:
            raise ValueError(f"Recovery row {line}: invalid target action")
        if "messages" not in record and not trace and not notes and action.action != "need_app":
            raise ValueError(f"Recovery row {line}: missing environment feedback")
        if action.action in {"call", "ask"} and (
            action.tool not in catalog.tools or not catalog.is_installed(action.tool, by_id[sid].installed_apps)
        ):
            raise ValueError(f"Recovery row {line}: target tool is unavailable")
        if action.action == "need_app" and action.app not in installed:
            raise ValueError(f"Recovery row {line}: target app is unavailable")
        if "messages" in record or "step" in record:
            messages = record.get("messages")
            caller_step = record.get("step")
            if type(caller_step) is not int or caller_step < 1:
                raise ValueError(f"Recovery row {line}: captured step must be a positive integer")
            if not isinstance(messages, list) or len(messages) != 2 or any(
                not isinstance(message, dict)
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message, role in zip(messages, ("system", "user"))
            ):
                raise ValueError(f"Recovery row {line}: expected captured system and user messages")
            step = caller_step - 1
        else:
            prepared = prepare_sample(by_id[sid])
            if record.get("request_text"):
                prepared.request_text = record["request_text"]
            step = len(trace)
            messages = caller_messages(prepared, catalog, focus, trace, notes=notes,
                                       step=step, max_steps=options["max_steps"],
                                       max_history_chars=options["max_history_chars"])
        if step >= options["max_steps"]:
            raise ValueError(f"Recovery row {line}: feedback exceeds caller step budget")
        rows.append(SFTExample("caller", sid, step, messages, target_text))
    return rows


def mix_examples(pools, weights, seed, *, recovery_max_copies=None):
    """Keep every gold example; sample other pools deterministically to target ratios."""
    if len(weights) != 3 or any(w < 0 for w in weights) or weights[0] <= 0:
        raise ValueError("Mixture requires three nonnegative weights and positive gold weight")
    rng = random.Random(seed)
    result = []
    counts = {}
    for name, weight in zip(("gold", "asr", "recovery"), weights):
        pool = pools[name]
        count = len(pools["gold"]) if name == "gold" else round(len(pools["gold"]) * weight / weights[0])
        if count and not pool:
            raise ValueError(f"{name} mixture is enabled but no examples are available; provide its file or set its weight to 0")
        requested = count
        if name == "recovery" and recovery_max_copies is not None:
            if recovery_max_copies < 1:
                raise ValueError("Recovery max copies must be positive")
            pool = list({(row.sample_id, row.step, row.target,
                          json.dumps(row.messages, sort_keys=True)): row for row in pool}.values())
            count = min(count, recovery_max_copies * len(pool))
        selected = list(pool) if name == "gold" else []
        while len(selected) < count:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            selected.extend(shuffled[:count - len(selected)])
        result.extend(selected)
        counts[name] = {"available": len(pool), "selected": len(selected)}
        if name == "recovery" and recovery_max_copies is not None:
            counts[name]["requested"] = requested
    rng.shuffle(result)
    return result, counts
