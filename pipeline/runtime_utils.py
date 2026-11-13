"""Final Search-First runtime helpers shared by training, evaluation, and inference.

This module intentionally contains no experiment code.  It fixes the prompt,
the adapter, the ASR transcript source, and the request-local prefix cache.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

from .common import REPO, sha

MODEL = os.environ.get("LIFEMATE_MODEL", "Qwen/Qwen3.5-4B")
CALLER_TOKENS = 220

# Prompt B is the final caller prompt.  The discovery variant below is the
# Search-First system prompt built from it by prompts.discovery_system().
PROMPT_B_SHA256 = "76eb6f20648c71b4d243932a9a18628b998e6373a6f2ffbb7d57eb468f3810b2"
DISCOVERY_PROMPT_SHA256 = "5268fcf5470fdfb2adb3eabb0eeb857cfc060f4413b81f8d3d4974bd4ec111ed"

# Final A100 Search-First adapter (continued LoRA, checkpoint-100).
ADAPTER_SHA256 = "927186df0ff6c7bb9deca0f1a9f960d14ff6f246f7c101e16069d7e974c9f8c8"

# Retriever source hash; the manifest records the same value.
RETRIEVER_SHA256 = "05b8ba54da9b85d3f95853c7ce7fd3e3f5ae15503ae1e1754b086c37529a784b"

RULE_A = (
    "- Copy ids exactly from a read result. Never invent ids. If an id is needed but you have not "
    "read it, call the app's read/search tool first (e.g. contacts.search, calendar.list_events, "
    "smart_home.get_status, alarm.list)."
)
RULE_B = (
    "- Copy ids exactly from a read result. Never invent ids. If an id is needed, call the app's "
    "read/search tool first, UNLESS a visible result in the execution trace for the current request "
    "already confirms the same target and entity type and provides all fields needed for the next "
    "action (in which case, skip the duplicate lookup and proceed directly). For actions based "
    "solely on conversation history without a read in the current request, still execute the "
    "required read first. Always execute a fresh read if the decision depends on dynamic state or "
    "if the user explicitly asks to check again. If target identity is ambiguous, ask for "
    "clarification even if names appear similar."
)


def apply_prompt_b(prompts) -> str:
    """Replace rule A with rule B exactly once and verify the resulting hash."""
    if prompts.CALLER_SYSTEM.count(RULE_A) != 1:
        raise RuntimeError("Expected exactly one caller rule A in prompts.py")
    prompt = prompts.CALLER_SYSTEM.replace(RULE_A, RULE_B)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest != PROMPT_B_SHA256:
        raise RuntimeError(f"Caller prompt hash mismatch: {digest}")
    prompts.CALLER_SYSTEM = prompt
    return digest


def configure():
    """Apply Prompt B and return the final Search-First discovery prompt hash."""
    import torch
    from lifemate import prompts

    torch.set_num_threads(2)
    if hashlib.sha256(prompts.CALLER_SYSTEM.encode()).hexdigest() != PROMPT_B_SHA256:
        apply_prompt_b(prompts)
    digest = hashlib.sha256(prompts.discovery_system(prompts.CALLER_SYSTEM).encode()).hexdigest()
    if digest != DISCOVERY_PROMPT_SHA256:
        raise RuntimeError(f"Search-First prompt hash mismatch: {digest}")
    return digest


def prepared_sample(sample, entry):
    """Replace ASR text for the final request and every user history turn."""
    current = (entry or {}).get("current", "").strip()
    if sample.audio and not current:
        raise ValueError(f"{sample.id}: missing current Gipformer transcript")
    history_gip = (entry or {}).get("history") or []
    history = []
    user_index = 0
    for turn in sample.history:
        updated = dict(turn)
        if turn.get("role") == "user":
            if turn.get("audio"):
                text = history_gip[user_index].strip() if user_index < len(history_gip) else ""
                if not text:
                    raise ValueError(f"{sample.id}: missing Gipformer history turn {user_index}")
                updated["text"] = text
            user_index += 1
        history.append(updated)
    text = current if sample.audio else sample.audio_text or sample.raw.get("text", "")
    return replace(sample, raw={**sample.raw, "audio_text": text, "history": history})


def adapter_path(explicit: str | Path | None = None) -> Path:
    """Locate the final adapter directory without silently picking another run."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("LIFEMATE_ADAPTER"):
        candidates.append(Path(os.environ["LIFEMATE_ADAPTER"]))
    candidates += [
        REPO / "models" / "caller_adapter",
        REPO / "models" / "search_first_adapter",
        REPO / "saved_models" / "caller_adapter",
        REPO / "saved_models" / "search_first_adapter",
        Path("/code/saved_models/caller_adapter"),
    ]
    for path in candidates:
        if (path / "adapter_model.safetensors").is_file():
            return path
    raise FileNotFoundError(
        "Could not find the final Search-First adapter. Copy checkpoint-100 to "
        f"models/caller_adapter or set LIFEMATE_ADAPTER. Expected SHA-256: {ADAPTER_SHA256}"
    )


def adapter_sha(path: str | Path) -> str:
    return sha(Path(path) / "adapter_model.safetensors")


def verify_adapter(path: str | Path, expected: str = ADAPTER_SHA256) -> str:
    digest = adapter_sha(path)
    if expected and digest != expected:
        raise RuntimeError(
            f"Adapter SHA-256 mismatch: expected {expected}, got {digest} ({path})"
        )
    return digest


class PrefixCacheCaller:
    """Request-local snapshot before the execution trace, with exact token matching."""

    marker = "\nEXECUTION TRACE FOR THE CURRENT REQUEST\n"

    def __init__(self, backend, delegate):
        self.backend = backend
        self.delegate = delegate
        self.snapshot = None
        self.prefix = None
        self.hits = self.misses = self.reused_tokens = 0
        self.errors = []
        self.last_hit = False
        self.last_reused_tokens = 0

    @property
    def last_generation(self):
        return {
            **getattr(self.delegate, "last_generation", {}),
            "cache_hit": self.last_hit,
            "reused_tokens": self.last_reused_tokens,
        }

    def finish_request(self):
        result = {
            "hits": self.hits,
            "misses": self.misses,
            "reused_tokens": self.reused_tokens,
            "errors": self.errors,
        }
        self.snapshot = self.prefix = None
        self.hits = self.misses = self.reused_tokens = 0
        self.errors = []
        return result

    def complete(self, messages, settings):
        import copy

        self.last_hit = False
        self.last_reused_tokens = 0

        backend = self.backend
        rendered = backend._render(messages)
        prefix_text, marker, _ = rendered.partition(self.marker)
        if not marker:
            self.misses += 1
            return self.delegate.complete(messages, settings)
        tokens = backend.tokenizer.encode(rendered, add_special_tokens=False)
        prefix = backend.tokenizer.encode(prefix_text, add_special_tokens=False)
        while prefix and prefix != tokens[: len(prefix)]:
            prefix.pop()
        if not prefix or len(prefix) >= len(tokens):
            self.misses += 1
            return self.delegate.complete(messages, settings)

        model = backend.model
        original_generate = model.generate

        def generate_with_prefix(**kwargs):
            try:
                if kwargs["input_ids"][0].tolist() != tokens:
                    raise RuntimeError("Rendered tokens differ from generation inputs")
                if self.snapshot is not None and self.prefix == prefix:
                    self.last_hit = True
                    self.last_reused_tokens = len(prefix)
                    self.hits += 1
                    self.reused_tokens += len(prefix)
                else:
                    self.misses += 1
                    self.snapshot = None
                    self.prefix = None
                    n = len(prefix)
                    # Generate one disposable token so Transformers initializes
                    # the model-specific cache and positional inputs.
                    prefix_kwargs = dict(kwargs)
                    for key in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"):
                        if key in prefix_kwargs:
                            prefix_kwargs[key] = prefix_kwargs[key][..., :n]
                    prefix_kwargs.update(max_new_tokens=1, return_dict_in_generate=True)
                    if kwargs.get("do_sample"):
                        devices = [backend.device] if backend.device.type == "cuda" else []
                        with backend.torch.random.fork_rng(devices=devices):
                            result = original_generate(**prefix_kwargs)
                    else:
                        result = original_generate(**prefix_kwargs)
                    snapshot = result.past_key_values
                    if snapshot is None or snapshot.get_seq_length() != n:
                        raise RuntimeError("Model did not return a cache at the exact prefix boundary")
                    self.snapshot, self.prefix = snapshot, prefix
                return original_generate(**kwargs, past_key_values=copy.deepcopy(self.snapshot))
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                raise

        previous_override = model.__dict__.get("generate")
        model.generate = generate_with_prefix
        try:
            return self.delegate.complete(messages, settings)
        finally:
            if previous_override is None:
                del model.generate
            else:
                model.generate = previous_override
