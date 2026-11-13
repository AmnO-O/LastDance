#!/usr/bin/env python3
"""Verify the final prompt, retriever, split, transcript, and adapter hashes.

Runs on CPU and does not load a model.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.runtime_utils import (  # noqa: E402
    ADAPTER_SHA256,
    DISCOVERY_PROMPT_SHA256,
    adapter_path,
    apply_prompt_b,
)
from pipeline.common import sha  # noqa: E402


def main() -> int:
    from lifemate import discovery, prompts

    apply_prompt_b(prompts)
    discovery_sha = hashlib.sha256(prompts.discovery_system(prompts.CALLER_SYSTEM).encode()).hexdigest()
    checks = [
        ("caller Prompt B", "76eb6f20648c71b4d243932a9a18628b998e6373a6f2ffbb7d57eb468f3810b2",
         hashlib.sha256(prompts.CALLER_SYSTEM.encode()).hexdigest()),
        ("search-first prompt", DISCOVERY_PROMPT_SHA256, discovery_sha),
        ("retriever source", "05b8ba54da9b85d3f95853c7ce7fd3e3f5ae15503ae1e1754b086c37529a784b",
         sha(discovery.__file__)),
        ("split ids", "402445331f22f3c5f7948f4bd64df7f3f85ea9bf31455f6d3a5f92dc554a09ee",
         sha(REPO / "data" / "split_ids.json")),
        ("validation ids", "e810b418d7018437a16892350e01cbee9526cf43ab4a190101433d4508498d2f",
         sha(REPO / "data" / "validation_200_ids.txt")),
    ]
    transcripts = REPO / "data" / "train_gipformer.json"
    if transcripts.is_file():
        checks.append(("transcripts", "81b2bb40dc5a3be532db29e8d3c14e194294d518ab0bc19e39e3605106cecfe6",
                       sha(transcripts)))
    else:
        print("transcripts  SKIP  (local file is absent)")
    try:
        path = adapter_path()
    except FileNotFoundError:
        print("adapter      SKIP  (models/caller_adapter is not populated)")
    else:
        checks.append(("adapter", ADAPTER_SHA256, sha(path / "adapter_model.safetensors")))

    failed = False
    for name, expected, actual in checks:
        ok = expected == actual
        failed |= not ok
        print(f"{name:20} {'OK' if ok else 'MISMATCH'}  {actual}")
        if not ok:
            print(f"{'':20} expected {expected}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
