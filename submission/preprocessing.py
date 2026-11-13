#!/usr/bin/env python3
"""Validate the mounted dataset and audio paths before inference starts."""
from __future__ import annotations

import json
import os
from pathlib import Path

SPLITS = ("private_test", "public_test", "train")


def find_split(data_root: Path) -> tuple[Path, str] | None:
    for root in (data_root, *data_root.parents[:1]):
        for split in SPLITS:
            split_dir = root / split
            if (split_dir / f"{split}.jsonl").is_file():
                return split_dir, split
    for split in SPLITS:
        if (data_root / f"{split}.jsonl").is_file():
            return data_root, split
    for jsonl in data_root.rglob("*.jsonl"):
        if jsonl.stem in SPLITS:
            return jsonl.parent, jsonl.stem
    return None


def main() -> None:
    print("preprocessing...", flush=True)
    data_root = Path(os.environ.get("DATA_DIR", "/data")).resolve()
    found = find_split(data_root) if data_root.is_dir() else None
    if found is None:
        print("preprocessing: 0 samples, 0 missing audio (no input dataset yet)", flush=True)
        return
    split_dir, split = found
    total = missing = 0
    with (split_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            audio = item.get("audio")
            if audio and not (data_root / split / audio).is_file() and not (split_dir / audio).is_file():
                missing += 1
    print(f"preprocessing: {total} samples, {missing} missing audio", flush=True)


if __name__ == "__main__":
    main()
