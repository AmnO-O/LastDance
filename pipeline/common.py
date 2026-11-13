"""Shared paths, data loading, and JSON helpers for the Search-First pipeline.

The pipeline is deliberately separate from the released data package.  Point it
at the directory that contains ``train/`` and ``public_test/`` with
``LIFEMATE_DATA_ROOT`` (or run it from inside that directory).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = REPO / "data"


def data_root() -> Path:
    """Return the dataset root that contains ``train/`` and ``public_test/``."""
    override = os.environ.get("LIFEMATE_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    from lifemate.data import find_data_root

    for start in (Path.cwd(), REPO):
        try:
            return find_data_root(start)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Could not locate the LifeMate data package. Set LIFEMATE_DATA_ROOT to the "
        "folder containing train/ and public_test/."
    )


def work_dir() -> Path:
    path = Path(os.environ.get("LIFEMATE_WORK", REPO / "work")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


WORK = Path(os.environ.get("LIFEMATE_WORK", REPO / "work")).expanduser()


def transcript_path() -> Path:
    override = os.environ.get("LIFEMATE_TRANSCRIPTS")
    path = Path(override) if override else DATA_DIR / "train_gipformer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Gipformer transcript file: {path}. Set LIFEMATE_TRANSCRIPTS."
        )
    return path


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path: str | Path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dump(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: str | Path, records) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def dataset():
    """Return ``(root, validation_ids, samples)`` for the released train split."""
    from lifemate.data import load_samples, read_ids

    root = data_root()
    ids_path = DATA_DIR / "validation_200_ids.txt"
    ids = read_ids(ids_path)
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError(f"Expected 200 unique validation IDs in {ids_path}")
    return root, ids, load_samples(root, "train")
