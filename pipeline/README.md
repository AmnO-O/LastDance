# Pipeline

Search-First prepare / train / evaluate entrypoints.

```bash
python -m pipeline.run_search_sft prepare --transcripts both
python -m pipeline.run_search_sft train --run-tag search_first
python -m pipeline.run_search_sft smoke-eval --adapter work/search_first/adapter
python -m pipeline.run_search_sft dev --adapter work/search_first/adapter
python -m pipeline.run_search_sft val --adapter work/search_first/adapter
```

## Files

- `common.py` — dataset root discovery, split metadata, transcript path, JSON
  helpers. Runtime artifacts go to `LIFEMATE_WORK` (default `work/`).
- `runtime_utils.py` — final prompt hashes, Prompt B application, adapter lookup
  and verification, Gipformer transcript substitution, prefix cache.
- `run_search_sft.py` — verified gold replay (`prepare`), LoRA training
  (`train`), and Dev/Val evaluation.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `LIFEMATE_DATA_ROOT` | auto-detect | folder containing `train/` and `public_test/` |
| `LIFEMATE_WORK` | `work/` | generated replay data, checkpoints, reports |
| `LIFEMATE_MODEL` | `Qwen/Qwen3.5-4B` | base model name/path |
| `LIFEMATE_ADAPTER` | `models/caller_adapter` | final adapter directory |
| `LIFEMATE_TRANSCRIPTS` | `data/train_gipformer.json` | ASR transcript JSON |

This folder contains only the final Search-First pipeline: no planner/roadmap
runs, teacher data, recovery data, corrections, prompt search, or soft prompts.
