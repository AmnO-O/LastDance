# Submission

Search-First offline Docker submission for ZAC 2026.

## Build

From the repository root:

```bash
docker build -f submission/Dockerfile -t zac2026-lifemate:v1 .
```

Or use `submission/build_and_export.sh` / `build_and_export.bat`.

Required model layout (see `../models/README.md`):

```text
models/
├── Qwen3.5-4B/
├── caller_adapter/       # SHA-256 927186...
└── asr_gipformer/
```

## Run

```bash
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/result:/result \
  zac2026-lifemate:v1
```

The image runs `predict.sh` → `preprocessing.py` → `predict.py` and writes
`/result/submission.jsonl`.

## Entrypoint behavior

- Finds `private_test`, `public_test`, or `train` under `/data`.
- Imports the split's own `zaic_tools` first, then the bundled fallback.
- Applies Prompt B and asserts the final `5268...` discovery prompt hash.
- Verifies the adapter SHA-256 by default.
- Passes Gipformer transcripts directly to the caller, which starts with zero open apps.
- Uses `max_searches=4`, `max_steps=12`, retrieval top-k 6.
- Emits `id`, `tool_calls`, `state_hash`, and per-sample `time` in milliseconds.

Useful flags for local checks:

```bash
python submission/predict.py --data-dir ./data --output ./submission.jsonl \
  --model-path ./models/Qwen3.5-4B --adapter-path ./models/caller_adapter \
  --asr-path ./models/asr_gipformer --limit 2
```

`--no-asr` uses `audio_text` only (train split debugging).
`--skip-adapter-check` allows non-final adapters for experiments.
