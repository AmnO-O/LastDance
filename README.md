# LifeMate — Search-First Agent

Final code for the Zalo AI Challenge 2026 LifeMate Search-First agent.

The agent starts with **no app schemas open**. At every step it can call
`sys.search_tools` (at most four searches) to discover a capability, open the
returned schema, and execute one action. Audio is transcribed by Gipformer
and the transcript goes directly to the caller.

## Final configuration

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.5-4B` |
| Adapter | A100 Search-First checkpoint-100 · `927186df...c9f8c8` |
| Caller prompt | Prompt B · `76eb6f20...f3810b2` |
| Search-First system prompt | `5268fcf5...4ec111ed` |
| Retriever | `intfloat/multilingual-e5-small` @ `614241f6...cc418b3` |
| Retriever source | `src/lifemate/discovery.py` · `05b8ba54...29a784b` |
| Searches / steps | `4` / `12` |
| Retrieval top-k | `6` |

The canonical machine-readable copy is `configs/search_first.json`.

## Layout

```text
ZIC_last/
├── src/lifemate/          Core package: runtime, caller, prompts, search, SFT, ASR
├── src/tests/             Offline unit/end-to-end tests
├── zaic_tools/            Tool simulator (same API as the released package)
├── pipeline/
│   ├── common.py          Paths, data loading, JSON helpers
│   ├── runtime_utils.py   Final prompt, adapter, transcripts, prefix cache
│   └── run_search_sft.py  prepare → train → smoke-eval / dev / val
├── submission/
│   ├── predict.py         Search-First inference entrypoint
│   ├── preprocessing.py   Input/audio validation
│   ├── Dockerfile         Build from the repository root
│   ├── predict.sh         BTC entrypoint
│   └── build_and_export.* Docker build/export helpers
├── configs/search_first.json
├── data/                  Split IDs, validation IDs, Gipformer transcripts
├── models/                Weights go here (not committed)
└── docs/                  Pipeline and reproduction notes
```

## Kaggle smoke test

[Open the 3-ID notebook](notebooks/lastdance_e2e_3_ids.ipynb). It clones this
repository and runs the submission predictor on real audio with the final adapter.

## Quick start

### 1. Point at the released data

```bash
export LIFEMATE_DATA_ROOT=/path/to/train+public_test   # contains train/ and public_test/
```

The data root must contain `train/train.jsonl`, `train/train_tools_catalog.json`,
`train/train_environment/`, and `train/zaic_tools/`.

`data/train_gipformer.json` is not committed because it contains ASR transcripts
derived from competition audio. Place your local copy at that path before running
`prepare`. Expected SHA-256:

```text
81b2bb40dc5a3be532db29e8d3c14e194294d518ab0bc19e39e3605106cecfe6
```

`scripts/check_hashes.py` skips this file when it is absent.

### 2. Verify the final hashes (no GPU)

```bash
python scripts/check_hashes.py
```

### 3. Prepare verified replay data (CPU)

```bash
python -m pipeline.run_search_sft prepare --transcripts both
```

This writes `work/gold_replay_manifest.json`, `work/train_examples.jsonl`, and
`work/dev_examples.jsonl`. Preparation rejects any trajectory that cannot be
reproduced with the real simulator.

### 4. Train the Search-First adapter (GPU)

```bash
python -m pipeline.run_search_sft train --run-tag search_first
```

Default recipe: fresh LoRA, 100 steps, save every 20, global batch 8, LR `1e-4`,
max length 4096, LoRA rank 16. Multi-GPU uses `torch.distributed.run` with up to
two workers. The adapter is written to `work/search_first/adapter/`.

### 5. Evaluate on the 20-case Dev split (GPU)

```bash
python -m pipeline.run_search_sft dev --adapter work/search_first/adapter
```

Use `smoke-eval` for `dev[:4]` or `val` for the fixed 200-case validation set.
Reports and per-case traces are written under `work/`.

### 6. Build the submission image

Put the weights in `models/`:

```text
models/
├── Qwen3.5-4B/
├── caller_adapter/       # final checkpoint-100, SHA-256 927186...
└── asr_gipformer/        # encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
```

Then:

```bash
docker build -f submission/Dockerfile -t zac2026-lifemate:v1 .
```

Run it with `/data` mounted read-only and `/result` writable:

```bash
docker run --gpus all -v /path/to/data:/data -v /path/to/result:/result zac2026-lifemate:v1
```

`submission/predict.py` applies Prompt B, asserts the 5268 discovery hash,
verifies the adapter SHA (unless `--skip-adapter-check` is passed), and writes
`/result/submission.jsonl` with `id`, `tool_calls`, `state_hash`, and `time`.

## Notes

- `data/` contains only small split/transcript metadata. The released dataset and
  model weights are intentionally not copied here.
- `work/` and `outputs/` are runtime artifacts and are ignored.
- Legacy planner code remains available for offline experiments.
- The adapter was trained with the previous discovery prompt (`fd0b...`) and is
  the selected A100 checkpoint. Inference uses the final `5268...` prompt; the
  prompt replay audit is documented in `docs/REPRODUCE.md`.
