# Reproduction notes

## Exact A100 environment

The selected checkpoint was produced with:

```text
torch        2.11.0+cu130
transformers 5.17.0
peft         0.21.2
unsloth      2026.11.4
```

`requirements.txt` gives compatible minimums for local work. For an exact
re-run, pin the versions above in the training image.

## Data and splits

| File | Contents |
| --- | --- |
| `data/split_ids.json` | 80 train IDs + 20 dev IDs |
| `data/validation_200_ids.txt` | fixed 200-case validation set |
| `data/train_gipformer.json` | Gipformer transcripts for train audio |

The released dataset is not included. Set `LIFEMATE_DATA_ROOT` to the folder
containing `train/` and `public_test/`. `train/zaic_tools/` is imported by the
runtime; the bundled `zaic_tools/` is the same simulator API for local tests.

## Final checkpoint provenance

The adapter `927186df...c9f8c8` is the A100 Search-First pilot checkpoint-100:

- fresh LoRA from `Qwen/Qwen3.5-4B`,
- 100 optimizer steps, save every 20, global batch 8, LR `1e-4`,
- trained with the previous discovery prompt `fd0b5681...`,
- selected for final inference after the prompt replay audit.

The current final prompt is `5268fcf5...` (Prompt B + `discovery_system`). The
smoke comparison held the adapter fixed and changed only the prompt, so the
prompt difference is isolated. A full 20-case Dev run with `5268...` is the
next quality gate; the 4-case smoke is not sufficient evidence.

## Reference results

Old-prompt full Dev20 (adapter `927186...`, prompt `fd0b...`):

```text
score          40.83
state hashes    9 / 20
searches       25
wrong writes    9
p95            30.76 s
```

New-prompt 4-case smoke (`dev[:4]`):

```text
score          25.0
state hashes    1 / 4
searches        3
wrong writes    1
missing writes  3
mean           11.39 s
p95            14.94 s
cache hits     63.6 %
```

The smoke set is three hard cases plus one trivial empty-gold case. Do not
generalize from it; run `dev` before accepting a prompt or adapter change.

## Replay data

`prepare` regenerates the verified replay rows from the released train split.
The earlier `revised_prompt_replay/` artifact (prompt `5268...`, 615 train rows
and 87 dev rows) is intentionally not copied into this repository; it is a
derived artifact and can be rebuilt with:

```bash
python -m pipeline.run_search_sft prepare --transcripts both
```

## Hash checks

`python scripts/check_hashes.py` verifies, without a GPU:

- caller Prompt B hash,
- Search-First discovery prompt hash,
- retriever source hash,
- adapter hash if `models/caller_adapter/` is present,
- split, validation, and transcript file hashes.
