# Data

Only small, stable metadata is stored here. The released dataset is not copied.

| File | Purpose |
| --- | --- |
| `split_ids.json` | 80 Search-First train IDs and 20 Dev IDs |
| `validation_200_ids.txt` | fixed 200-case validation IDs |
| `train_gipformer.json` | Gipformer ASR transcripts for train audio |

Set the dataset location explicitly:

```bash
export LIFEMATE_DATA_ROOT=/path/to/train+public_test
```

Expected layout:

```text
train+public_test/
├── train/
│   ├── train.jsonl
│   ├── train_tools_catalog.json
│   ├── train_environment/<id>/state.json
│   ├── audio/<id>/*.wav
│   └── zaic_tools/
└── public_test/
    └── ...
```

Generated replay rows, reports, and traces go to `work/`, not here.
