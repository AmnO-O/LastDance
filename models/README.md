# Models

Weights are not committed. Place them here before building the submission image.

```text
models/
├── Qwen3.5-4B/           # base model (config.json, tokenizer, safetensors)
├── caller_adapter/       # final Search-First adapter, checkpoint-100
└── asr_gipformer/        # encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
```

## Final adapter

Expected SHA-256 of `caller_adapter/adapter_model.safetensors`:

```text
927186df0ff6c7bb9deca0f1a9f960d14ff6f246f7c101e16069d7e974c9f8c8
```

The local source used for consolidation is:

```text
kaggle_output_sft_a_pilot/search_first_sft/pilot/checkpoints/checkpoint-100
```

Copy that directory to `models/caller_adapter/` (keep `adapter_config.json` and
`adapter_model.safetensors`; the tokenizer is loaded from the base model).
Alternatively set `LIFEMATE_ADAPTER` to the directory when running `pipeline`
or `submission/predict.py`.

## ASR

The local Gipformer export is in `gip/` and contains `encoder.onnx`,
`decoder.onnx`, `joiner.onnx`, and `tokens.txt`. Copy those four files to
`models/asr_gipformer/`.

## Base model

`predict.py` looks for `models/Qwen3.5-4B/`, then `saved_models/Qwen3.5-4B/`,
then `/code/saved_models/Qwen3.5-4B/`. The Dockerfile downloads the base model
only when it is absent; for a fully offline build, provide it locally.
