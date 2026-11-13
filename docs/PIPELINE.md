# Pipeline

## Search-First control loop

```text
Sample + installed apps
        │
        ▼
EmptyPlanner ──► focus_apps = []
        │
        ▼
Caller step (≤ 12)
  ├─ visible schemas: only sys.search_tools + discovered tools
  ├─ {"action":"call","tool":"sys.search_tools","arguments":{"query":...}}
  │      └─ BM25 + E5 dense retrieval → RRF → top 6 schemas opened
  ├─ {"action":"call","tool":"<open tool>","arguments":{...}}
  ├─ {"action":"ask","tool":"<open tool>","params":[...]}
  └─ {"action":"finish"}
        │
        ▼
ToolEnv (one session per sample) → tool_calls + state_hash
```

The final agent has no fallback compact schema directory and no `need_app`
action. If a required capability is not visible, the only path is
`sys.search_tools`. This is what the SFT data teaches.

## Replay preparation

`pipeline/run_search_sft.py prepare` does not trust the gold trajectory
blindly. For each train/dev sample it:

1. Replays the gold calls through a fresh `ToolEnv`.
2. Rejects the sample if the replayed calls or final `state_hash` differ from
   the label.
3. Runs the real Search-First agent with a `GoldCaller` that emits the gold
   action, including the search action needed to open an unseen tool.
4. Rejects the trace if search retrieval misses, budgets are exceeded, a target
   tool is unavailable, or an argument is not derivable from the observation.
5. Encodes every step with the final prompt and rejects overlength examples.

Only fully verified caller steps become SFT rows. Gold and Gipformer transcript
variants are deduplicated by rendered prompt.

## Training

`train` re-checks the prompt hash, retriever hash, and row checksums before
starting. It requires all three action kinds (`search`, `call`, `finish`) in the
train rows. The default final recipe is fresh LoRA, 100 steps, global batch 8,
LR `1e-4`, max length 4096, rank 16.

## Evaluation

`smoke-eval`, `dev`, and `val` evaluate with the same `PrefixCacheCaller` and
report:

- mean score and exact state-hash matches,
- search count,
- `wrong_writes` (predicted writes minus gold writes),
- `missing_writes` (gold writes minus predicted writes),
- mean / p95 / max latency, cache hit rate, and the adapter SHA.

`wrong_writes` alone is not a correctness signal; both write directions are
reported. The 4-case smoke set (`dev[:4]`) contains three hard failures plus one
trivial empty-gold case, so it is a regression check, not a benchmark. The
quality gate is the full 20-case Dev split.

## Inference

`submission/predict.py` rebuilds the same runtime:

- `AgentConfig(top_k=6, max_steps=12, tool_discovery=True, max_searches=4)`
- `EmptyPlanner`
- `Caller` with `max_new_tokens=220`
- Prompt B + `discovery_system` (SHA-256 `5268...`)
- final adapter (`927186...`)
- optional request-local prefix cache

One `ToolEnv` is created per sample and never reset, so every attempted call is
part of the submission.
