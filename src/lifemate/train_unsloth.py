"""Caller-only Unsloth LoRA training; run with python -m lifemate.train_unsloth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .catalog import Catalog
from .data import catalog_path, find_data_root, find_validation_ids, load_samples, read_ids
from .sft import build_examples
from .sft_mix import asr_examples, recovery_examples, mix_examples, planner_focus_examples


def encode_example(tokenizer, example, max_length: int):
    """Mask the exact inference prefix; reject truncation."""
    try:
        prompt_text = tokenizer.apply_chat_template(
            example.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except (TypeError, ValueError):
        prompt_text = tokenizer.apply_chat_template(
            example.messages, tokenize=False, add_generation_prompt=True,
        )
        for open_tag, close_tag in (("<think>", "</think>"), ("<thought>", "</thought>"), ("<reasoning>", "</reasoning>")):
            if open_tag in prompt_text and prompt_text.rfind(open_tag) > prompt_text.rfind(close_tag):
                raise ValueError(
                    f"Chat template defaulted to unclosed thinking mode ({open_tag}) without enable_thinking=False support; "
                    "direct JSON caller targets require thinking to be disabled."
                )
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(tok, "encode"):
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    else:
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tok(example.target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    full = list(prompt_ids) + list(target_ids)
    if len(full) > max_length:
        return None
    if not target_ids:
        raise ValueError("Example has no supervised response tokens")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt_ids) + list(target_ids),
    }


class ResponseCollator:
    """Right-pad input and label sequences without changing the loss mask."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(f["input_ids"]) for f in features)
        return {key: torch.tensor([
            list(f[key]) + [pad] * (length - len(f[key])) for f in features
        ], dtype=torch.long) for key, pad in (
            ("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)
        )}


def align_training_targets(pools, *, prefer_recovery=False):
    """Check all eligible rows before sampling or oversampling."""
    def key(row):
        return json.dumps(row.messages, sort_keys=True)

    def target(row):
        return json.dumps(json.loads(row.target), sort_keys=True)

    recovery = {}
    for row in pools["recovery"]:
        prompt = key(row)
        if recovery.setdefault(prompt, target(row)) != target(row):
            raise ValueError(f"Conflicting recovery targets for {row.sample_id}, step {row.step}")
    seen = dict(recovery)
    removed = 0
    for source in ("gold", "asr"):
        kept = []
        for row in pools[source]:
            prompt, label = key(row), target(row)
            if prefer_recovery and prompt in recovery and recovery[prompt] != label:
                removed += 1
                continue
            if seen.setdefault(prompt, label) != label:
                raise ValueError(f"Conflicting training targets for {row.sample_id}, step {row.step}")
            kept.append(row)
        pools[source] = kept
    return removed


def initialize_adapter(model, path):
    """Load adapter weights into the trainable Unsloth adapter before Trainer starts."""
    from peft import PeftConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    saved = PeftConfig.from_pretrained(str(path))
    current = model.peft_config["default"]
    for field in ("peft_type", "r", "lora_alpha", "rank_pattern", "alpha_pattern", "use_rslora", "use_dora"):
        if getattr(saved, field, None) != getattr(current, field, None):
            raise ValueError(f"Initial adapter configuration differs for {field}")
    expected = get_peft_model_state_dict(model, adapter_name="default")
    weights = load_peft_weights(str(path), device="cpu")
    if weights.keys() != expected.keys():
        # The inference export can omit the language_model wrapper in Qwen layer names.
        def normalized(name):
            return name.replace(".model.language_model.layers.", ".model.layers.")

        expected_names = {normalized(name): name for name in expected}
        normalized_weights = {normalized(name): value for name, value in weights.items()}
        if (len(expected_names) != len(expected) or len(normalized_weights) != len(weights)
                or expected_names.keys() != normalized_weights.keys()):
            raise ValueError("Initial adapter tensor names do not match the training model")
        weights = {name: normalized_weights[key] for key, name in expected_names.items()}
    for name, value in weights.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"Initial adapter tensor shape differs for {name}")
    set_peft_model_state_dict(model, weights, adapter_name="default")
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    for name, value in weights.items():
        if not loaded[name].detach().cpu().equal(value.to(dtype=loaded[name].dtype)):
            raise ValueError(f"Initial adapter weight did not load correctly: {name}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--validation-ids", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--revision", default="main")
    p.add_argument("--load-in-4bit", action="store_true", help="Train LoRA on a 4-bit base model")
    p.add_argument("--max-length", type=int, default=3500)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--caller-max-steps", type=int, default=12)
    p.add_argument("--history-chars", type=int, default=10000)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=None, help="Evaluation frequency in steps; defaults to save-steps")
    p.add_argument("--max-train-steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=13)
    restart = p.add_mutually_exclusive_group()
    restart.add_argument("--resume-from-checkpoint")
    restart.add_argument("--init-adapter", type=Path, help="Load adapter weights and start a fresh optimizer")
    p.add_argument("--prepare-only", action="store_true", help="Export examples without GPU dependencies")
    p.add_argument("--asr-transcripts", type=Path, help="Gipformer transcripts for training sample IDs")
    p.add_argument("--recovery-data", type=Path, help="Reviewed environment-feedback continuations JSONL")
    p.add_argument("--track-recovery-data", type=Path,
                   help="Record exact captured recovery rows used in training, per rank")
    p.add_argument("--planner-focus-rollouts", type=Path,
                   help="Add caller examples using observed planner focus from all-gold rollouts")
    p.add_argument("--prefer-recovery-targets", action="store_true",
                   help="Remove ordinary rows that conflict with verified recovery targets")
    p.add_argument("--mix", default="0.4,0.4,0.2", help="Gold,ASR,recovery proportions; 1,0,0 for clean baseline")
    p.add_argument("--recovery-max-copies", type=int, default=3,
                   help="Maximum copies of each distinct recovery example per prepared training dataset")
    return p


def main(argv=None):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args = parser().parse_args(argv)
    weights = tuple(float(w) for w in args.mix.split(","))
    if len(weights) != 3 or any(w < 0 for w in weights) or weights[0] <= 0:
        raise ValueError("--mix requires three nonnegative weights with positive gold weight")
    if weights[1] and args.asr_transcripts is None:
        raise ValueError("Default mixture requires --asr-transcripts; use --mix 1,0,0 for gold only")
    if weights[2] and args.recovery_data is None:
        raise ValueError("Default mixture requires --recovery-data with verified continuations")
    root = args.root or find_data_root()
    ids_path = args.validation_ids or find_validation_ids(root)
    if ids_path is None:
        raise ValueError("Supply --validation-ids to prevent training/validation leakage")
    held_out = set(read_ids(ids_path))
    samples = load_samples(root, "train")
    known = {s.id for s in samples}
    if not held_out or held_out - known:
        raise ValueError("Validation IDs must be nonempty and all belong to this train split")
    catalog = Catalog.load(catalog_path(root, "train"))
    groups = {"train": [s for s in samples if s.id not in held_out],
              "validation": [s for s in samples if s.id in held_out]}
    args.output.mkdir(parents=True, exist_ok=True)
    examples, summary = {}, {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    for name, group in groups.items():
        stats = {}
        examples[name] = build_examples(
            catalog, group, tasks=("caller",), top_k=args.top_k,
            max_steps=args.caller_max_steps, max_history_chars=args.history_chars,
            derivable_only=True, stats=stats,
        )
        if name == "train":
            options = dict(top_k=args.top_k, max_steps=args.caller_max_steps,
                           max_history_chars=args.history_chars)
            if args.planner_focus_rollouts:
                alternate = planner_focus_examples(group, catalog, args.planner_focus_rollouts, **options)
                if not alternate:
                    raise ValueError("No distinct caller examples matched all-gold planner rollouts")
                examples[name].extend(alternate)
                stats["planner_focus_examples"] = len(alternate)
            pools = {"gold": examples[name],
                     "asr": asr_examples(group, catalog, args.asr_transcripts, **options) if weights[1] else [],
                     "recovery": recovery_examples(group, catalog, args.recovery_data, **options) if weights[2] else []}
            stats["ordinary_rows_replaced"] = align_training_targets(
                pools, prefer_recovery=args.prefer_recovery_targets,
            )
            examples[name], stats["mixture"] = mix_examples(
                pools, weights, args.seed, recovery_max_copies=args.recovery_max_copies,
            )
        if not examples[name]:
            raise ValueError(f"No {name} examples remain")
        if local_rank == 0:
            path = args.output / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for example in examples[name]:
                    stream.write(json.dumps(example.as_row(), ensure_ascii=False) + "\n")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            sha = ""
        summary[name] = {"sample_ids": [s.id for s in group], "examples": len(examples[name]),
                         "sha256": sha, **stats}
    if local_rank == 0:
        manifest = args.output / "manifest.json"
        manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        if local_rank == 0:
            print(json.dumps({k: len(v) for k, v in examples.items()}))
        return 0

    return train_examples(args, examples, summary, local_rank)


def recovery_tracking_key(sample_id, step, messages, target):
    return json.dumps([sample_id, step, messages, target], sort_keys=True, ensure_ascii=False)


def train_examples(args, examples, summary, local_rank=0):
    manifest = args.output / "manifest.json"
    # Import Unsloth before Transformers/PEFT so its training patches take effect.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; use --prepare-only for local data inspection")
    torch.cuda.set_device(local_rank)
    bf16 = torch.cuda.is_bf16_supported()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, revision=args.revision, max_seq_length=args.max_length,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        load_in_4bit=args.load_in_4bit, load_in_16bit=not args.load_in_4bit, full_finetuning=False,
    )
    # Explicit language-only module suffixes avoid adapting the unused vision tower.
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
    vision_keywords = ("visual", "vision_tower", "vision_model", "image_encoder")
    targets = sorted(list({
        name.split(".")[-1] for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in suffixes
        and not any(kw in name.lower() for kw in vision_keywords)
    }))
    if not targets:
        raise ValueError("No language LoRA targets found; inspect the loaded model architecture")
    peft_kwargs = {
        "r": args.rank,
        "lora_alpha": 2 * args.rank,
        "lora_dropout": 0,
        "target_modules": targets,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": args.seed,
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
    }
    # Pre-freeze any vision parameters across any nested hierarchy.
    for name, param in model.named_parameters():
        if any(kw in name.lower() for kw in vision_keywords):
            param.requires_grad = False

    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    if args.init_adapter:
        initialize_adapter(model, args.init_adapter)

    # Post-freeze and assert no vision parameter is trainable.
    for name, param in model.named_parameters():
        if any(kw in name.lower() for kw in vision_keywords):
            param.requires_grad = False

    trainable_vision = [
        name for name, param in model.named_parameters()
        if param.requires_grad and any(kw in name.lower() for kw in vision_keywords)
    ]
    if trainable_vision:
        raise RuntimeError(f"Vision parameters remained trainable after adapter creation: {trainable_vision[:5]}")
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None) or 0
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token_id = pad_id
    if hasattr(tokenizer, "pad_token_id") and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = pad_id
    tracked = {}
    if args.track_recovery_data:
        from .data import load_jsonl
        for row in load_jsonl(args.track_recovery_data):
            key = recovery_tracking_key(row["sample_id"], row["step"] - 1,
                                        row["messages"], row["target"])
            tracked[key] = f'{row["sample_id"]}:{row["step"]}'
    available_tracked = set()
    datasets = {}
    for name, rows in examples.items():
        kept = []
        for row in rows:
            encoded = encode_example(tokenizer, row, args.max_length)
            if encoded is None:
                continue
            if tracked:
                tracking_id = ""
                if name == "train":
                    key = recovery_tracking_key(row.sample_id, row.step, row.messages, json.loads(row.target))
                    tracking_id = tracked.get(key, "")
                encoded["recovery_tracking_id"] = tracking_id
                if tracking_id:
                    available_tracked.add(tracking_id)
            kept.append(encoded)
        if not kept:
            raise ValueError(f"All {name} examples exceed --max-length")
        datasets[name] = Dataset.from_list(kept)
        summary[name].update(kept=len(kept), dropped_overlength=len(rows) - len(kept))
    if tracked:
        summary["tracked_recovery"] = {"eligible": sorted(available_tracked),
                                       "dropped": sorted(set(tracked.values()) - available_tracked)}
    summary["lora_targets"] = targets
    summary["versions"] = {name: __import__(name).__version__ for name in ("torch", "transformers", "peft", "unsloth")}
    if local_rank == 0:
        manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": str(args.output / "checkpoints"),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_train_steps,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "fp16": not bf16,
        "bf16": bf16,
        "logging_steps": 5,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps if args.eval_steps is not None else args.save_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 5,
        "prediction_loss_only": True,
        "report_to": "none",
        "seed": args.seed,
        "ddp_find_unused_parameters": False,
    }
    if "warmup_ratio" in sig:
        training_kwargs["warmup_ratio"] = 0.05
    else:
        warmup = max(1, int(args.max_train_steps * 0.05)) if args.max_train_steps > 0 else 10
        training_kwargs["warmup_steps"] = warmup
    collator = ResponseCollator(pad_id)
    trainer_class = Trainer
    callbacks = []
    if tracked:
        from collections import Counter

        consumed = Counter()
        base_collator = collator

        def tracking_collator(features):
            batch = base_collator(features)
            batch["recovery_tracking_ids"] = [feature["recovery_tracking_id"] for feature in features]
            return batch

        class TrackingTrainer(Trainer):
            def training_step(self, model, inputs, *args, **kwargs):
                ids = inputs.pop("recovery_tracking_ids")
                loss = super().training_step(model, inputs, *args, **kwargs)
                consumed.update(value for value in ids if value)
                return loss

            def prediction_step(self, model, inputs, *args, **kwargs):
                inputs.pop("recovery_tracking_ids", None)
                return super().prediction_step(model, inputs, *args, **kwargs)

        class TrackingCallback(TrainerCallback):
            def on_step_end(self, training_args, state, control, **kwargs):
                if state.global_step % args.save_steps == 0 or state.global_step == state.max_steps:
                    report = args.output / f"recovery_step{state.global_step}_rank{training_args.process_index}.json"
                    report.write_text(json.dumps(dict(consumed), indent=2), encoding="utf-8")

        training_kwargs["remove_unused_columns"] = False
        trainer_class = TrackingTrainer
        collator = tracking_collator
        callbacks.append(TrackingCallback())
    trainer = trainer_class(
        model=model, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=collator, callbacks=callbacks,
        args=TrainingArguments(**training_kwargs),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if local_rank == 0:
        adapter_dir = args.output / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.save_pretrained(str(adapter_dir))
        # Sanitize saved safetensors keys if Unsloth added .language_model.
        safetensors_path = adapter_dir / "adapter_model.safetensors"
        if safetensors_path.is_file():
            try:
                from safetensors.torch import load_file, save_file
                weights = load_file(str(safetensors_path))
                if any(".language_model." in k for k in weights):
                    cleaned = {k.replace(".language_model.", "."): v for k, v in weights.items()}
                    save_file(cleaned, str(safetensors_path))
                    print(f"Sanitized {len(cleaned)} adapter weights (removed .language_model.)", flush=True)
            except Exception as e:
                print(f"Warning: could not sanitize safetensors weights: {e}", flush=True)
        trainer.save_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
