"""Chat-model backends. Only the transformer backend is used by the offline solution."""
from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import torch


class ContextOverflow(RuntimeError):
    pass


@dataclass
class GenerationSettings:
    max_new_tokens: int = 320
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 13
    top_k: int | None = None
    max_context_tokens: int | None = None


class ChatBackend(Protocol):
    def complete(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str: ...


class AdapterView:
    """A view of one backend with the LoRA adapter optionally disabled or switched.

    Used when the planner should stay on the base model while the caller uses
    the fine-tuned adapter, or the rewriter uses a separate rewriter adapter.
    Keeping one loaded model avoids doubling GPU memory and lets all stages share
    the tokenizer and KV cache allocator.
    """

    def __init__(self, backend: ChatBackend, enabled: bool = True, adapter_name: str | None = None):
        self.backend = backend
        self.enabled = bool(enabled)
        self.adapter_name = adapter_name

    def complete(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        complete_with_adapter = getattr(self.backend, "complete_with_adapter", None)
        if complete_with_adapter is not None:
            try:
                return complete_with_adapter(messages, settings, enabled=self.enabled, adapter_name=self.adapter_name)
            except TypeError:
                return complete_with_adapter(messages, settings, enabled=self.enabled)
        return self.backend.complete(messages, settings)

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        counter = getattr(self.backend, "count_prompt_tokens", None)
        return counter(messages) if counter is not None else 0

    @property
    def last_generation(self) -> dict[str, Any]:
        return getattr(self.backend, "last_generation", {})


class ScriptedBackend:
    """Deterministic backend for tests and manual walkthroughs."""

    def __init__(self, responses: list[str] | Callable[[list[dict[str, str]], GenerationSettings], str]):
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        self.calls.append(messages)
        if callable(self.responses):
            return self.responses(messages, settings)
        if not self.responses:
            return json.dumps({"action": "finish"})
        return self.responses.pop(0)


class MockBackend:
    """A model that always emits one fixed response (smoke tests only)."""

    def __init__(self, response: str = '{"action":"finish"}'):
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        self.calls.append(messages)
        return self.response


def _load_peft_adapter(base_model: torch.nn.Module, adapter_path: str | Path) -> Any:
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")

    try:
        from peft import PeftConfig, PeftModel, set_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install `peft` to load a LoRA adapter") from exc

    # 1. Initialize adapter layers from config WITHOUT loading weights from disk.
    # This prevents PeftModel.from_pretrained from reading unmapped keys and warning.
    config = PeftConfig.from_pretrained(str(adapter_path))
    peft_model = PeftModel(base_model, config, adapter_name="default")

    # 2. Inspect adapter weights from safetensors or bin
    raw_weights: dict[str, Any] | None = None
    safetensors_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"

    if safetensors_path.is_file():
        try:
            from safetensors.torch import load_file
            raw_weights = load_file(str(safetensors_path))
        except ImportError:
            pass

    if raw_weights is None and bin_path.is_file():
        raw_weights = torch.load(str(bin_path), map_location="cpu", weights_only=True)

    if raw_weights is None:
        try:
            from peft import load_peft_weights
            raw_weights = load_peft_weights(str(adapter_path))
        except Exception:
            raw_weights = {}

    # 3. Strip .language_model. if saved by Unsloth
    clean_weights = {
        k.replace(".language_model.", "."): v for k, v in raw_weights.items()
    }

    # 4. Load cleaned adapter weights ONCE using PEFT's official adapter loader
    load_result = set_peft_model_state_dict(peft_model, clean_weights, adapter_name="default")
    if load_result and hasattr(load_result, "missing_keys"):
        lora_missing = [k for k in load_result.missing_keys if "lora_" in k]
        if lora_missing:
            raise RuntimeError(
                f"LoRA adapter loading FAILED: {len(lora_missing)} missing keys: {lora_missing[:5]}"
            )

    # 5. Strictly verify all LoRA parameters in the model received matching weights
    lora_param_names = [
        name for name, _ in peft_model.named_parameters() if "lora_" in name
    ]
    if not lora_param_names:
        raise RuntimeError(f"No LoRA parameters found in PeftModel! Check adapter_config.json at {adapter_path}")

    missing_lora_keys = []
    for name in lora_param_names:
        file_key = name.replace(".default.", ".")
        if file_key not in clean_weights and name not in clean_weights:
            missing_lora_keys.append(name)
        else:
            weight_key = name if name in clean_weights else file_key
            expected_shape = tuple(peft_model.get_parameter(name).shape)
            loaded_shape = tuple(clean_weights[weight_key].shape)
            if expected_shape != loaded_shape:
                raise RuntimeError(
                    f"Shape mismatch for LoRA weight {name!r}: expected {expected_shape}, loaded {loaded_shape}"
                )

    if missing_lora_keys:
        raise RuntimeError(
            f"LoRA adapter loading FAILED: {len(missing_lora_keys)} / {len(lora_param_names)} LoRA keys are missing! "
            f"Sample missing keys: {missing_lora_keys[:5]}"
        )

    return peft_model


def _add_peft_adapter(peft_model: Any, adapter_path: str | Path, adapter_name: str = "rewriter") -> None:
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Rewriter LoRA adapter directory not found: {adapter_path}")
    try:
        from peft import PeftConfig, set_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install `peft` to load a LoRA adapter") from exc

    config = PeftConfig.from_pretrained(str(adapter_path))
    peft_model.add_adapter(adapter_name, config)

    raw_weights: dict[str, Any] | None = None
    safetensors_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"

    if safetensors_path.is_file():
        try:
            from safetensors.torch import load_file
            raw_weights = load_file(str(safetensors_path))
        except ImportError:
            pass

    if raw_weights is None and bin_path.is_file():
        import torch
        raw_weights = torch.load(str(bin_path), map_location="cpu", weights_only=True)

    if raw_weights is None:
        try:
            from peft import load_peft_weights
            raw_weights = load_peft_weights(str(adapter_path))
        except Exception:
            raw_weights = {}

    clean_weights = {
        k.replace(".language_model.", "."): v for k, v in raw_weights.items()
    }
    set_peft_model_state_dict(peft_model, clean_weights, adapter_name=adapter_name)
    peft_model.set_adapter("default")


class TransformersBackend:
    """Greedy Hugging Face generation with the tokenizer's chat template.

    The model must fit the competition's parameter budget. The default is the
    non-thinking Qwen3-4B-Instruct-2507. Loading/importing ``transformers`` is
    deferred until this class is instantiated, so the runtime and scorer stay
    usable in a standard-library-only environment.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        *,
        tokenizer_name_or_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_input_tokens: int = 15200,
        attn_implementation: str | None = "sdpa",
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        adapter_path: str | None = None,
        rewriter_adapter_path: str | None = None,
        verify_adapter: bool = True,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError(
                "The transformers backend needs `transformers` and `torch` installed. "
                "Install them in the GPU environment or use a supplied backend."
            ) from exc
        self.torch = torch
        self.model_name = model_name_or_path
        self.max_input_tokens = int(max_input_tokens)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        try:
            from transformers import __version__ as transformers_version

            major = int(str(transformers_version).split(".", 1)[0])
        except Exception:  # pragma: no cover
            major = 4
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("4-bit loading needs transformers with BitsAndBytesConfig") from exc
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install `bitsandbytes` to use load_in_4bit=True") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(bnb_4bit_quant_type),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            if device == "auto":
                kwargs["device_map"] = "auto"
            elif torch.cuda.is_available():
                kwargs["device_map"] = {"": torch.device(device).index or 0}
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            self.device = next(self.model.parameters()).device
        else:
            if major >= 5:
                kwargs["dtype"] = torch_dtype
            else:
                kwargs["torch_dtype"] = torch_dtype
            if device == "auto":
                kwargs["device_map"] = "auto"
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
                self.device = next(self.model.parameters()).device
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
                self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
                self.model.to(self.device)
        self._peft_model = None
        self.adapter_delta: float | None = None
        if adapter_path:
            self.model = _load_peft_adapter(self.model, adapter_path)
            self.model.eval()
            self._peft_model = self.model
            if verify_adapter:
                self.adapter_delta = self.verify_adapter_effect()
        if rewriter_adapter_path:
            if self._peft_model is None:
                self.model = _load_peft_adapter(self.model, rewriter_adapter_path, adapter_name="rewriter")
                self.model.eval()
                self._peft_model = self.model
            else:
                _add_peft_adapter(self._peft_model, rewriter_adapter_path, adapter_name="rewriter")
        self.model.eval()
        self.attn_implementation = str(getattr(self.model.config, "_attn_implementation", "unknown"))
        try:
            self.supports_logits_to_keep = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):  # pragma: no cover
            self.supports_logits_to_keep = False
        self.last_prompt_tokens = 0
        self.last_generation: dict[str, Any] = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._thinking_kwarg: bool | None = None

    def _render(self, messages: list[dict[str, str]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._thinking_kwarg is not False:
            try:
                rendered = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
                self._thinking_kwarg = True
                return rendered
            except (TypeError, ValueError):
                self._thinking_kwarg = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        prompt = self._render(messages)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def complete(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
        use_adapter: bool = True,
        adapter_name: str | None = None,
    ) -> str:
        if self._peft_model is not None:
            if not use_adapter:
                disable = getattr(self._peft_model, "disable_adapter", None)
                if disable is not None:
                    with disable():
                        return self._complete(messages, settings)
            if adapter_name and hasattr(self._peft_model, "set_adapter"):
                self._peft_model.set_adapter(adapter_name)
                try:
                    return self._complete(messages, settings)
                finally:
                    self._peft_model.set_adapter("default")
        return self._complete(messages, settings)

    def complete_with_adapter(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
        enabled: bool = True,
        adapter_name: str | None = None,
    ) -> str:
        return self.complete(messages, settings, use_adapter=enabled, adapter_name=adapter_name)

    def verify_adapter_effect(self, prompt: str = "Test caller prompt for verification") -> float:
        """Verify that enabling the adapter changes model logits."""
        if self._peft_model is None:
            raise RuntimeError("No adapter is loaded on this backend.")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        torch = self.torch
        try:
            with torch.no_grad():
                disable = getattr(self._peft_model, "disable_adapter", None)
                if disable is None:
                    raise RuntimeError("disable_adapter context manager not supported on this model.")
                with disable():
                    out_disabled = self.model(**inputs).logits
                out_enabled = self.model(**inputs).logits
                diff = (out_enabled - out_disabled).abs().max().item()
        finally:
            inputs.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if diff == 0.0:
            import logging
            logging.getLogger("lifemate.llm").warning(
                "Adapter verification notice: max logit difference is 0.0 (adapter may be early in training or near identity)."
            )
        return diff

    def _complete(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        self.last_generation = {}
        prompt = self._render(messages)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        self.last_prompt_tokens = len(prompt_ids)
        self.last_generation["prompt_tokens"] = len(prompt_ids)
        if len(prompt_ids) > self.max_input_tokens:
            raise ContextOverflow(
                f"prompt is {len(prompt_ids)} tokens, over the configured limit {self.max_input_tokens}"
            )
        if settings.max_context_tokens is not None and len(prompt_ids) + settings.max_new_tokens > settings.max_context_tokens:
            raise ContextOverflow(f"Prompt and generation budget exceed {settings.max_context_tokens} tokens")
        torch = self.torch
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(settings.max_new_tokens),
            "do_sample": settings.temperature > 0,
            "temperature": max(1e-5, settings.temperature),
            "top_p": settings.top_p,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if settings.top_k is not None:
            generate_kwargs["top_k"] = settings.top_k
        # Qwen3 otherwise materialises logits for every prompt position, which
        # costs several GB at 4-8K tokens on a T4.
        if self.supports_logits_to_keep:
            generate_kwargs["logits_to_keep"] = 1
        try:
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**inputs, **generate_kwargs)
            new_tokens = output[0, inputs["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            count = new_tokens.numel()
            self.last_generation.update(new_tokens=count, generate_seconds=time.perf_counter() - started,
                                        hit_token_cap=count >= settings.max_new_tokens)
        finally:
            # Release per-call KV/activation blocks; T4-class cards fragment quickly.
            inputs.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return text


def load_transformers_backend_from_env() -> TransformersBackend:
    """Convenience for Docker: model path can be injected without changing code."""
    model = os.environ.get("LIFEMATE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    device = os.environ.get("LIFEMATE_DEVICE", "cuda")
    dtype = os.environ.get("LIFEMATE_DTYPE", "bfloat16")
    max_tokens = int(os.environ.get("LIFEMATE_MAX_INPUT_TOKENS", "15200"))
    load_in_4bit = os.environ.get("LIFEMATE_LOAD_IN_4BIT", "0") in {"1", "true", "True", "yes"}
    return TransformersBackend(
        model,
        device=device,
        dtype=dtype,
        max_input_tokens=max_tokens,
        load_in_4bit=load_in_4bit,
        adapter_path=os.environ.get("LIFEMATE_ADAPTER") or None,
    )
