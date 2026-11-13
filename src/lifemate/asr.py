"""Optional ASR adapters. Training data already has transcripts; public/private need one of these."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class FasterWhisperTranscriber:
    """faster-whisper / CTranslate2 adapter (GPU recommended).

    This is intentionally optional: the core planner/caller can be tested on the
    labeled train split without loading any ASR model.
    """

    def __init__(
        self,
        model_name_or_path: str = "large-v3",
        *,
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "vi",
        beam_size: int = 5,
        cache_path: str | Path | None = None,
        local_files_only: bool = False,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install `faster-whisper` to use FasterWhisperTranscriber") from exc
        self.model = WhisperModel(
            model_name_or_path,
            device=device,
            compute_type=compute_type,
            local_files_only=local_files_only,
        )
        self.language = language
        self.beam_size = int(beam_size)
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _cache_key(self, path: Path, cache_key: str) -> str:
        try:
            stat = path.stat()
            stamp = f"{int(stat.st_mtime)}:{stat.st_size}"
        except OSError:
            stamp = "0:0"
        return f"{cache_key}|{path.resolve()}|{stamp}"

    def transcribe(self, path: Path, cache_key: str = "") -> str:
        key = self._cache_key(path, cache_key)
        if key in self.cache:
            return self.cache[key]
        segments, _info = self.model.transcribe(
            str(path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        self.cache[key] = text
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.cache_path)
        return text


class CommandTranscriber:
    """Adapter for a custom offline ASR binary (e.g. Gipformer ONNX).

    The command must print one transcript line to stdout. ``{audio}`` is replaced
    with the absolute audio path.
    """

    def __init__(self, command: str, cache_path: str | Path | None = None):
        self.command = command
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def transcribe(self, path: Path, cache_key: str = "") -> str:
        key = f"{cache_key}|{path.resolve()}"
        if key in self.cache:
            return self.cache[key]
        cmd = self.command.replace("{audio}", str(path.resolve()))
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"ASR command failed ({proc.returncode}): {proc.stderr[-1000:]}")
        text = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        self.cache[key] = text
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return text


class SherpaOnnxTranscriber:
    """Offline Vietnamese ASR with a sherpa-onnx transducer (Gipformer).

    The int8 Gipformer model is ~73 MB and runs comfortably on CPU (RTF ~0.02 on
    a modern laptop), which leaves the GPU almost entirely to the LLM. Pass
    ``model_dir`` in the final image; ``download=True`` is for development only.
    """

    FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        repo_id: str = "g-group-ai-lab/gipformer1.5-68M-rnnt",
        revision: str | None = None,
        num_threads: int = 4,
        decoding_method: str = "modified_beam_search",
        max_active_paths: int = 4,
        cache_path: str | Path | None = None,
        lowercase: bool = True,
    ):
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install `sherpa-onnx` and `sherpa-onnx-core` to use this transcriber") from exc
        if model_dir is None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Pass model_dir or install huggingface_hub for the dev download path") from exc
            model_dir = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    allow_patterns=list(self.FILES),
                )
            )
        model_dir = Path(model_dir)
        def _find_file(base: str) -> Path:
            for candidate in (f"{base}.int8.onnx", f"{base}.onnx"):
                p = model_dir / candidate
                if p.is_file():
                    return p
            raise FileNotFoundError(f"Missing {base}.int8.onnx or {base}.onnx in {model_dir}")

        encoder_path = _find_file("encoder")
        decoder_path = _find_file("decoder")
        joiner_path = _find_file("joiner")
        tokens_path = model_dir / "tokens.txt"
        if not tokens_path.is_file():
            raise FileNotFoundError(f"Missing tokens.txt in {model_dir}")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder_path),
            decoder=str(decoder_path),
            joiner=str(joiner_path),
            tokens=str(tokens_path),
            num_threads=int(num_threads),
            sample_rate=16000,
            feature_dim=80,
            decoding_method=str(decoding_method),
            max_active_paths=int(max_active_paths),
        )
        self.lowercase = bool(lowercase)
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _cache_key(self, path: Path, cache_key: str) -> str:
        try:
            stat = path.stat()
            stamp = f"{int(stat.st_mtime)}:{stat.st_size}"
        except OSError:
            stamp = "0:0"
        return f"{cache_key}|{path.resolve()}|{stamp}"

    def transcribe(self, path: Path, cache_key: str = "") -> str:
        key = self._cache_key(path, cache_key)
        if key in self.cache:
            return self.cache[key]
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install `soundfile` to read audio for the sherpa-onnx transcriber") from exc
        samples, sample_rate = sf.read(str(path), dtype="float32")
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_streams([stream])
        text = " ".join(str(stream.result.text or "").split())
        if self.lowercase:
            text = text.lower()
        self.cache[key] = text
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.cache_path)
        return text

