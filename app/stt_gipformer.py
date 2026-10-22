"""Vietnamese speech-to-text using Gipformer (g-group-ai-lab/gipformer1.5-68M-rnnt).

Uses sherpa-onnx for fast on-device inference with Zipformer-Transducer architecture.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import Settings
from app.stt import SttError, Transcript

# On Windows, ensure sherpa-onnx's bundled onnxruntime DLL takes precedence over System32
if sys.platform == "win32":
    try:
        import ctypes
        import importlib.util
        spec = importlib.util.find_spec("sherpa_onnx")
        if spec and spec.origin:
            lib_dir = Path(spec.origin).parent / "lib"
            if lib_dir.is_dir():
                os.add_dll_directory(str(lib_dir))
                dll_path = lib_dir / "onnxruntime.dll"
                if dll_path.is_file():
                    ctypes.CDLL(str(dll_path))
    except Exception:
        pass

logger = logging.getLogger(__name__)

REPO_IDS = {
    "1": "g-group-ai-lab/gipformer-68M-rnnt",
    "1.5": "g-group-ai-lab/gipformer1.5-68M-rnnt",
}
TARGET_SAMPLE_RATE = 16000
FEATURE_DIM = 80

ONNX_FILES = {
    "fp32": {
        "encoder": "encoder.onnx",
        "decoder": "decoder.onnx",
        "joiner": "joiner.onnx",
    },
    "int8": {
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "joiner": "joiner.int8.onnx",
    },
}
TOKENS_FILE = "tokens.txt"


class GipformerTranscriber:
    """Local Vietnamese ASR engine using Gipformer 1.5 RNN-T with sherpa-onnx."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.version = str(settings.gipformer_version)
        self.quantize = settings.gipformer_quantize.lower()
        self.decoding_method = settings.gipformer_decoding_method
        self.num_threads = settings.gipformer_num_threads
        self.model_name = f"gipformer{self.version}-68M-rnnt"
        self._recognizer: Any = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.state = "idle"  # idle | loading | ready | error
        self.error: str | None = None

    @property
    def configured(self) -> bool:
        return self.state != "error"

    def is_available(self) -> bool:
        return self.state == "ready"

    def load(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        with self._load_lock:
            if self._recognizer is None:
                self.state = "loading"
                try:
                    import sherpa_onnx
                except ImportError as exc:
                    self.state, self.error = "error", "Chưa cài đặt thư viện sherpa-onnx."
                    logger.warning("Gipformer unavailable: %s", exc)
                    raise SttError(503, self.error) from exc

                model_paths = self._resolve_model_paths()
                try:
                    self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                        encoder=model_paths["encoder"],
                        decoder=model_paths["decoder"],
                        joiner=model_paths["joiner"],
                        tokens=model_paths["tokens"],
                        num_threads=self.num_threads,
                        sample_rate=TARGET_SAMPLE_RATE,
                        feature_dim=FEATURE_DIM,
                        decoding_method=self.decoding_method,
                    )
                    self.state, self.error = "ready", None
                    logger.info("Gipformer %s ready (quantize=%s, decoding=%s, threads=%d)",
                                self.model_name, self.quantize, self.decoding_method, self.num_threads)
                except Exception as exc:
                    self._recognizer = None
                    self.state, self.error = "error", f"Không thể tải model Gipformer: {exc}"
                    logger.exception("Loading Gipformer failed")
                    raise SttError(503, self.error) from exc
        return self._recognizer

    def _resolve_model_paths(self) -> dict[str, str]:
        filenames = dict(ONNX_FILES.get(self.quantize, ONNX_FILES["fp32"]), tokens=TOKENS_FILE)
        model_dir = self.settings.gipformer_model_dir.strip()
        if model_dir:
            directory = Path(model_dir).expanduser()
            if directory.is_dir():
                paths = {}
                for key, fname in filenames.items():
                    p = directory / fname
                    if not p.exists():
                        raise SttError(500, f"Thiếu file model Gipformer: {fname} tại {directory}")
                    paths[key] = str(p)
                return paths

        repo_id = REPO_IDS.get(self.version, REPO_IDS["1.5"])
        try:
            from huggingface_hub import hf_hub_download
            paths = {
                key: hf_hub_download(repo_id=repo_id, filename=fname)
                for key, fname in filenames.items()
            }
            return paths
        except Exception as exc:
            raise SttError(503, f"Không thể tải trọng số Gipformer từ HuggingFace ({repo_id}): {exc}") from exc

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
        """Transcribe raw PCM16 LE mono bytes."""
        if len(pcm) < 2:
            return ""
        recognizer = self.load()
        samples = np.frombuffer(pcm[:len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0
        if sample_rate != TARGET_SAMPLE_RATE:
            samples = _resample(samples, sample_rate, TARGET_SAMPLE_RATE)
        with self._infer_lock:
            stream = recognizer.create_stream()
            stream.accept_waveform(TARGET_SAMPLE_RATE, samples)
            recognizer.decode_streams([stream])
            return stream.result.text.strip()

    def transcribe(self, audio: bytes, mime_type: str, mode: str | None = None) -> Transcript:
        """Transcribe audio bytes (WAV, MP3, OGG, etc.)."""
        started = time.perf_counter()
        if not audio:
            raise SttError(400, "File âm thanh rỗng.")
        try:
            data, sr = sf.read(io.BytesIO(audio), dtype="float32")
        except Exception as exc:
            raise SttError(415, f"Gipformer không thể giải mã định dạng file: {exc}") from exc
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SAMPLE_RATE:
            data = _resample(data, sr, TARGET_SAMPLE_RATE)
        data = np.asarray(data, dtype=np.float32)

        recognizer = self.load()
        with self._infer_lock:
            stream = recognizer.create_stream()
            stream.accept_waveform(TARGET_SAMPLE_RATE, data)
            recognizer.decode_streams([stream])
            text = stream.result.text.strip()

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return Transcript(
            text=text,
            model=self.model_name,
            language_codes=["vi-VN"],
            mode=mode or "smart",
            latency_ms=latency_ms,
        )


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import scipy.signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        return scipy.signal.resample(audio, num_samples).astype(np.float32)
    except Exception:
        indices = np.linspace(0, len(audio) - 1, int(len(audio) * target_sr / orig_sr))
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
