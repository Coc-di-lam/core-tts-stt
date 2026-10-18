"""Text-to-speech with local VieNeu.

Device "auto" picks CUDA when PyTorch sees a GPU (PyTorch backend, batching/stream scheduler as
before) and otherwise ONNX Runtime on CPU, where int8 graphs are ~1.5x faster than fp32 with the
same intelligibility on a VNNI-capable CPU. `stream()` yields audio while it is generated, so
the first sound is ready in ~0.2 s on CPU instead of after the whole sentence.
"""
from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator

from app.config import Settings

logger = logging.getLogger(__name__)

# Silence around the clip so the DAC / Bluetooth amp ramp-up doesn't clip the first syllable.
LEADING_PAD_SEC = 0.15
TRAILING_PAD_SEC = 0.08
_CACHE_ENTRIES = 128
_MODES = {"vieneu-v3-turbo": "v3turbo", "vieneu-v3-nano": "v3nano"}


class TtsError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class Speech:
    wav: bytes
    duration_sec: float  # spoken audio, without padding
    voice: str
    latency_ms: float
    cache_hit: bool


def resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch not installed: CPU-only install
        return "cpu"


class VieneuSynthesizer:
    def __init__(self, settings: Settings, model: Any = None):
        self.settings = settings
        self._model = model
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self.state = "ready" if model is not None else "idle"  # idle | loading | ready | error
        self.error: str | None = None
        self.device = resolve_device(settings.tts_device) if model is None else "cpu"
        self.backend = getattr(model, "backend", None)
        self.default_voice = settings.tts_voice

    # ---- model ----
    def load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                mode = _MODES.get(self.settings.tts_model)
                if mode is None:
                    raise TtsError(500, f"TTS model không được hỗ trợ: {self.settings.tts_model}")
                self.state = "loading"
                started = time.perf_counter()
                try:
                    self._model = self._create(mode)
                    self.backend = getattr(self._model, "backend", "onnx")
                    self.default_voice = self._pick_default_voice(self._model)
                    with self._infer_lock:  # warm-up: the first inference pays graph/session setup
                        self._model.infer("Xin chào.", voice=self.default_voice)
                except Exception as exc:
                    self._model = None
                    self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
                    logger.exception("Loading VieNeu failed")
                    raise TtsError(503, f"VieNeu TTS không khả dụng: {type(exc).__name__}") from exc
                self.state, self.error = "ready", None
                logger.info("VieNeu %s ready on %s/%s in %.1fs", mode, self.device, self.backend,
                            time.perf_counter() - started)
        return self._model

    def _create(self, mode: str) -> Any:
        from vieneu import Vieneu

        settings = self.settings
        if mode == "v3nano":  # ONNX/CPU only
            self.device = "cpu"
            return Vieneu(mode="v3nano", threads=settings.tts_cpu_threads)
        if self.device == "cpu":
            return Vieneu(mode="v3turbo", device="cpu", precision=settings.tts_cpu_precision,
                          threads=settings.tts_cpu_threads)
        return Vieneu(mode="v3turbo", device=self.device, max_batch_size=settings.tts_gpu_max_batch_size,
                      max_streams=settings.tts_gpu_max_streams)

    def _pick_default_voice(self, model: Any) -> str:
        voices = [voice_id for _, voice_id in model.list_preset_voices()]
        if self.settings.tts_voice in voices or not voices:
            return self.settings.tts_voice
        logger.warning("TTS_VOICE %r is not a preset of %s; using %r", self.settings.tts_voice,
                       self.settings.tts_model, voices[0])
        return voices[0]

    def voices(self) -> list[dict[str, str]]:
        return [{"id": voice_id, "label": label.lstrip("⭐ ").strip()}
                for label, voice_id in self.load().list_preset_voices()]

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.load(), "sample_rate", 48_000))

    def _sampling(self) -> dict[str, float]:
        return {"temperature": self.settings.tts_temperature, "top_k": self.settings.tts_top_k,
                "top_p": self.settings.tts_top_p}

    def _prepare(self, text: str, voice: str | None) -> tuple[str, str, str]:
        text = text.strip()
        if not text:
            raise TtsError(400, "Văn bản không được để trống.")
        if len(text) > self.settings.tts_max_chars:
            raise TtsError(413, f"Văn bản vượt {self.settings.tts_max_chars} ký tự.")
        voice = (voice or self.default_voice).strip()
        key = hashlib.sha256(f"{self.settings.tts_model}:{voice}:{text}".encode()).hexdigest()
        return text, voice, key

    def _cache_get(self, key: str) -> bytes | None:
        wav = self._cache.get(key)
        if wav is not None:
            self._cache.move_to_end(key)
        return wav

    def _cache_put(self, key: str, wav: bytes) -> None:
        self._cache[key] = wav
        if len(self._cache) > _CACHE_ENTRIES:
            self._cache.popitem(last=False)

    # ---- whole clip ----
    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        text, voice, key = self._prepare(text, voice)
        started = time.perf_counter()
        cached = self._cache_get(key)
        if cached is not None:
            return Speech(cached, _wav_duration(cached), voice, 0.0, True)

        model = self.load()
        try:
            with self._infer_lock:
                waveform = model.infer(text, voice=voice, **self._sampling())
        except ValueError as exc:  # unknown preset voice
            raise TtsError(422, str(exc)[:300]) from exc
        except Exception as exc:
            logger.exception("VieNeu synthesis failed")
            raise TtsError(503, f"VieNeu TTS lỗi: {type(exc).__name__}") from exc

        waveform = _as_float32(waveform)
        if waveform.size == 0:
            raise TtsError(503, "VieNeu trả audio rỗng.")
        wav = _encode_wav(waveform, self.sample_rate)
        self._cache_put(key, wav)
        return Speech(wav, round(waveform.size / self.sample_rate, 3), voice,
                      round((time.perf_counter() - started) * 1000, 1), False)

    # ---- streaming ----
    def stream(self, text: str, voice: str | None = None, cancel: threading.Event | None = None) -> Iterator[bytes]:
        """Yield PCM 16-bit LE mono chunks at `sample_rate`, padded like `synthesize`.

        Validation errors raise before the first chunk. Runs in a worker thread; set `cancel`
        to stop generating once the client is gone.
        """
        import numpy as np

        text, voice, key = self._prepare(text, voice)
        model = self.load()
        rate = self.sample_rate
        cached = self._cache_get(key)
        if cached is not None:
            yield _wav_pcm(cached)
            return
        if voice not in {voice_id for _, voice_id in model.list_preset_voices()}:
            raise TtsError(422, f"Voice '{voice}' not found.")

        # ONNX/CPU runs one generation at a time; the CUDA stream scheduler serves concurrent streams.
        lock = self._infer_lock if self.backend != "pytorch" else _NullLock()
        parts: list[np.ndarray] = []
        with lock:
            yield _to_pcm16(np.zeros(int(rate * LEADING_PAD_SEC), dtype=np.float32))
            try:
                for chunk in model.infer_stream(text, voice=voice, **self._sampling()):
                    if cancel is not None and cancel.is_set():
                        return
                    chunk = _as_float32(chunk)
                    parts.append(chunk)
                    yield _to_pcm16(chunk)
            except Exception:
                logger.exception("VieNeu streaming failed")
                raise
        yield _to_pcm16(np.zeros(int(rate * TRAILING_PAD_SEC), dtype=np.float32))
        if parts:
            self._cache_put(key, _encode_wav(np.concatenate(parts), rate))


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _as_float32(waveform: Any):
    import numpy as np

    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def _to_pcm16(waveform) -> bytes:
    import numpy as np

    return (np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _encode_wav(waveform, sample_rate: int) -> bytes:
    import numpy as np
    import soundfile as sf

    padded = np.concatenate([np.zeros(int(sample_rate * LEADING_PAD_SEC), dtype=np.float32), waveform,
                             np.zeros(int(sample_rate * TRAILING_PAD_SEC), dtype=np.float32)])
    buffer = io.BytesIO()
    sf.write(buffer, padded, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _wav_pcm(wav: bytes) -> bytes:
    import soundfile as sf

    data, _ = sf.read(io.BytesIO(wav), dtype="int16")
    return data.astype("<i2").tobytes()


def _wav_duration(wav: bytes) -> float:
    import soundfile as sf

    info = sf.info(io.BytesIO(wav))
    return round(max(info.duration - LEADING_PAD_SEC - TRAILING_PAD_SEC, 0.0), 3)
