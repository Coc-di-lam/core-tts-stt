"""Text-to-speech: local VieNeu or ElevenLabs v4 Turbo (cloud).

`TTS_BACKEND=auto` runs VieNeu when PyTorch sees an NVIDIA GPU (best-quality settings) and
ElevenLabs v4 Turbo otherwise; "vieneu" / "elevenlabs" force one. Both backends expose the same interface.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

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
    except Exception:
        return "cpu"


def resolve_backend(settings: Settings) -> str:
    backend = settings.tts_backend.strip().lower()
    if backend in ("vieneu", "elevenlabs"):
        return backend
    if backend == "gemini":
        return "elevenlabs"
    if backend != "auto":
        raise ValueError(f"TTS_BACKEND phải là auto | vieneu | elevenlabs, không phải {settings.tts_backend!r}")
    if resolve_device("auto") == "cuda":
        return "vieneu"
    if shutil.which("nvidia-smi"):
        logger.warning("NVIDIA GPU found but PyTorch CUDA is not installed (`uv sync --extra cuda`); "
                       "using ElevenLabs TTS")
    return "elevenlabs"


def create_synthesizer(settings: Settings) -> "VieneuSynthesizer | ElevenLabsSynthesizer":
    if resolve_backend(settings) in ("elevenlabs", "gemini"):
        return ElevenLabsSynthesizer(settings)
    if settings.tts_backend.strip().lower() == "auto":
        settings = settings.model_copy(update={"tts_device": "cuda"})
    return VieneuSynthesizer(settings)


class _CachedSynthesizer:
    """Input validation and an LRU of finished clips, shared by both backends."""

    model_name = ""
    default_voice = ""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    def _prepare(self, text: str, voice: str | None) -> tuple[str, str, str]:
        text = text.strip()
        if not text:
            raise TtsError(400, "Văn bản không được để trống.")
        if len(text) > self.settings.tts_max_chars:
            raise TtsError(413, f"Văn bản vượt {self.settings.tts_max_chars} ký tự.")
        voice = (voice or self.default_voice).strip()
        key = hashlib.sha256(f"{self.model_name}:{voice}:{text}".encode()).hexdigest()
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


class ElevenLabsSynthesizer(_CachedSynthesizer):
    """ElevenLabs TTS over REST & Streaming API (supports Eleven v4 Turbo)."""

    provider = "elevenlabs"
    device = "cloud"
    backend = "elevenlabs"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        super().__init__(settings)
        self.model_name = settings.elevenlabs_tts_model
        self.sample_rate = settings.elevenlabs_tts_sample_rate
        self.output_format = settings.elevenlabs_tts_output_format
        self.client = client or httpx.Client(timeout=settings.elevenlabs_tts_timeout_sec)
        self.default_voice = settings.elevenlabs_tts_voice
        configured = bool(settings.elevenlabs_api_key)
        self.state = "ready" if configured else "error"
        self.error = None if configured else "Chưa cấu hình ELEVENLABS_API_KEY."
        self._cached_voices: list[dict[str, str]] | None = None

    def load(self) -> None:
        if self.error:
            raise TtsError(503, self.error)

    def voices(self) -> list[dict[str, str]]:
        if self._cached_voices is not None:
            return self._cached_voices
        if not self.settings.elevenlabs_api_key:
            return [{"id": self.default_voice, "label": "Nhu - Calm and Confident (Default)"}]

        url = f"{self.settings.elevenlabs_api_url.rstrip('/')}/v1/voices"
        try:
            resp = self.client.get(url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json().get("voices", [])
                voices_list = [
                    {"id": v["voice_id"], "label": f"{v.get('name', v['voice_id'])}"}
                    for v in data
                ]
                # Ensure default_voice is at the top of the list
                def_item = next((v for v in voices_list if v["id"] == self.default_voice), None)
                if def_item:
                    voices_list.remove(def_item)
                    if "(Default)" not in def_item["label"]:
                        def_item["label"] = f"{def_item['label']} (Default)"
                    voices_list.insert(0, def_item)
                else:
                    voices_list.insert(0, {"id": self.default_voice, "label": "Nhu - Calm and Confident (Default)"})
                self._cached_voices = voices_list
                return self._cached_voices
        except Exception as exc:
            logger.warning("Fetching ElevenLabs voices failed: %s", exc)

        return [{"id": self.default_voice, "label": "Nhu - Calm and Confident (Default)"}]

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self.settings.elevenlabs_api_key,
            "Content-Type": "application/json",
        }

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        text, voice, key = self._prepare(text, voice)
        cached = self._cache_get(key)
        if cached is not None:
            return Speech(cached, _wav_duration(cached), voice, 0.0, True)
        self.load()

        url = f"{self.settings.elevenlabs_api_url.rstrip('/')}/v1/text-to-speech/{voice}?output_format={self.output_format}"
        body = {
            "text": text,
            "model_id": self.model_name,
        }

        started = time.perf_counter()
        try:
            response = self.client.post(url, json=body, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise TtsError(504, "ElevenLabs TTS quá thời gian chờ.") from exc
        except httpx.HTTPError as exc:
            raise TtsError(502, f"Không gọi được ElevenLabs TTS: {type(exc).__name__}") from exc

        if response.status_code != 200:
            raise TtsError(502 if response.status_code >= 500 else response.status_code,
                           f"ElevenLabs TTS lỗi {response.status_code}: {_elevenlabs_error_message(response.text)}")

        waveform = _pcm_to_float32(response.content)
        if waveform.size == 0:
            raise TtsError(502, "ElevenLabs TTS trả audio rỗng.")
        wav = _encode_wav(waveform, self.sample_rate)
        self._cache_put(key, wav)
        return Speech(
            wav,
            round(waveform.size / self.sample_rate, 3),
            voice,
            round((time.perf_counter() - started) * 1000, 1),
            False,
        )

    def stream(self, text: str, voice: str | None = None, cancel: threading.Event | None = None) -> Iterator[bytes]:
        import numpy as np

        text, voice, key = self._prepare(text, voice)
        cached = self._cache_get(key)
        if cached is not None:
            yield _wav_pcm(cached)
            return
        self.load()

        rate = self.sample_rate
        url = f"{self.settings.elevenlabs_api_url.rstrip('/')}/v1/text-to-speech/{voice}/stream?output_format={self.output_format}"
        body = {
            "text": text,
            "model_id": self.model_name,
        }

        parts: list[bytes] = []
        try:
            with self.client.stream("POST", url, json=body, headers=self._headers()) as response:
                if response.status_code != 200:
                    err_text = response.read().decode("utf-8", "replace")
                    raise TtsError(
                        502 if response.status_code >= 500 else response.status_code,
                        f"ElevenLabs TTS stream lỗi {response.status_code}: {_elevenlabs_error_message(err_text)}",
                    )
                # Output leading silence padding
                yield _to_pcm16(np.zeros(int(rate * LEADING_PAD_SEC), dtype=np.float32))

                odd = b""
                for chunk in response.iter_bytes(chunk_size=4096):
                    if cancel is not None and cancel.is_set():
                        return
                    if not chunk:
                        continue
                    pcm = odd + chunk
                    cut = len(pcm) - len(pcm) % 2
                    pcm, odd = pcm[:cut], pcm[cut:]
                    if pcm:
                        parts.append(pcm)
                        yield pcm

        except httpx.TimeoutException as exc:
            raise TtsError(504, "ElevenLabs TTS quá thời gian chờ.") from exc
        except httpx.HTTPError as exc:
            raise TtsError(502, f"Không gọi được ElevenLabs TTS: {type(exc).__name__}") from exc

        # Output trailing silence padding
        yield _to_pcm16(np.zeros(int(rate * TRAILING_PAD_SEC), dtype=np.float32))
        if parts:
            full_pcm = b"".join(parts)
            self._cache_put(key, _encode_wav(_pcm_to_float32(full_pcm), rate))


# Backward-compatible alias
GeminiSynthesizer = ElevenLabsSynthesizer


class VieneuSynthesizer(_CachedSynthesizer):
    provider = "vieneu"

    def __init__(self, settings: Settings, model: Any = None, fallback: ElevenLabsSynthesizer | None = None):
        super().__init__(settings)
        self.model_name = settings.tts_model
        self._model = model
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.state = "ready" if model is not None else "idle"
        self.error: str | None = None
        self.device = resolve_device(settings.tts_device) if model is None else "cpu"
        self.backend = getattr(model, "backend", None)
        self.default_voice = settings.tts_voice
        self._fallback = fallback
        is_auto = settings.tts_backend.strip().lower() == "auto"
        if self._fallback is None and is_auto and settings.tts_fallback_online and settings.elevenlabs_api_key:
            self._fallback = ElevenLabsSynthesizer(settings)

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
                    with self._infer_lock:
                        self._model.infer("Xin chào.", voice=self.default_voice)
                except Exception as exc:
                    self._model = None
                    self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
                    logger.exception("Loading VieNeu failed")
                    if self.settings.tts_fallback_online and self._fallback:
                        logger.warning("VieNeu TTS không khả dụng (%s), tự động kích hoạt dự phòng ElevenLabs TTS", exc)
                        return None
                    raise TtsError(503, f"VieNeu TTS không khả dụng: {type(exc).__name__}") from exc
                self.state, self.error = "ready", None
                logger.info("VieNeu %s ready on %s/%s in %.1fs", mode, self.device, self.backend,
                            time.perf_counter() - started)
        return self._model

    def _create(self, mode: str) -> Any:
        from vieneu import Vieneu

        settings = self.settings
        if mode == "v3nano":
            self.device = "cpu"
            return Vieneu(mode="v3nano", threads=settings.tts_cpu_threads)
        if self.device == "cpu":
            return Vieneu(mode="v3turbo", device="cpu", precision=settings.tts_cpu_precision,
                          threads=settings.tts_cpu_threads)
        return Vieneu(mode="v3turbo", device=self.device, dtype=settings.tts_gpu_dtype,
                      max_batch_size=settings.tts_gpu_max_batch_size, max_streams=settings.tts_gpu_max_streams)

    def _pick_default_voice(self, model: Any) -> str:
        voices = [voice_id for _, voice_id in model.list_preset_voices()]
        if self.settings.tts_voice in voices or not voices:
            return self.settings.tts_voice
        logger.warning("TTS_VOICE %r is not a preset of %s; using %r", self.settings.tts_voice,
                       self.settings.tts_model, voices[0])
        return voices[0]

    def voices(self) -> list[dict[str, str]]:
        if self.state != "error":
            try:
                m = self.load()
                if m is not None:
                    return [{"id": voice_id, "label": label.lstrip("⭐ ").strip()}
                            for label, voice_id in m.list_preset_voices()]
            except Exception:
                pass
        if self._fallback:
            return self._fallback.voices()
        return []

    @property
    def sample_rate(self) -> int:
        if self.state == "error" and self._fallback:
            return self._fallback.sample_rate
        try:
            m = self.load()
            return int(getattr(m, "sample_rate", 48_000)) if m is not None else 48_000
        except Exception:
            if self._fallback:
                return self._fallback.sample_rate
            return 48_000

    def _sampling(self) -> dict[str, Any]:
        return {
            "temperature": self.settings.tts_temperature,
            "top_k": self.settings.tts_top_k,
            "top_p": self.settings.tts_top_p,
            "repetition_penalty": self.settings.tts_repetition_penalty,
            "apply_watermark": self.settings.tts_apply_watermark,
        }

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        text_prep, voice_prep, key = self._prepare(text, voice)
        started = time.perf_counter()
        cached = self._cache_get(key)
        if cached is not None:
            return Speech(cached, _wav_duration(cached), voice_prep, 0.0, True)

        if self.state == "error" and self.settings.tts_fallback_online and self._fallback:
            return self._fallback.synthesize(text, voice)

        try:
            model = self.load()
            if model is None and self.settings.tts_fallback_online and self._fallback:
                return self._fallback.synthesize(text, voice)
            preset_voices = {voice_id for _, voice_id in model.list_preset_voices()} if model is not None else set()
            if preset_voices and voice_prep not in preset_voices:
                raise TtsError(422, f"Voice '{voice_prep}' not found.")
            with self._infer_lock:
                waveform = model.infer(text_prep, voice=voice_prep, **self._sampling())
            waveform = _as_float32(waveform)
            if waveform.size == 0:
                raise TtsError(503, "VieNeu trả audio rỗng.")
            wav = _encode_wav(waveform, self.sample_rate)
            self._cache_put(key, wav)
            return Speech(wav, round(waveform.size / self.sample_rate, 3), voice_prep,
                          round((time.perf_counter() - started) * 1000, 1), False)
        except Exception as exc:
            if isinstance(exc, TtsError) and exc.status_code < 500:
                raise
            if self.settings.tts_fallback_online and self._fallback:
                logger.warning("VieNeu synthesize thất bại (%s), tự động chuyển sang ElevenLabs TTS", exc)
                return self._fallback.synthesize(text, voice)
            if isinstance(exc, TtsError):
                raise
            logger.exception("VieNeu synthesis failed")
            raise TtsError(503, f"VieNeu TTS lỗi: {type(exc).__name__}") from exc

    def stream(self, text: str, voice: str | None = None, cancel: threading.Event | None = None) -> Iterator[bytes]:
        import numpy as np

        text_prep, voice_prep, key = self._prepare(text, voice)
        cached = self._cache_get(key)
        if cached is not None:
            yield _wav_pcm(cached)
            return

        if self.state == "error" and self.settings.tts_fallback_online and self._fallback:
            logger.warning("VieNeu ở trạng thái lỗi, stream tự động chuyển sang ElevenLabs TTS")
            yield from self._fallback.stream(text, voice, cancel)
            return

        try:
            model = self.load()
            if model is None and self.settings.tts_fallback_online and self._fallback:
                yield from self._fallback.stream(text, voice, cancel)
                return
            preset_voices = {voice_id for _, voice_id in model.list_preset_voices()} if model is not None else set()
            if preset_voices and voice_prep not in preset_voices:
                raise TtsError(422, f"Voice '{voice_prep}' not found.")

            rate = self.sample_rate
            lock = self._infer_lock if self.backend != "pytorch" else _NullLock()
            parts: list[np.ndarray] = []
            with lock:
                yield _to_pcm16(np.zeros(int(rate * LEADING_PAD_SEC), dtype=np.float32))
                try:
                    for chunk in model.infer_stream(text_prep, voice=voice_prep, **self._sampling()):
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
        except Exception as exc:
            if isinstance(exc, TtsError) and exc.status_code < 500:
                raise
            if self.settings.tts_fallback_online and self._fallback:
                logger.warning("VieNeu stream thất bại (%s), tự động chuyển sang ElevenLabs TTS", exc)
                yield from self._fallback.stream(text, voice, cancel)
                return
            if isinstance(exc, TtsError):
                raise
            raise


def _elevenlabs_error_message(text: str) -> str:
    try:
        data = json.loads(text)
        detail = data.get("detail") or data.get("error") or {}
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)[:300]
        return str(detail)[:300]
    except Exception:
        return text[:300]


def _pcm_to_float32(audio: bytes):
    """Raw PCM16 LE -> float32; also accepts a WAV in case returned with header."""
    import numpy as np
    import soundfile as sf

    if audio[:4] == b"RIFF":
        data, _ = sf.read(io.BytesIO(audio), dtype="float32")
        return np.asarray(data, dtype=np.float32).reshape(-1)
    audio = audio[:len(audio) - len(audio) % 2]
    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0


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
