"""Speech-to-text through the Gemini transcription model (Interactions API)."""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import httpx

from app.config import Settings

SUPPORTED_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/mp3", "audio/mpeg", "audio/aiff", "audio/aac",
    "audio/ogg", "audio/flac", "audio/m4a", "audio/mp4", "audio/x-m4a", "audio/l16", "audio/opus",
    "audio/webm", "video/webm",
}
_EXTENSION_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mp3", ".aiff": "audio/aiff", ".aac": "audio/aac",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/opus", ".flac": "audio/flac",
    ".m4a": "audio/m4a", ".webm": "audio/webm",
}
_MODES = {"smart", "verbatim"}


class SttError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class Transcript:
    text: str
    model: str
    language_codes: list[str]
    mode: str
    latency_ms: float


def resolve_mime_type(content_type: str | None, filename: str | None) -> str:
    """Normalize the upload's MIME type; browsers send e.g. 'audio/webm;codecs=opus'."""
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime in SUPPORTED_MIME_TYPES:
        return "audio/webm" if mime == "video/webm" else mime
    suffix = "." + (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if suffix in _EXTENSION_MIME:
        return _EXTENSION_MIME[suffix]
    raise SttError(415, f"Định dạng audio không được hỗ trợ: {content_type or filename or 'unknown'}")


class GeminiTranscriber:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.stt_timeout_sec)

    @property
    def configured(self) -> bool:
        return bool(self.settings.gemini_api_key)

    async def transcribe(self, audio: bytes, mime_type: str, language_codes: list[str] | None = None,
                         mode: str | None = None, vocabulary: list[str] | None = None) -> Transcript:
        if not self.configured:
            raise SttError(503, "Chưa cấu hình GEMINI_API_KEY.")
        if not audio:
            raise SttError(400, "File âm thanh rỗng.")
        if len(audio) > self.settings.stt_max_bytes:
            raise SttError(413, f"Audio vượt giới hạn {self.settings.stt_max_bytes // (1024 * 1024)} MB.")
        mode = (mode or self.settings.stt_mode).strip().lower()
        if mode not in _MODES:
            raise SttError(422, "mode phải là 'smart' hoặc 'verbatim'.")
        languages = self.settings.language_codes if language_codes is None else language_codes

        transcription_config: dict = {"mode": mode if mode == "smart" else {"type": "verbatim"}}
        if languages:
            transcription_config["language_codes"] = languages
        if vocabulary:
            transcription_config["custom_vocabulary"] = vocabulary[:1000]
        body = {
            "model": self.settings.stt_model,
            "input": [{"type": "audio", "data": base64.b64encode(audio).decode("ascii"), "mime_type": mime_type}],
            "generation_config": {"transcription_config": transcription_config},
        }

        started = time.perf_counter()
        try:
            response = await self.client.post(
                self.settings.gemini_api_url, json=body,
                headers={"x-goog-api-key": self.settings.gemini_api_key},
            )
        except httpx.TimeoutException as exc:
            raise SttError(504, "Gemini transcribe quá thời gian chờ.") from exc
        except httpx.HTTPError as exc:
            raise SttError(502, f"Không gọi được Gemini: {type(exc).__name__}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        if response.status_code != 200:
            raise SttError(502 if response.status_code >= 500 else response.status_code,
                           f"Gemini trả lỗi {response.status_code}: {_error_message(response)}")
        payload = response.json()
        if payload.get("status") not in (None, "completed"):
            raise SttError(502, f"Gemini interaction chưa hoàn tất: {payload.get('status')}")
        return Transcript(text=_output_text(payload), model=payload.get("model") or self.settings.stt_model,
                          language_codes=languages, mode=mode, latency_ms=latency_ms)

    @property
    def active_provider(self) -> str:
        return "gemini"

    @property
    def active_model(self) -> str:
        return self.settings.stt_model

    async def aclose(self) -> None:
        await self.client.aclose()


class ResilientTranscriber:
    """Combines Gipformer (local SOTA Vietnamese ASR) with automatic fallback to Gemini STT."""

    def __init__(self, settings: Settings, gipformer: Any | None = None, gemini: GeminiTranscriber | None = None):
        self.settings = settings
        self.gemini = gemini or GeminiTranscriber(settings)
        if gipformer is not None:
            self.gipformer = gipformer
        else:
            try:
                from app.stt_gipformer import GipformerTranscriber
                self.gipformer = GipformerTranscriber(settings)
            except Exception as exc:
                self.gipformer = None

    @property
    def configured(self) -> bool:
        backend = self.settings.stt_backend.strip().lower()
        gip_ok = bool(self.gipformer and self.gipformer.configured and self.gipformer.is_available())
        gem_ok = bool(self.gemini and self.gemini.configured)
        if backend == "gemini":
            return gem_ok
        if backend == "gipformer":
            return gip_ok
        return gip_ok or gem_ok

    @property
    def active_provider(self) -> str:
        backend = self.settings.stt_backend.strip().lower()
        if backend == "gemini":
            return "gemini"
        if backend == "gipformer":
            return "gipformer"
        if self.gipformer is not None and self.gipformer.configured and self.gipformer.is_available():
            return "gipformer"
        return "gemini"

    @property
    def active_model(self) -> str:
        backend = self.settings.stt_backend.strip().lower()
        if backend == "gemini":
            return self.settings.stt_model
        if backend == "gipformer":
            return getattr(self.gipformer, "model_name", "gipformer1.5-68M-rnnt")
        if self.gipformer is not None and self.gipformer.configured and self.gipformer.is_available():
            return self.gipformer.model_name
        return self.settings.stt_model

    async def transcribe(self, audio: bytes, mime_type: str, language_codes: list[str] | None = None,
                         mode: str | None = None, vocabulary: list[str] | None = None) -> Transcript:
        backend = self.settings.stt_backend.strip().lower()
        if backend == "gemini":
            return await self.gemini.transcribe(audio, mime_type, language_codes, mode, vocabulary)

        if backend == "gipformer":
            if self.gipformer is None or not self.gipformer.is_available():
                raise SttError(503, "Gipformer STT chưa sẵn sàng hoặc chưa nạp xong.")
            import asyncio
            return await asyncio.to_thread(self.gipformer.transcribe, audio, mime_type, mode)

        # backend == "auto":
        if self.gipformer is not None and self.gipformer.is_available():
            try:
                import asyncio
                return await asyncio.to_thread(self.gipformer.transcribe, audio, mime_type, mode)
            except Exception as exc:
                if not self.gemini.configured:
                    raise
                import logging
                logging.getLogger(__name__).warning("Gipformer transcription failed (%s), falling back to Gemini STT", exc)

        return await self.gemini.transcribe(audio, mime_type, language_codes, mode, vocabulary)

    async def aclose(self) -> None:
        await self.gemini.aclose()


def create_transcriber(settings: Settings) -> GeminiTranscriber | ResilientTranscriber:
    gemini = GeminiTranscriber(settings)
    if settings.stt_backend.strip().lower() == "gemini":
        return gemini
    return ResilientTranscriber(settings, gemini=gemini)


def _output_text(payload: dict) -> str:
    parts = [
        content.get("text", "")
        for step in payload.get("steps") or []
        if step.get("type") == "model_output"
        for content in step.get("content") or []
        if content.get("type") == "text"
    ]
    return " ".join(part.strip() for part in parts if part.strip())


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        return str(error.get("message") or error)[:300]
    except ValueError:
        return response.text[:300]
