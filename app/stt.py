"""Speech-to-text through the ElevenLabs transcription model (Scribe v2 API)."""
from __future__ import annotations

import io
import logging
import mimetypes
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, split_csv

logger = logging.getLogger(__name__)

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
_MIME_EXTENSION = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/flac": ".flac", "audio/m4a": ".m4a",
    "audio/aac": ".aac", "audio/opus": ".opus", "audio/webm": ".webm", "video/webm": ".webm",
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


def _map_language_code(codes: list[str] | str | None) -> str | None:
    if not codes:
        return None
    code = codes[0] if isinstance(codes, list) else codes
    code = code.strip().lower()
    if code in ("auto", "all", ""):
        return None
    if code in ("vi", "vi-vn", "vie", "vietnamese"):
        return "vie"
    if code in ("en", "en-us", "eng", "english"):
        return "eng"
    return code.split("-")[0]


class ElevenLabsTranscriber:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.stt_timeout_sec)

    @property
    def configured(self) -> bool:
        return bool(self.settings.elevenlabs_api_key)

    async def transcribe(self, audio: bytes, mime_type: str, language_codes: list[str] | None = None,
                         mode: str | None = None, vocabulary: list[str] | None = None) -> Transcript:
        if not self.configured:
            raise SttError(503, "Chưa cấu hình ELEVENLABS_API_KEY.")
        if not audio:
            raise SttError(400, "File âm thanh rỗng.")
        if len(audio) > self.settings.stt_max_bytes:
            raise SttError(413, f"Audio vượt giới hạn {self.settings.stt_max_bytes // (1024 * 1024)} MB.")
        mode = (mode or self.settings.stt_mode).strip().lower()
        if mode not in _MODES:
            raise SttError(422, "mode phải là 'smart' hoặc 'verbatim'.")

        languages = self.settings.language_codes if language_codes is None else language_codes
        lang_code = _map_language_code(languages) or self.settings.stt_language_code

        ext = _MIME_EXTENSION.get(mime_type, ".wav")
        files = {"file": (f"audio{ext}", audio, mime_type)}
        data: dict[str, Any] = {
            "model_id": self.settings.stt_model,
            "no_verbatim": "true" if mode == "smart" else "false",
        }
        if lang_code:
            data["language_code"] = lang_code
        if vocabulary:
            data["keyterms"] = vocabulary

        url = f"{self.settings.elevenlabs_api_url.rstrip('/')}/v1/speech-to-text"
        started = time.perf_counter()
        try:
            response = await self.client.post(
                url,
                data=data,
                files=files,
                headers={"xi-api-key": self.settings.elevenlabs_api_key},
            )
        except httpx.TimeoutException as exc:
            raise SttError(504, "ElevenLabs Scribe transcribe quá thời gian chờ.") from exc
        except httpx.HTTPError as exc:
            raise SttError(502, f"Không gọi được ElevenLabs: {type(exc).__name__}") from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if response.status_code != 200:
            raise SttError(502 if response.status_code >= 500 else response.status_code,
                           f"ElevenLabs trả lỗi {response.status_code}: {_error_message(response)}")

        payload = response.json()
        text = str(payload.get("text") or "").strip()
        if mode == "smart":
            from app.fluency import _strip_fillers
            text = _strip_fillers(text)
        return Transcript(
            text=text,
            model=self.settings.stt_model,
            language_codes=languages if languages else [lang_code] if lang_code else ["vie"],
            mode=mode,
            latency_ms=latency_ms,
        )

    @property
    def active_provider(self) -> str:
        return "elevenlabs"

    @property
    def active_model(self) -> str:
        return self.settings.stt_model

    async def aclose(self) -> None:
        await self.client.aclose()


# Backward-compatible alias
GeminiTranscriber = ElevenLabsTranscriber


class ResilientTranscriber:
    """Combines Gipformer (local SOTA Vietnamese ASR) with ElevenLabs Scribe v2 STT."""

    def __init__(self, settings: Settings, gipformer: Any | None = None, online: ElevenLabsTranscriber | None = None):
        self.settings = settings
        self.online = online or ElevenLabsTranscriber(settings)
        if gipformer is not None:
            self.gipformer = gipformer
        else:
            try:
                from app.stt_gipformer import GipformerTranscriber
                self.gipformer = GipformerTranscriber(settings)
            except Exception:
                self.gipformer = None

    @property
    def configured(self) -> bool:
        backend = self.settings.stt_backend.strip().lower()
        gip_ok = bool(self.gipformer and self.gipformer.configured and self.gipformer.is_available())
        online_ok = bool(self.online and self.online.configured)
        if backend in ("elevenlabs", "gemini"):
            return online_ok
        if backend == "gipformer":
            return gip_ok
        return gip_ok or online_ok

    @property
    def active_provider(self) -> str:
        backend = self.settings.stt_backend.strip().lower()
        if backend in ("elevenlabs", "gemini"):
            return "elevenlabs"
        if backend == "gipformer":
            return "gipformer"
        if self.gipformer is not None and self.gipformer.configured and self.gipformer.is_available():
            return "gipformer"
        return "elevenlabs"

    @property
    def active_model(self) -> str:
        backend = self.settings.stt_backend.strip().lower()
        if backend in ("elevenlabs", "gemini"):
            return self.settings.stt_model
        if backend == "gipformer":
            return getattr(self.gipformer, "model_name", "gipformer1.5-68M-rnnt")
        if self.gipformer is not None and self.gipformer.configured and self.gipformer.is_available():
            return self.gipformer.model_name
        return self.settings.stt_model

    async def transcribe(self, audio: bytes, mime_type: str, language_codes: list[str] | None = None,
                         mode: str | None = None, vocabulary: list[str] | None = None) -> Transcript:
        backend = self.settings.stt_backend.strip().lower()
        if backend in ("elevenlabs", "gemini"):
            return await self.online.transcribe(audio, mime_type, language_codes, mode, vocabulary)

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
                if not self.online.configured:
                    raise
                logger.warning("Gipformer transcription failed (%s), falling back to ElevenLabs Scribe STT", exc)

        return await self.online.transcribe(audio, mime_type, language_codes, mode, vocabulary)

    async def aclose(self) -> None:
        await self.online.aclose()


def create_transcriber(settings: Settings) -> ElevenLabsTranscriber | ResilientTranscriber:
    online = ElevenLabsTranscriber(settings)
    backend = settings.stt_backend.strip().lower()
    if backend in ("elevenlabs", "gemini"):
        return online
    return ResilientTranscriber(settings, online=online)


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
        detail = data.get("detail") or data.get("error") or {}
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)[:300]
        return str(detail)[:300]
    except Exception:
        return response.text[:300]
